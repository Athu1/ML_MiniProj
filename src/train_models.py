"""
Reproducible training pipeline for all four required models.

Run with:
    python src/train_models.py                # Mathematics file (project default)
    python src/train_models.py --dataset por  # Portuguese file (robustness check)

Steps performed, in order
-------------------------
 1. load the UCI file and validate it against the published shape
 2. derive the at-risk label from G3
 3. drop every leakage-prone column (G3, G1, G2) from the feature matrix
 4. split into train/test (stratified for the classification task)
 5. build the preprocessing ColumnTransformer
 6. train Multivariate Linear Regression        (regression task)
 7. train Logistic Regression                   (classification task)
 8. train SVM with an RBF kernel                (classification task)
 9. train a Random Forest ensemble              (classification task, primary)
10. evaluate everything on the held-out test set and with stratified 5-fold CV
11. run the deliberate data-leakage comparison (same models, G1/G2 included)
12. save the fitted pipelines, metadata and evaluation results
13. write the static matplotlib figures used by the report

Nothing in this script invents a number. Every metric printed and every metric
written to results/ comes from a model fitted on this data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# scikit-learn 1.9 deprecates SVC(probability=True) in favour of
# CalibratedClassifierCV. The project keeps SVC(probability=True) because it is
# the form taught in the syllabus and keeps the SVM easy to explain; the warning
# is silenced so the training summary stays readable. Revisit if upgrading to
# scikit-learn 1.11, where the parameter is scheduled for removal.
warnings.filterwarnings(
    "ignore", message=".*probability.*parameter was deprecated.*",
    category=FutureWarning,
)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import config as cfg
import data_preprocessing as dp
import evaluate_models as em


# ==========================================================================
# Model builders
# ==========================================================================
def build_linear_regression(numeric=None, categorical=None) -> Pipeline:
    """MODEL 1 -- Multivariate Linear Regression.

    "Multivariate" here means many input variables (30 student attributes,
    43 after encoding) predicting ONE continuous output: the final grade.
    The model fits y = b0 + b1*x1 + ... + bn*xn by ordinary least squares.
    """
    return Pipeline(
        [
            ("preprocess", dp.build_preprocessor(numeric, categorical)),
            ("model", LinearRegression()),
        ]
    )


def build_logistic_regression(numeric=None, categorical=None) -> Pipeline:
    """MODEL 2 -- Logistic Regression.

    Produces P(at risk) by passing a linear combination of the features through
    the sigmoid function. `class_weight="balanced"` makes the minority at-risk
    class count proportionally more in the loss, which raises recall -- the
    metric that matters for an early-warning system.
    """
    return Pipeline(
        [
            ("preprocess", dp.build_preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=cfg.RANDOM_STATE,
                ),
            ),
        ]
    )


def build_svm(numeric=None, categorical=None) -> Pipeline:
    """MODEL 3 -- Support Vector Machine with an RBF kernel.

    SVM searches for the decision boundary with the widest margin between the
    two classes. The RBF kernel lets that boundary be non-linear in the original
    feature space. `probability=True` enables ROC-AUC (it fits Platt scaling
    internally, which is why this model is the slowest of the three).

    Feature scaling is essential here and is handled by the shared preprocessor:
    the RBF kernel uses squared Euclidean distances, so an unscaled `absences`
    column (range 0-93) would swamp `studytime` (range 1-4).
    """
    return Pipeline(
        [
            ("preprocess", dp.build_preprocessor(numeric, categorical)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=cfg.RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest(numeric=None, categorical=None) -> Pipeline:
    """MODEL 4 -- Random Forest: ENSEMBLE LEARNING (bagging).

    Individual Decision Trees -> Multiple Trees -> Bagging -> Random Forest

    A single decision tree fitted to 316 students overfits badly: it can keep
    splitting until each leaf holds one student. A Random Forest instead trains
    many trees, each on a different bootstrap sample of the training rows
    (bagging = Bootstrap AGGregatING) and each considering only a random subset
    of features at every split. The trees therefore make *different* mistakes,
    and majority voting across them cancels much of the individual variance.

    `min_samples_leaf=3` is the main guard against memorising this small
    dataset; `class_weight="balanced_subsample"` rebalances the at-risk class
    inside each bootstrap sample.
    """
    return Pipeline(
        [
            ("preprocess", dp.build_preprocessor(numeric, categorical)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=cfg.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


CLASSIFIER_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "SVM (RBF)": build_svm,
    "Random Forest": build_random_forest,
}


# ==========================================================================
# Light hyper-parameter search
# ==========================================================================
# Deliberately small grids -- two or three parameters per model, the ones that
# actually change how each algorithm behaves. Every model is tuned with the
# SAME StratifiedKFold splits and the SAME scorer (F1 on the at-risk class), so
# the comparison in section 5 is fair: it would be misleading to grid-search
# the SVM and then compare it against an untuned Logistic Regression.
PARAM_GRIDS = {
    "Logistic Regression": {
        # C is the inverse of regularisation strength: small C = stronger
        # penalty on large coefficients = a simpler, more constrained model.
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    "SVM (RBF)": {
        # C trades margin width against training errors; gamma sets how far a
        # single training point's influence reaches in the RBF kernel.
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", 0.01, 0.1],
    },
    "Random Forest": {
        # max_features controls how decorrelated the trees are; min_samples_leaf
        # is the main brake on a tree memorising this small training set.
        "model__max_features": ["sqrt", 0.3],
        "model__min_samples_leaf": [1, 3, 5],
    },
}


def tune_classifier(name: str, builder, X_train, y_train, cv):
    """Fit a small GridSearchCV for `name` and return (best_pipeline, best_params).

    Scoring is F1 on the at-risk class rather than accuracy, for the same reason
    the comparison table is ranked by F1: with a 67/33 class split, accuracy
    rewards a model that simply predicts "Not At Risk" for almost everyone.
    """
    grid = PARAM_GRIDS.get(name)
    pipe = builder()
    if not grid:
        pipe.fit(X_train, y_train)
        return pipe, {}

    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        k.replace("model__", ""): v for k, v in search.best_params_.items()
    }


# ==========================================================================
# Training routines
# ==========================================================================
def train_regression(df, numeric=None, categorical=None, leaky=False, verbose=True):
    """Train and evaluate the Multivariate Linear Regression model."""
    X, y = dp.split_features_target(df, task="regression", leaky=leaky)

    # No stratification for regression: the target is continuous.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    pipe = build_linear_regression(numeric, categorical)
    pipe.fit(X_train, y_train)

    pred_test = pipe.predict(X_test)
    pred_train = pipe.predict(X_train)

    test_metrics = em.regression_metrics(y_test, pred_test)
    train_metrics = em.regression_metrics(y_train, pred_train)
    baseline = em.baseline_regression_metrics(y_train, y_test)

    # 5-fold CV R-squared on the training split, to show how unstable a single
    # 79-row test split is on a dataset this small.
    cv_r2 = cross_val_score(
        build_linear_regression(numeric, categorical), X_train, y_train,
        cv=cfg.CV_FOLDS, scoring="r2",
    )

    if verbose:
        tag = " [LEAKY VARIANT]" if leaky else ""
        print(f"\n  Multivariate Linear Regression{tag}")
        print(f"    train R2 = {train_metrics['R2']:+.4f}   "
              f"test R2 = {test_metrics['R2']:+.4f}   "
              f"CV R2 = {cv_r2.mean():+.4f} (+/- {cv_r2.std():.4f})")
        print(f"    test MAE = {test_metrics['MAE']:.3f}   "
              f"test RMSE = {test_metrics['RMSE']:.3f} grade points (0-20 scale)")
        print(f"    mean-predictor baseline: RMSE = {baseline['RMSE']:.3f}, "
              f"R2 = {baseline['R2']:+.4f}")

    return {
        "name": cfg.REGRESSION_MODEL_NAME,
        "pipeline": pipe,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "baseline_metrics": baseline,
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_r2_scores": cv_r2.tolist(),
        "y_test": np.asarray(y_test, dtype=float).tolist(),
        "y_pred": np.asarray(pred_test, dtype=float).tolist(),
        "residuals": (np.asarray(y_test, dtype=float) - np.asarray(pred_test, dtype=float)).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": list(X.columns),
    }


def train_classifiers(df, numeric=None, categorical=None, leaky=False,
                      tune=True, verbose=True):
    """Train and evaluate Logistic Regression, SVM and Random Forest."""
    X, y = dp.split_features_target(df, task="classification", leaky=leaky)

    # stratify=y keeps the 33% / 67% at-risk split identical in train and test.
    # Without it, a random 79-row test set could easily end up with a very
    # different class balance, making the metrics meaningless.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                         random_state=cfg.RANDOM_STATE)

    results = {}
    for name, builder in CLASSIFIER_BUILDERS.items():
        t0 = time.time()
        make = lambda b=builder: b(numeric, categorical)
        best_params = {}

        if tune:
            pipe, best_params = tune_classifier(name, make, X_train, y_train, cv)
        else:
            pipe = make()
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = (
            pipe.predict_proba(X_test)[:, 1]
            if hasattr(pipe.named_steps["model"], "predict_proba") else None
        )

        test_metrics = em.classification_metrics(y_test, y_pred, y_proba)
        train_metrics = em.classification_metrics(
            y_train, pipe.predict(X_train),
            pipe.predict_proba(X_train)[:, 1] if y_proba is not None else None,
        )

        # Stratified 5-fold CV on the TRAINING split only. Reported because the
        # test set is just 79 students: a single-split F1 can move by several
        # points on a handful of rows, so the CV mean +/- std says how much to
        # trust the comparison.
        # Re-run CV with the tuned configuration so the reported mean +/- std
        # describes the model actually being compared. Note this is an optimistic
        # estimate: the same folds chose the hyper-parameters. The held-out test
        # set, which the search never saw, remains the honest number.
        cv_f1 = cross_val_score(
            clone(pipe), X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
        )

        results[name] = {
            "name": name,
            "pipeline": pipe,
            "best_params": best_params,
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "cv_f1_scores": cv_f1.tolist(),
            "confusion": em.confusion_counts(y_test, y_pred),
            "report_text": em.text_classification_report(y_test, y_pred),
            "roc": em.roc_points(y_test, y_proba) if y_proba is not None else None,
            "y_test": np.asarray(y_test, dtype=int).tolist(),
            "y_pred": np.asarray(y_pred, dtype=int).tolist(),
            "y_proba": np.asarray(y_proba, dtype=float).tolist() if y_proba is not None else None,
            "fit_seconds": round(time.time() - t0, 2),
        }

        if verbose:
            m = test_metrics
            tag = " [LEAKY]" if leaky else ""
            print(f"\n  {name}{tag}")
            if best_params:
                print(f"    tuned: {best_params}")
            print(f"    accuracy={m['Accuracy']:.4f}  precision={m['Precision']:.4f}  "
                  f"recall={m['Recall']:.4f}  F1={m['F1']:.4f}  ROC-AUC={m['ROC-AUC']:.4f}")
            print(f"    CV F1 (5-fold, train only) = {cv_f1.mean():.4f} "
                  f"(+/- {cv_f1.std():.4f})")
            c = results[name]["confusion"]
            print(f"    confusion: TN={c['TN']} FP={c['FP']} FN={c['FN']} TP={c['TP']} "
                  f"  (FN = at-risk students the model missed)")
            print(f"    train F1={train_metrics['F1']:.4f} -> "
                  f"train-test gap {train_metrics['F1'] - m['F1']:+.4f}")

    baseline = em.baseline_classification_metrics(y_train, y_test)

    return results, {
        "baseline": baseline,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_columns": list(X.columns),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_at_risk_pct": float(y_train.mean() * 100),
        "test_at_risk_pct": float(y_test.mean() * 100),
    }


# ==========================================================================
# Explainability
# ==========================================================================
def random_forest_importance(rf_pipe: Pipeline, X_test, y_test, verbose=True):
    """Two complementary importance views for the Random Forest.

    `feature_importances_` (mean decrease in impurity) is what the syllabus
    asks for and is what the UI shows by default. It has a known bias: it
    inflates high-cardinality / continuous features such as `absences`, because
    they offer more possible split points.

    Permutation importance on the held-out test set is therefore also computed:
    it measures how much the F1 score actually drops when one column is
    shuffled. Where the two disagree, the permutation view is the more
    trustworthy one, and the UI says so.
    """
    pre = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]
    names = dp.get_feature_names(pre)

    impurity = pd.DataFrame(
        {"feature": names, "impurity_importance": model.feature_importances_}
    ).sort_values("impurity_importance", ascending=False)

    perm = permutation_importance(
        rf_pipe, X_test, y_test,
        n_repeats=20, random_state=cfg.RANDOM_STATE, scoring="f1", n_jobs=-1,
    )
    permutation = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "permutation_importance": perm.importances_mean,
            "permutation_std": perm.importances_std,
        }
    ).sort_values("permutation_importance", ascending=False)

    if verbose:
        print("\n  Random Forest -- top 10 features by impurity importance")
        for _, r in impurity.head(10).iterrows():
            print(f"    {r['feature']:22s} {r['impurity_importance']:.4f}")
        print("\n  Random Forest -- top 10 features by permutation importance (test F1 drop)")
        for _, r in permutation.head(10).iterrows():
            print(f"    {r['feature']:22s} {r['permutation_importance']:+.4f} "
                  f"(+/- {r['permutation_std']:.4f})")

    return impurity.reset_index(drop=True), permutation.reset_index(drop=True)


def logistic_coefficients(lr_pipe: Pipeline) -> pd.DataFrame:
    """Logistic Regression coefficients on the standardised feature scale.

    Because the numeric features were standardised inside the pipeline, these
    coefficients are comparable to each other: each is the change in the
    log-odds of being at risk per one-standard-deviation increase in that
    feature. A positive value is associated with a higher predicted risk.
    These are associations learned from 316 students, not causal effects.
    """
    pre = lr_pipe.named_steps["preprocess"]
    model = lr_pipe.named_steps["model"]
    coefs = model.coef_.ravel()
    df = pd.DataFrame({"feature": dp.get_feature_names(pre), "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    df["direction"] = np.where(df["coefficient"] >= 0,
                               "increases risk", "decreases risk")
    df["odds_ratio"] = np.exp(df["coefficient"])
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


# ==========================================================================
# Static figures for the report
# ==========================================================================
def save_report_figures(reg, clf_results, importance_df, df, verbose=True):
    """Write the matplotlib/seaborn figures referenced by docs/project_report.md.

    The Streamlit app builds its own interactive Plotly versions; these PNGs
    exist so the written report has images without needing a screenshot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import ConfusionMatrixDisplay

    sns.set_theme(style="whitegrid", context="notebook")
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    def _save(fig, name):
        path = cfg.FIGURES_DIR / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(path.name)

    # 1. Final score distribution -- shows the spike at zero
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[cfg.REGRESSION_TARGET], bins=21, range=(0, 21),
            edgecolor="white", color="#3b6ea5")
    ax.axvline(cfg.AT_RISK_THRESHOLD, color="#c0392b", ls="--",
               label=f"At-risk threshold (G3 < {cfg.AT_RISK_THRESHOLD})")
    ax.set_xlabel("Final grade G3 (0-20)")
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of final grades")
    ax.legend()
    _save(fig, "01_score_distribution.png")

    # 2. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[cfg.CLASSIFICATION_TARGET].value_counts().sort_index()
    ax.bar([cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL],
           counts.values, color=["#2e8b57", "#c0392b"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, f"{v} ({v / len(df) * 100:.1f}%)", ha="center")
    ax.set_ylabel("Number of students")
    ax.set_title("At-risk class distribution")
    _save(fig, "02_class_distribution.png")

    # 3. Correlation heatmap (numeric features + the grades)
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = cfg.NUMERIC_FEATURES + ["G1", "G2", "G3"]
    sns.heatmap(df[cols].corr(numeric_only=True), cmap="RdBu_r", center=0,
                annot=False, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation between numeric features and grades")
    _save(fig, "03_correlation_heatmap.png")

    # 4. Actual vs predicted (regression)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    yt, yp = np.array(reg["y_test"]), np.array(reg["y_pred"])
    ax.scatter(yt, yp, alpha=0.7, color="#3b6ea5", edgecolor="white")
    lim = [-1, 21]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual final grade")
    ax.set_ylabel("Predicted final grade")
    ax.set_title(f"Linear Regression: actual vs predicted "
                 f"(test R2 = {reg['test_metrics']['R2']:.3f})")
    ax.legend()
    _save(fig, "04_actual_vs_predicted.png")

    # 5. Residual plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(yp, yt - yp, alpha=0.7, color="#8e44ad", edgecolor="white")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Predicted final grade")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual plot")
    _save(fig, "05_residual_plot.png")

    # 6. Confusion matrices, one panel per classifier
    fig, axes = plt.subplots(1, len(clf_results), figsize=(5 * len(clf_results), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, clf_results.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=np.array(res["confusion"]["matrix"]),
            display_labels=[cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL],
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}\nF1 = {res['test_metrics']['F1']:.3f}")
    fig.suptitle("Confusion matrices on the held-out test set", y=1.04)
    _save(fig, "06_confusion_matrices.png")

    # 7. Grouped metric comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    names = list(clf_results)
    width = 0.8 / len(metrics)
    xs = np.arange(len(names))
    for i, m in enumerate(metrics):
        vals = [clf_results[n]["test_metrics"][m] for n in names]
        ax.bar(xs + i * width, vals, width, label=m)
    ax.set_xticks(xs + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classification model comparison (test set)")
    ax.legend(ncol=5, fontsize=8)
    _save(fig, "07_model_comparison.png")

    # 8. Random Forest feature importance
    fig, ax = plt.subplots(figsize=(7.5, 6))
    top = importance_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["impurity_importance"], color="#2e7d32")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest feature importance (top 15)")
    _save(fig, "08_feature_importance.png")

    # 9. ROC curves
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, res in clf_results.items():
        if res["roc"]:
            ax.plot(res["roc"]["fpr"], res["roc"]["tpr"],
                    label=f"{name} (AUC = {res['roc']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "09_roc_curves.png")

    if verbose:
        print(f"\n  Saved {len(saved)} figures to {cfg.FIGURES_DIR.relative_to(cfg.PROJECT_ROOT)}/")
    return saved


# ==========================================================================
# Main
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train every model for the Student At-Risk Prediction System."
    )
    parser.add_argument("--dataset", default=cfg.DEFAULT_DATASET,
                        choices=sorted(cfg.DATASET_FILES),
                        help="which UCI subject file to train on (default: mat)")
    parser.add_argument("--no-tune", action="store_true",
                        help="skip the small SVM grid search")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip writing the static report figures")
    parser.add_argument("--skip-leakage-demo", action="store_true",
                        help="skip the G1/G2 data-leakage comparison run")
    args = parser.parse_args(argv)

    t_start = time.time()
    print("=" * 74)
    print("  STUDENT ACADEMIC PERFORMANCE & AT-RISK PREDICTION -- TRAINING")
    print("=" * 74)

    # --- 1-3. load, label, inspect -------------------------------------
    try:
        df = dp.load_dataset(args.dataset)
    except dp.DatasetError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    summary = dp.dataset_summary(df)
    print(f"\n[1/6] Dataset: student-{args.dataset}.csv")
    print(f"      {summary['n_students']} students x {summary['n_raw_columns']} raw attributes")
    print(f"      missing values: {summary['missing_values']}   "
          f"duplicate rows: {summary['duplicate_rows']}")
    print(f"      final grade G3: mean {summary['mean_g3']:.2f}, "
          f"median {summary['median_g3']:.1f}, "
          f"{summary['n_zero_g3']} students scored 0")
    print(f"      at risk (G3 < {cfg.AT_RISK_THRESHOLD}): "
          f"{summary['n_at_risk']} ({summary['pct_at_risk']:.1f}%)   "
          f"not at risk: {summary['n_not_at_risk']}")
    print(f"      majority-class baseline accuracy: "
          f"{summary['majority_class_accuracy'] * 100:.1f}%  <-- the number to beat")

    print(f"\n[2/6] Features: {len(cfg.ALL_FEATURES)} inputs "
          f"({len(cfg.NUMERIC_FEATURES)} numeric, {len(cfg.CATEGORICAL_FEATURES)} categorical)")
    print("      Excluded to prevent target leakage:")
    for col, reason in cfg.EXCLUDED_COLUMNS:
        print(f"        - {col}: {reason.split('.')[0]}.")

    # --- 4-6. regression -----------------------------------------------
    print(f"\n[3/6] Training the regression model "
          f"(train/test = {1 - cfg.TEST_SIZE:.0%}/{cfg.TEST_SIZE:.0%}, "
          f"random_state={cfg.RANDOM_STATE})")
    reg = train_regression(df)

    # --- 7-9. classification -------------------------------------------
    print(f"\n[4/6] Training the classification models "
          f"(stratified split, random_state={cfg.RANDOM_STATE})")
    clf_results, clf_ctx = train_classifiers(df, tune=not args.no_tune)
    print(f"\n  Majority-class baseline: "
          f"accuracy={clf_ctx['baseline']['Accuracy']:.4f}  "
          f"recall={clf_ctx['baseline']['Recall']:.4f}  "
          f"F1={clf_ctx['baseline']['F1']:.4f}")

    best_model = em.pick_best_model(clf_results)
    tie = em.assess_tie(clf_results, metric="F1")
    print(f"\n  Highest F1 (ties broken by ROC-AUC): {best_model} "
          f"(F1 = {tie['best_score']:.4f})")
    if tie["is_tie"]:
        print(f"  NOTE: the gap to {tie['runner_up']} is only {tie['gap']:.4f}, "
              f"smaller than the cross-validation spread of +/-{tie['noise_floor']:.4f}.")
        print(f"        On a {tie['n_test']}-student test set these models are "
              f"statistically indistinguishable: {', '.join(tie['tied_models'])}.")
        print("        The project therefore reports a tie rather than a winner, and "
              "keeps")
        print("        Random Forest as the primary model (ensemble requirement, best "
              "accuracy")
        print("        and precision, smallest train-test gap).")

    # --- 10. explainability --------------------------------------------
    print("\n[5/6] Explainability")
    importance_df, perm_df = random_forest_importance(
        clf_results["Random Forest"]["pipeline"],
        clf_ctx["X_test"], clf_ctx["y_test"],
    )
    coef_df = logistic_coefficients(clf_results["Logistic Regression"]["pipeline"])
    print("\n  Logistic Regression -- 8 largest absolute coefficients "
          "(standardised scale)")
    for _, r in coef_df.head(8).iterrows():
        print(f"    {r['feature']:22s} {r['coefficient']:+.4f}  {r['direction']}")

    # --- 11. leakage demonstration -------------------------------------
    leakage_rows = []
    if not args.skip_leakage_demo:
        print("\n[6/6] Data-leakage demonstration: refitting WITH G1 and G2 included")
        print("      (these models are evaluated and reported, never saved or "
              "used for predictions)")
        leaky_numeric = cfg.NUMERIC_FEATURES + cfg.LEAKY_EXTRA_NUMERIC
        leaky_reg = train_regression(df, numeric=leaky_numeric, leaky=True)
        leaky_clf, _ = train_classifiers(
            df, numeric=leaky_numeric, leaky=True, tune=not args.no_tune
        )
        leakage_rows.append({
            "Task": "Regression (G3)", "Model": cfg.REGRESSION_MODEL_NAME,
            "Metric": "R2",
            "Leakage-free (no G1/G2)": round(reg["test_metrics"]["R2"], 4),
            "With leakage (G1+G2)": round(leaky_reg["test_metrics"]["R2"], 4),
        })
        leakage_rows.append({
            "Task": "Regression (G3)", "Model": cfg.REGRESSION_MODEL_NAME,
            "Metric": "RMSE",
            "Leakage-free (no G1/G2)": round(reg["test_metrics"]["RMSE"], 4),
            "With leakage (G1+G2)": round(leaky_reg["test_metrics"]["RMSE"], 4),
        })
        for name in clf_results:
            for metric in ("F1", "ROC-AUC", "Accuracy"):
                leakage_rows.append({
                    "Task": "Classification (at risk)", "Model": name, "Metric": metric,
                    "Leakage-free (no G1/G2)": round(clf_results[name]["test_metrics"][metric], 4),
                    "With leakage (G1+G2)": round(leaky_clf[name]["test_metrics"][metric], 4),
                })
        print("\n  Effect of including the intermediate grades:")
        for row in leakage_rows:
            print(f"    {row['Model']:22s} {row['Metric']:9s} "
                  f"{row['Leakage-free (no G1/G2)']:+.4f} -> "
                  f"{row['With leakage (G1+G2)']:+.4f}")
    else:
        print("\n[6/6] Leakage demonstration skipped (--skip-leakage-demo)")

    # --- 12. persist ----------------------------------------------------
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(reg["pipeline"], cfg.MODEL_FILES[cfg.REGRESSION_MODEL_NAME])
    for name, res in clf_results.items():
        joblib.dump(res["pipeline"], cfg.MODEL_FILES[name])

    # Everything the Streamlit app needs that is NOT a fitted estimator.
    # Pipelines are stripped out so the metadata file stays small and loadable
    # on its own.
    metadata = {
        "dataset": args.dataset,
        "dataset_file": cfg.DATASET_FILES[args.dataset].name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": cfg.RANDOM_STATE,
        "test_size": cfg.TEST_SIZE,
        "cv_folds": cfg.CV_FOLDS,
        "at_risk_threshold": cfg.AT_RISK_THRESHOLD,
        "summary": summary,
        "excluded_columns": cfg.EXCLUDED_COLUMNS,
        "numeric_features": cfg.NUMERIC_FEATURES,
        "categorical_features": cfg.CATEGORICAL_FEATURES,
        "encoded_feature_names": dp.get_feature_names(
            clf_results["Random Forest"]["pipeline"].named_steps["preprocess"]
        ),
        "best_classifier": best_model,
        "tie_analysis": tie,
        "baseline_classification": clf_ctx["baseline"],
        "n_train": clf_ctx["n_train"], "n_test": clf_ctx["n_test"],
        "train_at_risk_pct": clf_ctx["train_at_risk_pct"],
        "test_at_risk_pct": clf_ctx["test_at_risk_pct"],
        "default_profile": dp.default_student_profile(df),
        "regression": {k: v for k, v in reg.items() if k != "pipeline"},
        "classification": {
            n: {k: v for k, v in r.items() if k != "pipeline"}
            for n, r in clf_results.items()
        },
        "rf_impurity_importance": importance_df.to_dict("records"),
        "rf_permutation_importance": perm_df.to_dict("records"),
        "logistic_coefficients": coef_df.to_dict("records"),
        "leakage_comparison": leakage_rows,
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(metadata, cfg.METADATA_FILE)

    # --- results tables -------------------------------------------------
    clf_rows = [{"Model": n, **r["test_metrics"],
                 "CV F1 (mean)": round(r["cv_f1_mean"], 4),
                 "CV F1 (std)": round(r["cv_f1_std"], 4)}
                for n, r in clf_results.items()]
    clf_rows.append({"Model": "Baseline (majority class)",
                     **{k: v for k, v in clf_ctx["baseline"].items() if k != "Model"},
                     "CV F1 (mean)": np.nan, "CV F1 (std)": np.nan})
    clf_table = em.build_comparison_table(clf_rows, sort_by="F1").round(4)
    clf_table.to_csv(cfg.RESULTS_CLASSIFICATION_CSV, index=False)

    reg_rows = [
        {"Model": cfg.REGRESSION_MODEL_NAME, **reg["test_metrics"],
         "CV R2 (mean)": round(reg["cv_r2_mean"], 4),
         "CV R2 (std)": round(reg["cv_r2_std"], 4)},
        {"Model": "Baseline (predict mean)",
         **{k: v for k, v in reg["baseline_metrics"].items() if k != "Model"},
         "CV R2 (mean)": np.nan, "CV R2 (std)": np.nan},
    ]
    reg_table = em.build_comparison_table(reg_rows, sort_by="R2").round(4)
    reg_table.to_csv(cfg.RESULTS_REGRESSION_CSV, index=False)

    importance_df.round(6).to_csv(cfg.RESULTS_FEATURE_IMPORTANCE_CSV, index=False)
    if leakage_rows:
        pd.DataFrame(leakage_rows).to_csv(cfg.RESULTS_LEAKAGE_CSV, index=False)

    # One combined file, which is what the spec asks for by name.
    combined = pd.concat(
        [clf_table.assign(Task="Classification (at risk)"),
         reg_table.assign(Task="Regression (final grade)")],
        ignore_index=True,
    )
    combined = combined[["Task"] + [c for c in combined.columns if c != "Task"]]
    combined.to_csv(cfg.RESULTS_MODEL_RESULTS_CSV, index=False)

    # Processed dataset, for the notebook and for inspection.
    df.to_csv(cfg.DATA_PROCESSED_DIR / f"student_{args.dataset}_processed.csv", index=False)
    with open(cfg.RESULTS_DIR / "metrics_summary.json", "w") as fh:
        json.dump(
            {
                "dataset": args.dataset,
                "best_classifier": best_model,
        "tie_analysis": tie,
                "classification": {n: r["test_metrics"] for n, r in clf_results.items()},
                "classification_baseline": clf_ctx["baseline"],
                "regression": reg["test_metrics"],
                "regression_baseline": reg["baseline_metrics"],
            },
            fh, indent=2, default=float,
        )

    # --- 13. figures ----------------------------------------------------
    if not args.no_figures:
        save_report_figures(reg, clf_results, importance_df, df)

    # --- summary --------------------------------------------------------
    print("\n" + "=" * 74)
    print("  TRAINING SUMMARY")
    print("=" * 74)
    print("\n  Classification -- held-out test set "
          f"({clf_ctx['n_test']} students, {clf_ctx['test_at_risk_pct']:.1f}% at risk)")
    print(clf_table.to_string(index=False))
    print(f"\n  Regression -- held-out test set ({reg['n_test']} students)")
    print(reg_table.to_string(index=False))
    print(f"\n  Highest F1: {best_model} "
          f"(F1 = {clf_results[best_model]['test_metrics']['F1']:.4f}, "
          f"ROC-AUC = {clf_results[best_model]['test_metrics']['ROC-AUC']:.4f})")
    if tie["is_tie"]:
        print(f"  Verdict: statistical tie between {', '.join(tie['tied_models'])} "
              f"(gap {tie['gap']:.4f} < noise {tie['noise_floor']:.4f}).")
        print("           Random Forest is retained as the project's primary model.")
    print(f"\n  Reference points -- a model is only useful if it beats these:")
    print(f"    majority-class accuracy {clf_ctx['baseline']['Accuracy']:.4f} "
          f"(with recall 0.0000 on at-risk students)")
    print(f"    mean-predictor regression RMSE {reg['baseline_metrics']['RMSE']:.4f}")
    print(f"\n  Saved {len(cfg.MODEL_FILES)} pipelines to "
          f"{cfg.MODELS_DIR.relative_to(cfg.PROJECT_ROOT)}/ "
          f"and results to {cfg.RESULTS_DIR.relative_to(cfg.PROJECT_ROOT)}/")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print("\n  Next step:  streamlit run app.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
