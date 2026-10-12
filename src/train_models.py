"""
Reproducible training pipeline for all four models.

Run with:
    python src/train_models.py

Steps performed, in order
-------------------------
 1. load the CarDekho file and validate it (shape, checksum)
 2. clean it: coerce max_power to a number, derive age and brand, drop the exact
    duplicate listings, derive the premium label from the price
 3. keep the 11 allowed features; price, label, name and year are stripped out
 4. split into train/test (stratified for the classification task)
 5. build the preprocessing ColumnTransformer
 6. train Multivariate Linear Regression on log-price   (regression task)
 7. train Logistic Regression                           (classification task)
 8. train an RBF SVM                                    (classification task)
 9. train a Random Forest ensemble                      (classification task)
10. evaluate on the held-out test set and with stratified 5-fold CV
11. choose a decision threshold from OUT-OF-FOLD training predictions
12. check probability calibration
13. run the duplicate-leakage comparison (same models, duplicates kept)
14. save the fitted pipelines, metadata and result tables
15. write the static figures used by the report

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import config as cfg
import data_preprocessing as dp
import evaluate_models as em


# ==========================================================================
# Model builders
# ==========================================================================
def build_linear_regression() -> TransformedTargetRegressor:
    """MODEL 1 -- Multivariate Linear Regression, fitted to log-price.

    "Multivariate" means many input variables (11 car attributes, 37 columns
    after encoding) predicting ONE continuous output: the asking price. The
    model fits  log(price) = b0 + b1*x1 + ... + bn*xn  by ordinary least squares.

    Why log-price: prices are heavily right-skewed (median 4 lakh, maximum 100
    lakh, skewness above 5), and a car's price tends to change by a PERCENTAGE
    per year of age rather than by a fixed number of rupees. A straight line fits
    a percentage relationship on the log scale, and a log target also stops a
    few luxury cars dominating the squared-error loss. The wrapper converts
    predictions back to rupees, so every metric downstream is in rupees.
    """
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [("preprocess", dp.build_preprocessor()),
             ("model", LinearRegression())]
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def build_logistic_regression() -> Pipeline:
    """MODEL 2 -- Logistic Regression.

    Produces P(premium) by passing a linear combination of the features through
    the sigmoid function. No class re-weighting is used: unlike an early-warning
    problem, nothing here makes a missed premium car worse than a false alarm,
    so there is no reason to distort the loss toward the minority class.
    """
    return Pipeline(
        [("preprocess", dp.build_preprocessor()),
         ("model", LogisticRegression(max_iter=2000, random_state=cfg.RANDOM_STATE))]
    )


def build_svm() -> Pipeline:
    """MODEL 3 -- Support Vector Machine with an RBF kernel.

    SVM searches for the decision boundary with the widest margin between the
    two classes. The RBF kernel lets that boundary be non-linear in the original
    feature space. `probability=True` enables ROC-AUC (it fits Platt scaling
    internally, which is why this is the slowest model to fit).

    Feature scaling is essential and is handled by the shared preprocessor: the
    RBF kernel uses squared Euclidean distances, so an unscaled `km_driven`
    (hundreds of thousands) would swamp `seats` (2 to 14).
    """
    return Pipeline(
        [("preprocess", dp.build_preprocessor()),
         ("model", SVC(kernel="rbf", probability=True,
                       random_state=cfg.RANDOM_STATE))]
    )


def build_random_forest() -> Pipeline:
    """MODEL 4 -- Random Forest: ENSEMBLE LEARNING (bagging).

    Individual Decision Trees -> Multiple Trees -> Bagging -> Random Forest

    A single decision tree fitted to about 5,500 cars overfits: it can keep
    splitting until each leaf holds one car. A Random Forest instead trains many
    trees, each on a different bootstrap sample of the training rows (bagging =
    Bootstrap AGGregatING) and each considering only a random subset of features
    at every split. The trees therefore make *different* mistakes, and majority
    voting across them cancels much of the individual variance.
    """
    return Pipeline(
        [("preprocess", dp.build_preprocessor()),
         ("model", RandomForestClassifier(
             n_estimators=400, min_samples_leaf=3, max_features="sqrt",
             random_state=cfg.RANDOM_STATE, n_jobs=-1))]
    )


CLASSIFIER_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "SVM (RBF)": build_svm,
    "Random Forest": build_random_forest,
}

# ==========================================================================
# Light hyper-parameter search
# ==========================================================================
# Deliberately small grids -- the parameters that actually change how each
# algorithm behaves. Every model is tuned with the SAME StratifiedKFold splits
# and the SAME scorer (F1 on the premium class), so the comparison is fair: it
# would be misleading to grid-search the SVM and then compare it against an
# untuned Logistic Regression.
PARAM_GRIDS = {
    "Logistic Regression": {
        # C is the inverse of regularisation strength: small C = stronger penalty.
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "SVM (RBF)": {
        # C trades margin width against training errors; gamma sets how far a
        # single training point's influence reaches in the RBF kernel.
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", 0.01, 0.1],
    },
    "Random Forest": {
        # max_features controls how decorrelated the trees are; min_samples_leaf
        # is the main brake on a tree memorising the training set.
        "model__max_features": ["sqrt", 0.3],
        "model__min_samples_leaf": [1, 3, 5],
    },
}


def tune_classifier(name, builder, X_train, y_train, cv):
    """Fit a small GridSearchCV for `name` and return (best_pipeline, best_params).

    Scoring is F1 on the premium class rather than accuracy: with about 30%
    premium cars, accuracy rewards a model that answers "Standard" too often.
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
# Out-of-fold probabilities (for choosing the decision threshold)
# ==========================================================================
def out_of_fold_probabilities(pipe, X_train, y_train, cv):
    """P(premium) for each training car, predicted by a model that did not see it.

    This exists so the decision threshold can be chosen without touching the
    test set. `cross_val_predict` refits the pipeline on each fold and predicts
    the held-out fold, so every probability returned here comes from a model
    fitted without that car. A cut-off chosen from these numbers and then
    applied, unchanged, to the test set keeps the test set clean -- picking it on
    the test set would be tuning a parameter on the data used to report the
    result.
    """
    return cross_val_predict(
        clone(pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]


# ==========================================================================
# Training routines
# ==========================================================================
def train_regression(df, verbose=True, label=""):
    """Train and evaluate the Multivariate Linear Regression model."""
    X, y = dp.split_features_target(df, task="regression")

    # No stratification for regression: the target is continuous.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    model = build_linear_regression()
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)

    # Metrics are computed in lakh rupees so MAE and RMSE read naturally.
    lakh = float(cfg.LAKH)
    test_metrics = em.regression_metrics(y_test / lakh, pred_test / lakh)
    train_metrics = em.regression_metrics(y_train / lakh, pred_train / lakh)
    # R-squared on the log scale -- the scale the model was actually fitted on.
    test_metrics["R2_log"] = float(
        em.regression_metrics(np.log(y_test), np.log(pred_test))["R2"]
    )
    baseline = em.baseline_regression_metrics(y_train / lakh, y_test / lakh)

    cv_r2 = cross_val_score(
        build_linear_regression(), X_train, y_train,
        cv=cfg.CV_FOLDS, scoring="r2",
    )

    if verbose:
        print(f"\n  Multivariate Linear Regression{label}")
        print(f"    train R2 = {train_metrics['R2']:+.4f}   "
              f"test R2 = {test_metrics['R2']:+.4f}   "
              f"CV R2 = {cv_r2.mean():+.4f} (+/- {cv_r2.std():.4f})")
        print(f"    test R2 on log-price = {test_metrics['R2_log']:+.4f}")
        print(f"    test MAE = {test_metrics['MAE']:.3f} lakh   "
              f"RMSE = {test_metrics['RMSE']:.3f} lakh   "
              f"median error = {test_metrics['MedAPE']:.1f}% of the price")
        print(f"    mean-predictor baseline: RMSE = {baseline['RMSE']:.3f} lakh, "
              f"R2 = {baseline['R2']:+.4f}")

    return {
        "name": cfg.REGRESSION_MODEL_NAME,
        "pipeline": model,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "baseline_metrics": baseline,
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_r2_scores": cv_r2.tolist(),
        "y_test": (np.asarray(y_test, dtype=float) / lakh).tolist(),
        "y_pred": (np.asarray(pred_test, dtype=float) / lakh).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": list(X.columns),
    }


def train_classifiers(df, tune=True, verbose=True, label=""):
    """Train and evaluate Logistic Regression, SVM and Random Forest."""
    X, y = dp.split_features_target(df, task="classification")

    # stratify=y keeps the minority premium share identical in train and test.
    # Without it a random test set could end up with a noticeably different
    # class balance, making the metrics hard to compare.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y,
    )
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                         random_state=cfg.RANDOM_STATE)

    results = {}
    for name, builder in CLASSIFIER_BUILDERS.items():
        t0 = time.time()
        best_params = {}
        if tune:
            pipe, best_params = tune_classifier(name, builder, X_train, y_train, cv)
        else:
            pipe = builder()
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        test_metrics = em.classification_metrics(y_test, y_pred, y_proba)
        train_metrics = em.classification_metrics(
            y_train, pipe.predict(X_train), pipe.predict_proba(X_train)[:, 1]
        )

        # Re-run CV with the tuned configuration so the reported mean +/- std
        # describes the model actually being compared. This is an optimistic
        # estimate -- the same folds chose the hyper-parameters -- so the
        # held-out test set, which the search never saw, remains the honest
        # number.
        cv_f1 = cross_val_score(clone(pipe), X_train, y_train, cv=cv,
                                scoring="f1", n_jobs=-1)

        # ---- decision threshold, chosen WITHOUT the test set ----------
        oof = out_of_fold_probabilities(pipe, X_train, y_train, cv)
        threshold_info = em.choose_threshold(y_train, oof)
        tuned_metrics = em.metrics_at_threshold(
            y_test, y_proba, threshold_info["threshold"]
        )
        sweep = em.threshold_sweep(y_train, oof)
        calibration = em.calibration_report(y_test, y_proba)
        calibration["summary"] = em.describe_calibration(calibration)
        from sklearn.metrics import brier_score_loss
        calibration["oof_brier"] = float(brier_score_loss(y_train, oof))

        results[name] = {
            "name": name,
            "pipeline": pipe,
            "best_params": best_params,
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "cv_f1_scores": cv_f1.tolist(),
            "threshold": threshold_info,
            "tuned_metrics": tuned_metrics,
            "threshold_sweep": sweep,
            "threshold_curve": em.threshold_curve(y_test, y_proba),
            "calibration": calibration,
            "confusion": em.confusion_counts(y_test, y_pred),
            "report_text": em.text_classification_report(y_test, y_pred),
            "roc": em.roc_points(y_test, y_proba),
            "y_test": np.asarray(y_test, dtype=int).tolist(),
            "y_pred": np.asarray(y_pred, dtype=int).tolist(),
            "y_proba": np.asarray(y_proba, dtype=float).tolist(),
            "fit_seconds": round(time.time() - t0, 2),
        }

        if verbose:
            m = test_metrics
            print(f"\n  {name}{label}")
            if best_params:
                print(f"    tuned: {best_params}")
            print(f"    accuracy={m['Accuracy']:.4f}  precision={m['Precision']:.4f}  "
                  f"recall={m['Recall']:.4f}  F1={m['F1']:.4f}  "
                  f"ROC-AUC={m['ROC-AUC']:.4f}")
            print(f"    CV F1 (5-fold, train only) = {cv_f1.mean():.4f} "
                  f"(+/- {cv_f1.std():.4f})")
            c = results[name]["confusion"]
            print(f"    confusion: TN={c['TN']} FP={c['FP']} FN={c['FN']} TP={c['TP']}")
            print(f"    train F1={train_metrics['F1']:.4f} -> "
                  f"train-test gap {train_metrics['F1'] - m['F1']:+.4f}")
            tm = tuned_metrics
            print(f"    threshold: {cfg.DEFAULT_THRESHOLD:.2f} (default) -> "
                  f"{threshold_info['threshold']:.3f} (chosen on out-of-fold train "
                  f"predictions, FN:FP = {threshold_info['fn_cost_ratio']:g}:1)")
            print(f"      at the tuned cut-off: precision={tm['Precision']:.4f} "
                  f"recall={tm['Recall']:.4f} F1={tm['F1']:.4f} "
                  f"accuracy={tm['Accuracy']:.4f}")
            print(f"    calibration: {calibration['summary']}")
            print(f"      Brier {calibration['brier_score']:.4f} vs "
                  f"{calibration['brier_baseline']:.4f} for a constant base-rate "
                  f"predictor"
                  + ("  <-- WORSE than the constant predictor"
                     if calibration['brier_score'] > calibration['brier_baseline']
                     else ""))

    baseline = em.baseline_classification_metrics(y_train, y_test)
    return results, {
        "baseline": baseline,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_columns": list(X.columns),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_premium_pct": float(y_train.mean() * 100),
        "test_premium_pct": float(y_test.mean() * 100),
    }


# ==========================================================================
# The duplicate-leakage comparison
# ==========================================================================
def duplicates_in_test(df) -> dict:
    """How many test cars have an exact twin in the training set.

    Uses the same split the classification models use. A "twin" is a row whose
    every feature and price match a training row -- the model has effectively
    been shown the answer.
    """
    X, y = dp.split_features_target(df, task="classification")
    key = pd.concat([X, df[cfg.REGRESSION_TARGET]], axis=1)
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    train_keys = set(map(tuple, key.loc[X_train.index].astype(str).values))
    test_rows = key.loc[X_test.index].astype(str).values
    twins = sum(tuple(r) in train_keys for r in test_rows)
    return {"n_test": int(len(X_test)), "n_twins": int(twins),
            "pct_twins": float(twins / len(X_test) * 100)}


def duplicate_leakage_comparison(raw, reg, clf_results, tune, verbose=True):
    """Refit every model with the duplicate listings LEFT IN, and compare.

    The honest models drop the 1,202 exact duplicates before splitting. Here the
    same pipeline is run on the data with them kept, so a car can appear in both
    the training and the test set. The difference in the metrics is the inflation
    that duplicate leakage buys on paper. These models are reported and then
    discarded: never saved, never used for a prediction.
    """
    df_dup = dp.clean(raw, dedupe=False)
    twins = duplicates_in_test(df_dup)
    if verbose:
        print(f"      {twins['n_twins']} of {twins['n_test']} test cars "
              f"({twins['pct_twins']:.1f}%) have an exact twin in the training set")
    reg_dup = train_regression(df_dup, verbose=False)
    clf_dup, _ = train_classifiers(df_dup, tune=tune, verbose=False)

    rows = [
        {"Task": "Regression", "Model": cfg.REGRESSION_MODEL_NAME, "Metric": "R2",
         "Duplicates removed (reported)": round(reg["test_metrics"]["R2"], 4),
         "Duplicates kept": round(reg_dup["test_metrics"]["R2"], 4)},
        {"Task": "Regression", "Model": cfg.REGRESSION_MODEL_NAME, "Metric": "MAE (lakh)",
         "Duplicates removed (reported)": round(reg["test_metrics"]["MAE"], 4),
         "Duplicates kept": round(reg_dup["test_metrics"]["MAE"], 4)},
    ]
    for name in clf_results:
        for metric in ("F1", "ROC-AUC", "Accuracy"):
            rows.append({
                "Task": "Classification", "Model": name, "Metric": metric,
                "Duplicates removed (reported)":
                    round(clf_results[name]["test_metrics"][metric], 4),
                "Duplicates kept": round(clf_dup[name]["test_metrics"][metric], 4),
            })
    return rows, twins


# ==========================================================================
# Explainability
# ==========================================================================
def random_forest_importance(rf_pipe, X_test, y_test, verbose=True):
    """Two complementary importance views for the Random Forest.

    `feature_importances_` (mean decrease in impurity) is what the syllabus
    asks for. It has a known bias: it favours continuous / many-valued features
    because they offer more possible split points. Permutation importance on the
    held-out test set -- how much F1 falls when one ORIGINAL column is shuffled --
    is therefore also computed, and where the two disagree the permutation view
    is the more trustworthy.
    """
    pre = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]
    names = dp.get_feature_names(pre)

    impurity = pd.DataFrame(
        {"feature": names, "impurity_importance": model.feature_importances_}
    ).sort_values("impurity_importance", ascending=False)

    perm = permutation_importance(
        rf_pipe, X_test, y_test, n_repeats=20,
        random_state=cfg.RANDOM_STATE, scoring="f1", n_jobs=-1,
    )
    permutation = pd.DataFrame(
        {"feature": list(X_test.columns),
         "permutation_importance": perm.importances_mean,
         "permutation_std": perm.importances_std}
    ).sort_values("permutation_importance", ascending=False)

    if verbose:
        print("\n  Random Forest -- top 8 by impurity importance (encoded columns)")
        for _, r in impurity.head(8).iterrows():
            print(f"    {r['feature']:26s} {r['impurity_importance']:.4f}")
        print("\n  Random Forest -- top 8 by permutation importance (test F1 drop)")
        for _, r in permutation.head(8).iterrows():
            print(f"    {r['feature']:26s} {r['permutation_importance']:+.4f} "
                  f"(+/- {r['permutation_std']:.4f})")
    return impurity.reset_index(drop=True), permutation.reset_index(drop=True)


def logistic_coefficients(lr_pipe) -> pd.DataFrame:
    """Logistic Regression coefficients on the standardised feature scale.

    Numeric features were standardised inside the pipeline, so these are
    comparable: each is the change in the log-odds of being premium per one
    standard deviation of that feature. A positive value is associated with a
    higher predicted probability of premium. These are associations learned from
    about 5,500 cars, not causal effects.
    """
    pre = lr_pipe.named_steps["preprocess"]
    model = lr_pipe.named_steps["model"]
    coefs = model.coef_.ravel()
    df = pd.DataFrame({"feature": dp.get_feature_names(pre), "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    df["direction"] = np.where(df["coefficient"] >= 0,
                               "raises premium odds", "lowers premium odds")
    df["odds_ratio"] = np.exp(df["coefficient"])
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def regression_coefficients(reg) -> pd.DataFrame:
    """Linear Regression coefficients on the log-price scale (standardised).

    exp(coefficient) - 1 is the proportional change in price for one standard
    deviation of the feature, holding the others fixed.
    """
    inner = reg.regressor_
    names = dp.get_feature_names(inner.named_steps["preprocess"])
    coefs = inner.named_steps["model"].coef_.ravel()
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["pct_change_per_sd"] = (np.exp(df["coefficient"]) - 1) * 100
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


# ==========================================================================
# Static figures for the report
# ==========================================================================
def save_report_figures(reg, clf_results, importance_df, df, verbose=True):
    """Write the matplotlib/seaborn figures referenced by the written report.

    The Streamlit app builds its own interactive Plotly versions; these PNGs
    exist so the report has images without needing a screenshot.
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
        fig.savefig(cfg.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    price_lakh = df[cfg.REGRESSION_TARGET] / cfg.LAKH

    # 1. Price distribution on a log axis -- the skew is the point
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.logspace(np.log10(price_lakh.min()), np.log10(price_lakh.max()), 45)
    ax.hist(price_lakh, bins=bins, edgecolor="white", color="#3b6ea5")
    ax.set_xscale("log")
    ax.axvline(cfg.PREMIUM_THRESHOLD / cfg.LAKH, color="#c0392b", ls="--",
               label=f"Premium threshold ({cfg.PREMIUM_THRESHOLD / cfg.LAKH:g} lakh)")
    ax.set_xlabel("Selling price (lakh rupees, log scale)")
    ax.set_ylabel("Number of cars")
    ax.set_title("Distribution of used-car prices")
    ax.legend()
    _save(fig, "01_price_distribution.png")

    # 2. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[cfg.CLASSIFICATION_TARGET].value_counts().sort_index()
    ax.bar([cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL], counts.values,
           color=["#7f8c8d", "#2a78d6"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v} ({v / len(df) * 100:.1f}%)", ha="center")
    ax.set_ylabel("Number of cars")
    ax.set_title("Premium vs standard class distribution")
    _save(fig, "02_class_distribution.png")

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cols = cfg.NUMERIC_FEATURES + [cfg.REGRESSION_TARGET]
    sns.heatmap(df[cols].corr(numeric_only=True), cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation between numeric features and price")
    _save(fig, "03_correlation_heatmap.png")

    # 4. Actual vs predicted (lakh), log axes
    fig, ax = plt.subplots(figsize=(6, 5.5))
    yt, yp = np.array(reg["y_test"]), np.array(reg["y_pred"])
    ax.scatter(yt, yp, alpha=0.4, s=14, color="#3b6ea5")
    lim = [min(yt.min(), yp.min()) * 0.9, max(yt.max(), yp.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price (lakh rupees, log scale)")
    ax.set_ylabel("Predicted price (lakh rupees, log scale)")
    ax.set_title(f"Linear Regression: actual vs predicted "
                 f"(test R2 = {reg['test_metrics']['R2']:.3f})")
    ax.legend()
    _save(fig, "04_actual_vs_predicted.png")

    # 5. Residual plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(yp, yt - yp, alpha=0.4, s=14, color="#8e44ad")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Predicted price (lakh rupees, log scale)")
    ax.set_ylabel("Residual (actual - predicted, lakh)")
    ax.set_title("Residual plot")
    _save(fig, "05_residual_plot.png")

    # 6. Confusion matrices
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
        ax.bar(xs + i * width, [clf_results[n]["test_metrics"][m] for n in names],
               width, label=m)
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
    ax.set_title("Random Forest feature importance (top 15 encoded columns)")
    _save(fig, "08_feature_importance.png")

    # 9. ROC curves
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, res in clf_results.items():
        ax.plot(res["roc"]["fpr"], res["roc"]["tpr"],
                label=f"{name} (AUC = {res['roc']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "09_roc_curves.png")

    if verbose:
        print(f"\n  Saved {len(saved)} figures to "
              f"{cfg.FIGURES_DIR.relative_to(cfg.PROJECT_ROOT)}/")
    return saved


# ==========================================================================
# Main
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train every model for the Used Car Price & Premium system."
    )
    parser.add_argument("--no-tune", action="store_true",
                        help="skip the small grid searches")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip writing the static report figures")
    parser.add_argument("--skip-duplicate-demo", action="store_true",
                        help="skip the duplicate-leakage comparison run")
    args = parser.parse_args(argv)

    t_start = time.time()
    print("=" * 74)
    print("  USED CAR PRICE & PREMIUM PREDICTION -- TRAINING")
    print("=" * 74)

    # --- 1-2. load, validate, clean -------------------------------------
    try:
        raw = dp.load_raw()
        df = dp.clean(raw)
    except dp.DatasetError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    summary = dp.dataset_summary(raw, df)
    print(f"\n[1/6] Dataset: {cfg.DATASET_FILE.name}")
    print(f"      {summary['n_raw_rows']} listings x {summary['n_raw_columns']} columns")
    print(f"      {summary['n_duplicates_removed']} exact duplicate listings "
          f"({summary['pct_duplicates']:.1f}%) removed BEFORE splitting "
          f"-> {summary['n_cars']} unique cars")
    print(f"      missing: {summary['n_rows_missing_specs']} cars lack mileage, "
          f"engine and seats; {summary['n_missing_max_power']} lack max_power "
          f"(imputed inside the pipeline)")
    print(f"      price: median {summary['median_price'] / cfg.LAKH:.1f} lakh, "
          f"mean {summary['mean_price'] / cfg.LAKH:.1f} lakh, "
          f"max {summary['max_price'] / cfg.LAKH:.0f} lakh "
          f"(skewness {summary['price_skew']:.1f})")
    print(f"      premium (price >= {cfg.PREMIUM_THRESHOLD / cfg.LAKH:g} lakh): "
          f"{summary['n_premium']} ({summary['pct_premium']:.1f}%)   "
          f"standard: {summary['n_standard']}")
    print(f"      majority-class baseline accuracy: "
          f"{summary['majority_class_accuracy'] * 100:.1f}%  <-- the number to beat")

    print(f"\n[2/6] Features: {len(cfg.ALL_FEATURES)} inputs "
          f"({len(cfg.NUMERIC_FEATURES)} numeric, "
          f"{len(cfg.CATEGORICAL_FEATURES)} categorical)")
    print("      Excluded to prevent leakage or redundancy:")
    for col, reason in cfg.EXCLUDED_COLUMNS:
        print(f"        - {col}: {reason.split('.')[0]}.")

    # --- regression ------------------------------------------------------
    print(f"\n[3/6] Training the regression model "
          f"(train/test = {1 - cfg.TEST_SIZE:.0%}/{cfg.TEST_SIZE:.0%}, "
          f"random_state={cfg.RANDOM_STATE})")
    reg = train_regression(df)

    # --- classification --------------------------------------------------
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
              f"smaller than the cross-validation spread of "
              f"+/-{tie['noise_floor']:.4f}.")
        print(f"        On a {tie['n_test']}-car test set these models are "
              f"statistically indistinguishable: {', '.join(tie['tied_models'])}.")
    else:
        print(f"  The gap to {tie['runner_up']} is {tie['gap']:.4f}, larger than the "
              f"cross-validation spread of +/-{tie['noise_floor']:.4f}: the ranking "
              f"is not just noise.")

    # --- explainability --------------------------------------------------
    print("\n[5/6] Explainability")
    importance_df, perm_df = random_forest_importance(
        clf_results["Random Forest"]["pipeline"],
        clf_ctx["X_test"], clf_ctx["y_test"],
    )
    coef_df = logistic_coefficients(clf_results["Logistic Regression"]["pipeline"])
    reg_coef_df = regression_coefficients(reg["pipeline"])
    print("\n  Linear Regression -- 6 largest coefficients (log-price, standardised)")
    for _, r in reg_coef_df.head(6).iterrows():
        print(f"    {r['feature']:26s} {r['coefficient']:+.4f}  "
              f"({r['pct_change_per_sd']:+.1f}% price per SD)")

    # --- duplicate-leakage demonstration ---------------------------------
    leakage_rows, twins = [], None
    if not args.skip_duplicate_demo:
        print("\n[6/6] Duplicate-leakage demonstration: refitting WITH the "
              "duplicate listings kept")
        print("      (these models are evaluated and reported, never saved or "
              "used for predictions)")
        leakage_rows, twins = duplicate_leakage_comparison(
            raw, reg, clf_results, tune=not args.no_tune
        )
        print("\n  Effect of leaving the duplicates in:")
        for row in leakage_rows:
            print(f"    {row['Model']:22s} {row['Metric']:11s} "
                  f"{row['Duplicates removed (reported)']:+.4f} -> "
                  f"{row['Duplicates kept']:+.4f}")
    else:
        print("\n[6/6] Duplicate-leakage demonstration skipped")

    # --- persist ---------------------------------------------------------
    for d in (cfg.MODELS_DIR, cfg.RESULTS_DIR, cfg.DATA_PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    joblib.dump(reg["pipeline"], cfg.MODEL_FILES[cfg.REGRESSION_MODEL_NAME])
    for name, res in clf_results.items():
        joblib.dump(res["pipeline"], cfg.MODEL_FILES[name])

    metadata = {
        "dataset_file": cfg.DATASET_FILE.name,
        "dataset_sha256": dp.file_sha256(),
        "checksum_matches": dp.file_sha256() == cfg.EXPECTED_SHA256,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": cfg.RANDOM_STATE,
        "test_size": cfg.TEST_SIZE,
        "cv_folds": cfg.CV_FOLDS,
        "premium_threshold": cfg.PREMIUM_THRESHOLD,
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
        "train_premium_pct": clf_ctx["train_premium_pct"],
        "test_premium_pct": clf_ctx["test_premium_pct"],
        "default_profile": dp.default_car_profile(df),
        "feature_ranges": dp.feature_ranges(df),
        "known_brands": dp.known_brands(df),
        "regression": {k: v for k, v in reg.items() if k != "pipeline"},
        "classification": {
            n: {k: v for k, v in r.items() if k != "pipeline"}
            for n, r in clf_results.items()
        },
        "rf_impurity_importance": importance_df.to_dict("records"),
        "rf_permutation_importance": perm_df.to_dict("records"),
        "logistic_coefficients": coef_df.to_dict("records"),
        "regression_coefficients": reg_coef_df.to_dict("records"),
        "duplicate_leakage": leakage_rows,
        "duplicate_twins": twins,
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(metadata, cfg.METADATA_FILE)

    # --- results tables --------------------------------------------------
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

    thr_rows, cal_rows = [], []
    for name, res in clf_results.items():
        ti, tm, base = res["threshold"], res["tuned_metrics"], res["test_metrics"]
        thr_rows.append({
            "Model": name, "Threshold": cfg.DEFAULT_THRESHOLD,
            "Chosen by": f"default ({cfg.DEFAULT_THRESHOLD})",
            "Precision": round(base["Precision"], 4), "Recall": round(base["Recall"], 4),
            "F1": round(base["F1"], 4), "Accuracy": round(base["Accuracy"], 4)})
        thr_rows.append({
            "Model": name, "Threshold": round(ti["threshold"], 4),
            "Chosen by": f"expected cost, FN:FP = {ti['fn_cost_ratio']:g}:1",
            "Precision": round(tm["Precision"], 4), "Recall": round(tm["Recall"], 4),
            "F1": round(tm["F1"], 4), "Accuracy": round(tm["Accuracy"], 4)})
        c = res["calibration"]
        cal_rows.append({
            "Model": name, "Brier score": round(c["brier_score"], 4),
            "Brier (constant base-rate predictor)": round(c["brier_baseline"], 4),
            "Beats constant predictor": c["brier_score"] < c["brier_baseline"],
            "Mean calibration error": round(c["mean_abs_calibration_error"], 4),
            "P(premium) min": round(c["proba_min"], 4),
            "P(premium) max": round(c["proba_max"], 4),
            "Share in [0.3, 0.7]": round(c["frac_in_mid_band"], 4),
            "Share within 0.1 of 0.5": round(c["frac_near_default"], 4),
            "Verdict": c["summary"]})
    pd.DataFrame(thr_rows).to_csv(cfg.RESULTS_THRESHOLD_CSV, index=False)
    pd.DataFrame(cal_rows).to_csv(cfg.RESULTS_CALIBRATION_CSV, index=False)

    combined = pd.concat(
        [clf_table.assign(Task="Classification (premium)"),
         reg_table.assign(Task="Regression (price)")], ignore_index=True)
    combined = combined[["Task"] + [c for c in combined.columns if c != "Task"]]
    combined.to_csv(cfg.RESULTS_MODEL_RESULTS_CSV, index=False)

    df.to_csv(cfg.DATA_PROCESSED_DIR / "cars_processed.csv", index=False)
    with open(cfg.RESULTS_DIR / "metrics_summary.json", "w") as fh:
        json.dump(
            {"best_classifier": best_model,
             "classification": {n: r["test_metrics"] for n, r in clf_results.items()},
             "classification_baseline": clf_ctx["baseline"],
             "regression": reg["test_metrics"],
             "regression_baseline": reg["baseline_metrics"]},
            fh, indent=2, default=float)

    if not args.no_figures:
        save_report_figures(reg, clf_results, importance_df, df)

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 74)
    print("  TRAINING SUMMARY")
    print("=" * 74)
    print(f"\n  Classification -- held-out test set "
          f"({clf_ctx['n_test']} cars, {clf_ctx['test_premium_pct']:.1f}% premium)")
    print(clf_table.to_string(index=False))
    print(f"\n  Regression -- held-out test set ({reg['n_test']} cars, errors in lakh rupees)")
    print(reg_table.to_string(index=False))
    print(f"\n  Highest F1: {best_model} "
          f"(F1 = {clf_results[best_model]['test_metrics']['F1']:.4f}, "
          f"ROC-AUC = {clf_results[best_model]['test_metrics']['ROC-AUC']:.4f})")
    if tie["is_tie"]:
        print(f"  Verdict: statistical tie between {', '.join(tie['tied_models'])} "
              f"(gap {tie['gap']:.4f} < noise {tie['noise_floor']:.4f}).")
        print("           Random Forest is retained as the project's primary model.")
    print("\n  Reference points -- a model is only useful if it beats these:")
    print(f"    majority-class accuracy {clf_ctx['baseline']['Accuracy']:.4f} "
          f"(with recall 0.0000 on premium cars)")
    print(f"    mean-predictor regression RMSE "
          f"{reg['baseline_metrics']['RMSE']:.4f} lakh")
    print(f"\n  Saved {len(cfg.MODEL_FILES)} pipelines to "
          f"{cfg.MODELS_DIR.relative_to(cfg.PROJECT_ROOT)}/ "
          f"and results to {cfg.RESULTS_DIR.relative_to(cfg.PROJECT_ROOT)}/")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print("\n  Next step:  streamlit run app.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
