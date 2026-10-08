"""
Evaluation metrics for both project tasks.

Kept in its own module so the training script, the notebook and the Streamlit
app all compute metrics the same way -- there is no second, slightly different
definition of "F1" anywhere in the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config as cfg


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE and R-squared.

    RMSE is reported alongside MAE because it is in the same units as the grade
    (0-20) but penalises large errors more heavily -- useful here, where the
    38 students with G3 = 0 produce a handful of very large residuals.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def baseline_regression_metrics(y_train, y_test) -> dict:
    """Metrics for the 'always predict the training mean' baseline.

    Any honest R-squared claim needs this reference point: a model that cannot
    beat the mean predictor has learned nothing.
    """
    mean_pred = np.full(len(y_test), float(np.mean(y_train)))
    out = regression_metrics(y_test, mean_pred)
    out["Model"] = "Baseline (predict mean)"
    return out


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy, precision, recall, F1 and (when probabilities exist) ROC-AUC.

    Precision/recall/F1 are reported for the POSITIVE class, which this project
    defines as "At Risk" (label 1). That is deliberate: the costly mistake in an
    early-warning system is failing to flag a student who does need help, so
    recall on the at-risk class is the metric that matters most.
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    metrics["ROC-AUC"] = (
        float(roc_auc_score(y_true, y_proba)) if y_proba is not None else np.nan
    )
    return metrics


def baseline_classification_metrics(y_train, y_test) -> dict:
    """The majority-class baseline: always predict 'Not At Risk'.

    Its accuracy is the floor every model must clear before accuracy means
    anything. Its recall on the at-risk class is 0, which is exactly why
    accuracy alone is a misleading score on this dataset.
    """
    majority = int(pd.Series(y_train).mode().iloc[0])
    y_pred = np.full(len(y_test), majority)
    out = classification_metrics(y_test, y_pred, y_proba=None)
    out["Model"] = "Baseline (majority class)"
    return out


def confusion_counts(y_true, y_pred) -> dict:
    """TN / FP / FN / TP as plain integers, labelled for the UI."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "matrix": cm.tolist(),
        "TN": int(tn),  # correctly identified as Not At Risk
        "FP": int(fp),  # false alarm: flagged At Risk but actually passed
        "FN": int(fn),  # missed case: not flagged but actually At Risk
        "TP": int(tp),  # correctly identified as At Risk
    }


def text_classification_report(y_true, y_pred) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL],
        zero_division=0,
        digits=3,
    )


def roc_points(y_true, y_proba) -> dict:
    """FPR/TPR curve points plus AUC, for the ROC chart."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------
def build_comparison_table(rows: list[dict], sort_by: str = "F1") -> pd.DataFrame:
    """Tidy comparison DataFrame with Model first and the metrics after."""
    df = pd.DataFrame(rows)
    ordered = ["Model"] + [c for c in df.columns if c != "Model"]
    df = df[ordered]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False, na_position="last")
    return df.reset_index(drop=True)


def pick_best_model(results: dict, primary: str = "F1", secondary: str = "ROC-AUC") -> str:
    """Select the best classifier by F1, breaking ties with ROC-AUC.

    F1 leads rather than accuracy because the classes are imbalanced (about one
    student in three is at risk), so a model can score well on accuracy while
    missing most of the students the system exists to find.
    """
    def key(name):
        m = results[name]["test_metrics"]
        p = m.get(primary, float("nan"))
        s = m.get(secondary, float("nan"))
        return (
            -1 if p != p else p,       # NaN sorts last
            -1 if s != s else s,
        )

    return max(results, key=key)


def assess_tie(results: dict, metric: str = "F1") -> dict:
    """Decide whether the models are actually distinguishable on this dataset.

    Ranking three models by a single test-set F1 is only meaningful if the gaps
    between them are larger than the run-to-run noise. This compares the spread
    of the top scores against the cross-validation standard deviation, which is
    a direct estimate of that noise.

    If the gap between the best and second-best model is smaller than the
    largest CV standard deviation among them, the difference is inside the noise
    floor and the project reports a statistical tie instead of crowning a
    winner. The test set here is only 79 students, so one or two reclassified
    students move F1 by several points.
    """
    scored = sorted(
        ((name, res["test_metrics"].get(metric, float("nan")))
         for name, res in results.items()),
        key=lambda kv: (kv[1] != kv[1], -kv[1]),  # NaN last, then descending
    )
    best_name, best_score = scored[0]
    runner_name, runner_score = scored[1] if len(scored) > 1 else (None, float("nan"))

    gap = abs(best_score - runner_score) if runner_name else float("nan")
    noise = max(
        (results[n].get("cv_f1_std", 0.0) or 0.0) for n in results
    )
    n_test = len(results[best_name]["y_test"])

    tied = [n for n, sc in scored if sc == sc and abs(best_score - sc) <= noise]

    return {
        "metric": metric,
        "ranking": [{"model": n, metric: sc} for n, sc in scored],
        "best_model": best_name,
        "best_score": float(best_score),
        "runner_up": runner_name,
        "runner_up_score": float(runner_score) if runner_name else None,
        "gap": float(gap),
        "noise_floor": float(noise),
        "is_tie": bool(gap == gap and gap <= noise and len(tied) > 1),
        "tied_models": tied,
        "n_test": int(n_test),
    }
