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


# ==========================================================================
# Decision threshold selection
# ==========================================================================
def expected_cost(y_true, y_proba, threshold: float, fn_cost: float) -> float:
    """Cost of applying `threshold`, in units of "one false alarm".

    A false negative (a missed at-risk student) counts `fn_cost` times as much
    as a false positive. Cost is returned per student so it is comparable
    across differently sized sets.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return (fp + fn_cost * fn) / len(y_true)


def choose_threshold(y_true, y_proba, fn_cost: float = None,
                     grid: np.ndarray = None) -> dict:
    """Pick the probability cut-off that minimises expected cost.

    IMPORTANT -- which data this may be called on. The threshold is a
    parameter chosen from data, so selecting it on the test set would be
    exactly the kind of leakage this project is built to avoid. It is therefore
    chosen from *out-of-fold predictions on the training split* (see
    `out_of_fold_probabilities` in train_models.py) and only then applied,
    unchanged, to the held-out test set.

    Ties are broken toward the HIGHER threshold, which is the conservative
    choice: of two cut-offs with equal cost, the one that raises fewer alarms
    is preferred.
    """
    fn_cost = cfg.FN_COST_RATIO if fn_cost is None else fn_cost
    y_proba = np.asarray(y_proba, dtype=float)
    if grid is None:
        # Candidate cut-offs are the midpoints between observed probabilities:
        # anything between two adjacent values produces an identical labelling,
        # so there is nothing to gain from a finer grid.
        uniq = np.unique(y_proba)
        grid = np.unique(np.concatenate([[0.0], (uniq[:-1] + uniq[1:]) / 2, [1.0]])) \
            if len(uniq) > 1 else np.array([0.5])

    costs = np.array([expected_cost(y_true, y_proba, t, fn_cost) for t in grid])
    best = float(grid[np.where(costs == costs.min())[0][-1]])  # highest on a tie

    return {
        "threshold": best,
        "fn_cost_ratio": float(fn_cost),
        "cost_at_chosen": float(costs.min()),
        "cost_at_default": float(
            expected_cost(y_true, y_proba, cfg.DEFAULT_THRESHOLD, fn_cost)
        ),
        "n_candidates": int(len(grid)),
    }


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    """Classification metrics obtained by cutting `y_proba` at `threshold`."""
    pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
    out = classification_metrics(y_true, pred, y_proba)
    out["Threshold"] = float(threshold)
    return out


def threshold_sweep(y_true, y_proba, ratios=None) -> list[dict]:
    """How the chosen threshold and the resulting metrics move with the cost ratio.

    Reported so the cost ratio in config.py is visibly a choice with
    consequences, rather than a number buried in a constant.
    """
    ratios = cfg.THRESHOLD_SWEEP_RATIOS if ratios is None else ratios
    rows = []
    for r in ratios:
        sel = choose_threshold(y_true, y_proba, fn_cost=r)
        m = metrics_at_threshold(y_true, y_proba, sel["threshold"])
        rows.append({
            "FN:FP cost ratio": r,
            "Chosen threshold": round(sel["threshold"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "Accuracy": round(m["Accuracy"], 4),
        })
    return rows


def threshold_curve(y_true, y_proba, fn_cost: float = None) -> dict:
    """Precision, recall, F1 and expected cost across the full threshold range.

    Powers the chart that shows why the default 0.5 is not special.
    """
    fn_cost = cfg.FN_COST_RATIO if fn_cost is None else fn_cost
    grid = np.linspace(0.05, 0.95, 91)
    rows = {"threshold": [], "precision": [], "recall": [], "f1": [], "cost": []}
    for t in grid:
        m = metrics_at_threshold(y_true, y_proba, t)
        rows["threshold"].append(float(t))
        rows["precision"].append(m["Precision"])
        rows["recall"].append(m["Recall"])
        rows["f1"].append(m["F1"])
        rows["cost"].append(expected_cost(y_true, y_proba, t, fn_cost))
    rows["fn_cost_ratio"] = float(fn_cost)
    return rows


# ==========================================================================
# Probability calibration
# ==========================================================================
def calibration_report(y_true, y_proba, n_bins: int = None) -> dict:
    """Is a predicted "70% risk" actually borne out 70% of the time?

    Returns the calibration curve (mean predicted probability against observed
    frequency, per bin), the Brier score, and -- importantly for this project --
    the *observed range* of the predicted probabilities.

    That range matters because the application displays these numbers as
    percentages with a meter, which implies they are calibrated probabilities.
    If a model's output never leaves, say, 0.34 to 0.78, then it is a compressed
    relative score and the interface should say so rather than imply a
    confidence the model never expresses.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    n_bins = cfg.CALIBRATION_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    # `strategy="quantile"` puts an equal number of students in each bin rather
    # than slicing the probability axis evenly. With compressed probabilities
    # the uniform strategy leaves most bins empty.
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
    )

    base_rate = float(y_true.mean())
    return {
        "mean_predicted": [float(v) for v in mean_pred],
        "observed_frequency": [float(v) for v in frac_pos],
        "n_bins_returned": int(len(mean_pred)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        # Brier for a constant predictor at the base rate: the reference point.
        "brier_baseline": float(brier_score_loss(
            y_true, np.full_like(y_proba, base_rate)
        )),
        "proba_min": float(y_proba.min()),
        "proba_max": float(y_proba.max()),
        "proba_range": float(y_proba.max() - y_proba.min()),
        "frac_in_mid_band": float(np.mean((y_proba >= 0.3) & (y_proba <= 0.7))),
        "frac_near_default": float(np.mean(
            np.abs(y_proba - cfg.DEFAULT_THRESHOLD) <= 0.1
        )),
        "base_rate": base_rate,
        # Mean absolute gap between predicted and observed, across bins: a
        # single-number summary of how far off the diagonal the curve sits.
        "mean_abs_calibration_error": float(
            np.mean(np.abs(np.array(mean_pred) - np.array(frac_pos)))
        ) if len(mean_pred) else float("nan"),
    }


def describe_calibration(report: dict) -> str:
    """One honest sentence about whether these probabilities can be read as such."""
    rng, mace = report["proba_range"], report["mean_abs_calibration_error"]
    mid = report["frac_in_mid_band"]
    if rng < 0.5 or mid > 0.9:
        verdict = (
            "heavily compressed -- read these as relative risk scores, not as "
            "calibrated probabilities"
        )
    elif mace > 0.15:
        verdict = "poorly calibrated -- the stated percentage is not reliable"
    elif mace > 0.08:
        verdict = "roughly calibrated, with visible deviation in places"
    else:
        verdict = "reasonably well calibrated on this test set"
    return (
        f"range {report['proba_min']:.2f}-{report['proba_max']:.2f}, "
        f"{mid:.0%} of students between 0.3 and 0.7, "
        f"mean calibration error {mace:.3f} -> {verdict}"
    )
