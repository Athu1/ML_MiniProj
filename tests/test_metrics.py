"""Metric definitions, baselines, tie analysis, thresholds and calibration."""

import numpy as np
import pytest


def test_positive_class_is_at_risk(em):
    """Precision/recall/F1 must be reported for the at-risk class, not class 0.

    If this flips, every precision and recall figure in the documentation means
    the opposite of what it says.
    """
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]   # 1 TP, 1 FP, 1 FN
    m = em.classification_metrics(y_true, y_pred)
    assert m["Precision"] == pytest.approx(0.5)
    assert m["Recall"] == pytest.approx(0.5)


def test_majority_baseline_has_zero_recall(em):
    """The README's central argument about accuracy depends on this."""
    y_train = [0] * 70 + [1] * 30
    y_test = [0] * 7 + [1] * 3
    b = em.baseline_classification_metrics(y_train, y_test)
    assert b["Accuracy"] == pytest.approx(0.7)
    assert b["Recall"] == 0.0
    assert b["F1"] == 0.0


def test_rmse_is_never_below_mae(em):
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 20, 200)
    p = y + rng.normal(0, 3, 200)
    m = em.regression_metrics(y, p)
    assert m["RMSE"] >= m["MAE"]
    assert m["RMSE"] == pytest.approx(np.sqrt(m["MSE"]))


def test_r2_of_the_mean_predictor_is_about_zero(em):
    y = np.array([1.0, 5.0, 9.0, 13.0])
    m = em.regression_metrics(y, np.full(4, y.mean()))
    assert m["R2"] == pytest.approx(0.0)


def test_confusion_counts_are_labelled_the_right_way_round(em):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    c = em.confusion_counts(y_true, y_pred)
    assert (c["TN"], c["FP"], c["FN"], c["TP"]) == (1, 1, 1, 1)


def test_tie_is_declared_when_the_gap_is_inside_the_noise(em):
    results = {
        "A": {"test_metrics": {"F1": 0.561, "ROC-AUC": 0.72},
              "cv_f1_std": 0.08, "y_test": [0] * 79},
        "B": {"test_metrics": {"F1": 0.558, "ROC-AUC": 0.69},
              "cv_f1_std": 0.08, "y_test": [0] * 79},
    }
    t = em.assess_tie(results)
    assert t["is_tie"] is True
    assert t["gap"] < t["noise_floor"]
    assert set(t["tied_models"]) == {"A", "B"}


def test_tie_is_not_declared_when_the_gap_is_real(em):
    results = {
        "A": {"test_metrics": {"F1": 0.90, "ROC-AUC": 0.95},
              "cv_f1_std": 0.01, "y_test": [0] * 79},
        "B": {"test_metrics": {"F1": 0.50, "ROC-AUC": 0.60},
              "cv_f1_std": 0.01, "y_test": [0] * 79},
    }
    t = em.assess_tie(results)
    assert t["is_tie"] is False
    assert t["best_model"] == "A"


# -------------------------------------------------------------------------
# Decision threshold
# -------------------------------------------------------------------------
def test_higher_fn_cost_never_raises_the_threshold(em):
    """Valuing missed students more must make the model flag MORE, not fewer.

    Recall is monotonically non-decreasing in the cost ratio; equivalently the
    chosen cut-off is non-increasing. If this fails, the cost model is wired up
    backwards.
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, 400), 0.01, 0.99)
    thresholds = [em.choose_threshold(y, p, fn_cost=r)["threshold"]
                  for r in (1, 2, 3, 5, 10)]
    assert thresholds == sorted(thresholds, reverse=True), thresholds


def test_cost_ratio_of_one_is_plain_error_minimisation(em):
    """With FN and FP equally costly, cost is just the error count."""
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    c = em.expected_cost(y, p, threshold=0.5, fn_cost=1.0)
    assert c == pytest.approx(0.0)
    c_bad = em.expected_cost(y, p, threshold=0.95, fn_cost=1.0)
    assert c_bad == pytest.approx(0.5)   # both positives now missed


def test_false_negatives_are_weighted_by_the_ratio(em):
    """Deliberately ASYMMETRIC: 1 false positive against 2 false negatives.

    With equal counts of each error, swapping the two weights gives the same
    total, so a symmetric case cannot detect the weights being wired up
    backwards. (An earlier version of this test used 1 FP and 1 FN and failed to
    catch exactly that mutation.)
    """
    #        FP           FN           FN         correct
    y = [0,            1,           1,          0]
    p = [0.9,          0.1,         0.2,        0.1]
    # fn_cost=1 -> (1 + 1*2)/4 = 0.75 ; fn_cost=3 -> (1 + 3*2)/4 = 1.75
    assert em.expected_cost(y, p, 0.5, fn_cost=1.0) == pytest.approx(0.75)
    assert em.expected_cost(y, p, 0.5, fn_cost=3.0) == pytest.approx(1.75)
    # And with the weights swapped the two would be 2.25 and 1.25 — different,
    # so this case genuinely distinguishes the orientation.
    assert em.expected_cost(y, p, 0.5, fn_cost=3.0) != pytest.approx(1.25)


def test_chosen_threshold_is_at_least_as_good_as_the_default(em):
    """By construction the selected cut-off cannot cost more than 0.5 does."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.35 + 0.25 * y + rng.normal(0, 0.12, 300), 0.01, 0.99)
    sel = em.choose_threshold(y, p, fn_cost=3.0)
    assert sel["cost_at_chosen"] <= sel["cost_at_default"] + 1e-12


def test_metrics_at_threshold_reproduce_the_default_prediction(em):
    y = [0, 0, 1, 1]
    p = [0.2, 0.6, 0.4, 0.8]
    m = em.metrics_at_threshold(y, p, 0.5)
    direct = em.classification_metrics(y, [0, 1, 0, 1], p)
    assert m["F1"] == pytest.approx(direct["F1"])
    assert m["Threshold"] == 0.5


def test_threshold_sweep_is_ordered_by_recall(em):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.35 + 0.25 * y + rng.normal(0, 0.12, 300), 0.01, 0.99)
    rows = em.threshold_sweep(y, p)
    recalls = [r["Recall"] for r in rows]
    assert recalls == sorted(recalls), recalls


# -------------------------------------------------------------------------
# Calibration
# -------------------------------------------------------------------------
def test_a_well_calibrated_model_scores_low_error(em):
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)   # outcomes match the stated risk
    rep = em.calibration_report(y, p, n_bins=5)
    assert rep["mean_abs_calibration_error"] < 0.05
    assert rep["brier_score"] < rep["brier_baseline"]


def test_a_compressed_model_is_flagged_as_such(em):
    """A model whose output never leaves a narrow band must be called out."""
    rng = np.random.default_rng(5)
    p = rng.uniform(0.44, 0.56, 500)              # says nothing useful
    y = rng.integers(0, 2, 500)
    rep = em.calibration_report(y, p, n_bins=5)
    assert rep["proba_range"] < 0.5
    assert "compressed" in em.describe_calibration(rep)


def test_brier_baseline_is_the_constant_base_rate_predictor(em):
    y = np.array([0] * 70 + [1] * 30)
    rep = em.calibration_report(y, np.full(100, 0.3), n_bins=2)
    assert rep["base_rate"] == pytest.approx(0.3)
    assert rep["brier_score"] == pytest.approx(rep["brier_baseline"])
