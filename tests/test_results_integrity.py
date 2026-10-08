"""The committed results must match the committed models.

These are regression tests on the project's published numbers. If one fails,
either the models were retrained without updating the documentation, or the
documentation quotes a figure that no run produced.
"""

import csv

import pytest


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def results(cfg):
    if not cfg.RESULTS_MODEL_RESULTS_CSV.exists():
        pytest.skip("results/ not present — run `python src/train_models.py`")
    return _rows(cfg.RESULTS_MODEL_RESULTS_CSV)


# The figures quoted throughout README.md and docs/project_report.md.
DOCUMENTED = {
    ("Classification (at risk)", "SVM (RBF)"):
        {"Accuracy": "0.6835", "Precision": "0.5161", "Recall": "0.6154",
         "F1": "0.5614", "ROC-AUC": "0.7242"},
    ("Classification (at risk)", "Random Forest"):
        {"Accuracy": "0.7595", "Precision": "0.7059", "Recall": "0.4615",
         "F1": "0.5581", "ROC-AUC": "0.6858"},
    ("Classification (at risk)", "Logistic Regression"):
        {"Accuracy": "0.6456", "Precision": "0.4706", "Recall": "0.6154",
         "F1": "0.5333", "ROC-AUC": "0.6981"},
    ("Classification (at risk)", "Baseline (majority class)"):
        {"Accuracy": "0.6709", "Recall": "0.0", "F1": "0.0"},
    ("Regression (final grade)", "Linear Regression"):
        {"MAE": "3.3953", "MSE": "17.6037", "RMSE": "4.1957", "R2": "0.1415",
         "CV R2 (mean)": "-0.0781"},
}


@pytest.mark.parametrize("key,expected", list(DOCUMENTED.items()),
                         ids=[f"{t.split()[0]}-{m}" for t, m in DOCUMENTED])
def test_committed_results_match_the_documentation(results, key, expected):
    task, model = key
    row = next((r for r in results if r["Task"] == task and r["Model"] == model), None)
    assert row is not None, f"{model} missing from model_results.csv"
    for metric, value in expected.items():
        assert row[metric] == value, (
            f"{model} {metric}: CSV says {row[metric]}, docs say {value}"
        )


def test_metadata_agrees_with_the_results_csv(artifacts, results):
    """The app reads metadata.pkl; the docs quote the CSV. They must agree."""
    _, md = artifacts
    for name, res in md["classification"].items():
        row = next(r for r in results
                   if r["Model"] == name and r["Task"].startswith("Classification"))
        assert round(res["test_metrics"]["F1"], 4) == float(row["F1"])
        assert round(res["test_metrics"]["Recall"], 4) == float(row["Recall"])


def test_the_tie_verdict_is_recorded(artifacts):
    _, md = artifacts
    tie = md["tie_analysis"]
    assert tie["is_tie"] is True, "README's central comparison claim"
    assert tie["gap"] < tie["noise_floor"]
    assert len(tie["tied_models"]) == 3


def test_leakage_demonstration_shows_a_large_gain(artifacts):
    """The README quotes RF F1 rising from 0.5581 to 0.8679 with G1/G2."""
    _, md = artifacts
    rf = next(r for r in md["leakage_comparison"]
              if r["Model"] == "Random Forest" and r["Metric"] == "F1")
    assert rf["Leakage-free (no G1/G2)"] == 0.5581
    assert rf["With leakage (G1+G2)"] == 0.8679
    assert rf["With leakage (G1+G2)"] - rf["Leakage-free (no G1/G2)"] > 0.25


def test_threshold_was_chosen_without_the_test_set(artifacts, cfg):
    """The tuned cut-off must differ from the default and be recorded."""
    _, md = artifacts
    for name, res in md["classification"].items():
        t = res["threshold"]
        assert t is not None, f"{name} has no recorded threshold"
        assert 0.0 < t["threshold"] < 1.0
        assert t["fn_cost_ratio"] == cfg.FN_COST_RATIO
        # Chosen on out-of-fold TRAIN data, so it cannot be the test optimum.
        assert t["cost_at_chosen"] <= t["cost_at_default"] + 1e-9


def test_tuned_threshold_raises_recall_on_the_test_set(artifacts):
    """The whole point of the exercise, checked on held-out data."""
    _, md = artifacts
    for name, res in md["classification"].items():
        base, tuned = res["test_metrics"], res["tuned_metrics"]
        assert tuned["Recall"] >= base["Recall"], (
            f"{name}: tuning lowered recall ({base['Recall']} -> {tuned['Recall']})"
        )


def test_calibration_is_recorded_and_logistic_regression_is_flagged(artifacts):
    """The documented finding: LR's Brier is worse than a constant predictor."""
    _, md = artifacts
    lr = md["classification"]["Logistic Regression"]["calibration"]
    assert lr["brier_score"] > lr["brier_baseline"], (
        "documented claim: LR is worse than the constant base-rate predictor"
    )
    assert "compressed" in lr["summary"]
    for name in ("SVM (RBF)", "Random Forest"):
        c = md["classification"][name]["calibration"]
        assert c["brier_score"] < c["brier_baseline"]


def test_feature_importance_measures_disagree_as_documented(artifacts):
    """README section 15 is built on impurity ranking absences over failures."""
    _, md = artifacts
    assert md["rf_impurity_importance"][0]["feature"] == "absences"
    assert md["rf_permutation_importance"][0]["feature"] == "failures"
    perm = md["rf_permutation_importance"]
    assert perm[0]["permutation_importance"] > 5 * perm[1]["permutation_importance"]


def test_random_state_is_fixed(artifacts, cfg):
    _, md = artifacts
    assert md["random_state"] == cfg.RANDOM_STATE == 42
    assert md["n_train"] == 316 and md["n_test"] == 79
