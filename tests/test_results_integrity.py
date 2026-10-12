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
        pytest.skip("results/ not present -- run `python src/train_models.py`")
    return _rows(cfg.RESULTS_MODEL_RESULTS_CSV)


# The figures quoted throughout README.md and docs/project_report.md.
DOCUMENTED = {
    ("Classification (premium)", "Random Forest"):
        {"Accuracy": "0.9127", "Precision": "0.8543", "Recall": "0.848",
         "F1": "0.8512", "ROC-AUC": "0.9736"},
    ("Classification (premium)", "SVM (RBF)"):
        {"Accuracy": "0.9105", "Precision": "0.8381", "Recall": "0.8627",
         "F1": "0.8502", "ROC-AUC": "0.9695"},
    ("Classification (premium)", "Logistic Regression"):
        {"Accuracy": "0.9062", "Precision": "0.839", "Recall": "0.8431",
         "F1": "0.8411", "ROC-AUC": "0.9611"},
    ("Classification (premium)", "Baseline (majority class)"):
        {"Accuracy": "0.7056", "Recall": "0.0", "F1": "0.0"},
    ("Regression (price)", "Linear Regression"):
        {"MAE": "0.9201", "MSE": "3.0951", "RMSE": "1.7593", "R2": "0.8589",
         "MedAPE": "15.9731", "R2_log": "0.8685", "CV R2 (mean)": "0.8527"},
    ("Regression (price)", "Baseline (predict mean)"):
        {"MAE": "2.7531", "RMSE": "4.6879", "R2": "-0.002", "MedAPE": "46.2371"},
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
    reg = next(r for r in results if r["Task"].startswith("Regression")
               and r["Model"] == "Linear Regression")
    assert round(md["regression"]["test_metrics"]["R2"], 4) == float(reg["R2"])


def test_models_beat_the_baselines(artifacts):
    _, md = artifacts
    base_acc = md["baseline_classification"]["Accuracy"]
    for name, res in md["classification"].items():
        assert res["test_metrics"]["Accuracy"] > base_acc + 0.15, name
        assert res["test_metrics"]["F1"] > 0.8, name
    reg, base = md["regression"]["test_metrics"], md["regression"]["baseline_metrics"]
    assert reg["R2"] > 0.8 > base["R2"]
    assert reg["RMSE"] < base["RMSE"] / 2


def test_the_tie_verdict_is_recorded(artifacts):
    """Random Forest and SVM are inside the noise; logistic regression is not."""
    _, md = artifacts
    tie = md["tie_analysis"]
    assert tie["is_tie"] is True
    assert tie["gap"] < tie["noise_floor"]
    assert set(tie["tied_models"]) == {"Random Forest", "SVM (RBF)"}
    assert tie["best_model"] == "Random Forest"


def test_duplicate_leakage_demonstration_inflates_every_model(artifacts):
    """Keeping duplicates must flatter each model, or the demo shows nothing."""
    _, md = artifacts
    rows = md["duplicate_leakage"]
    f1 = [r for r in rows if r["Metric"] == "F1"]
    assert len(f1) == 3
    for r in f1:
        assert r["Duplicates kept"] > r["Duplicates removed (reported)"], r["Model"]
    rf = next(r for r in f1 if r["Model"] == "Random Forest")
    assert rf["Duplicates removed (reported)"] == 0.8512
    assert rf["Duplicates kept"] == 0.8988
    tw = md["duplicate_twins"]
    assert tw["n_twins"] == 332 and tw["n_test"] == 1626


def test_threshold_was_chosen_without_the_test_set(artifacts, cfg):
    _, md = artifacts
    for name, res in md["classification"].items():
        t = res["threshold"]
        assert 0.0 < t["threshold"] < 1.0
        assert t["fn_cost_ratio"] == cfg.FN_COST_RATIO
        # Chosen on out-of-fold TRAIN data, so it cannot be the test optimum.
        assert t["cost_at_chosen"] <= t["cost_at_default"] + 1e-9


def test_symmetric_cost_leaves_the_threshold_near_one_half(artifacts):
    """With 1:1 costs there is no reason to move far from 0.5."""
    _, md = artifacts
    for name, res in md["classification"].items():
        assert abs(res["threshold"]["threshold"] - 0.5) < 0.1, name


def test_every_model_beats_the_constant_predictor_on_calibration(artifacts):
    _, md = artifacts
    for name, res in md["classification"].items():
        c = res["calibration"]
        assert c["brier_score"] < c["brier_baseline"], name
    rf = md["classification"]["Random Forest"]["calibration"]
    lr = md["classification"]["Logistic Regression"]["calibration"]
    assert rf["brier_score"] < lr["brier_score"], "documented: RF is best calibrated"


def test_feature_importance_leaders_are_as_documented(artifacts):
    _, md = artifacts
    assert md["rf_impurity_importance"][0]["feature"] == "age"
    perm = md["rf_permutation_importance"]
    assert [p["feature"] for p in perm[:2]] == ["age", "max_power"]
    assert perm[0]["permutation_importance"] > perm[1]["permutation_importance"]


def test_split_sizes_and_seed(artifacts, cfg):
    _, md = artifacts
    assert md["random_state"] == cfg.RANDOM_STATE == 42
    assert md["n_train"] == 5540 and md["n_test"] == 1386
    assert md["checksum_matches"] is True
