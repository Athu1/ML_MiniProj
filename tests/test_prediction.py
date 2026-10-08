"""The prediction path the Streamlit form uses, including its error handling."""

import numpy as np
import pytest


def test_all_four_pipelines_load(artifacts, cfg):
    models, _ = artifacts
    assert set(models) == set(cfg.MODEL_FILES)


def test_prediction_returns_a_grade_in_range_and_a_verdict(artifacts, cfg):
    import prediction as pr
    models, md = artifacts
    r = pr.predict_student(dict(md["default_profile"]), models, md)
    assert 0 <= r["predicted_score"] <= cfg.GRADE_MAX
    assert r["risk_label"] in (cfg.POSITIVE_CLASS_LABEL, cfg.NEGATIVE_CLASS_LABEL)
    assert 0.0 <= r["risk_probability"] <= 1.0
    assert len(r["per_model"]) == 3


def test_the_saved_pipeline_applies_training_time_preprocessing(artifacts, cfg):
    """A single-row form submission must go through the fitted transformer.

    This is what guarantees the app cannot drift from training: the pipeline
    carries its own preprocessing, so the same raw dict always maps to the same
    encoded row.
    """
    import prediction as pr
    models, md = artifacts
    row = pr.build_input_row(dict(md["default_profile"]), md["default_profile"])
    pre = models["Random Forest"].named_steps["preprocess"]
    assert list(row.columns) == cfg.ALL_FEATURES
    assert pre.transform(row).shape == (1, len(md["encoded_feature_names"]))


def test_missing_features_are_filled_from_the_population_profile(artifacts, cfg):
    import prediction as pr
    models, md = artifacts
    row = pr.build_input_row({"failures": 2}, md["default_profile"])
    assert row.shape == (1, len(cfg.ALL_FEATURES))
    assert int(row.iloc[0]["failures"]) == 2


@pytest.mark.parametrize("bad", [{}, None, "not a dict"])
def test_empty_or_wrong_input_is_rejected(artifacts, bad):
    import prediction as pr
    models, md = artifacts
    with pytest.raises(pr.InvalidStudentInputError):
        pr.predict_student(bad, models, md)


@pytest.mark.parametrize("value", ["abc", float("nan"), float("inf"), None])
def test_non_numeric_or_infinite_numbers_are_rejected(artifacts, value):
    import prediction as pr
    models, md = artifacts
    profile = dict(md["default_profile"])
    profile["absences"] = value
    with pytest.raises(pr.InvalidStudentInputError):
        pr.predict_student(profile, models, md)


def test_unknown_category_is_reported_but_still_predicts(artifacts):
    """The app warns about this instead of failing — so it must not raise."""
    import prediction as pr
    models, md = artifacts
    profile = dict(md["default_profile"], Mjob="astronaut")
    r = pr.predict_student(profile, models, md)
    assert r["unknown_categories"], "an unseen level should be reported"
    assert 0 <= r["predicted_score"] <= 20


def test_unknown_classifier_is_rejected(artifacts):
    import prediction as pr
    models, md = artifacts
    with pytest.raises(pr.InvalidStudentInputError, match="Unknown classifier"):
        pr.predict_student(dict(md["default_profile"]), models, md,
                           classifier="Deep Neural Net")


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2])
def test_out_of_range_threshold_is_rejected(artifacts, bad):
    import prediction as pr
    models, md = artifacts
    with pytest.raises(pr.InvalidStudentInputError, match="between 0 and 1"):
        pr.predict_student(dict(md["default_profile"]), models, md, threshold=bad)


def test_threshold_changes_the_verdict_for_a_boundary_student(artifacts, cfg):
    """A student whose probability sits between the two cut-offs must flip."""
    import prediction as pr
    models, md = artifacts
    profile = dict(md["default_profile"])
    tuned = md["classification"]["Random Forest"]["threshold"]["threshold"]

    lo = pr.predict_student(profile, models, md, classifier="Random Forest",
                            threshold=tuned)
    hi = pr.predict_student(profile, models, md, classifier="Random Forest",
                            threshold=None)
    p = lo["risk_probability"]
    if tuned <= p < cfg.DEFAULT_THRESHOLD:
        assert lo["at_risk"] and not hi["at_risk"], (
            "a probability between the cut-offs must give different verdicts"
        )
        assert lo["threshold_decided_it"] and hi["threshold_decided_it"]
    assert lo["threshold_used"] == pytest.approx(tuned)
    assert hi["threshold_used"] == pytest.approx(cfg.DEFAULT_THRESHOLD)
    assert lo["threshold_is_tuned"] and not hi["threshold_is_tuned"]


def test_a_worse_student_profile_never_lowers_predicted_risk(artifacts):
    """Sanity check on direction: more failures must not reduce estimated risk.

    Not a causal claim about students — a monotonicity check that the model is
    wired up the right way round, since `failures` is its dominant feature.
    """
    import prediction as pr
    models, md = artifacts
    base = dict(md["default_profile"], failures=0)
    worse = dict(md["default_profile"], failures=3)
    p0 = pr.predict_student(base, models, md, classifier="Random Forest")
    p3 = pr.predict_student(worse, models, md, classifier="Random Forest")
    assert p3["risk_probability"] > p0["risk_probability"]
    assert p3["predicted_score"] < p0["predicted_score"]


def test_missing_model_files_give_an_actionable_error(cfg, monkeypatch, tmp_path):
    import prediction as pr
    monkeypatch.setitem(cfg.MODEL_FILES, "Random Forest", tmp_path / "gone.pkl")
    with pytest.raises(pr.ModelsNotTrainedError, match="train_models"):
        pr.load_artifacts()


def test_corrupted_model_file_gives_an_actionable_error(cfg, monkeypatch, tmp_path):
    import prediction as pr
    junk = tmp_path / "junk.pkl"
    junk.write_bytes(b"not a pickle")
    monkeypatch.setitem(cfg.MODEL_FILES, "Random Forest", junk)
    with pytest.raises(pr.ModelsNotTrainedError, match="could not be loaded"):
        pr.load_artifacts()


def test_influential_features_are_global_and_labelled(artifacts):
    import prediction as pr
    models, md = artifacts
    r = pr.predict_student(dict(md["default_profile"]), models, md)
    feats = pr.top_influential_features(md, r["input_row"], top_n=5)
    assert len(feats) == 5
    assert feats[0]["feature"] == "failures", "documented as the top feature"
    assert all(isinstance(f["student_value"], str) for f in feats), (
        "values must be formatted for display, not raw numpy reprs"
    )
