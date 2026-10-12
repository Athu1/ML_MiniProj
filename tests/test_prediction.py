"""The prediction path the Streamlit form uses, including its error handling."""

import pytest


def test_all_four_pipelines_load(artifacts, cfg):
    models, _ = artifacts
    assert set(models) == set(cfg.MODEL_FILES)


def test_prediction_returns_a_positive_price_and_a_verdict(artifacts, cfg):
    import prediction as pr
    models, md = artifacts
    r = pr.predict_car(dict(md["default_profile"]), models, md)
    assert r["predicted_price"] > 0
    assert r["label"] in (cfg.POSITIVE_CLASS_LABEL, cfg.NEGATIVE_CLASS_LABEL)
    assert 0.0 <= r["premium_probability"] <= 1.0
    assert len(r["per_model"]) == 3


def test_the_saved_pipeline_applies_training_time_preprocessing(artifacts, cfg):
    import prediction as pr
    models, md = artifacts
    row = pr.build_input_row(dict(md["default_profile"]), md["default_profile"])
    pre = models["Random Forest"].named_steps["preprocess"]
    assert list(row.columns) == cfg.ALL_FEATURES
    assert pre.transform(row).shape == (1, len(md["encoded_feature_names"]))


def test_missing_features_are_filled_from_the_population_profile(artifacts, cfg):
    import prediction as pr
    _, md = artifacts
    row = pr.build_input_row({"age": 11}, md["default_profile"])
    assert row.shape == (1, len(cfg.ALL_FEATURES))
    assert int(row.iloc[0]["age"]) == 11


@pytest.mark.parametrize("bad", [{}, None, "not a dict"])
def test_empty_or_wrong_input_is_rejected(artifacts, bad):
    import prediction as pr
    models, md = artifacts
    with pytest.raises(pr.InvalidCarInputError):
        pr.predict_car(bad, models, md)


@pytest.mark.parametrize("value", ["abc", float("nan"), float("inf"), None, -5])
def test_bad_numbers_are_rejected(artifacts, value):
    import prediction as pr
    models, md = artifacts
    profile = dict(md["default_profile"])
    profile["km_driven"] = value
    with pytest.raises(pr.InvalidCarInputError):
        pr.predict_car(profile, models, md)


def test_unknown_brand_is_reported_but_still_predicts(artifacts):
    import prediction as pr
    models, md = artifacts
    r = pr.predict_car(dict(md["default_profile"], brand="Zeppelin"), models, md)
    assert any("Zeppelin" in u for u in r["unknown_categories"])
    assert r["predicted_price"] > 0


def test_out_of_range_inputs_are_flagged(artifacts):
    import prediction as pr
    models, md = artifacts
    r = pr.predict_car(dict(md["default_profile"], km_driven=9_000_000), models, md)
    assert r["out_of_range"], "an odometer far beyond the data must be flagged"
    r = pr.predict_car(dict(md["default_profile"]), models, md)
    assert not r["out_of_range"]


def test_unknown_classifier_is_rejected(artifacts):
    import prediction as pr
    models, md = artifacts
    with pytest.raises(pr.InvalidCarInputError, match="Unknown classifier"):
        pr.predict_car(dict(md["default_profile"]), models, md, classifier="Deep Net")


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2])
def test_out_of_range_threshold_is_rejected(artifacts, bad):
    import prediction as pr
    models, md = artifacts
    with pytest.raises(pr.InvalidCarInputError, match="between 0 and 1"):
        pr.predict_car(dict(md["default_profile"]), models, md, threshold=bad)


def test_threshold_controls_the_verdict(artifacts):
    """A cut-off just above and just below the probability must flip the label."""
    import prediction as pr
    models, md = artifacts
    p = pr.predict_car(dict(md["default_profile"]), models, md,
                       classifier="Random Forest")["premium_probability"]
    below = pr.predict_car(dict(md["default_profile"]), models, md,
                           classifier="Random Forest", threshold=max(p - 0.01, 0.001))
    above = pr.predict_car(dict(md["default_profile"]), models, md,
                           classifier="Random Forest", threshold=min(p + 0.01, 0.999))
    assert below["premium"] and not above["premium"]
    assert below["threshold_is_tuned"] and above["threshold_is_tuned"]


def test_older_cars_are_priced_lower(artifacts):
    """Direction check, not a causal claim: age is the dominant feature."""
    import prediction as pr
    models, md = artifacts
    young = pr.predict_car(dict(md["default_profile"], age=2), models, md)
    old = pr.predict_car(dict(md["default_profile"], age=14), models, md)
    assert young["predicted_price"] > old["predicted_price"]
    assert young["premium_probability"] > old["premium_probability"]


def test_more_power_is_priced_higher(artifacts):
    import prediction as pr
    models, md = artifacts
    weak = pr.predict_car(dict(md["default_profile"], max_power=60), models, md)
    strong = pr.predict_car(dict(md["default_profile"], max_power=180), models, md)
    assert strong["predicted_price"] > weak["predicted_price"]


def test_price_is_returned_in_rupees_not_log_units(artifacts):
    """The regression pipeline is wrapped in log/exp; the app must see rupees."""
    import prediction as pr
    models, md = artifacts
    r = pr.predict_car(dict(md["default_profile"]), models, md)
    assert 30_000 < r["predicted_price"] < 20_000_000
    assert r["predicted_price_lakh"] == pytest.approx(r["predicted_price"] / 100_000)


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
    r = pr.predict_car(dict(md["default_profile"]), models, md)
    feats = pr.top_influential_features(md, r["input_row"], top_n=5)
    assert len(feats) == 5
    assert feats[0]["feature"] == "age", "documented as the top feature"
    assert all(isinstance(f["car_value"], str) for f in feats)
