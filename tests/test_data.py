"""Dataset loading, validation and the documented data characteristics."""

import pytest


def test_raw_dataset_has_the_documented_shape(raw):
    assert raw.shape == (8128, 12)


def test_checksum_matches_the_documented_file(dp, cfg):
    assert dp.file_sha256() == cfg.EXPECTED_SHA256


def test_loader_refuses_a_file_of_the_wrong_shape(dp, tmp_path, cfg, monkeypatch):
    """A truncated or substituted download must fail loudly, not train quietly."""
    truncated = tmp_path / "cardekho.csv"
    rows = cfg.DATASET_FILE.read_text().splitlines()
    truncated.write_text("\n".join(rows[:50]))
    monkeypatch.setattr(cfg, "DATASET_FILE", truncated)
    with pytest.raises(dp.DatasetError, match="Refusing to train"):
        dp.load_raw()


def test_missing_file_gives_an_actionable_message(dp, tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(cfg, "DATASET_FILE", tmp_path / "absent.csv")
    with pytest.raises(dp.DatasetError, match="not found"):
        dp.load_raw()


def test_a_changed_checksum_warns_but_does_not_fail(dp, tmp_path, cfg, monkeypatch):
    """Re-saving in a spreadsheet changes bytes, not data: warn, don't refuse."""
    resaved = tmp_path / "cardekho.csv"
    resaved.write_text(cfg.DATASET_FILE.read_text() + "\n")
    monkeypatch.setattr(cfg, "DATASET_FILE", resaved)
    with pytest.warns(UserWarning, match="checksum"):
        raw = dp.load_raw()
    assert raw.shape == (8128, 12)


def test_documented_duplicate_count(dp, raw):
    d = dp.duplicate_summary(raw)
    assert d["n_duplicates"] == 1202
    assert round(d["pct_duplicates"], 1) == 14.8


def test_cleaned_dataset_has_no_duplicates_and_the_documented_size(df):
    assert len(df) == 6926
    assert int(df.duplicated().sum()) == 0


def test_documented_data_characteristics_still_hold(df, raw, dp):
    s = dp.dataset_summary(raw, df)
    assert s["n_cars"] == 6926
    assert s["n_premium"] == 2041
    assert round(s["pct_premium"], 2) == 29.47
    assert round(s["majority_class_accuracy"], 4) == 0.7053
    assert s["median_price"] == 400_000
    assert round(s["price_skew"], 2) == 5.57
    assert s["n_rows_missing_specs"] == 208
    assert s["n_missing_max_power"] == 206


def test_brand_extraction_fixes_two_word_brands(df):
    brands = set(df["brand"])
    assert {"Land Rover", "Force Motors", "Ashok Leyland"} <= brands
    assert not {"Land", "Force", "Ashok"} & brands
    assert df["brand"].nunique() == 32


def test_max_power_text_value_becomes_missing_not_a_crash(df):
    assert df["max_power"].dtype.kind == "f"
    assert df["max_power"].isna().sum() == 206


def test_documented_correlations_with_log_price(df):
    import numpy as np
    lp = np.log(df["selling_price"])
    assert round(df["age"].corr(lp), 3) == -0.719
    assert round(df["max_power"].corr(lp), 3) == 0.654


def test_encoding_produces_the_documented_column_count(dp, df):
    X, _ = dp.split_features_target(df, task="classification")
    assert X.shape[1] == 11, "README documents 11 input features"
    pre = dp.build_preprocessor().fit(X)
    assert len(dp.get_feature_names(pre)) == 37, "README documents 37 encoded columns (full data)"


def test_unseen_category_does_not_raise(dp, df):
    """The web form can submit anything; the pipeline must not crash."""
    X, _ = dp.split_features_target(df, task="classification")
    pre = dp.build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row.loc[row.index[0], "brand"] = "Zeppelin"
    row.loc[row.index[0], "fuel"] = "Hydrogen"
    out = pre.transform(row)
    assert out.shape[1] == len(dp.get_feature_names(pre))


def test_rare_brands_are_pooled_into_one_column(dp, df):
    X, _ = dp.split_features_target(df, task="classification")
    names = dp.get_feature_names(dp.build_preprocessor().fit(X))
    assert any("infrequent" in n for n in names)
    assert not any(n == "brand_Daewoo" for n in names), "rare brand should be pooled"
