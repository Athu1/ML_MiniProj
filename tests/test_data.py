"""Dataset loading, validation and the documented data characteristics."""

import pytest


def test_dataset_matches_the_shape_uci_documents(dp):
    raw = dp.load_raw("mat")
    assert raw.shape == (395, 33)


def test_loader_refuses_a_file_of_the_wrong_shape(dp, tmp_path, cfg, monkeypatch):
    """A truncated or substituted download must fail loudly, not train quietly."""
    truncated = tmp_path / "student-mat.csv"
    rows = cfg.DATASET_FILES["mat"].read_text().splitlines()
    truncated.write_text("\n".join(rows[:50]))
    monkeypatch.setitem(cfg.DATASET_FILES, "mat", truncated)

    with pytest.raises(dp.DatasetError, match="UCI documents"):
        dp.load_raw("mat")


def test_missing_file_gives_an_actionable_message(dp, tmp_path, cfg, monkeypatch):
    monkeypatch.setitem(cfg.DATASET_FILES, "mat", tmp_path / "absent.csv")
    with pytest.raises(dp.DatasetError, match="not found"):
        dp.load_raw("mat")


def test_unknown_dataset_name_is_rejected(dp):
    with pytest.raises(dp.DatasetError, match="Unknown dataset"):
        dp.load_raw("nonexistent")


def test_both_separators_parse(dp, cfg):
    """The published files use ';' but some mirrors re-save with ','."""
    assert dp.load_raw("mat").shape[1] == 33   # semicolon-delimited
    assert dp.load_raw("por").shape[1] == 33   # comma-delimited mirror


def test_no_missing_values_or_duplicates(df):
    assert int(df.isna().sum().sum()) == 0
    assert int(df.duplicated().sum()) == 0


def test_documented_data_characteristics_still_hold(df, dp):
    """The specific numbers the README and report quote about the data."""
    s = dp.dataset_summary(df)
    assert s["n_students"] == 395
    assert s["n_at_risk"] == 130
    assert round(s["pct_at_risk"], 1) == 32.9
    assert s["n_zero_g3"] == 38
    assert round(s["majority_class_accuracy"], 4) == 0.6709


def test_the_zero_grade_records_are_the_documented_artefacts(df):
    """README and report claim all 38 have 0 absences AND a non-zero G1."""
    zeros = df[df["G3"] == 0]
    assert len(zeros) == 38
    assert (zeros["absences"] == 0).all(), "claim: all 38 record zero absences"
    assert (zeros["G1"] > 0).all(), "claim: all 38 have a non-zero G1"


def test_the_absences_correlation_claim(df):
    """README claims +0.034 overall but -0.213 excluding the zero-grade group."""
    overall = df["absences"].corr(df["G3"])
    graded = df[df["G3"] > 0]
    excl = graded["absences"].corr(graded["G3"])
    assert round(overall, 3) == 0.034
    assert round(excl, 3) == -0.213
    assert overall > 0 > excl, "the documented sign reversal is gone"


def test_encoding_produces_the_documented_column_count(dp, df):
    X, _ = dp.split_features_target(df, task="classification")
    pre = dp.build_preprocessor().fit(X)
    assert X.shape[1] == 30, "README documents 30 input features"
    assert len(dp.get_feature_names(pre)) == 43, "README documents 43 encoded columns"


def test_unseen_category_encodes_to_zeros_instead_of_raising(dp, df):
    """The web form can submit anything; the pipeline must not crash."""
    import pandas as pd
    X, _ = dp.split_features_target(df, task="classification")
    pre = dp.build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row.loc[row.index[0], "Mjob"] = "astronaut"
    out = pre.transform(row)
    assert out.shape[1] == len(dp.get_feature_names(pre))
