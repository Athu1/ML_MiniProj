"""Tests for the project's central claim: no target leakage.

If any test in this file fails, the headline metrics in README.md are not
measuring what they say they measure. These are the most important tests here.
"""

import pytest


FORBIDDEN = {"G1", "G2", "G3", "at_risk"}


def test_grade_columns_are_not_in_the_configured_feature_list(cfg):
    """The feature lists in config.py must not name any grade column."""
    overlap = FORBIDDEN.intersection(cfg.ALL_FEATURES)
    assert not overlap, f"grade columns configured as model inputs: {overlap}"


def test_classification_feature_matrix_has_no_grade_columns(dp, df):
    X, _ = dp.split_features_target(df, task="classification")
    overlap = FORBIDDEN.intersection(X.columns)
    assert not overlap, f"leaked into X: {overlap}"


def test_regression_feature_matrix_has_no_grade_columns(dp, df):
    """G3 is the regression target, so it must not also be an input."""
    X, y = dp.split_features_target(df, task="regression")
    overlap = FORBIDDEN.intersection(X.columns)
    assert not overlap, f"leaked into X: {overlap}"
    assert y.name == "G3"


def test_leakage_guard_actually_raises(dp, df, monkeypatch, cfg):
    """The assertion must FIRE, not merely exist.

    A guard that never triggers is not a guard. This adds `G2` to the
    configured numeric features and asserts the build then fails, which is the
    only way to know the check is live.
    """
    monkeypatch.setattr(cfg, "NUMERIC_FEATURES",
                        list(cfg.NUMERIC_FEATURES) + ["G2"])
    with pytest.raises(AssertionError, match="Leakage guard"):
        dp.split_features_target(df, task="classification")


def test_leaky_variant_is_opt_in_only(dp, df):
    """`leaky=True` is the documented escape hatch for the demonstration.

    It must include the grades when asked (otherwise the leakage comparison in
    the README is not measuring leakage) and exclude them by default.
    """
    clean, _ = dp.split_features_target(df, task="regression", leaky=False)
    leaky, _ = dp.split_features_target(df, task="regression", leaky=True)
    assert "G1" not in clean.columns and "G2" not in clean.columns
    assert "G1" in leaky.columns and "G2" in leaky.columns
    assert "G3" not in leaky.columns, "the target must never be an input"


def test_target_is_derived_from_g3_at_the_documented_threshold(dp, df, cfg):
    expected = (df["G3"] < cfg.AT_RISK_THRESHOLD).astype(int)
    assert (df[cfg.CLASSIFICATION_TARGET] == expected).all()
    assert cfg.AT_RISK_THRESHOLD == 10, "README documents a pass mark of 10"


def test_preprocessor_is_fitted_on_training_rows_only(dp, df, cfg):
    """Scaling statistics must come from the training split, not the full data.

    Fitting a scaler before splitting is the most common silent leak in a
    student ML project. This checks the pipeline's scaler mean matches the
    TRAIN mean and not the whole-dataset mean.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    X, y = dp.split_features_target(df, task="classification")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    pre = dp.build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["num"].named_steps["scaler"]

    idx = cfg.NUMERIC_FEATURES.index("absences")
    assert np.isclose(scaler.mean_[idx], X_train["absences"].mean())
    assert not np.isclose(scaler.mean_[idx], X["absences"].mean()), (
        "scaler mean equals the FULL dataset mean — preprocessing leaked"
    )
