"""Tests for the project's central claim: no target leakage.

If any test in this file fails, the headline metrics in README.md are not
measuring what they say they measure. These are the most important tests here.
"""

import pytest

FORBIDDEN = {"selling_price", "is_premium", "name", "year"}


def test_forbidden_columns_are_not_in_the_configured_feature_list(cfg):
    overlap = FORBIDDEN.intersection(cfg.ALL_FEATURES)
    assert not overlap, f"configured as model inputs: {overlap}"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_feature_matrix_has_no_target_derived_columns(dp, df, task):
    X, _ = dp.split_features_target(df, task=task)
    overlap = FORBIDDEN.intersection(X.columns)
    assert not overlap, f"leaked into X: {overlap}"


def test_targets_are_the_documented_columns(dp, df, cfg):
    _, y_reg = dp.split_features_target(df, task="regression")
    _, y_clf = dp.split_features_target(df, task="classification")
    assert y_reg.name == "selling_price"
    assert y_clf.name == "is_premium"


@pytest.mark.parametrize("bad", ["selling_price", "name", "year"])
def test_leakage_guard_actually_raises(dp, df, monkeypatch, cfg, bad):
    """The assertion must FIRE, not merely exist."""
    monkeypatch.setattr(cfg, "NUMERIC_FEATURES", list(cfg.NUMERIC_FEATURES) + [bad])
    with pytest.raises(AssertionError, match="Leakage guard"):
        dp.split_features_target(df, task="classification")


def test_premium_label_is_derived_from_price_at_the_documented_threshold(df, cfg):
    expected = (df["selling_price"] >= cfg.PREMIUM_THRESHOLD).astype(int)
    assert (df[cfg.CLASSIFICATION_TARGET] == expected).all()
    assert cfg.PREMIUM_THRESHOLD == 600_000, "README documents a 6 lakh cut-off"


def test_preprocessor_is_fitted_on_training_rows_only(dp, df, cfg):
    """Scaling statistics must come from the training split, not the full data."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    X, y = dp.split_features_target(df, task="classification")
    X_train, _, _, _ = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    pre = dp.build_preprocessor().fit(X_train)
    num = pre.named_transformers_["num"]
    idx = cfg.NUMERIC_FEATURES.index("km_driven")
    # The imputer fills with the training median, the scaler centres on the
    # training mean of the imputed column.
    train_imputed = num.named_steps["imputer"].transform(X_train[cfg.NUMERIC_FEATURES])
    assert np.isclose(num.named_steps["scaler"].mean_[idx], train_imputed[:, idx].mean())
    assert not np.isclose(num.named_steps["scaler"].mean_[idx], X["km_driven"].mean()), (
        "scaler mean equals the FULL dataset mean -- preprocessing leaked"
    )


def test_duplicates_are_removed_before_any_split(df):
    assert int(df.duplicated().sum()) == 0
