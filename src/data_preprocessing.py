"""
Data loading, validation, cleaning, target construction and the preprocessing
pipeline.

Design rules enforced here
--------------------------
1. The raw file is validated against the shape and checksum this project was
   built on, so a truncated or substituted file fails loudly instead of being
   trained on quietly.
2. Exact duplicate listings are removed BEFORE the train/test split. The raw
   file contains 1,202 of them (about 15%); left in, the same car lands in both
   the training and the test set and the test score measures memory, not skill.
   The effect is measured, not assumed -- see `train_models.py`.
3. The premium label is derived from the price, and the price, the label and the
   free-text name are then removed from the feature matrix. No model ever sees
   them as an input.
4. The preprocessor is a scikit-learn ColumnTransformer wrapped inside each
   model's Pipeline, so `fit` only ever sees training rows: the scaler's mean and
   the encoder's category list are never computed from test data.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg

# The first word of `name` is the brand, except where the brand is two words.
BRAND_FIXES = {"Land": "Land Rover", "Force": "Force Motors", "Ashok": "Ashok Leyland"}


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or does not match what was documented."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def file_sha256(path=None) -> str:
    path = cfg.DATASET_FILE if path is None else path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(validate: bool = True) -> pd.DataFrame:
    """Load the raw CSV exactly as published, with only the one column renamed.

    The shape is a hard check. The checksum is a warning rather than an error,
    because a student who opens the file in a spreadsheet and re-saves it changes
    the bytes without changing the data; the warning says the documented results
    may then differ.
    """
    path = cfg.DATASET_FILE
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}\n"
            "Place cardekho.csv in data/raw/ (see the Dataset section of README.md)."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"{path.name} could not be read as CSV: {exc}") from exc

    df = df.rename(columns=cfg.RAW_RENAMES)

    if validate:
        if df.shape != cfg.EXPECTED_RAW_SHAPE:
            raise DatasetError(
                f"{path.name} has shape {df.shape} but this project was built on "
                f"{cfg.EXPECTED_RAW_SHAPE}. Refusing to train on a file that does "
                "not match."
            )
        if file_sha256(path) != cfg.EXPECTED_SHA256:
            warnings.warn(
                f"{path.name} has the expected shape but its checksum differs from "
                "the file the documented results were produced from. Results may "
                "differ slightly.",
                stacklevel=2,
            )
    return df


def duplicate_summary(raw: pd.DataFrame) -> dict:
    """How many exact duplicate listings the raw file holds."""
    dup = raw.duplicated(keep="first")
    return {
        "n_rows": int(len(raw)),
        "n_duplicates": int(dup.sum()),
        "pct_duplicates": float(dup.mean() * 100),
        "n_unique": int((~dup).sum()),
    }


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(raw: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """Turn the raw listings into a modelling table.

    * `max_power` is stored as text in the raw file (one value is a blank space),
      so it is coerced to a number; anything unparseable becomes missing and is
      imputed later, inside the pipeline.
    * `age` replaces `year`; `brand` is cut from `name`.
    * Exact duplicate rows are dropped first when `dedupe` is True. `dedupe=False`
      exists only to power the duplicate-leakage demonstration.
    """
    df = raw.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    if dedupe:
        df = df.drop_duplicates(keep="first")

    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")
    df["age"] = cfg.REFERENCE_YEAR - df["year"]

    first_word = df["name"].str.split().str[0]
    df["brand"] = first_word.replace(BRAND_FIXES)

    df[cfg.CLASSIFICATION_TARGET] = (
        df[cfg.REGRESSION_TARGET] >= cfg.PREMIUM_THRESHOLD
    ).astype(int)

    return df.reset_index(drop=True)


def load_dataset(dedupe: bool = True) -> pd.DataFrame:
    """Load, validate and clean. The usual entry point."""
    return clean(load_raw(), dedupe=dedupe)


# --------------------------------------------------------------------------
# Feature / target split
# --------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame, task: str = "classification"):
    """Return (X, y) with every target-derived column stripped from X.

    task = "classification" -> y is the 0/1 premium label
    task = "regression"     -> y is the selling price in rupees
    """
    if task == "classification":
        y = df[cfg.CLASSIFICATION_TARGET]
    elif task == "regression":
        y = df[cfg.REGRESSION_TARGET]
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    X = df[list(cfg.NUMERIC_FEATURES) + list(cfg.CATEGORICAL_FEATURES)].copy()

    # Belt-and-braces: assert nothing derived from the target survived into X.
    forbidden = {cfg.REGRESSION_TARGET, cfg.CLASSIFICATION_TARGET, "name", "year"}
    present = forbidden.intersection(X.columns)
    if present:
        raise AssertionError(
            f"Leakage guard tripped: {sorted(present)} must not be model inputs."
        )

    return X, y


# --------------------------------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------------------------------
def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the shared preprocessing ColumnTransformer.

    Numeric branch   : median imputation -> StandardScaler
    Brand branch     : OneHotEncoder that pools rare and unseen brands
    Other categoricals: most-frequent imputation -> OneHotEncoder

    Median imputation matters here: 221 cars are missing mileage, engine and
    seats together, and 215 are missing max_power. A median is not pulled by the
    ten-million-rupee luxury cars the way a mean would be.

    Scaling is required for Logistic Regression and the SVM: `km_driven` runs to
    the hundreds of thousands while `seats` is 2-14, and the RBF kernel is a
    function of squared distance, so without scaling the odometer alone would
    dominate every prediction. Random Forest does not need it but shares the
    preprocessor so one definition has to be explained.

    `min_frequency` pools any brand with fewer than MIN_BRAND_FREQUENCY training
    cars (and any brand never seen) into one "infrequent" column. `drop="if_binary"`
    keeps one column per two-level feature such as transmission.
    """
    numeric_features = list(
        cfg.NUMERIC_FEATURES if numeric_features is None else numeric_features
    )
    categorical_features = list(
        cfg.CATEGORICAL_FEATURES if categorical_features is None
        else categorical_features
    )
    other_cats = [c for c in categorical_features if c != "brand"]

    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler())]
    )
    brand_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(
             handle_unknown="infrequent_if_exist",
             min_frequency=cfg.MIN_BRAND_FREQUENCY,
             sparse_output=False,
         ))]
    )
    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(
             drop="if_binary", handle_unknown="ignore", sparse_output=False,
         ))]
    )

    transformers = [("num", numeric_pipe, numeric_features)]
    if "brand" in categorical_features:
        transformers.append(("brand", brand_pipe, ["brand"]))
    if other_cats:
        transformers.append(("cat", cat_pipe, other_cats))

    return ColumnTransformer(
        transformers=transformers, remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Readable post-encoding feature names, for importance/coefficient charts."""
    return [str(n) for n in fitted_preprocessor.get_feature_names_out()]


# --------------------------------------------------------------------------
# Helpers used by the UI and the reports
# --------------------------------------------------------------------------
def dataset_summary(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard metric cards."""
    prem = df[cfg.CLASSIFICATION_TARGET]
    price = df[cfg.REGRESSION_TARGET]
    dup = duplicate_summary(raw)
    return {
        "n_raw_rows": dup["n_rows"],
        "n_duplicates_removed": dup["n_duplicates"],
        "pct_duplicates": dup["pct_duplicates"],
        "n_cars": int(len(df)),
        "n_raw_columns": int(raw.shape[1]),
        "n_model_features": len(cfg.ALL_FEATURES),
        "n_numeric_features": len(cfg.NUMERIC_FEATURES),
        "n_categorical_features": len(cfg.CATEGORICAL_FEATURES),
        "n_premium": int(prem.sum()),
        "n_standard": int(len(prem) - prem.sum()),
        "pct_premium": float(prem.mean() * 100),
        "majority_class_accuracy": float(max(prem.mean(), 1 - prem.mean())),
        "median_price": float(price.median()),
        "mean_price": float(price.mean()),
        "min_price": float(price.min()),
        "max_price": float(price.max()),
        "price_skew": float(price.skew()),
        "missing_cells_clean": int(df[cfg.ALL_FEATURES].isna().sum().sum()),
        "n_rows_missing_specs": int(
            df[["mileage", "engine", "seats"]].isna().all(axis=1).sum()
        ),
        "n_missing_max_power": int(df["max_power"].isna().sum()),
    }


def default_car_profile(df: pd.DataFrame) -> dict:
    """A 'typical car' used to pre-fill the form.

    Numeric features take the median, categorical features the mode.
    """
    profile = {}
    for col in cfg.NUMERIC_FEATURES:
        profile[col] = float(df[col].median())
    for col in cfg.CATEGORICAL_FEATURES:
        profile[col] = str(df[col].mode().iloc[0])
    profile["age"] = int(round(profile["age"]))
    profile["seats"] = int(round(profile["seats"]))
    profile["engine"] = int(round(profile["engine"]))
    return profile


def feature_ranges(df: pd.DataFrame) -> dict:
    """Observed min / max of each numeric feature, to warn on extrapolation."""
    return {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in cfg.NUMERIC_FEATURES
    }


def known_brands(df: pd.DataFrame) -> list[str]:
    """Brands ordered by how many cars they have, for the form dropdown."""
    return df["brand"].value_counts().index.tolist()
