"""
Data loading, validation, target construction and the preprocessing pipeline.

Design rules enforced here
--------------------------
1. The raw file is validated against the shape UCI documents, so a truncated or
   substituted download fails loudly instead of being trained on quietly.
2. The at-risk label is derived from G3 and then G3/G1/G2 are *removed* from the
   feature matrix. No model in this project ever sees a grade as an input
   (except the explicitly-labelled leakage demonstration).
3. The preprocessor is a scikit-learn ColumnTransformer wrapped inside each
   model's Pipeline. Because it lives inside the Pipeline, `fit` only ever sees
   training rows -- the scaler's mean/std and the encoder's category list are
   never computed from test data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or does not match the UCI spec."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(dataset: str = cfg.DEFAULT_DATASET, validate: bool = True) -> pd.DataFrame:
    """Load one of the UCI Student Performance files.

    The published files use ';' as the separator. Some mirrors re-save them with
    ','; both are handled by sniffing which separator yields the 33 documented
    columns, so provenance differences do not silently produce a 1-column frame.
    """
    if dataset not in cfg.DATASET_FILES:
        raise DatasetError(
            f"Unknown dataset {dataset!r}. Choose one of {sorted(cfg.DATASET_FILES)}."
        )

    path = cfg.DATASET_FILES[dataset]
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}\n"
            "Place the UCI Student Performance CSV files in data/raw/ "
            "(see the Dataset section of README.md for the download link)."
        )

    df = None
    for sep in (";", ","):
        candidate = pd.read_csv(path, sep=sep)
        if candidate.shape[1] == 33:
            df = candidate
            break
    if df is None:
        raise DatasetError(
            f"{path.name} did not parse into the 33 columns documented by UCI "
            "with either ';' or ',' as the separator. The file may be corrupted."
        )

    # Object columns in the raw files are quoted ("GP"); strip stray quotes/space.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    if validate:
        expected = cfg.EXPECTED_SHAPES.get(dataset)
        if expected and df.shape != expected:
            raise DatasetError(
                f"{path.name} has shape {df.shape} but UCI documents {expected}. "
                "Refusing to train on a file that does not match the published "
                "dataset."
            )

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Append the binary `at_risk` label derived from the final grade G3.

    Threshold: G3 < 10 on the 0-20 Portuguese scale, i.e. below the pass mark.
    """
    out = df.copy()
    out[cfg.CLASSIFICATION_TARGET] = (
        out[cfg.REGRESSION_TARGET] < cfg.AT_RISK_THRESHOLD
    ).astype(int)
    return out


def load_dataset(dataset: str = cfg.DEFAULT_DATASET) -> pd.DataFrame:
    """Load a raw file and attach the at-risk target. The usual entry point."""
    return add_target(load_raw(dataset))


# --------------------------------------------------------------------------
# Feature / target split
# --------------------------------------------------------------------------
def split_features_target(
    df: pd.DataFrame, task: str = "classification", leaky: bool = False
):
    """Return (X, y) with every leakage-prone column stripped from X.

    Parameters
    ----------
    task   : "classification" -> y is the at_risk label
             "regression"     -> y is the final grade G3
    leaky  : when True, G1 and G2 are *kept* in X. This exists only to power the
             side-by-side "what data leakage does to your metrics" comparison and
             is never used for the models the application predicts with.
    """
    if task == "classification":
        y = df[cfg.CLASSIFICATION_TARGET]
    elif task == "regression":
        y = df[cfg.REGRESSION_TARGET]
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    numeric = list(cfg.NUMERIC_FEATURES)
    if leaky:
        numeric = numeric + list(cfg.LEAKY_EXTRA_NUMERIC)

    X = df[numeric + list(cfg.CATEGORICAL_FEATURES)].copy()

    # Belt-and-braces: assert no target-derived column survived into X.
    forbidden = {cfg.REGRESSION_TARGET, cfg.CLASSIFICATION_TARGET}
    if not leaky:
        forbidden |= set(cfg.LEAKY_EXTRA_NUMERIC)
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
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the shared preprocessing ColumnTransformer.

    Numeric branch      : median imputation -> StandardScaler
    Categorical branch  : most-frequent imputation -> OneHotEncoder

    `scale_numeric=False` is available for Random Forest, which is invariant to
    monotonic rescaling of its inputs, but the project keeps scaling on for all
    models so that exactly one preprocessing definition has to be explained and
    audited. Scaling is *required* for Logistic Regression and SVM: both are
    distance/penalty based, and `absences` (0-93) would otherwise dominate
    `studytime` (1-4) purely because of its units.

    `drop="if_binary"` keeps one column per yes/no feature, which avoids
    redundant collinear dummies and makes the Logistic Regression coefficients
    directly readable. `handle_unknown="ignore"` means a category the model never
    saw during training is encoded as all-zeros rather than raising -- important
    for the Streamlit form, where a user could submit an unexpected value.
    """
    numeric_features = list(
        cfg.NUMERIC_FEATURES if numeric_features is None else numeric_features
    )
    categorical_features = list(
        cfg.CATEGORICAL_FEATURES if categorical_features is None
        else categorical_features
    )

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Readable post-encoding feature names, for importance/coefficient charts."""
    return [str(n) for n in fitted_preprocessor.get_feature_names_out()]


# --------------------------------------------------------------------------
# Small helpers used by the UI / reports
# --------------------------------------------------------------------------
def dataset_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard metric cards."""
    at_risk = df[cfg.CLASSIFICATION_TARGET]
    return {
        "n_students": int(len(df)),
        "n_raw_columns": int(df.shape[1] - 1),  # minus the derived at_risk column
        "n_model_features": len(cfg.ALL_FEATURES),
        "n_numeric_features": len(cfg.NUMERIC_FEATURES),
        "n_categorical_features": len(cfg.CATEGORICAL_FEATURES),
        "n_at_risk": int(at_risk.sum()),
        "n_not_at_risk": int(len(at_risk) - at_risk.sum()),
        "pct_at_risk": float(at_risk.mean() * 100),
        "majority_class_accuracy": float(max(at_risk.mean(), 1 - at_risk.mean())),
        "mean_g3": float(df[cfg.REGRESSION_TARGET].mean()),
        "median_g3": float(df[cfg.REGRESSION_TARGET].median()),
        "n_zero_g3": int((df[cfg.REGRESSION_TARGET] == 0).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def default_student_profile(df: pd.DataFrame) -> dict:
    """A 'typical student' used to pre-fill the form and to fill hidden features.

    Numeric features take the dataset median, categorical features the mode.
    Being explicit about this matters: the UI tells the user that features not
    shown on the form are held at these population-typical values.
    """
    profile = {}
    for col in cfg.NUMERIC_FEATURES:
        profile[col] = int(round(float(df[col].median())))
    for col in cfg.CATEGORICAL_FEATURES:
        profile[col] = str(df[col].mode().iloc[0])
    return profile
