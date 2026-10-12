"""
Loading the trained pipelines and predicting the price of a single car.

The Streamlit app never trains anything. It loads the pipelines saved by
`src/train_models.py`; if they are missing it says so and stops. Because each
saved object is a full scikit-learn Pipeline, the preprocessing applied to a
form submission is byte-for-byte the preprocessing used during training -- the
same imputer statistics, the same scaler mean/std, the same category list.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

import config as cfg


# An unseen categorical level is an expected, handled condition here: the
# encoder maps it to all-zeros and `unknown_categories()` reports it to the
# user. scikit-learn's warning about it would otherwise be printed on every
# prediction, so it is silenced at this one call site only.
warnings.filterwarnings(
    "ignore", message=".*unknown categories.*", category=UserWarning,
    module=r"sklearn\.preprocessing\._encoders",
)


class ModelsNotTrainedError(RuntimeError):
    """Raised when the saved model files are missing or unreadable."""


class InvalidCarInputError(ValueError):
    """Raised when a submitted car profile cannot be scored."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def models_available() -> bool:
    return cfg.METADATA_FILE.exists() and all(
        p.exists() for p in cfg.MODEL_FILES.values()
    )


def missing_artifacts() -> list[str]:
    missing = [p.name for p in cfg.MODEL_FILES.values() if not p.exists()]
    if not cfg.METADATA_FILE.exists():
        missing.append(cfg.METADATA_FILE.name)
    return missing


def load_artifacts() -> tuple[dict, dict]:
    """Return (models, metadata), both loaded from disk.

    Raises ModelsNotTrainedError with an actionable message rather than letting
    a FileNotFoundError or an unpickling error reach the user as a traceback.
    """
    if not models_available():
        raise ModelsNotTrainedError(
            "Trained models were not found in models/. Missing: "
            + ", ".join(missing_artifacts())
            + ". Run `python src/train_models.py` first."
        )
    try:
        models = {name: joblib.load(path) for name, path in cfg.MODEL_FILES.items()}
        metadata = joblib.load(cfg.METADATA_FILE)
    except Exception as exc:  # corrupted pickle, version mismatch, truncated file
        raise ModelsNotTrainedError(
            "The saved model files could not be loaded "
            f"({type(exc).__name__}: {exc}). They may have been written by a "
            "different scikit-learn version or left incomplete. Re-run "
            "`python src/train_models.py` to rebuild them."
        ) from exc
    return models, metadata


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def build_input_row(car: dict, default_profile: dict) -> pd.DataFrame:
    """Turn a form submission into the single-row DataFrame the pipelines expect.

    Features the form does not expose are filled from `default_profile` (the
    dataset median for numeric columns, the mode for categorical ones).
    """
    if not isinstance(car, dict) or not car:
        raise InvalidCarInputError("No car details were provided.")

    row = {}
    for col in cfg.ALL_FEATURES:
        value = car.get(col, default_profile.get(col))
        if value is None:
            raise InvalidCarInputError(
                f"No value available for the required feature '{col}'."
            )
        row[col] = value

    for col in cfg.NUMERIC_FEATURES:
        label = cfg.FEATURE_LABELS.get(col, col)
        try:
            row[col] = float(row[col])
        except (TypeError, ValueError) as exc:
            raise InvalidCarInputError(
                f"'{label}' must be a number, got {row[col]!r}."
            ) from exc
        if not np.isfinite(row[col]):
            raise InvalidCarInputError(f"'{label}' must be a finite number.")
        if col in ("age", "km_driven", "engine", "max_power", "seats") and row[col] < 0:
            raise InvalidCarInputError(f"'{label}' cannot be negative.")

    for col in cfg.CATEGORICAL_FEATURES:
        row[col] = str(row[col])

    return pd.DataFrame([row])[cfg.ALL_FEATURES]


def unknown_categories(car_row: pd.DataFrame, metadata: dict) -> list[str]:
    """Categorical values never seen in training (encoded as all-zeros / pooled).

    An unseen brand is pooled with the rare brands; any other unseen level is
    encoded as all-zeros. Either way the prediction runs but leans on the rest.
    """
    known_brands = set(metadata.get("known_brands", []))
    out = []
    for col in cfg.CATEGORICAL_FEATURES:
        value = str(car_row.iloc[0][col])
        label = cfg.FEATURE_LABELS.get(col, col)
        if col == "brand":
            if known_brands and value not in known_brands:
                out.append(f"{label} = {value!r}")
        elif value not in cfg.CATEGORICAL_LABELS.get(col, {}):
            out.append(f"{label} = {value!r}")
    return out


def out_of_range_features(car_row: pd.DataFrame, metadata: dict) -> list[str]:
    """Numeric inputs outside the range the models were trained on."""
    out = []
    for col, rng in metadata.get("feature_ranges", {}).items():
        v = float(car_row.iloc[0][col])
        if v < rng["min"] or v > rng["max"]:
            out.append(
                f"{cfg.FEATURE_LABELS.get(col, col)} = {v:g} "
                f"(trained on {rng['min']:g} to {rng['max']:g})"
            )
    return out


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_premium_probability(pipeline, X: pd.DataFrame) -> float | None:
    """P(premium) from a classifier, or None if the model cannot express one."""
    model = pipeline.named_steps.get("model", pipeline)
    if hasattr(model, "predict_proba"):
        return float(pipeline.predict_proba(X)[0, 1])
    if hasattr(model, "decision_function"):
        margin = float(np.ravel(pipeline.decision_function(X))[0])
        return float(1.0 / (1.0 + np.exp(-margin)))
    return None


def predict_car(car: dict, models: dict, metadata: dict,
                classifier: str | None = None,
                threshold: float | None = None) -> dict:
    """Price one car with the regression model and classify it with all three.

    Returns the predicted price in rupees, the chosen classifier's verdict and
    probability, and every classifier's verdict so the UI can show where the
    models agree and where they do not.
    """
    classifier = classifier or metadata.get("best_classifier", "Random Forest")
    if classifier not in models:
        raise InvalidCarInputError(f"Unknown classifier {classifier!r}.")
    if threshold is not None and not (0.0 < float(threshold) < 1.0):
        raise InvalidCarInputError(
            f"Decision threshold must be strictly between 0 and 1, got {threshold!r}."
        )

    X = build_input_row(car, metadata.get("default_profile", {}))

    try:
        price = float(models[cfg.REGRESSION_MODEL_NAME].predict(X)[0])
    except Exception as exc:
        raise InvalidCarInputError(
            f"The regression model could not price this car "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    per_model = {}
    for name in cfg.CLASSIFICATION_MODEL_NAMES:
        pipe = models.get(name)
        if pipe is None:
            continue
        try:
            proba = predict_premium_probability(pipe, X)
            if threshold is None or proba is None:
                label = int(pipe.predict(X)[0])
            else:
                label = int(proba >= float(threshold))
        except Exception as exc:
            raise InvalidCarInputError(
                f"{name} could not score this car ({type(exc).__name__}: {exc})."
            ) from exc
        # SVC labels by the sign of its decision function but reports a separately
        # Platt-scaled probability, so near the boundary they can disagree. The
        # label is kept (every reported metric uses it) and the clash is flagged.
        per_model[name] = {
            "premium": bool(label),
            "label": cfg.POSITIVE_CLASS_LABEL if label else cfg.NEGATIVE_CLASS_LABEL,
            "probability": proba,
            "probability_disagrees_with_label": (
                proba is not None and bool(label) != bool(proba >= 0.5)
            ),
        }

    chosen = per_model[classifier]
    votes = sum(1 for v in per_model.values() if v["premium"])

    alt_threshold = (
        cfg.DEFAULT_THRESHOLD if threshold is not None
        else (metadata.get("classification", {}).get(classifier, {})
              .get("threshold") or {}).get("threshold")
    )
    alt_label = None
    if alt_threshold is not None and chosen["probability"] is not None:
        alt_label = bool(chosen["probability"] >= float(alt_threshold))

    return {
        "input_row": X,
        "predicted_price": price,
        "predicted_price_lakh": price / cfg.LAKH,
        "classifier": classifier,
        "premium": chosen["premium"],
        "label": chosen["label"],
        "premium_probability": chosen["probability"],
        "per_model": per_model,
        "models_agree": votes in (0, len(per_model)),
        "premium_votes": votes,
        "probability_disagrees_with_label": chosen["probability_disagrees_with_label"],
        "threshold_used": (float(threshold) if threshold is not None
                           else cfg.DEFAULT_THRESHOLD),
        "threshold_is_tuned": threshold is not None,
        "alt_threshold": (float(alt_threshold) if alt_threshold is not None else None),
        "alt_premium": alt_label,
        "threshold_decided_it": (
            alt_label is not None and alt_label != chosen["premium"]
        ),
        "n_models": len(per_model),
        "unknown_categories": unknown_categories(X, metadata),
        "out_of_range": out_of_range_features(X, metadata),
        # The regression model's own view of the 6-lakh line, which can disagree
        # with the classifiers. Surfacing the disagreement is the honest choice.
        "regression_implies_premium": bool(price >= cfg.PREMIUM_THRESHOLD),
    }


def top_influential_features(metadata: dict, car_row: pd.DataFrame,
                             top_n: int = 6) -> list[dict]:
    """The inputs the Random Forest relies on most, with this car's values.

    A GLOBAL view (which inputs matter across all cars), not a per-car
    attribution. Permutation importance is used because it is measured on
    held-out data.
    """
    perm = metadata.get("rf_permutation_importance") or []
    if not perm:
        perm = [
            {"feature": r["feature"], "permutation_importance": r["impurity_importance"]}
            for r in metadata.get("rf_impurity_importance", [])
        ]

    out = []
    for rec in perm[:top_n]:
        feature = rec["feature"]
        if feature not in car_row.columns:
            continue
        value = car_row.iloc[0][feature]
        if isinstance(value, (int, float, np.integer, np.floating)):
            readable = f"{float(value):g}"
        else:
            readable = cfg.CATEGORICAL_LABELS.get(feature, {}).get(value, str(value))
        out.append({
            "feature": feature,
            "label": cfg.FEATURE_LABELS.get(feature, feature),
            "importance": float(rec.get("permutation_importance", 0.0)),
            "car_value": readable,
        })
    return out
