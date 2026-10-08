"""
Loading the trained pipelines and predicting for a single student.

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


class InvalidStudentInputError(ValueError):
    """Raised when a submitted student profile cannot be scored."""


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
def build_input_row(student: dict, default_profile: dict) -> pd.DataFrame:
    """Turn a form submission into the single-row DataFrame the pipelines expect.

    Features the form does not expose are filled from `default_profile` (the
    dataset median for numeric columns, the mode for categorical ones). The UI
    states that this is happening, so a prediction is never silently based on
    values the user did not supply and does not know about.
    """
    if not isinstance(student, dict) or not student:
        raise InvalidStudentInputError("No student details were provided.")

    row = {}
    for col in cfg.ALL_FEATURES:
        value = student.get(col, default_profile.get(col))
        if value is None:
            raise InvalidStudentInputError(
                f"No value available for the required feature '{col}'."
            )
        row[col] = value

    # Numeric columns must actually be numeric and finite.
    for col in cfg.NUMERIC_FEATURES:
        try:
            row[col] = float(row[col])
        except (TypeError, ValueError) as exc:
            raise InvalidStudentInputError(
                f"'{cfg.FEATURE_LABELS.get(col, col)}' must be a number, "
                f"got {row[col]!r}."
            ) from exc
        if not np.isfinite(row[col]):
            raise InvalidStudentInputError(
                f"'{cfg.FEATURE_LABELS.get(col, col)}' must be a finite number."
            )

    # Categorical columns are passed through as strings. An unexpected level is
    # not an error: OneHotEncoder(handle_unknown="ignore") encodes it as
    # all-zeros, which the UI warns about rather than refusing outright.
    for col in cfg.CATEGORICAL_FEATURES:
        row[col] = str(row[col])

    return pd.DataFrame([row])[cfg.ALL_FEATURES]


def unknown_categories(student_row: pd.DataFrame, metadata: dict) -> list[str]:
    """List categorical values that never appeared in the training data.

    Such a value is encoded as all-zeros, so the prediction still works but
    leans entirely on the other features. Worth telling the user about.
    """
    known = metadata.get("encoded_feature_names", [])
    unknown = []
    for col in cfg.CATEGORICAL_FEATURES:
        value = str(student_row.iloc[0][col])
        expected = cfg.CATEGORICAL_LABELS.get(col, {})
        # A binary column is encoded as a single "<col>_<level>" feature, so the
        # label map is the reliable source of the levels seen during training.
        if expected and value not in expected:
            unknown.append(f"{cfg.FEATURE_LABELS.get(col, col)} = {value!r}")
        elif not expected and f"{col}_{value}" not in known:
            unknown.append(f"{cfg.FEATURE_LABELS.get(col, col)} = {value!r}")
    return unknown


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_risk_probability(pipeline, X: pd.DataFrame) -> float | None:
    """P(at risk) from a classifier, or None if the model cannot express one.

    Falls back to `decision_function` passed through a logistic squash when a
    model exposes only a margin. The fallback is monotonic in the margin, so it
    ranks students correctly, but it is not a calibrated probability -- the UI
    labels it accordingly.
    """
    model = pipeline.named_steps.get("model", pipeline)
    if hasattr(model, "predict_proba"):
        return float(pipeline.predict_proba(X)[0, 1])
    if hasattr(model, "decision_function"):
        margin = float(np.ravel(pipeline.decision_function(X))[0])
        return float(1.0 / (1.0 + np.exp(-margin)))
    return None


def predict_student(student: dict, models: dict, metadata: dict,
                    classifier: str | None = None) -> dict:
    """Score one student with the regression model and all three classifiers.

    Returns the predicted grade, the chosen classifier's verdict and
    probability, and every classifier's verdict so the UI can show where the
    models agree and where they do not -- which is more honest than presenting
    one model's answer as "the" answer.
    """
    classifier = classifier or metadata.get("best_classifier", "Random Forest")
    if classifier not in models:
        raise InvalidStudentInputError(f"Unknown classifier {classifier!r}.")

    X = build_input_row(student, metadata.get("default_profile", {}))

    try:
        raw_score = float(models[cfg.REGRESSION_MODEL_NAME].predict(X)[0])
    except Exception as exc:
        raise InvalidStudentInputError(
            f"The regression model could not score this student "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    # Linear regression is unbounded, so it can return a value outside 0-20.
    # Both numbers are kept: the UI shows the clipped one and mentions the raw
    # value when clipping changed it, rather than hiding the model's behaviour.
    score = float(np.clip(raw_score, 0, cfg.GRADE_MAX))

    per_model = {}
    for name in cfg.CLASSIFICATION_MODEL_NAMES:
        pipe = models.get(name)
        if pipe is None:
            continue
        try:
            label = int(pipe.predict(X)[0])
            proba = predict_risk_probability(pipe, X)
        except Exception as exc:
            raise InvalidStudentInputError(
                f"{name} could not score this student "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        # SVC decides its label from the sign of the decision function, while
        # its probability comes from a separate Platt-scaling fit. Near the
        # boundary the two can disagree -- a student can be labelled "At Risk"
        # with an estimated probability just under 50%. The label is kept as the
        # model's real decision (it is what every reported metric is computed
        # from) and the disagreement is flagged so the UI can explain it instead
        # of looking like a bug.
        proba_disagrees = (
            proba is not None and bool(label) != bool(proba >= 0.5)
        )
        per_model[name] = {
            "at_risk": bool(label),
            "label": cfg.POSITIVE_CLASS_LABEL if label else cfg.NEGATIVE_CLASS_LABEL,
            "probability": proba,
            "probability_disagrees_with_label": proba_disagrees,
        }

    chosen = per_model[classifier]
    votes = sum(1 for v in per_model.values() if v["at_risk"])

    return {
        "input_row": X,
        "predicted_score": score,
        "predicted_score_raw": raw_score,
        "score_was_clipped": abs(raw_score - score) > 1e-9,
        "classifier": classifier,
        "at_risk": chosen["at_risk"],
        "risk_label": chosen["label"],
        "risk_probability": chosen["probability"],
        "per_model": per_model,
        "models_agree": votes in (0, len(per_model)),
        "at_risk_votes": votes,
        "probability_disagrees_with_label": chosen["probability_disagrees_with_label"],
        "n_models": len(per_model),
        "unknown_categories": unknown_categories(X, metadata),
        # The regression model's own view of the pass mark, which can disagree
        # with the classifiers. Surfacing the disagreement is the honest choice.
        "regression_implies_at_risk": bool(score < cfg.AT_RISK_THRESHOLD),
    }


def top_influential_features(metadata: dict, student_row: pd.DataFrame,
                             top_n: int = 6) -> list[dict]:
    """The features the Random Forest relies on most, with this student's values.

    This is a GLOBAL view -- it describes which inputs the model leans on across
    all students, not a per-student attribution, and the UI says so. Permutation
    importance is used rather than impurity importance because it is measured on
    held-out data and is not inflated by a feature simply having many distinct
    values.
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
        if feature not in student_row.columns:
            continue
        value = student_row.iloc[0][feature]
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            fallback = f"{numeric:g}"
            readable = cfg.ORDINAL_SCALES.get(feature, {}).get(int(numeric), fallback)
        else:
            readable = cfg.CATEGORICAL_LABELS.get(feature, {}).get(value, str(value))
        out.append({
            "feature": feature,
            "label": cfg.FEATURE_LABELS.get(feature, feature),
            "importance": float(rec.get("permutation_importance", 0.0)),
            "student_value": readable,
        })
    return out
