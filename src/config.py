"""
Central configuration for the Student Academic Performance & At-Risk
Prediction System.

Everything that another module might want to agree on -- paths, the random
seed, the at-risk threshold, and (most importantly) which columns are allowed
to be used as model inputs -- lives here, so there is exactly one place to
look when answering "what did this project actually train on?".
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# The two files published with the UCI Student Performance dataset.
# "mat" (Mathematics, 395 students) is the project default -- see README for why.
DATASET_FILES = {
    "mat": DATA_RAW_DIR / "student-mat.csv",
    "por": DATA_RAW_DIR / "student-por.csv",
}
DEFAULT_DATASET = "mat"

# Row/column counts documented by UCI. Used as an integrity check at load time
# so a corrupted or silently-substituted file is caught instead of trained on.
EXPECTED_SHAPES = {"mat": (395, 33), "por": (649, 33)}

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
REGRESSION_TARGET = "G3"          # final grade, 0-20 scale
CLASSIFICATION_TARGET = "at_risk"  # derived: 1 if G3 < AT_RISK_THRESHOLD
AT_RISK_THRESHOLD = 10             # Portuguese pass mark on a 0-20 scale
GRADE_MAX = 20

# Class 1 = "At Risk" is the positive class for precision/recall/F1/ROC-AUC.
POSITIVE_CLASS_LABEL = "At Risk"
NEGATIVE_CLASS_LABEL = "Not At Risk"

# --------------------------------------------------------------------------
# Decision threshold
# --------------------------------------------------------------------------
# The probability above which a student is labelled At Risk. 0.5 is the
# scikit-learn default and is what `predict()` uses, but it is an arbitrary
# choice, not a derived one: on this dataset a third of the test students sit
# within 0.1 of it, so for those students the verdict is decided by this
# constant rather than by the model.
DEFAULT_THRESHOLD = 0.5

# How many false alarms one missed at-risk student is considered worth.
#
# This is a POLICY judgement, not a statistical one, and it belongs in the
# open. The reasoning: the cost of a false alarm is a tutor spending a
# conversation on a student who was going to pass anyway; the cost of a false
# negative is a struggling student receiving no help at all. Those are not
# equal, so a threshold chosen to maximise accuracy (or even F1, which weights
# precision and recall equally) encodes the wrong preference.
#
# 3 is deliberately modest -- it says "we would accept three unnecessary
# conversations to catch one more at-risk student". A school with more tutoring
# capacity would raise it; one with less would lower it. The training script
# reports a sweep over several ratios so the sensitivity to this number is
# visible rather than hidden.
FN_COST_RATIO = 3.0
THRESHOLD_SWEEP_RATIOS = [1.0, 2.0, 3.0, 5.0, 10.0]

# Bins for the calibration curve. Kept small because the test set is only 79
# students: 10 bins would leave several bins with one or two students and a
# meaninglessly spiky curve.
CALIBRATION_BINS = 5

# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
# Columns deliberately kept OUT of the model inputs. Each entry is
# (column, reason) and the reason is surfaced in the README and in the UI, so
# the exclusion decisions are auditable rather than buried in code.
EXCLUDED_COLUMNS = [
    ("G3", "This IS the regression target and the source of the at-risk label. "
           "Using it as an input would be direct target leakage."),
    ("at_risk", "This IS the classification target, derived from G3."),
    ("G2", "Second-period grade. Correlates r=0.90 with G3 on the Mathematics "
           "file. It is an intermediate assessment of the very outcome we are "
           "predicting, so including it both leaks the target and destroys the "
           "early-warning use case (a grade this late leaves no time to act)."),
    ("G1", "First-period grade. Correlates r=0.80 with G3. Excluded for the "
           "same reason as G2."),
]

# Ordinal/count variables treated as numeric. The Likert-style 1-5 and 0-4
# columns are ordered scales, so treating them as numeric (rather than
# one-hot encoding each level) keeps the model small and the coefficients
# interpretable. This choice is documented in the README.
NUMERIC_FEATURES = [
    "age",         # 15-22
    "Medu",        # mother's education, 0-4 (ordinal)
    "Fedu",        # father's education, 0-4 (ordinal)
    "traveltime",  # home->school travel time, 1-4 (ordinal)
    "studytime",   # weekly study time, 1-4 (ordinal)
    "failures",    # number of past class failures, 0-4
    "famrel",      # quality of family relationships, 1-5 (ordinal)
    "freetime",    # free time after school, 1-5 (ordinal)
    "goout",       # going out with friends, 1-5 (ordinal)
    "Dalc",        # workday alcohol consumption, 1-5 (ordinal)
    "Walc",        # weekend alcohol consumption, 1-5 (ordinal)
    "health",      # current health status, 1-5 (ordinal)
    "absences",    # school absences, 0-93
]

CATEGORICAL_FEATURES = [
    "school",      # GP / MS
    "sex",         # F / M
    "address",     # U (urban) / R (rural)
    "famsize",     # LE3 / GT3
    "Pstatus",     # T (together) / A (apart)
    "Mjob",        # 5 levels
    "Fjob",        # 5 levels
    "reason",      # 4 levels
    "guardian",    # 3 levels
    "schoolsup",   # extra educational support, yes/no
    "famsup",      # family educational support, yes/no
    "paid",        # extra paid classes, yes/no
    "activities",  # extra-curricular activities, yes/no
    "nursery",     # attended nursery school, yes/no
    "higher",      # wants higher education, yes/no
    "internet",    # internet access at home, yes/no
    "romantic",    # in a romantic relationship, yes/no
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The "leaky" variant used ONLY for the side-by-side leakage demonstration in
# section 6 of the app. It is never used for the models the app predicts with.
LEAKY_EXTRA_NUMERIC = ["G1", "G2"]

# --------------------------------------------------------------------------
# Human-readable labels and help text for the prediction form
# --------------------------------------------------------------------------
# Only features a user could plausibly know about a real student are exposed in
# the UI form. The rest are filled from dataset medians/modes (see
# src/prediction.py), which is stated in the UI.
FORM_FEATURES = [
    "age", "sex", "studytime", "failures", "absences", "Medu", "Fedu",
    "famsize", "Pstatus", "guardian", "traveltime", "schoolsup", "famsup",
    "paid", "activities", "higher", "internet", "romantic", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "traveltime": "Home-to-school travel time",
    "studytime": "Weekly study time",
    "failures": "Number of past class failures",
    "famrel": "Quality of family relationships",
    "freetime": "Free time after school",
    "goout": "Going out with friends",
    "Dalc": "Workday alcohol consumption",
    "Walc": "Weekend alcohol consumption",
    "health": "Current health status",
    "absences": "School absences (days)",
    "school": "School",
    "sex": "Sex",
    "address": "Home address type",
    "famsize": "Family size",
    "Pstatus": "Parents' cohabitation status",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason for choosing this school",
    "guardian": "Guardian",
    "schoolsup": "Extra educational support (school)",
    "famsup": "Family educational support",
    "paid": "Extra paid classes",
    "activities": "Extra-curricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants to take higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "G1": "First period grade",
    "G2": "Second period grade",
    "G3": "Final grade (target)",
}

# Ordinal scales: {feature: {raw_value: readable_label}}
ORDINAL_SCALES = {
    "Medu": {0: "0 - none", 1: "1 - primary (4th grade)", 2: "2 - 5th to 9th grade",
             3: "3 - secondary", 4: "4 - higher education"},
    "Fedu": {0: "0 - none", 1: "1 - primary (4th grade)", 2: "2 - 5th to 9th grade",
             3: "3 - secondary", 4: "4 - higher education"},
    "traveltime": {1: "1 - under 15 min", 2: "2 - 15 to 30 min",
                   3: "3 - 30 min to 1 hour", 4: "4 - over 1 hour"},
    "studytime": {1: "1 - under 2 hours", 2: "2 - 2 to 5 hours",
                  3: "3 - 5 to 10 hours", 4: "4 - over 10 hours"},
    "famrel": {1: "1 - very bad", 2: "2 - bad", 3: "3 - average",
               4: "4 - good", 5: "5 - excellent"},
    "freetime": {1: "1 - very low", 2: "2 - low", 3: "3 - medium",
                 4: "4 - high", 5: "5 - very high"},
    "goout": {1: "1 - very low", 2: "2 - low", 3: "3 - medium",
              4: "4 - high", 5: "5 - very high"},
    "Dalc": {1: "1 - very low", 2: "2 - low", 3: "3 - medium",
             4: "4 - high", 5: "5 - very high"},
    "Walc": {1: "1 - very low", 2: "2 - low", 3: "3 - medium",
             4: "4 - high", 5: "5 - very high"},
    "health": {1: "1 - very bad", 2: "2 - bad", 3: "3 - average",
               4: "4 - good", 5: "5 - very good"},
}

CATEGORICAL_LABELS = {
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"},
    "famsize": {"LE3": "3 or fewer members", "GT3": "More than 3 members"},
    "Pstatus": {"T": "Living together", "A": "Living apart"},
    "guardian": {"mother": "Mother", "father": "Father", "other": "Other"},
    "Mjob": {"at_home": "At home", "health": "Health care", "other": "Other",
             "services": "Civil services", "teacher": "Teacher"},
    "Fjob": {"at_home": "At home", "health": "Health care", "other": "Other",
             "services": "Civil services", "teacher": "Teacher"},
    "reason": {"home": "Close to home", "reputation": "School reputation",
               "course": "Course preference", "other": "Other"},
}
# yes/no columns share one label map
for _c in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher",
           "internet", "romantic"]:
    CATEGORICAL_LABELS[_c] = {"yes": "Yes", "no": "No"}

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
REGRESSION_MODEL_NAME = "Linear Regression"
CLASSIFICATION_MODEL_NAMES = ["Logistic Regression", "SVM (RBF)", "Random Forest"]

MODEL_FILES = {
    "Linear Regression": MODELS_DIR / "regression_model.pkl",
    "Logistic Regression": MODELS_DIR / "logistic_model.pkl",
    "SVM (RBF)": MODELS_DIR / "svm_model.pkl",
    "Random Forest": MODELS_DIR / "random_forest_model.pkl",
}
METADATA_FILE = MODELS_DIR / "metadata.pkl"

RESULTS_CLASSIFICATION_CSV = RESULTS_DIR / "classification_results.csv"
RESULTS_REGRESSION_CSV = RESULTS_DIR / "regression_results.csv"
RESULTS_LEAKAGE_CSV = RESULTS_DIR / "leakage_comparison.csv"
RESULTS_FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
RESULTS_MODEL_RESULTS_CSV = RESULTS_DIR / "model_results.csv"
RESULTS_THRESHOLD_CSV = RESULTS_DIR / "threshold_tuning.csv"
RESULTS_CALIBRATION_CSV = RESULTS_DIR / "calibration.csv"
