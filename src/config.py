"""
Central configuration for the Used Car Price & Premium Prediction System.

Everything another module might want to agree on -- paths, the random seed, the
"premium" price threshold, and (most importantly) which columns are allowed to
be used as model inputs -- lives here, so there is exactly one place to look
when answering "what did this project actually train on?".
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

DATASET_FILE = DATA_RAW_DIR / "cardekho.csv"

# Integrity checks applied when the file is loaded. The file is a copy of the
# widely circulated CarDekho used-car listings, not a UCI-style published
# dataset with an official specification, so there is no authority to quote.
# The shape and checksum below are what THIS project was developed and tested
# on; if either differs the loader refuses to proceed rather than silently
# training on a different file.
EXPECTED_RAW_SHAPE = (8128, 12)
EXPECTED_SHA256 = (
    "13680430728ec823242429b5a6826e7e42e38fd379e90d972841d84869e19603"
)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
REGRESSION_TARGET = "selling_price"     # asking price in Indian rupees
CLASSIFICATION_TARGET = "is_premium"    # derived: 1 if price >= PREMIUM_THRESHOLD

# A "premium" car is one listed at 6 lakh rupees or more (1 lakh = 100,000).
# 6 lakh is a round figure chosen BEFORE any model was fitted, because it leaves
# a learnable minority class (about 30% of listings, close to the 33% minority
# in a typical early-warning problem). It was not tuned to improve a metric, and
# it is a modelling convention rather than an industry definition: a different
# threshold would define a different task. The classification label is derived
# from the regression target, exactly as a pass/fail label would be derived from
# a grade, so neither the price nor the label may ever be an input.
PREMIUM_THRESHOLD = 600_000
LAKH = 100_000

POSITIVE_CLASS_LABEL = "Premium"
NEGATIVE_CLASS_LABEL = "Standard"

# The newest model year in the data. Age = REFERENCE_YEAR - year. A fixed
# constant (not today's date) so results do not change from year to year.
REFERENCE_YEAR = 2020

# --------------------------------------------------------------------------
# Decision threshold
# --------------------------------------------------------------------------
# The probability above which a car is labelled Premium. 0.5 is the
# scikit-learn default and what `predict()` uses; it is a convention, not a
# result.
DEFAULT_THRESHOLD = 0.5

# How many false alarms one missed premium car is considered worth.
#
# Unlike a medical or early-warning problem there is NO stated reason here to
# treat the two errors differently: wrongly calling a standard car premium and
# wrongly calling a premium car standard are about equally unhelpful. So the
# default is 1.0 (minimise total errors) and nothing about the domain justifies
# a higher number. The training script still reports a sweep over other ratios
# so the sensitivity of the chosen cut-off to this assumption is visible.
FN_COST_RATIO = 1.0
THRESHOLD_SWEEP_RATIOS = [0.5, 1.0, 2.0, 3.0, 5.0]

# Bins for the calibration curve. Kept small because the test set is only about
# 1,400 cars and equal-count bins must each hold enough of them to be meaningful.
CALIBRATION_BINS = 8

# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
# Columns deliberately kept OUT of the model inputs. Each entry is
# (column, reason); the reasons are surfaced in the README and the UI, so the
# exclusion decisions are auditable rather than buried in code.
EXCLUDED_COLUMNS = [
    ("selling_price", "This IS the regression target and the source of the "
                      "premium label. Using it as an input would be direct "
                      "target leakage."),
    ("is_premium", "This IS the classification target, derived from the price."),
    ("name", "Free text such as 'Maruti Swift Dzire VDI'. Almost every value is "
             "near-unique, so a model could memorise individual listings. Only "
             "its first word, the brand, is kept as a categorical feature."),
    ("year", "Replaced by `age` = 2020 - year. They carry identical "
             "information, and keeping both would add a perfectly collinear "
             "column that makes Linear Regression coefficients unstable."),
]

NUMERIC_FEATURES = [
    "age",         # years since manufacture, derived from `year`
    "km_driven",   # odometer reading, km
    "mileage",     # fuel economy, km/l (km/kg for CNG and LPG)
    "engine",      # engine displacement, cc
    "max_power",   # maximum power, bhp
    "seats",       # number of seats
]

CATEGORICAL_FEATURES = [
    "brand",         # first word of `name`; 32 brands, many very rare
    "fuel",          # Diesel / Petrol / CNG / LPG
    "seller_type",   # Individual / Dealer / Trustmark Dealer
    "transmission",  # Manual / Automatic
    "owner",         # First / Second / Third / Fourth & Above / Test Drive Car
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A brand with fewer than this many cars in the TRAINING split is pooled into a
# single "infrequent" column by the encoder, and so is any brand never seen in
# training. This keeps 32 mostly tiny brand columns from overfitting, and means
# a brand typed into the web form that the model has never seen degrades
# gracefully instead of failing.
MIN_BRAND_FREQUENCY = 30

# Raw column names that need renaming on load.
RAW_RENAMES = {"mileage(km/ltr/kg)": "mileage"}

# --------------------------------------------------------------------------
# Labels and help text for the prediction form
# --------------------------------------------------------------------------
FEATURE_LABELS = {
    "age": "Age (years)",
    "km_driven": "Distance driven (km)",
    "mileage": "Fuel economy (km/l)",
    "engine": "Engine size (cc)",
    "max_power": "Maximum power (bhp)",
    "seats": "Number of seats",
    "brand": "Brand",
    "fuel": "Fuel type",
    "seller_type": "Seller type",
    "transmission": "Transmission",
    "owner": "Owner history",
    "year": "Year of manufacture",
    "selling_price": "Selling price (target)",
}

CATEGORICAL_LABELS = {
    "fuel": {"Diesel": "Diesel", "Petrol": "Petrol", "CNG": "CNG", "LPG": "LPG"},
    "seller_type": {"Individual": "Individual", "Dealer": "Dealer",
                    "Trustmark Dealer": "Trustmark Dealer"},
    "transmission": {"Manual": "Manual", "Automatic": "Automatic"},
    "owner": {"First Owner": "First owner", "Second Owner": "Second owner",
              "Third Owner": "Third owner",
              "Fourth & Above Owner": "Fourth owner or more",
              "Test Drive Car": "Test drive car"},
}

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
RESULTS_LEAKAGE_CSV = RESULTS_DIR / "duplicate_leakage.csv"
RESULTS_FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
RESULTS_MODEL_RESULTS_CSV = RESULTS_DIR / "model_results.csv"
RESULTS_THRESHOLD_CSV = RESULTS_DIR / "threshold_tuning.csv"
RESULTS_CALIBRATION_CSV = RESULTS_DIR / "calibration.csv"
