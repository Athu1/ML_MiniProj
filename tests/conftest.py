"""Shared fixtures.

`src/` is put on the path here rather than in each test file, matching how
`train_models.py` and `app.py` import the project's modules.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def cfg():
    import config
    return config


@pytest.fixture(scope="session")
def dp():
    import data_preprocessing
    return data_preprocessing


@pytest.fixture(scope="session")
def em():
    import evaluate_models
    return evaluate_models


@pytest.fixture(scope="session")
def df(dp):
    """The full dataset with the at-risk target attached."""
    return dp.load_dataset("mat")


@pytest.fixture(scope="session")
def artifacts():
    """Trained models plus metadata, skipping the whole module if absent.

    The model files are committed, so this normally loads. Skipping rather than
    failing keeps `pytest` useful on a checkout where someone has cleared
    models/ but not yet re-run training.
    """
    import prediction
    if not prediction.models_available():
        pytest.skip("trained models not present — run `python src/train_models.py`")
    return prediction.load_artifacts()
