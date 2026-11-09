import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from predictive_ml.artifact import FEATURE_COLUMNS, save_artifact
from predictive_ml.train import build_preprocessor


def make_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic dataset shaped like AI4I 2020 (same columns/dtypes)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 10, n),
            "Tool wear": rng.integers(0, 250, n),
        },
        columns=FEATURE_COLUMNS,
    )
    # Failures driven by high torque * tool wear, so the model has signal.
    y = ((X["Torque"] * X["Tool wear"]) > 6000).astype(int)
    return X, y


@pytest.fixture(scope="session")
def fitted_pipeline():
    X, y = make_frame()
    model = make_pipeline(build_preprocessor(X), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model


@pytest.fixture
def model_file(tmp_path, fitted_pipeline):
    return save_artifact(fitted_pipeline, 0.5, tmp_path / "model.joblib")


@pytest.fixture
def valid_body():
    return {
        "type": "L",
        "airTemperature": 298.1,
        "processTemperature": 308.6,
        "rotationalSpeed": 1551,
        "torque": 42.8,
        "toolWear": 0,
    }
