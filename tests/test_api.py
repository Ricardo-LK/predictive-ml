import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from predictive_ml.api import create_app
from predictive_ml.artifact import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    ModelArtifactError,
    load_artifact,
    save_artifact,
)
from predictive_ml.train import build_preprocessor

from .conftest import make_frame


@pytest.fixture
def client(model_file):
    with TestClient(create_app(model_file)) as c:
        yield c


def test_health(client):
    res = client.get("/health")

    assert res.status_code == 200
    assert res.json() == {"status": "alive"}


def test_predict(client, valid_body):
    res = client.post("/predict", json=valid_body)
    data = res.json()

    assert res.status_code == 200
    assert set(data) == {"prediction", "failureProbability", "threshold"}
    assert data["prediction"] in [0, 1]
    assert 0 <= data["failureProbability"] <= 1
    assert data["threshold"] == 0.5
    assert data["prediction"] == int(data["failureProbability"] >= data["threshold"])


def test_predict_matches_pipeline_and_applies_threshold(tmp_path, fitted_pipeline, valid_body):
    expected = fitted_pipeline.predict_proba(
        pd.DataFrame([["L", 298.1, 308.6, 1551, 42.8, 0]], columns=FEATURE_COLUMNS)
    )[0, 1]

    for threshold, prediction in [(0.0, 1), (1.0, int(expected >= 1.0))]:
        path = save_artifact(fitted_pipeline, threshold, tmp_path / f"m{threshold}.joblib")
        with TestClient(create_app(path)) as c:
            data = c.post("/predict", json=valid_body).json()
        assert data["failureProbability"] == pytest.approx(expected)
        assert data["prediction"] == prediction


@pytest.mark.parametrize(
    "change",
    [
        {"type": "X"},  # unknown product type would be silently one-hot ignored
        {"type": "l"},
        {"airTemperature": -5},
        {"airTemperature": 0},
        {"rotationalSpeed": -1},
        {"rotationalSpeed": 1551.5},
        {"torque": "high"},
        {"toolWear": None},
        {"Air temperature [K]": 298.1},  # unexpected / unit-suffixed field name
    ],
)
def test_predict_rejects_invalid_input(client, valid_body, change):
    res = client.post("/predict", json={**valid_body, **change})

    assert res.status_code == 422
    assert "detail" in res.json()


def test_predict_rejects_missing_field(client, valid_body):
    del valid_body["torque"]

    res = client.post("/predict", json=valid_body)

    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", "torque"]


def test_predict_rejects_malformed_json(client):
    res = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})

    assert res.status_code == 422


def test_startup_fails_clearly_when_model_missing(tmp_path):
    app = create_app(tmp_path / "missing.joblib")
    with pytest.raises(ModelArtifactError, match="not found"), TestClient(app):
        pass


def test_rejects_model_trained_on_other_feature_names(tmp_path):
    """Regression: the API sends unit-less ucimlrepo names; a model fitted on
    e.g. 'Air temperature [K]' must be refused at startup, not mis-predict."""
    X, y = make_frame()
    X = X.rename(columns={c: f"{c} [unit]" for c in FEATURE_COLUMNS[1:]})
    model = make_pipeline(build_preprocessor(X), LogisticRegression(max_iter=1000)).fit(X, y)
    path = save_artifact(model, 0.5, tmp_path / "bad.joblib")

    with pytest.raises(ModelArtifactError, match="trained on features"):
        load_artifact(path)


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="committed model not present")
def test_committed_model_matches_api_contract(valid_body):
    """Regression guard for the shipped artifact: feature names/order, a
    'Type' encoder that knows L/M/H, and a usable threshold."""
    artifact = load_artifact(DEFAULT_MODEL_PATH)
    model = artifact["model"]

    assert list(model.feature_names_in_) == FEATURE_COLUMNS
    encoder = model[0].named_transformers_["categ"][-1]
    assert sorted(encoder.categories_[0]) == ["H", "L", "M"]

    with TestClient(create_app(DEFAULT_MODEL_PATH)) as c:
        res = c.post("/predict", json=valid_body)
    assert res.status_code == 200
    assert res.json()["threshold"] == artifact["threshold"]
