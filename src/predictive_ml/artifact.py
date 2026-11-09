"""Shared feature schema and model-artifact persistence.

The feature names below are the column names returned by ``ucimlrepo`` for the
UCI AI4I 2020 dataset (id=601). They do not include unit suffixes; units are
documented here and in the API schema instead.
"""

import os
from pathlib import Path

import joblib

PROJ_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJ_ROOT / "models" / "predictive_model.joblib"
MODEL_PATH_ENV = "PREDICTIVE_ML_MODEL_PATH"

# Column name -> unit, in the order the model was trained on.
FEATURE_UNITS = {
    "Type": None,  # product quality variant: L, M or H
    "Air temperature": "K",
    "Process temperature": "K",
    "Rotational speed": "rpm",
    "Torque": "Nm",
    "Tool wear": "min",
}
FEATURE_COLUMNS = list(FEATURE_UNITS)
PRODUCT_TYPES = ("L", "M", "H")


class ModelArtifactError(RuntimeError):
    """Raised when the model artifact is missing or incompatible."""


def model_path() -> Path:
    """Model location, overridable with the PREDICTIVE_ML_MODEL_PATH env var."""
    return Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))


def save_artifact(model, threshold: float, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": float(threshold)}, path)
    return path


def load_artifact(path: Path | None = None) -> dict:
    """Load and validate a ``{"model", "threshold"}`` artifact."""
    path = Path(path) if path is not None else model_path()

    if not path.is_file():
        raise ModelArtifactError(
            f"Model artifact not found at '{path}'. Train one with "
            f"'uv run predictive-ml train' or set {MODEL_PATH_ENV}."
        )

    try:
        artifact = joblib.load(path)
    except Exception as exc:  # unpickling errors vary by library version
        raise ModelArtifactError(f"Could not load model artifact '{path}': {exc}") from exc

    if not isinstance(artifact, dict) or not {"model", "threshold"} <= artifact.keys():
        raise ModelArtifactError(
            f"'{path}' is not a predictive-ml artifact: expected a dict with "
            "'model' and 'threshold' keys."
        )

    model = artifact["model"]
    if not hasattr(model, "predict_proba"):
        raise ModelArtifactError(f"Model in '{path}' does not support predict_proba.")

    trained_features = list(getattr(model, "feature_names_in_", []))
    if trained_features != FEATURE_COLUMNS:
        raise ModelArtifactError(
            f"Model in '{path}' was trained on features {trained_features}, "
            f"but the API sends {FEATURE_COLUMNS}."
        )

    threshold = float(artifact["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ModelArtifactError(f"Invalid threshold {threshold} in '{path}'.")

    return {"model": model, "threshold": threshold}
