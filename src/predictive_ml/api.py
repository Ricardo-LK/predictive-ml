from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from predictive_ml.artifact import FEATURE_COLUMNS, load_artifact


class PredictionRequest(BaseModel):
    # Reject unknown fields so client/server contract drift fails loudly.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    airTemperature: float = Field(gt=0, description="Air temperature [K]")
    processTemperature: float = Field(gt=0, description="Process temperature [K]")
    rotationalSpeed: int = Field(ge=0, description="Rotational speed [rpm]")
    torque: float = Field(ge=0, description="Torque [Nm]")
    toolWear: int = Field(ge=0, description="Tool wear [min]")

    def to_frame(self) -> pd.DataFrame:
        row = {
            "Type": self.type,
            "Air temperature": self.airTemperature,
            "Process temperature": self.processTemperature,
            "Rotational speed": self.rotationalSpeed,
            "Torque": self.torque,
            "Tool wear": self.toolWear,
        }
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    failureProbability: float
    threshold: float


def create_app(model_path: Path | None = None) -> FastAPI:
    """Build the API. The model is loaded (and validated) at startup, so a
    missing or incompatible artifact stops the server instead of serving
    wrong predictions."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        artifact = load_artifact(model_path)
        app.state.model = artifact["model"]
        app.state.threshold = artifact["threshold"]
        yield

    app = FastAPI(title="predictive-ml inference API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(data: PredictionRequest) -> PredictionResponse:
        probability = float(app.state.model.predict_proba(data.to_frame())[:, 1][0])
        threshold = app.state.threshold
        return PredictionResponse(
            prediction=int(probability >= threshold),
            failureProbability=probability,
            threshold=threshold,
        )

    return app


app = create_app()
