import joblib
import pandas as pd

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

PROJ_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJ_ROOT / "models" / "predictive_model.joblib"

app = FastAPI()

class PredictionRequest(BaseModel):
    type: str
    airTemperature: float
    processTemperature: float
    rotationalSpeed: int
    torque: float
    toolWear: int


artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]

@app.get("/health")
def health():
    return {
        "status": "alive"
    }


@app.post("/predict")
def predict(data: PredictionRequest):
    X = pd.DataFrame([
        {
            "Type": data.type,
            "Air temperature": data.airTemperature,
            "Process temperature": data.processTemperature,
            "Rotational speed": data.rotationalSpeed,
            "Torque": data.torque,
            "Tool wear": data.toolWear,
        }
    ])

    probability = model.predict_proba(X)[:, 1][0]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "failureProbability": float(probability),
        "threshold": threshold,
    }
