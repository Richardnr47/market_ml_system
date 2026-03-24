import joblib
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import PredictionResponse
from src.pipelines.inference_pipeline import run_inference_pipeline

app = FastAPI(title="Market ML System")

state_model = joblib.load("models/snapshots/latest/state_model.joblib")
predictor = joblib.load("models/snapshots/latest/predictor.joblib")
feature_cols = joblib.load("models/snapshots/latest/feature_cols.joblib")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: list[dict]) -> PredictionResponse:
    df = pd.DataFrame(payload)
    result = run_inference_pipeline(
        input_df=df,
        feature_cols=feature_cols,
        state_model=state_model,
        predictor=predictor,
        price_col="close",
        window_size=20,
    )
    return PredictionResponse(**result)