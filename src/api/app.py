import joblib
import pandas as pd
from fastapi import FastAPI

from src.pipelines.inference_pipeline import run_inference_pipeline

app = FastAPI(title="Market ML System")

print("[API] Loading artifacts...")
state_model = joblib.load("models/snapshots/latest/state_model.joblib")
predictor = joblib.load("models/snapshots/latest/predictor.joblib")
feature_cols = joblib.load("models/snapshots/latest/feature_cols.joblib")
print("[API] Artifacts loaded")


@app.get("/health")
def health() -> dict:
    print("[API] /health called")
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: list[dict]) -> dict:
    print("[API] /predict called")
    print(f"[API] Payload rows: {len(payload)}")

    df = pd.DataFrame(payload)

    result = run_inference_pipeline(
        input_df=df,
        feature_cols=feature_cols,
        state_model=state_model,
        predictor=predictor,
        price_col="close",
        window_size=20,
    )
    print("[API] Returning prediction result")
    return result