import joblib
import pandas as pd
from fastapi import FastAPI

from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.logging import setup_logger

logger = setup_logger(
    name="api",
    log_dir="logs",
    log_file="api.log",
    verbose=True,
    log_to_file=True,
)

app = FastAPI(title="Market ML System")

logger.info("[API] Loading artifacts...")
state_model = joblib.load("models/snapshots/latest/state_model.joblib")
predictor = joblib.load("models/snapshots/latest/predictor.joblib")
feature_cols = joblib.load("models/snapshots/latest/feature_cols.joblib")
logger.info("[API] Artifacts loaded")


@app.get("/health")
def health() -> dict:
    logger.info("[API] /health called")
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: list[dict]) -> dict:
    logger.info("[API] /predict called with %s rows", len(payload))
    df = pd.DataFrame(payload)

    result = run_inference_pipeline(
        input_df=df,
        feature_cols=feature_cols,
        state_model=state_model,
        predictor=predictor,
        price_col="close",
        window_size=20,
        logger=logger,
    )
    return result