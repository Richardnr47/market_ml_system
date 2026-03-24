import joblib
import pandas as pd

from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.logging import setup_logger


def main() -> None:
    logger = setup_logger(
        name="inference_script",
        log_dir="logs",
        log_file="inference.log",
        verbose=True,
        log_to_file=True,
    )

    df = pd.read_csv("data/raw/market_data.csv")

    state_model = joblib.load("models/snapshots/latest/state_model.joblib")
    predictor = joblib.load("models/snapshots/latest/predictor.joblib")
    feature_cols = joblib.load("models/snapshots/latest/feature_cols.joblib")

    result = run_inference_pipeline(
        input_df=df.tail(100).copy(),
        feature_cols=feature_cols,
        state_model=state_model,
        predictor=predictor,
        price_col="close",
        window_size=20,
        logger=logger,
    )

    print(result)


if __name__ == "__main__":
    main()