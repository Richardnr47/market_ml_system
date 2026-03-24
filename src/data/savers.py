from pathlib import Path
import pandas as pd
import logging


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    logger: logging.Logger | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension for save_dataframe: {path.suffix}")

    if logger:
        logger.info("[SAVE] Saved dataframe to %s", path)