from pathlib import Path
import json
import joblib
import logging


def save_joblib(obj, path: str | Path, logger: logging.Logger | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info("[ARTIFACT] Saving joblib artifact to %s", path)
    joblib.dump(obj, path)


def save_json(data: dict, path: str | Path, logger: logging.Logger | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info("[ARTIFACT] Saving JSON artifact to %s", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)