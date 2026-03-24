from pathlib import Path
import json
import joblib


def save_joblib(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ARTIFACT] Saving joblib artifact to: {path}")
    joblib.dump(obj, path)
    print(f"[ARTIFACT] Saved: {path}")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ARTIFACT] Saving JSON artifact to: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[ARTIFACT] Saved: {path}")