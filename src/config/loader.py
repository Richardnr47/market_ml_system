from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    print(f"[CONFIG] Loading YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"[CONFIG] Loaded YAML successfully: {path}")
    return data