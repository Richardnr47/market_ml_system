import pandas as pd
import numpy as np
import logging


def extract_feature_importance(
    predictor,
    feature_names: list[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    model = predictor.model

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Predictor model does not expose feature_importances_")

    importances = np.asarray(model.feature_importances_, dtype=float)

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature importance length mismatch: got {len(importances)} importances "
            f"for {len(feature_names)} feature names"
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if logger:
        logger.info("[IMPORTANCE] Top feature importances:\n%s", df.head(20).to_string(index=False))

    return df