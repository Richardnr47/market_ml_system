from __future__ import annotations

import pandas as pd
import logging


def log_fold_summary(
    logger: logging.Logger,
    fold_metrics: list[dict],
    enabled: bool = True,
) -> None:
    if not enabled or not fold_metrics:
        return

    df = pd.DataFrame(fold_metrics)
    logger.info("[REPORT] Fold metrics summary:")
    logger.info("\n%s", df.to_string(index=False))

    numeric_cols = [c for c in df.columns if c != "fold"]
    if numeric_cols:
        summary = df[numeric_cols].agg(["mean", "std", "min", "max"])
        logger.info("[REPORT] Aggregate summary:")
        logger.info("\n%s", summary.to_string())