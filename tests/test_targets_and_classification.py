import numpy as np
import pandas as pd

from src.features.targets import build_event_label_target
from src.modeling.evaluation import classification_metrics


def test_build_event_label_target_marks_up_event() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A"] * 6,
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="D"),
            "close": [100.0, 101.0, 102.0, 103.0, 99.0, 100.0],
            "high": [100.5, 101.5, 104.5, 103.5, 99.5, 100.5],
        }
    )
    out = build_event_label_target(
        df=df,
        price_col="close",
        horizon=2,
        up_pct=0.03,
        ticker_col="ticker",
        high_col="high",
    )

    valid = out["target_event_up"].dropna().to_list()
    assert valid[:3] == [1.0, 1.0, 0.0]


def test_classification_metrics_contains_requested_keys() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    y_score = np.array([0.9, 0.2, 0.8, 0.1, 0.4, 0.3], dtype=float)
    timestamps = np.array(
        pd.to_datetime(
            [
                "2026-01-01 09:30",
                "2026-01-01 09:30",
                "2026-01-01 09:45",
                "2026-01-01 09:45",
                "2026-01-01 10:00",
                "2026-01-01 10:00",
            ]
        )
    )

    m = classification_metrics(
        y_true=y_true,
        y_score=y_score,
        timestamps=timestamps,
        threshold=0.5,
        top_k_per_timestamp=1,
    )

    assert "classification_precision" in m
    assert "classification_recall" in m
    assert "classification_pr_auc" in m
    assert "precision_at_k" in m
    assert "hit_rate_top_ranked" in m
