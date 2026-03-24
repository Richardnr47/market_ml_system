import numpy as np

from src.backtesting.walk_forward import generate_walk_forward_splits


def test_generate_walk_forward_splits_aligns_timestamp_boundaries() -> None:
    timestamps = np.array(
        [
            np.datetime64("2026-01-01T09:30"),
            np.datetime64("2026-01-01T09:30"),
            np.datetime64("2026-01-01T09:45"),
            np.datetime64("2026-01-01T09:45"),
            np.datetime64("2026-01-01T10:00"),
            np.datetime64("2026-01-01T10:00"),
            np.datetime64("2026-01-01T10:15"),
            np.datetime64("2026-01-01T10:15"),
            np.datetime64("2026-01-01T10:30"),
            np.datetime64("2026-01-01T10:30"),
        ]
    )

    splits = generate_walk_forward_splits(
        n_samples=len(timestamps),
        train_min_size=3,
        test_size=3,
        n_splits=2,
        timestamps=timestamps,
    )

    assert [(s.train_start, s.train_end, s.test_start, s.test_end) for s in splits] == [
        (0, 4, 4, 8),
    ]

    split = splits[0]
    assert timestamps[split.train_end - 1] < timestamps[split.test_start]


def test_generate_walk_forward_splits_requires_sorted_timestamps() -> None:
    timestamps = np.array(
        [
            np.datetime64("2026-01-01T09:45"),
            np.datetime64("2026-01-01T09:30"),
        ]
    )

    try:
        generate_walk_forward_splits(
            n_samples=len(timestamps),
            train_min_size=1,
            test_size=1,
            n_splits=1,
            timestamps=timestamps,
        )
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsorted timestamps")
