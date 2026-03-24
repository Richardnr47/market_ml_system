from dataclasses import dataclass
import logging

import numpy as np


@dataclass
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_walk_forward_splits(
    n_samples: int,
    train_min_size: int,
    test_size: int,
    n_splits: int,
    timestamps: np.ndarray | None = None,
    logger: logging.Logger | None = None,
) -> list[WalkForwardSplit]:
    if logger:
        logger.info("[SPLIT] Generating walk-forward splits...")

    if n_samples <= 0:
        return []

    if train_min_size <= 0:
        raise ValueError("train_min_size must be positive")

    if test_size <= 0:
        raise ValueError("test_size must be positive")

    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    splits: list[WalkForwardSplit] = []

    if timestamps is None:
        train_end = train_min_size

        for split_idx in range(n_splits):
            test_start = train_end
            test_end = test_start + test_size

            if test_end > n_samples:
                if logger:
                    logger.warning(
                        "[SPLIT] Stopping at split %s because test_end=%s exceeds n_samples=%s",
                        split_idx,
                        test_end,
                        n_samples,
                    )
                break

            split = WalkForwardSplit(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            splits.append(split)

            if logger:
                logger.info(
                    "[SPLIT] Fold %s train=[%s:%s] test=[%s:%s]",
                    split_idx,
                    split.train_start,
                    split.train_end,
                    split.test_start,
                    split.test_end,
                )

            train_end = test_end

        if logger:
            logger.info("[SPLIT] Total generated folds: %s", len(splits))

        return splits

    ts = np.asarray(timestamps)
    if ts.ndim != 1:
        ts = ts.reshape(-1)

    if len(ts) != n_samples:
        raise ValueError(
            f"timestamps length mismatch: expected {n_samples}, got {len(ts)}"
        )

    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("timestamps must be sorted in ascending order")

    group_starts = np.concatenate(([0], np.flatnonzero(ts[1:] != ts[:-1]) + 1))
    group_ends = np.concatenate((group_starts[1:], [n_samples]))

    def align_to_group_end(target_end: int) -> int:
        group_idx = int(np.searchsorted(group_ends, target_end, side="left"))
        if group_idx >= len(group_ends):
            return n_samples + 1
        return int(group_ends[group_idx])

    train_end = align_to_group_end(train_min_size)

    for split_idx in range(n_splits):
        test_start = train_end
        test_end = align_to_group_end(test_start + test_size)

        if test_end > n_samples:
            if logger:
                logger.warning(
                    "[SPLIT] Stopping at split %s because test_end=%s exceeds n_samples=%s",
                    split_idx,
                    test_end,
                    n_samples,
                )
            break

        split = WalkForwardSplit(
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)

        if logger:
            logger.info(
                "[SPLIT] Fold %s train=[%s:%s] test=[%s:%s] | "
                "train_ts=[%s -> %s] test_ts=[%s -> %s]",
                split_idx,
                split.train_start,
                split.train_end,
                split.test_start,
                split.test_end,
                ts[split.train_start],
                ts[split.train_end - 1],
                ts[split.test_start],
                ts[split.test_end - 1],
            )

        train_end = test_end

    if logger:
        logger.info("[SPLIT] Total generated folds: %s", len(splits))

    return splits