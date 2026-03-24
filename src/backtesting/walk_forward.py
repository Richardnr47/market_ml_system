from dataclasses import dataclass
import logging


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
    logger: logging.Logger | None = None,
) -> list[WalkForwardSplit]:
    if logger:
        logger.info("[SPLIT] Generating walk-forward splits...")

    splits: list[WalkForwardSplit] = []
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