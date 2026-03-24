from dataclasses import dataclass


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
) -> list[WalkForwardSplit]:
    splits: list[WalkForwardSplit] = []
    train_end = train_min_size

    for _ in range(n_splits):
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        splits.append(
            WalkForwardSplit(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_end = test_end

    return splits