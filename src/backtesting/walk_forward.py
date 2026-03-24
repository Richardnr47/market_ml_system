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
    print("[SPLIT] Generating walk-forward splits...")
    print(f"[SPLIT] n_samples={n_samples}")
    print(f"[SPLIT] train_min_size={train_min_size}")
    print(f"[SPLIT] test_size={test_size}")
    print(f"[SPLIT] requested n_splits={n_splits}")

    splits: list[WalkForwardSplit] = []
    train_end = train_min_size

    for split_idx in range(n_splits):
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_samples:
            print(f"[SPLIT] Stopping at split {split_idx}: test_end={test_end} > n_samples={n_samples}")
            break

        split = WalkForwardSplit(
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)

        print(
            f"[SPLIT] Fold {split_idx}: "
            f"train=[{split.train_start}:{split.train_end}] "
            f"test=[{split.test_start}:{split.test_end}]"
        )

        train_end = test_end

    print(f"[SPLIT] Total generated folds: {len(splits)}")
    return splits