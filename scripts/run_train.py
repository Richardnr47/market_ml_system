import sys

from src.pipelines.train_pipeline import run_train_pipeline


def main() -> None:
    config_path = "configs/train.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    result = run_train_pipeline(config_path)
    print(result)


if __name__ == "__main__":
    main()