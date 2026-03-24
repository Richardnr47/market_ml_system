from src.pipelines.train_pipeline import run_train_pipeline


def main() -> None:
    result = run_train_pipeline("configs/train.yaml")
    print(result)


if __name__ == "__main__":
    main()