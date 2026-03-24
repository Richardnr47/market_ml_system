from src.pipelines.train_pipeline import run_train_pipeline


if __name__ == "__main__":
    result = run_train_pipeline("configs/train.yaml")
    print(result)