from src.pipelines.train_pipeline import run_train_pipeline


def main() -> None:
    print("[SCRIPT] Running training entrypoint...")
    result = run_train_pipeline("configs/train.yaml")
    print("[SCRIPT] Training finished")
    print(f"[SCRIPT] Final summary: {result}")


if __name__ == "__main__":
    main()