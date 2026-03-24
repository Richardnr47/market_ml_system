from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.backtesting.walk_forward import generate_walk_forward_splits
from src.config.loader import load_yaml
from src.data.validation import validate_market_df
from src.features.builders import build_market_features
from src.features.targets import build_forward_return_target
from src.features.windows import flatten_windows, make_windows
from src.modeling.evaluation import regression_metrics
from src.modeling.predictors import LGBMReturnPredictor
from src.modeling.state_models import GMMStateModel
from src.registry.artifact_store import save_joblib, save_json
from src.utils.debug import DebugPrinter


def run_train_pipeline(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    runtime_cfg = cfg.get("runtime", {})
    dbg = DebugPrinter(enabled=runtime_cfg.get("verbose", True))

    dbg.banner("TRAIN PIPELINE START")

    with dbg.timer("DATA", "Loading raw dataset"):
        input_file = cfg["dataset"]["input_file"]
        dbg.log("DATA", f"Reading CSV: {input_file}")
        df = pd.read_csv(input_file)
        dbg.log("DATA", f"Loaded rows={len(df)}, cols={len(df.columns)}")
        dbg.log("DATA", f"Columns: {list(df.columns)}")

    with dbg.timer("VALIDATION", "Validating raw dataset"):
        validate_market_df(df)

    with dbg.timer("DATA", "Parsing and sorting timestamps"):
        timestamp_col = cfg["dataset"]["timestamp_col"]
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        dbg.log("DATA", f"Timestamp range: {df[timestamp_col].min()} -> {df[timestamp_col].max()}")

    with dbg.timer("FEATURES", "Building market features"):
        df = build_market_features(
            df=df,
            price_col=cfg["dataset"]["price_col"],
            return_lags=cfg["features"]["return_lags"],
            vol_windows=cfg["features"]["vol_windows"],
            momentum_windows=cfg["features"]["momentum_windows"],
            volume_windows=cfg["features"]["volume_windows"],
        )

    with dbg.timer("TARGET", "Building training targets"):
        df = build_forward_return_target(
            df=df,
            price_col=cfg["dataset"]["price_col"],
            horizon=cfg["dataset"]["target_horizon"],
        )

    feature_cols = [
        c for c in df.columns
        if c.startswith("ret_")
        or c.startswith("rv_")
        or c.startswith("mom_")
        or c.startswith("vol_z_")
        or c in {"hl_range", "oc_change", "body_to_range"}
    ]

    dbg.log("FEATURES", f"Selected feature columns: {len(feature_cols)}")
    if runtime_cfg.get("show_feature_preview", True):
        dbg.log("FEATURES", f"Feature preview: {feature_cols[:10]}")

    if runtime_cfg.get("show_nan_counts", True):
        nan_total = int(df[feature_cols + ["target_forward_return"]].isna().sum().sum())
        dbg.log("FEATURES", f"NaN count in features+target block: {nan_total}")

    with dbg.timer("WINDOWS", "Creating rolling windows"):
        wb = make_windows(
            df=df,
            feature_cols=feature_cols,
            target_col="target_forward_return",
            timestamp_col=timestamp_col,
            window_size=cfg["windows"]["size"],
            stride=cfg["windows"]["stride"],
        )

    with dbg.timer("WINDOWS", "Flattening rolling windows"):
        X = flatten_windows(wb.X)
        y = wb.y
        ts = wb.timestamps
        if runtime_cfg.get("show_shapes", True):
            dbg.log("WINDOWS", f"Flat X shape: {X.shape}")
            dbg.log("WINDOWS", f"y shape: {y.shape}")
            dbg.log("WINDOWS", f"timestamps shape: {ts.shape}")

    with dbg.timer("SPLIT", "Generating walk-forward validation splits"):
        splits = generate_walk_forward_splits(
            n_samples=len(y),
            train_min_size=cfg["validation"]["train_min_size"],
            test_size=cfg["validation"]["test_size"],
            n_splits=cfg["validation"]["n_splits"],
        )

    if runtime_cfg.get("show_split_summary", True):
        dbg.log("SPLIT", f"Generated {len(splits)} splits")

    all_fold_metrics = []

    experiment_name = cfg["experiment"]["name"]
    dbg.log("MLFLOW", f"Setting experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        dbg.log("MLFLOW", "Started MLflow run")
        mlflow.log_params({
            "window_size": cfg["windows"]["size"],
            "target_horizon": cfg["dataset"]["target_horizon"],
            "state_n_components": cfg["state_model"]["n_components"],
            "predictor_type": cfg["predictor"]["type"],
        })

        final_state_model = None
        final_predictor = None

        for fold_idx, split in enumerate(splits):
            dbg.line()
            dbg.log("FOLD", f"Starting fold {fold_idx}")

            X_train = X[split.train_start:split.train_end]
            y_train = y[split.train_start:split.train_end]
            X_test = X[split.test_start:split.test_end]
            y_test = y[split.test_start:split.test_end]

            dbg.log("FOLD", f"Train slice: [{split.train_start}:{split.train_end}]")
            dbg.log("FOLD", f"Test slice: [{split.test_start}:{split.test_end}]")
            dbg.log("FOLD", f"Train shape: X={X_train.shape}, y={y_train.shape}")
            dbg.log("FOLD", f"Test shape: X={X_test.shape}, y={y_test.shape}")

            with dbg.timer("STATE", f"Fold {fold_idx} fit state model"):
                state_model = GMMStateModel(
                    n_components=cfg["state_model"]["n_components"],
                    random_state=cfg["state_model"]["random_state"],
                ).fit(X_train)

            with dbg.timer("STATE", f"Fold {fold_idx} transform train/test with state model"):
                train_states = state_model.predict(X_train)
                train_probs = state_model.predict_proba(X_train)
                test_states = state_model.predict(X_test)
                test_probs = state_model.predict_proba(X_test)

                X_train_meta = np.column_stack([X_train, train_states, train_probs])
                X_test_meta = np.column_stack([X_test, test_states, test_probs])

                dbg.log("STATE", f"X_train_meta shape: {X_train_meta.shape}")
                dbg.log("STATE", f"X_test_meta shape: {X_test_meta.shape}")

            with dbg.timer("PREDICTOR", f"Fold {fold_idx} fit predictor"):
                predictor = LGBMReturnPredictor(
                    learning_rate=cfg["predictor"]["learning_rate"],
                    n_estimators=cfg["predictor"]["n_estimators"],
                    num_leaves=cfg["predictor"]["num_leaves"],
                    max_depth=cfg["predictor"]["max_depth"],
                    subsample=cfg["predictor"]["subsample"],
                    colsample_bytree=cfg["predictor"]["colsample_bytree"],
                    random_state=cfg["predictor"]["random_state"],
                ).fit(X_train_meta, y_train)

            with dbg.timer("PREDICTOR", f"Fold {fold_idx} predict on test"):
                preds = predictor.predict(X_test_meta)

            with dbg.timer("EVAL", f"Fold {fold_idx} evaluate predictions"):
                metrics = regression_metrics(y_test, preds)
                metrics["fold"] = fold_idx
                all_fold_metrics.append(metrics)

            if runtime_cfg.get("show_fold_metrics", True):
                dbg.log("FOLD", f"Fold {fold_idx} metrics: {metrics}")

            mlflow.log_metrics({
                f"fold_{fold_idx}_{k}": v for k, v in metrics.items() if k != "fold"
            })

            final_state_model = state_model
            final_predictor = predictor

        with dbg.timer("SUMMARY", "Aggregating fold metrics"):
            avg_mae = float(np.mean([m["mae"] for m in all_fold_metrics]))
            avg_r2 = float(np.mean([m["r2"] for m in all_fold_metrics]))

            summary = {
                "avg_mae": avg_mae,
                "avg_r2": avg_r2,
                "n_folds": len(all_fold_metrics),
                "feature_cols": feature_cols,
            }

            dbg.log("SUMMARY", f"avg_mae={avg_mae:.6f}")
            dbg.log("SUMMARY", f"avg_r2={avg_r2:.6f}")
            dbg.log("SUMMARY", f"n_folds={len(all_fold_metrics)}")

            mlflow.log_metrics({
                "avg_mae": avg_mae,
                "avg_r2": avg_r2,
            })

        with dbg.timer("ARTIFACT", "Saving artifacts"):
            out_dir = Path(cfg["artifacts"]["output_dir"])
            save_joblib(final_state_model, out_dir / "state_model.joblib")
            save_joblib(final_predictor, out_dir / "predictor.joblib")
            save_joblib(feature_cols, out_dir / "feature_cols.joblib")
            save_json(summary, out_dir / "summary.json")
            save_json(cfg, out_dir / "config_snapshot.json")

        dbg.log("MLFLOW", "MLflow run complete")

    dbg.banner("TRAIN PIPELINE FINISHED")
    dbg.log("RESULT", str(summary))
    return summary