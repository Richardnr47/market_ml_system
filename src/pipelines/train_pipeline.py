from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtesting.walk_forward import generate_walk_forward_splits
from src.config.loader import load_yaml
from src.data.savers import save_dataframe
from src.data.validation import validate_market_df
from src.features.builders import build_market_features
from src.features.naming import build_flat_feature_names
from src.features.selectors import select_feature_columns
from src.features.targets import build_forward_return_target
from src.features.windows import flatten_windows, make_windows
from src.modeling.evaluation import (
    baseline_metrics,
    correlation_metrics,
    directional_metrics,
    optimize_long_only_threshold,
    regression_metrics,
    trading_metrics,
)
from src.modeling.importance import extract_feature_importance
from src.modeling.predictors import LGBMReturnPredictor
from src.modeling.state_models import GMMStateModel
from src.registry.artifact_store import save_joblib, save_json
from src.utils.debug import DebugPrinter
from src.utils.logging import setup_logger
from src.utils.reporting import log_fold_summary


def run_train_pipeline(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    runtime_cfg = cfg.get("runtime", {})

    logger = setup_logger(
        name="train_pipeline",
        log_dir=runtime_cfg.get("log_dir", "logs"),
        log_file=runtime_cfg.get("log_file", "train.log"),
        verbose=runtime_cfg.get("verbose", True),
        log_to_file=runtime_cfg.get("log_to_file", True),
    )
    dbg = DebugPrinter(logger=logger, enabled=True)

    dbg.banner("TRAIN PIPELINE START")

    try:
        # -------------------------------------------------
        # Load + validate raw data
        # -------------------------------------------------
        with dbg.timer("DATA", "Loading raw dataset"):
            input_file = cfg["dataset"]["input_file"]
            df = pd.read_csv(input_file)
            dbg.log("DATA", f"Loaded rows={len(df)}, cols={len(df.columns)}")
            dbg.debug("DATA", f"Columns: {list(df.columns)}")

        with dbg.timer("VALIDATION", "Validating raw dataset"):
            validate_market_df(df, logger=logger)

        ticker_col = cfg["dataset"].get("ticker_col", "ticker")
        timestamp_col = cfg["dataset"]["timestamp_col"]
        price_col = cfg["dataset"]["price_col"]
        target_horizon = cfg["dataset"]["target_horizon"]
        window_size = cfg["windows"]["size"]
        stride = cfg["windows"]["stride"]

        feature_flags = cfg["features"]["include"]
        state_enabled = cfg["state_model"].get("enabled", True)
        trading_cfg = cfg.get("trading", {})
        trading_enabled = trading_cfg.get("enabled", True)
        trading_threshold = float(trading_cfg.get("prediction_threshold", 0.001))
        trading_cost_bps = float(trading_cfg.get("transaction_cost_bps", 2.0))
        trading_periods_per_year = int(trading_cfg.get("periods_per_year", 252 * 26))
        # Strategy is intentionally fixed to long-only.
        trading_long_only = True
        trading_selection_mode = str(trading_cfg.get("selection_mode", "threshold"))
        trading_top_k_per_timestamp = int(trading_cfg.get("top_k_per_timestamp", 5))
        trading_optimize_threshold = bool(trading_cfg.get("optimize_threshold", True))
        trading_optimize_objective = str(
            trading_cfg.get("optimize_objective", "trading_total_net_return")
        )
        trading_threshold_grid = [
            float(x) for x in trading_cfg.get("threshold_grid", [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003])
        ]
        recommendations_top_k = int(trading_cfg.get("recommendations_top_k", 10))

        if trading_cfg.get("long_only", True) is not True:
            dbg.warning(
                "TRADING",
                f"Config requested long_only={trading_cfg.get('long_only')} "
                "but strategy is fixed to long-only",
            )

        with dbg.timer("DATA", "Parsing and sorting timestamps"):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values([ticker_col, timestamp_col]).reset_index(drop=True)
            dbg.log(
                "DATA",
                f"Timestamp range: {df[timestamp_col].min()} -> {df[timestamp_col].max()}",
            )

        # -------------------------------------------------
        # Features
        # -------------------------------------------------
        with dbg.timer("FEATURES", "Building market features"):
            df = build_market_features(
                df=df,
                price_col=price_col,
                return_lags=cfg["features"]["return_lags"],
                vol_windows=cfg["features"]["vol_windows"],
                momentum_windows=cfg["features"]["momentum_windows"],
                volume_windows=cfg["features"]["volume_windows"],
                feature_flags=feature_flags,
                ticker_col=ticker_col,
                timestamp_col=timestamp_col,
                logger=logger,
            )

        # -------------------------------------------------
        # Targets
        # -------------------------------------------------
        with dbg.timer("TARGET", "Building training targets"):
            df = build_forward_return_target(
                df=df,
                price_col=price_col,
                horizon=target_horizon,
                ticker_col=ticker_col,
                logger=logger,
            )

        valid_target = df["target_forward_return"].dropna()
        dbg.log("TARGET", f"Valid target count: {len(valid_target)}")

        if len(valid_target) > 0:
            dbg.log("TARGET", f"Target mean: {valid_target.mean():.6f}")
            dbg.log("TARGET", f"Target std: {valid_target.std():.6f}")
            dbg.log("TARGET", f"Target min: {valid_target.min():.6f}")
            dbg.log("TARGET", f"Target max: {valid_target.max():.6f}")
        else:
            dbg.warning("TARGET", "No valid target values after target creation")

        # -------------------------------------------------
        # Feature selection
        # -------------------------------------------------
        feature_cols = select_feature_columns(
            df_columns=df.columns.tolist(),
            feature_flags=feature_flags,
        )

        dbg.log("FEATURES", f"Selected feature columns: {len(feature_cols)}")

        if runtime_cfg.get("show_feature_preview", True):
            dbg.log("FEATURES", f"Feature preview: {feature_cols[:10]}")

        if runtime_cfg.get("show_nan_counts", True):
            nan_total = int(df[feature_cols + ["target_forward_return"]].isna().sum().sum())
            dbg.log("FEATURES", f"NaN count in features+target block: {nan_total}")

        # -------------------------------------------------
        # Build windows
        # -------------------------------------------------
        with dbg.timer("WINDOWS", "Creating rolling windows"):
            wb = make_windows(
                df=df,
                feature_cols=feature_cols,
                target_col="target_forward_return",
                timestamp_col=timestamp_col,
                ticker_col=ticker_col,
                window_size=window_size,
                stride=stride,
                logger=logger,
            )

        ticker_window_counts = pd.Series(wb.tickers).value_counts().sort_values(ascending=False)
        dbg.log(
            "WINDOWS",
            f"Top ticker window counts:\n{ticker_window_counts.head(10).to_string()}",
        )
        dbg.log("WINDOWS", f"Total windows created: {len(wb.y)}")
        dbg.log("WINDOWS", f"Unique tickers in windows: {pd.Series(wb.tickers).nunique()}")

        # -------------------------------------------------
        # Flatten windows
        # -------------------------------------------------
        with dbg.timer("WINDOWS", "Flattening rolling windows"):
            X_flat = flatten_windows(wb.X, logger=logger)
            y = wb.y
            ts = wb.timestamps
            tickers = wb.tickers

            if runtime_cfg.get("show_shapes", True):
                dbg.log("WINDOWS", f"Flat X shape: {X_flat.shape}")
                dbg.log("WINDOWS", f"y shape: {y.shape}")
                dbg.log("WINDOWS", f"timestamps shape: {ts.shape}")
                dbg.log("WINDOWS", f"tickers shape: {tickers.shape}")

        # Keep all ticker windows in global chronological order before validation.
        with dbg.timer("WINDOWS", "Sorting window samples by timestamp"):
            ts_series = pd.to_datetime(pd.Series(ts), utc=True)
            ts = ts_series.dt.tz_localize(None).to_numpy()
            sort_order = np.argsort(ts, kind="stable")

            X_flat = X_flat[sort_order]
            y = y[sort_order]
            ts = ts[sort_order]
            tickers = tickers[sort_order]

            dbg.log("WINDOWS", f"Sorted window timestamp range: {ts[0]} -> {ts[-1]}")
            dbg.log("WINDOWS", f"Unique window timestamps: {pd.Index(ts).nunique()}")

        # -------------------------------------------------
        # Feature names
        # -------------------------------------------------
        flat_feature_names = build_flat_feature_names(
            feature_cols=feature_cols,
            window_size=window_size,
        )

        assert X_flat.shape[1] == len(flat_feature_names), (
            f"Mismatch: X_flat has {X_flat.shape[1]} columns but "
            f"{len(flat_feature_names)} feature names"
        )

        if state_enabled:
            state_feature_names = ["state_label"] + [
                f"state_proba_{i}" for i in range(cfg["state_model"]["n_components"])
            ]
        else:
            state_feature_names = []

        full_feature_names = flat_feature_names + state_feature_names

        # -------------------------------------------------
        # Splits
        # -------------------------------------------------
        with dbg.timer("SPLIT", "Generating walk-forward validation splits"):
            splits = generate_walk_forward_splits(
                n_samples=len(y),
                train_min_size=cfg["validation"]["train_min_size"],
                test_size=cfg["validation"]["test_size"],
                n_splits=cfg["validation"]["n_splits"],
                timestamps=ts,
                logger=logger,
            )

        if runtime_cfg.get("show_split_summary", True):
            dbg.log("SPLIT", f"Generated {len(splits)} splits")

        if not splits:
            raise ValueError("No valid walk-forward splits generated")

        # -------------------------------------------------
        # MLflow setup
        # -------------------------------------------------
        experiment_name = cfg["experiment"]["name"]
        mlflow.set_experiment(experiment_name)
        dbg.log("MLFLOW", f"Experiment set to: {experiment_name}")

        all_fold_metrics: list[dict] = []
        final_state_model = None
        final_predictor = None
        importance_df = None
        final_fold_pred_df: pd.DataFrame | None = None
        oof_y_true: list[np.ndarray] = []
        oof_y_pred: list[np.ndarray] = []
        oof_tickers: list[np.ndarray] = []
        oof_timestamps: list[np.ndarray] = []
        optimized_threshold = trading_threshold

        with mlflow.start_run():
            dbg.log("MLFLOW", "Started MLflow run")
            mlflow.log_params(
                {
                    "window_size": window_size,
                    "target_horizon": target_horizon,
                    "state_model_enabled": state_enabled,
                    "state_n_components": cfg["state_model"].get("n_components", 0),
                    "predictor_type": cfg["predictor"]["type"],
                    "n_feature_cols": len(feature_cols),
                    "n_flat_features": len(flat_feature_names),
                    "feature_flags": str(feature_flags),
                    "trading_enabled": trading_enabled,
                    "trading_threshold": trading_threshold,
                    "trading_cost_bps": trading_cost_bps,
                    "trading_long_only": trading_long_only,
                    "trading_selection_mode": trading_selection_mode,
                    "trading_top_k_per_timestamp": trading_top_k_per_timestamp,
                    "trading_optimize_threshold": trading_optimize_threshold,
                    "trading_optimize_objective": trading_optimize_objective,
                }
            )

            fold_iterator = splits
            if runtime_cfg.get("use_tqdm", True):
                fold_iterator = tqdm(splits, desc="Walk-forward folds")

            # -------------------------------------------------
            # Fold loop
            # -------------------------------------------------
            for fold_idx, split in enumerate(fold_iterator):
                dbg.line()
                dbg.log("FOLD", f"Starting fold {fold_idx}")

                try:
                    X_train = X_flat[split.train_start : split.train_end]
                    y_train = y[split.train_start : split.train_end]

                    X_test = X_flat[split.test_start : split.test_end]
                    y_test = y[split.test_start : split.test_end]

                    ts_test = ts[split.test_start : split.test_end]
                    tickers_test = tickers[split.test_start : split.test_end]

                    dbg.log("FOLD", f"Train shape: X={X_train.shape}, y={y_train.shape}")
                    dbg.log("FOLD", f"Test shape: X={X_test.shape}, y={y_test.shape}")

                    if state_enabled:
                        with dbg.timer("STATE", f"Fold {fold_idx} fit state model"):
                            state_model = GMMStateModel(
                                n_components=cfg["state_model"]["n_components"],
                                random_state=cfg["state_model"]["random_state"],
                                logger=logger,
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
                    else:
                        dbg.log("STATE", "State model disabled; using flat features only")
                        state_model = None
                        X_train_meta = X_train
                        X_test_meta = X_test

                    assert X_train_meta.shape[1] == len(full_feature_names), (
                        f"Mismatch after state features: {X_train_meta.shape[1]} vs {len(full_feature_names)}"
                    )

                    with dbg.timer("PREDICTOR", f"Fold {fold_idx} fit predictor"):
                        predictor = LGBMReturnPredictor(
                            learning_rate=cfg["predictor"]["learning_rate"],
                            n_estimators=cfg["predictor"]["n_estimators"],
                            num_leaves=cfg["predictor"]["num_leaves"],
                            max_depth=cfg["predictor"]["max_depth"],
                            subsample=cfg["predictor"]["subsample"],
                            colsample_bytree=cfg["predictor"]["colsample_bytree"],
                            random_state=cfg["predictor"]["random_state"],
                            logger=logger,
                        ).fit(X_train_meta, y_train)

                    with dbg.timer("PREDICTOR", f"Fold {fold_idx} predict on test"):
                        preds = predictor.predict(X_test_meta)

                    with dbg.timer("EVAL", f"Fold {fold_idx} evaluate predictions"):
                        reg_metrics = regression_metrics(y_test, preds, logger=logger)
                        base_metrics = baseline_metrics(y_test, y_train, logger=logger)
                        dir_metrics = directional_metrics(y_test, preds, logger=logger)
                        corr_metrics = correlation_metrics(y_test, preds, logger=logger)
                        if trading_enabled:
                            trade_metrics, trade_arrays = trading_metrics(
                                y_true=y_test,
                                y_pred=preds,
                                tickers=tickers_test,
                                timestamps=ts_test,
                                threshold=trading_threshold,
                                transaction_cost_bps=trading_cost_bps,
                                periods_per_year=trading_periods_per_year,
                                selection_mode=trading_selection_mode,
                                top_k_per_timestamp=trading_top_k_per_timestamp,
                                logger=logger,
                            )
                        else:
                            trade_metrics = {}
                            trade_arrays = {
                                "signal": np.zeros_like(y_test, dtype=int),
                                "gross_return": np.zeros_like(y_test, dtype=float),
                                "cost_return": np.zeros_like(y_test, dtype=float),
                                "net_return": np.zeros_like(y_test, dtype=float),
                            }

                        metrics = {
                            **reg_metrics,
                            **base_metrics,
                            **dir_metrics,
                            **corr_metrics,
                            **trade_metrics,
                            "fold": fold_idx,
                            "train_size": int(len(X_train)),
                            "test_size": int(len(X_test)),
                        }
                        all_fold_metrics.append(metrics)

                    if runtime_cfg.get("show_fold_metrics", True):
                        dbg.log("FOLD", f"Fold {fold_idx} metrics: {metrics}")

                    mlflow.log_metrics(
                        {
                            f"fold_{fold_idx}_{k}": float(v)
                            for k, v in metrics.items()
                            if isinstance(v, (int, float, np.floating)) and k != "fold"
                        }
                    )

                    with dbg.timer("SAVE", f"Fold {fold_idx} save predictions"):
                        signal_arr = np.zeros_like(y_test, dtype=int)
                        gross_return_arr = np.zeros_like(y_test, dtype=float)
                        cost_return_arr = np.zeros_like(y_test, dtype=float)
                        net_return_arr = np.zeros_like(y_test, dtype=float)

                        if trading_enabled:
                            sorted_index = trade_arrays["sorted_index"]
                            signal_arr[sorted_index] = trade_arrays["signal"]
                            gross_return_arr[sorted_index] = trade_arrays["gross_return"]
                            cost_return_arr[sorted_index] = trade_arrays["cost_return"]
                            net_return_arr[sorted_index] = trade_arrays["net_return"]

                        fold_pred_df = pd.DataFrame(
                            {
                                "fold": fold_idx,
                                "ticker": tickers_test,
                                "timestamp": ts_test,
                                "y_true": y_test,
                                "y_pred": preds,
                                "y_true_sign": (y_test > 0).astype(int),
                                "y_pred_sign": (preds > 0).astype(int),
                                "signal": signal_arr,
                                "signal_action": np.where(
                                    signal_arr > 0,
                                    "long",
                                    np.where(signal_arr < 0, "short", "flat"),
                                ),
                                "expected_edge": preds - trading_threshold,
                                "gross_return": gross_return_arr,
                                "cost_return": cost_return_arr,
                                "net_return": net_return_arr,
                                "pred_error": preds - y_test,
                                "abs_error": np.abs(preds - y_test),
                            }
                        )

                        pred_dir = Path(cfg["artifacts"]["fold_predictions_dir"])
                        pred_format = cfg["artifacts"].get("fold_predictions_format", "parquet").lower()
                        pred_path = pred_dir / f"fold_{fold_idx}_predictions.{pred_format}"

                        save_dataframe(fold_pred_df, pred_path, logger=logger)
                        final_fold_pred_df = fold_pred_df.copy()

                    oof_y_true.append(y_test)
                    oof_y_pred.append(preds)
                    oof_tickers.append(tickers_test)
                    oof_timestamps.append(ts_test)

                    final_state_model = state_model
                    final_predictor = predictor

                except Exception as fold_exc:
                    logger.exception("[FOLD] Fold %s failed: %s", fold_idx, fold_exc)
                    raise

            # -------------------------------------------------
            # Summary after all folds
            # -------------------------------------------------
            optimized_trade_metrics: dict[str, float] | None = None
            threshold_search_df: pd.DataFrame | None = None
            if trading_enabled and trading_optimize_threshold and oof_y_true:
                with dbg.timer("TRADING", "Optimizing prediction threshold on OOF folds"):
                    y_oof = np.concatenate(oof_y_true)
                    pred_oof = np.concatenate(oof_y_pred)
                    tickers_oof = np.concatenate(oof_tickers)
                    ts_oof = np.concatenate(oof_timestamps)

                    (
                        optimized_threshold,
                        optimized_trade_metrics,
                        threshold_search_df,
                    ) = optimize_long_only_threshold(
                        y_true=y_oof,
                        y_pred=pred_oof,
                        tickers=tickers_oof,
                        timestamps=ts_oof,
                        transaction_cost_bps=trading_cost_bps,
                        periods_per_year=trading_periods_per_year,
                        selection_mode=trading_selection_mode,
                        top_k_per_timestamp=trading_top_k_per_timestamp,
                        thresholds=trading_threshold_grid,
                        objective=trading_optimize_objective,
                        logger=logger,
                    )

                    dbg.log(
                        "TRADING",
                        (
                            f"Optimized threshold={optimized_threshold:.6f} "
                            f"objective={trading_optimize_objective} "
                            f"value={optimized_trade_metrics.get(trading_optimize_objective, np.nan):.6f}"
                        ),
                    )

                    if threshold_search_df is not None:
                        search_path = (
                            Path(cfg["artifacts"]["output_dir"]) / "threshold_search.csv"
                        )
                        save_dataframe(threshold_search_df, search_path, logger=logger)

            with dbg.timer("SUMMARY", "Aggregating fold metrics"):
                avg_mae = float(np.mean([m["mae"] for m in all_fold_metrics]))
                avg_r2 = float(np.mean([m["r2"] for m in all_fold_metrics]))
                avg_directional_accuracy = float(
                    np.nanmean([m["directional_accuracy"] for m in all_fold_metrics])
                )
                avg_pearson_corr = float(np.nanmean([m["pearson_corr"] for m in all_fold_metrics]))
                avg_spearman_corr = float(np.nanmean([m["spearman_corr"] for m in all_fold_metrics]))
                avg_baseline_zero_mae = float(
                    np.mean([m["baseline_zero_mae"] for m in all_fold_metrics])
                )
                avg_baseline_mean_mae = float(
                    np.mean([m["baseline_mean_mae"] for m in all_fold_metrics])
                )
                if trading_enabled:
                    avg_trading_total_net_return = float(
                        np.nanmean([m["trading_total_net_return"] for m in all_fold_metrics])
                    )
                    avg_trading_sharpe = float(
                        np.nanmean([m["trading_sharpe"] for m in all_fold_metrics])
                    )
                    avg_trading_max_drawdown = float(
                        np.nanmean([m["trading_max_drawdown"] for m in all_fold_metrics])
                    )
                    avg_trading_trade_count = float(
                        np.nanmean([m["trading_trade_count"] for m in all_fold_metrics])
                    )
                    avg_trading_win_rate = float(
                        np.nanmean([m["trading_win_rate"] for m in all_fold_metrics])
                    )
                else:
                    avg_trading_total_net_return = np.nan
                    avg_trading_sharpe = np.nan
                    avg_trading_max_drawdown = np.nan
                    avg_trading_trade_count = np.nan
                    avg_trading_win_rate = np.nan

                summary = {
                    "avg_mae": avg_mae,
                    "avg_r2": avg_r2,
                    "avg_directional_accuracy": avg_directional_accuracy,
                    "avg_pearson_corr": avg_pearson_corr,
                    "avg_spearman_corr": avg_spearman_corr,
                    "avg_baseline_zero_mae": avg_baseline_zero_mae,
                    "avg_baseline_mean_mae": avg_baseline_mean_mae,
                    "avg_trading_total_net_return": avg_trading_total_net_return,
                    "avg_trading_sharpe": avg_trading_sharpe,
                    "avg_trading_max_drawdown": avg_trading_max_drawdown,
                    "avg_trading_trade_count": avg_trading_trade_count,
                    "avg_trading_win_rate": avg_trading_win_rate,
                    "n_folds": len(all_fold_metrics),
                    "feature_cols": feature_cols,
                    "state_model_enabled": state_enabled,
                    "trading_enabled": trading_enabled,
                    "trading_threshold_used": trading_threshold,
                    "trading_optimized_threshold": optimized_threshold,
                    "trading_selection_mode": trading_selection_mode,
                }

                if optimized_trade_metrics is not None:
                    summary["optimized_trading_total_net_return"] = float(
                        optimized_trade_metrics["trading_total_net_return"]
                    )
                    summary["optimized_trading_sharpe"] = float(
                        optimized_trade_metrics["trading_sharpe"]
                    )
                    summary["optimized_trading_max_drawdown"] = float(
                        optimized_trade_metrics["trading_max_drawdown"]
                    )
                    summary["optimized_trading_win_rate"] = float(
                        optimized_trade_metrics["trading_win_rate"]
                    )

                dbg.log("SUMMARY", f"avg_mae={avg_mae:.6f}")
                dbg.log("SUMMARY", f"avg_r2={avg_r2:.6f}")
                dbg.log("SUMMARY", f"avg_directional_accuracy={avg_directional_accuracy:.6f}")
                dbg.log("SUMMARY", f"avg_pearson_corr={avg_pearson_corr:.6f}")
                dbg.log("SUMMARY", f"avg_spearman_corr={avg_spearman_corr:.6f}")
                dbg.log("SUMMARY", f"avg_baseline_zero_mae={avg_baseline_zero_mae:.6f}")
                dbg.log("SUMMARY", f"avg_baseline_mean_mae={avg_baseline_mean_mae:.6f}")
                if trading_enabled:
                    dbg.log("SUMMARY", f"avg_trading_total_net_return={avg_trading_total_net_return:.6f}")
                    dbg.log("SUMMARY", f"avg_trading_sharpe={avg_trading_sharpe:.6f}")
                    dbg.log("SUMMARY", f"avg_trading_max_drawdown={avg_trading_max_drawdown:.6f}")
                    dbg.log("SUMMARY", f"avg_trading_trade_count={avg_trading_trade_count:.2f}")
                    dbg.log("SUMMARY", f"avg_trading_win_rate={avg_trading_win_rate:.6f}")
                    dbg.log("SUMMARY", f"trading_threshold_used={trading_threshold:.6f}")
                    dbg.log("SUMMARY", f"trading_optimized_threshold={optimized_threshold:.6f}")
                    if optimized_trade_metrics is not None:
                        dbg.log(
                            "SUMMARY",
                            (
                                "optimized_trading_total_net_return="
                                f"{optimized_trade_metrics['trading_total_net_return']:.6f}"
                            ),
                        )
                        dbg.log(
                            "SUMMARY",
                            f"optimized_trading_sharpe={optimized_trade_metrics['trading_sharpe']:.6f}",
                        )
                dbg.log("SUMMARY", f"n_folds={len(all_fold_metrics)}")

                mlflow.log_metrics(
                    {
                        "avg_mae": avg_mae,
                        "avg_r2": avg_r2,
                        "avg_directional_accuracy": avg_directional_accuracy,
                        "avg_pearson_corr": avg_pearson_corr,
                        "avg_spearman_corr": avg_spearman_corr,
                        "avg_baseline_zero_mae": avg_baseline_zero_mae,
                        "avg_baseline_mean_mae": avg_baseline_mean_mae,
                        "avg_trading_total_net_return": avg_trading_total_net_return,
                        "avg_trading_sharpe": avg_trading_sharpe,
                        "avg_trading_max_drawdown": avg_trading_max_drawdown,
                        "avg_trading_trade_count": avg_trading_trade_count,
                        "avg_trading_win_rate": avg_trading_win_rate,
                        "trading_threshold_used": trading_threshold,
                        "trading_optimized_threshold": optimized_threshold,
                    }
                )

                if optimized_trade_metrics is not None:
                    mlflow.log_metrics(
                        {
                            "optimized_trading_total_net_return": float(
                                optimized_trade_metrics["trading_total_net_return"]
                            ),
                            "optimized_trading_sharpe": float(
                                optimized_trade_metrics["trading_sharpe"]
                            ),
                            "optimized_trading_max_drawdown": float(
                                optimized_trade_metrics["trading_max_drawdown"]
                            ),
                            "optimized_trading_win_rate": float(
                                optimized_trade_metrics["trading_win_rate"]
                            ),
                        }
                    )

            if runtime_cfg.get("show_fold_table", True):
                log_fold_summary(logger=logger, fold_metrics=all_fold_metrics, enabled=True)

            if trading_enabled and final_fold_pred_df is not None and not final_fold_pred_df.empty:
                with dbg.timer("TRADING", "Building top-picks recommendations"):
                    recommendation_threshold = optimized_threshold
                    latest_ts = final_fold_pred_df["timestamp"].max()
                    latest_slice = final_fold_pred_df[final_fold_pred_df["timestamp"] == latest_ts].copy()

                    latest_slice = latest_slice.sort_values("y_pred", ascending=False)
                    if trading_selection_mode == "top_k_per_timestamp":
                        long_candidates = latest_slice[
                            latest_slice["y_pred"] > recommendation_threshold
                        ].head(recommendations_top_k)
                    else:
                        long_candidates = latest_slice[
                            latest_slice["y_pred"] > recommendation_threshold
                        ].head(recommendations_top_k)

                    if not long_candidates.empty:
                        rec_df = long_candidates[
                            [
                                "timestamp",
                                "ticker",
                                "y_pred",
                                "expected_edge",
                                "signal",
                                "signal_action",
                            ]
                        ].copy()
                        rec_df = rec_df.rename(columns={"y_pred": "predicted_return"})
                        rec_df["expected_edge"] = rec_df["predicted_return"] - recommendation_threshold
                        rec_df["threshold_used"] = recommendation_threshold
                    else:
                        rec_df = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "ticker",
                                "predicted_return",
                                "expected_edge",
                                "signal",
                                "signal_action",
                                "threshold_used",
                            ]
                        )

                    rec_path = (
                        Path(cfg["artifacts"]["output_dir"]) / "latest_recommendations.csv"
                    )
                    save_dataframe(rec_df, rec_path, logger=logger)

                    summary["recommendation_timestamp"] = str(latest_ts)
                    summary["recommendation_count"] = int(len(rec_df))
                    summary["top_recommendation_tickers"] = rec_df["ticker"].tolist()
                    summary["recommendation_threshold_used"] = recommendation_threshold

                    if not rec_df.empty:
                        dbg.log("TRADING", "Top recommendations:")
                        dbg.log("TRADING", rec_df.to_string(index=False))
                    else:
                        dbg.log(
                            "TRADING",
                            "No long recommendations passed threshold on latest timestamp",
                        )

            # -------------------------------------------------
            # Feature importance
            # -------------------------------------------------
            with dbg.timer("IMPORTANCE", "Extracting and saving feature importance"):
                if final_predictor is None:
                    raise ValueError("final_predictor is None; cannot compute feature importance")

                importance_df = extract_feature_importance(
                    predictor=final_predictor,
                    feature_names=full_feature_names,
                    logger=logger,
                )

                importance_path = Path(cfg["artifacts"]["output_dir"]) / "feature_importance.parquet"
                save_dataframe(importance_df, importance_path, logger=logger)

                dbg.log(
                    "IMPORTANCE",
                    f"Top 20 features:\n{importance_df.head(20).to_string(index=False)}",
                )

                summary["top_10_features"] = importance_df.head(10)["feature"].tolist()

            # -------------------------------------------------
            # Save artifacts
            # -------------------------------------------------
            with dbg.timer("ARTIFACT", "Saving artifacts"):
                out_dir = Path(cfg["artifacts"]["output_dir"])

                if final_predictor is None:
                    raise ValueError("final_predictor is None; training did not complete")

                if final_state_model is not None:
                    save_joblib(final_state_model, out_dir / "state_model.joblib", logger=logger)
                else:
                    dbg.log("ARTIFACT", "State model disabled or unavailable; skipping state_model.joblib")

                save_joblib(final_predictor, out_dir / "predictor.joblib", logger=logger)
                save_joblib(feature_cols, out_dir / "feature_cols.joblib", logger=logger)
                save_json(summary, out_dir / "summary.json", logger=logger)
                save_json(cfg, out_dir / "config_snapshot.json", logger=logger)

            dbg.log("MLFLOW", "MLflow run complete")

        dbg.banner("TRAIN PIPELINE FINISHED")
        dbg.log("RESULT", str(summary))
        return summary

    except Exception as exc:
        logger.exception("[PIPELINE] Training pipeline failed: %s", exc)
        raise