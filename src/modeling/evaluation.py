import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import logging


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if logger:
        logger.info("[EVAL] Regression metrics: %s", metrics)

    return metrics


def baseline_metrics(
    y_true: np.ndarray,
    y_train: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    zero_pred = np.zeros_like(y_true, dtype=float)
    mean_pred = np.full_like(y_true, fill_value=float(np.mean(y_train)), dtype=float)

    metrics = {
        "baseline_zero_mae": float(mean_absolute_error(y_true, zero_pred)),
        "baseline_zero_r2": float(r2_score(y_true, zero_pred)),
        "baseline_mean_mae": float(mean_absolute_error(y_true, mean_pred)),
        "baseline_mean_r2": float(r2_score(y_true, mean_pred)),
    }

    if logger:
        logger.info("[EVAL] Baseline metrics: %s", metrics)

    return metrics


def directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    true_up = y_true > 0
    pred_up = y_pred > 0

    direction_acc = float((true_up == pred_up).mean())

    predicted_positive_mask = pred_up
    predicted_negative_mask = ~pred_up

    positive_hit_rate = (
        float((y_true[predicted_positive_mask] > 0).mean())
        if predicted_positive_mask.sum() > 0
        else np.nan
    )
    negative_hit_rate = (
        float((y_true[predicted_negative_mask] <= 0).mean())
        if predicted_negative_mask.sum() > 0
        else np.nan
    )

    metrics = {
        "directional_accuracy": direction_acc,
        "predicted_positive_rate": float(pred_up.mean()),
        "predicted_negative_rate": float((~pred_up).mean()),
        "positive_hit_rate": float(positive_hit_rate) if not np.isnan(positive_hit_rate) else np.nan,
        "negative_hit_rate": float(negative_hit_rate) if not np.isnan(negative_hit_rate) else np.nan,
    }

    if logger:
        logger.info("[EVAL] Directional metrics: %s", metrics)

    return metrics


def correlation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)

    pearson = y_true_s.corr(y_pred_s, method="pearson")
    spearman = y_true_s.corr(y_pred_s, method="spearman")

    metrics = {
        "pearson_corr": float(pearson) if pd.notna(pearson) else np.nan,
        "spearman_corr": float(spearman) if pd.notna(spearman) else np.nan,
    }

    if logger:
        logger.info("[EVAL] Correlation metrics: %s", metrics)

    return metrics


def _build_long_only_signals(
    eval_df: pd.DataFrame,
    threshold: float,
    selection_mode: str,
    top_k_per_timestamp: int,
) -> np.ndarray:
    if selection_mode == "top_k_per_timestamp":
        if top_k_per_timestamp <= 0:
            raise ValueError("top_k_per_timestamp must be positive for top_k_per_timestamp mode")

        ranked = eval_df.groupby("timestamp")["y_pred"].rank(method="first", ascending=False)
        signal = ((ranked <= top_k_per_timestamp) & (eval_df["y_pred"] > threshold)).astype(int)
        return signal.to_numpy(dtype=int)

    # Default/fallback mode is threshold.
    return (eval_df["y_pred"] > threshold).astype(int).to_numpy(dtype=int)


def trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tickers: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    transaction_cost_bps: float,
    periods_per_year: int = 252 * 26,
    selection_mode: str = "threshold",
    top_k_per_timestamp: int = 5,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    eval_df = pd.DataFrame(
        {
            "ticker": tickers,
            "timestamp": pd.to_datetime(timestamps),
            "y_true": y_true.astype(float),
            "y_pred": y_pred.astype(float),
        }
    ).sort_values(["timestamp", "ticker"], kind="stable")

    signal = _build_long_only_signals(
        eval_df=eval_df,
        threshold=threshold,
        selection_mode=selection_mode,
        top_k_per_timestamp=top_k_per_timestamp,
    )
    eval_df["signal"] = signal

    cost_rate = transaction_cost_bps / 10000.0
    eval_df["position_change"] = (
        eval_df.groupby("ticker")["signal"].diff().fillna(eval_df["signal"]).abs()
    )
    eval_df["gross_return"] = eval_df["signal"] * eval_df["y_true"]
    eval_df["cost_return"] = eval_df["position_change"] * cost_rate
    eval_df["net_return"] = eval_df["gross_return"] - eval_df["cost_return"]

    ts_returns = (
        eval_df.groupby("timestamp")[["gross_return", "cost_return", "net_return"]].mean().reset_index()
    )
    portfolio_net = ts_returns["net_return"].to_numpy(dtype=float)

    equity_curve = np.cumprod(1.0 + portfolio_net)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve / running_peak) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else np.nan

    exposure_mask = eval_df["signal"].to_numpy(dtype=int) != 0
    n_trades = int(eval_df["position_change"].sum())

    mean_net = float(np.mean(portfolio_net)) if len(portfolio_net) > 0 else np.nan
    std_net = float(np.std(portfolio_net, ddof=0)) if len(portfolio_net) > 0 else np.nan
    sharpe = (
        float((mean_net / std_net) * np.sqrt(periods_per_year))
        if std_net is not None and std_net > 0
        else np.nan
    )

    metrics = {
        "trading_total_net_return": float(np.sum(portfolio_net)),
        "trading_avg_net_return": mean_net,
        "trading_sharpe": sharpe,
        "trading_max_drawdown": max_drawdown,
        "trading_exposure_rate": float(np.mean(exposure_mask)) if len(exposure_mask) > 0 else np.nan,
        "trading_trade_count": float(n_trades),
        "trading_win_rate": (
            float(np.mean(eval_df.loc[exposure_mask, "net_return"] > 0))
            if np.any(exposure_mask)
            else np.nan
        ),
    }

    extras = {
        "signal": eval_df["signal"].to_numpy(dtype=int),
        "gross_return": eval_df["gross_return"].to_numpy(dtype=float),
        "cost_return": eval_df["cost_return"].to_numpy(dtype=float),
        "net_return": eval_df["net_return"].to_numpy(dtype=float),
        "portfolio_net_return": portfolio_net,
        "sorted_index": eval_df.index.to_numpy(dtype=int),
    }

    if logger:
        logger.info("[EVAL] Trading metrics: %s", metrics)

    return metrics, extras


def optimize_long_only_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tickers: np.ndarray,
    timestamps: np.ndarray,
    transaction_cost_bps: float,
    periods_per_year: int,
    selection_mode: str,
    top_k_per_timestamp: int,
    thresholds: list[float],
    objective: str = "trading_total_net_return",
    logger: logging.Logger | None = None,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    if not thresholds:
        raise ValueError("thresholds list cannot be empty")

    rows: list[dict[str, float]] = []
    best_threshold = thresholds[0]
    best_metrics: dict[str, float] | None = None
    best_objective = -np.inf

    for threshold in thresholds:
        metrics, _ = trading_metrics(
            y_true=y_true,
            y_pred=y_pred,
            tickers=tickers,
            timestamps=timestamps,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
            selection_mode=selection_mode,
            top_k_per_timestamp=top_k_per_timestamp,
            logger=None,
        )
        row = {"threshold": float(threshold), **metrics}
        rows.append(row)

        score = row.get(objective, np.nan)
        if np.isnan(score):
            continue
        if score > best_objective:
            best_objective = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No valid threshold produced a finite optimization score")

    result_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    if logger:
        logger.info(
            "[EVAL] Threshold optimization complete. objective=%s best_threshold=%.6f best_score=%.6f",
            objective,
            best_threshold,
            best_objective,
        )

    return best_threshold, best_metrics, result_df