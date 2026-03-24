# Market ML System

Production-oriented machine learning system for market-state modeling, return prediction, experiment tracking, and model serving.

This repository demonstrates how to build an end-to-end ML pipeline for financial time series: from raw market data and feature engineering to state modeling, walk-forward validation, artifact versioning, local experiment tracking, and API inference.

---

## Why this project matters

Many market ML projects stop at notebook experiments. This one is structured as a reproducible system:

- configuration-driven training
- leakage-aware temporal validation
- explicit artifact management
- model serving through FastAPI
- local experiment tracking with MLflow
- clear separation between training, inference, and API layers

The goal is not just to train a model, but to package the workflow the way a production-minded ML engineer would.

---

## System overview

The pipeline combines feature engineering, unsupervised regime/state modeling, supervised return prediction, and operational inference.

```text
Raw market data (CSV)
        |
        v
Validate schema and data quality
        |
        v
Feature engineering
(returns, volatility, momentum, candle structure, volume regime, flow proxies)
        |
        v
Rolling window construction + flattening
        |
        +-------------------------------+
        |                               |
        v                               v
State model (GMM)                 Walk-forward splits
        |                               |
        v                               v
State IDs + probabilities         Fold-wise training and evaluation
        \_____________________________ /
                                      v
                        Meta-features into predictor (LightGBM)
                                      |
                                      v
                           Forward return prediction
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
        Save model artifacts                    Log runs and metrics
        (joblib / json / parquet)              in MLflow
                 |
                 v
      FastAPI loads artifacts for /predict