# Market ML System

Produktionsinriktat ML-system för marknadsdata med:
- window-baserad feature engineering
- state modeling (GMM)
- return prediction (LightGBM)
- walk-forward validering
- artifact-hantering och API-serving
- experimentspårning i MLflow

## Arkitektur (textdiagram)

```text
Raw market data (CSV)
        |
        v
Validate data schema/quality
        |
        v
Feature builder (returns, vol, momentum, candle, volume zscore)
        |
        v
Rolling windows + flatten
        |
        +------------------------------+
        |                              |
        v                              v
State model (GMM)                 Walk-forward splits
        |                              |
        v                              v
State id + state probabilities    Fold-wise training/evaluation
        \___________________________  /
                                    \/
                          Meta-features to predictor (LightGBM)
                                    |
                                    v
                            Forward return prediction
                                    |
             +----------------------+----------------------+
             |                                             |
             v                                             v
      Save artifacts (joblib/json)                 Log params/metrics (MLflow)
             |                                             |
             v                                             v
models/snapshots/latest/*                         local MLflow tracking UI
             |
             v
FastAPI service loads artifacts -> /predict
```

## Quickstart (train, serve, predict)

### 1) Miljö och installation

Krav:
- Python 3.11+

Installera beroenden:

```bash
pip install -e .
```

Eller med dev-verktyg:

```bash
pip install -e ".[dev]"
```

### 2) Forbered data

Traingspipen forvantar sig en CSV i:

`data/raw/market_data.csv`

Obligatoriska kolumner:
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Input-fil och ovriga hyperparametrar konfigureras i `configs/train.yaml`.

### 3) Trana modellen

```bash
python scripts/run_train.py
```

Alternativt via Make:

```bash
make train
```

Detta kor hela trankedjan och sparar artifacts i `models/snapshots/latest/`.

### 4) Starta API-server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Alternativt via Make:

```bash
make serve
```

Kontrollera health-endpoint:

```bash
curl http://127.0.0.1:8000/health
```

### 5) Gor en prediktion

#### A) Via script

```bash
python scripts/run_inference.py
```

Skriptet:
- laser `data/raw/market_data.csv`
- laddar artifacts fran `models/snapshots/latest/`
- kor inferens pa senaste 100 rader

#### B) Via API `/predict`

Skicka en JSON-lista med marknadsrader (minst tillräckligt efter feature/dropna för `window_size`, default 20).

Exempel:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "[{\"timestamp\":\"2026-01-01T10:00:00\",\"open\":100,\"high\":101,\"low\":99,\"close\":100.5,\"volume\":1200}]"
```

## Artifact-struktur

Efter träning sparas artefakter i `models/snapshots/latest/`:

```text
models/
  snapshots/
    latest/
      state_model.joblib      # GMM state model
      predictor.joblib        # LightGBM predictor
      feature_cols.joblib     # lista med featurekolumner i rätt ordning
      summary.json            # summering: avg_mae, avg_r2, n_folds, features
      config_snapshot.json    # konfigurationssnapshot från train-korning
```

Hur de används:
- API (`src/api/app.py`) laddar `state_model.joblib`, `predictor.joblib`, `feature_cols.joblib` vid startup.
- Inference-script (`scripts/run_inference.py`) laddar samma artifacts for lokal batch-inferens.

## Experimentflöde i MLflow

Träning i `src/pipelines/train_pipeline.py` gör följande i MLflow:

1. Sätter experiment via `mlflow.set_experiment(...)` (namn från `configs/train.yaml`).
2. Startar en run med `mlflow.start_run()`.
3. Loggar centrala parametrar:
   - `window_size`
   - `target_horizon`
   - `state_n_components`
   - `predictor_type`
4. Loggar fold-metrics per walk-forward split (t.ex. `fold_0_mae`, `fold_0_r2`, ...).
5. Loggar aggregat-metrics:
   - `avg_mae`
   - `avg_r2`

Starta MLflow UI lokalt:

```bash
mlflow ui
```

Och öppna:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

1. `pip install -e .`
2. Verifiera att `data/raw/market_data.csv` finns med rätt kolumner.
3. `python scripts/run_train.py`
4. `uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000`
5. `python scripts/run_inference.py` eller POST till `/predict`
6. `mlflow ui` för att validera experiment och metrics

Om allt fungerar har du:
- tränad modell
- sparade artifacts
- fungerande prediktions-API
- spårbara experiment