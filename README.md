# Yurg — BTC/ETH 15m Crypto Prediction Pipeline

End-to-end pipeline for crypto OHLCV ingestion, feature engineering, model training, and inference.  
Focus: clean engineering, time-series correctness, and reproducible workflows (not trading alpha).

## What It Does
- Pulls BTCUSDT/ETHUSDT OHLCV (15m + 1h) from Binance public REST.
- Stores raw candles in Postgres.
- Builds 15m features with 1h context + cross-coin signals.
- Trains a 3-class model (down / flat / up) per coin using PyTorch (GPU required).
- Stores predictions with model versioning.

## Quickstart
1) Start Postgres:
```bash
docker compose -f docker-compose.yaml up -d
```

2) Create `.env` (example):
```env
DATABASE_URL=postgresql+psycopg2://admin:1234@localhost:5433/yurg
BINANCE_BASE_URL=https://api.binance.com
INGEST_LOOKBACK_DAYS_15M=730
INGEST_LOOKBACK_DAYS_1H=730
FEATURES_THETA=0.001
LOG_LEVEL=INFO
```

3) Create venv + install deps:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4) Ingest candles:
```bash
python3 -m src.ingest
```

5) Build features:
```bash
python3 -m src.features --full
```

6) Train models (GPU required):
```bash
python3 -m model.train_model --symbol BTCUSDT --model-version v1
python3 -m model.train_model --symbol ETHUSDT --model-version v1
```
`model.train_model` runs walk-forward validation by default. To skip CV: `--cv none`.

7) Predict latest bars:
```bash
python3 -m model.predict --symbol BTCUSDT --model-version v1
python3 -m model.predict --symbol ETHUSDT --model-version v1
```

To populate metrics on historical data (so the dashboard has something to evaluate), backfill predictions:
```bash
python3 -m model.predict --mode backtest --limit 20000 --symbol BTCUSDT --model-version v1
python3 -m model.predict --mode backtest --limit 20000 --symbol ETHUSDT --model-version v1
```

## Visualization (Streamlit)
Run an interactive dashboard (price + targets + predictions):
```bash
streamlit run viz/streamlit_app.py
```

## Data Layout
- `sql/schema.sql`: tables for raw OHLCV, features, predictions
- `sql/views.sql`: analytics views (daily summary, rolling volatility, signal distribution)
- `src/ingest.py`: Binance → Postgres
- `src/features.py`: features + labels
- `model/train_model.py`: PyTorch training (GPU-only)
- `model/predict.py`: inference → `predictions_15m`
- `viz/streamlit_app.py`: dashboard

## Notes
- All timestamps are candle close time (UTC).
- Predictions are stored in `predictions_15m` with `model_version`.
