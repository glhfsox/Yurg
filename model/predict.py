from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from psycopg2.extras import execute_values

from src.db import ensure_coin_ids, get_engine
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

INDEX_TO_CLASS = {0: -1, 1: 0, 2: 1}


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    symbol: str
    coin_id: int
    model_version: str
    feature_columns: list[str]
    hidden_dims: list[int]
    dropout: float
    scaler_mean: np.ndarray
    scaler_std: np.ndarray


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    return torch.device("cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and store predictions_15m.")
    parser.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], default="BTCUSDT")
    parser.add_argument(
        "--model-version",
        default=os.environ.get("MODEL_VERSION"),
        help="Must match the model files in models/.",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest"],
        default="live",
        help="live: predict only future rows (target_class IS NULL); backtest: predict labeled history.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute predictions even if they already exist for this model_version.",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def load_model(symbol: str, model_version: str, device: torch.device) -> LoadedModel:
    if not model_version:
        raise ValueError("--model-version is required (or set MODEL_VERSION)")
    model_path = Path("models") / f"{symbol.lower()}_{model_version}.pt"
    meta_path = Path("models") / f"{symbol.lower()}_{model_version}.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text())
    hidden_dims = list(meta.get("model_params", {}).get("hidden_dims", [256, 128]))
    dropout = float(meta.get("model_params", {}).get("dropout", 0.2))
    scaler_mean = np.array(meta.get("model_params", {}).get("scaler_mean", []), dtype="float64")
    scaler_std = np.array(meta.get("model_params", {}).get("scaler_std", []), dtype="float64")
    if scaler_mean.size == 0 or scaler_std.size == 0:
        raise RuntimeError("Scaler parameters not found in model metadata.")

    model = MLP(input_dim=len(FEATURE_COLUMNS), hidden_dims=hidden_dims, dropout=dropout).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return LoadedModel(
        model=model,
        symbol=symbol,
        coin_id=int(meta["coin_id"]),
        model_version=str(meta["model_version"]),
        feature_columns=list(meta.get("feature_columns", FEATURE_COLUMNS)),
        hidden_dims=hidden_dims,
        dropout=dropout,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def load_candidates(
    engine,
    *,
    coin_id: int,
    model_version: str,
    limit: int,
    mode: str,
    overwrite: bool,
) -> pd.DataFrame:
    cols = ["ts", *FEATURE_COLUMNS]
    target_filter = "IS NULL" if mode == "live" else "IS NOT NULL"
    predicted_filter = "" if overwrite else "AND p.id IS NULL"
    query = f"""
        SELECT f.{", f.".join(cols)}
        FROM features_15m f
        LEFT JOIN predictions_15m p
          ON p.coin_id = f.coin_id
         AND p.ts = f.ts
         AND p.model_version = %(model_version)s
        WHERE f.coin_id = %(coin_id)s
          AND f.target_class {target_filter}
          {predicted_filter}
        ORDER BY f.ts DESC
        LIMIT %(limit)s
    """
    df = pd.read_sql_query(
        query,
        engine,
        params={"coin_id": coin_id, "model_version": model_version, "limit": int(limit)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def insert_predictions(engine, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO predictions_15m
          (coin_id, ts, model_version, predicted_class, prob_down, prob_flat, prob_up)
        VALUES %s
        ON CONFLICT (coin_id, ts, model_version)
        DO UPDATE SET
          predicted_class = EXCLUDED.predicted_class,
          prob_down = EXCLUDED.prob_down,
          prob_flat = EXCLUDED.prob_flat,
          prob_up = EXCLUDED.prob_up
    """
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    finally:
        conn.close()


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xb)
    return torch.softmax(logits, dim=1).cpu().numpy()


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    device = require_cuda()
    loaded = load_model(args.symbol, args.model_version, device)

    engine = get_engine()
    coin_ids = ensure_coin_ids(engine, [args.symbol])
    coin_id = coin_ids[args.symbol]
    if coin_id != loaded.coin_id:
        logger.warning("coin_id mismatch (db=%s, model=%s); continuing", coin_id, loaded.coin_id)

    if args.limit is None:
        limit = 1 if args.mode == "live" else 8000
    else:
        limit = int(args.limit)

    df = load_candidates(
        engine,
        coin_id=coin_id,
        model_version=loaded.model_version,
        limit=limit,
        mode=args.mode,
        overwrite=bool(args.overwrite),
    )
    if df.empty:
        logger.info(
            "No rows to predict for %s (mode=%s, model_version=%s).",
            args.symbol,
            args.mode,
            loaded.model_version,
        )
        return 0

    df = df.dropna(subset=FEATURE_COLUMNS)
    if df.empty:
        logger.info("All candidate rows have missing features; run features.py again.")
        return 0

    X = df[FEATURE_COLUMNS].to_numpy(dtype="float64")
    X = standardize_transform(X, loaded.scaler_mean, loaded.scaler_std)
    proba = predict_proba(loaded.model, X, device)
    pred_idx = np.argmax(proba, axis=1)
    pred_class = np.vectorize(INDEX_TO_CLASS.get)(pred_idx).astype(int)

    rows: list[tuple[Any, ...]] = []
    for i, ts in enumerate(df["ts"]):
        rows.append(
            (
                coin_id,
                ts.to_pydatetime().astimezone(timezone.utc),
                loaded.model_version,
                int(pred_class[i]),
                float(proba[i, 0]),
                float(proba[i, 1]),
                float(proba[i, 2]),
            )
        )

    insert_predictions(engine, rows)
    for row in rows:
        logger.info(
            "%s %s pred=%s p=[down=%.4f flat=%.4f up=%.4f] model=%s",
            args.symbol,
            row[1].isoformat(),
            row[3],
            row[4],
            row[5],
            row[6],
            row[2],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
