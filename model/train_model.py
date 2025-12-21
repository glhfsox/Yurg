from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.db import ensure_coin_ids, get_engine
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

CLASS_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}


@dataclass(frozen=True)
class TrainMetadata:
    symbol: str
    coin_id: int
    model_version: str
    theta: float
    trained_at_utc: str
    feature_columns: list[str]
    class_to_index: dict[int, int]
    metrics: dict[str, Any]
    model_params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-class PyTorch model from features_15m.")
    parser.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], default="BTCUSDT")
    parser.add_argument(
        "--theta",
        type=float,
        default=float(os.environ.get("FEATURES_THETA", "0.001")),
        help="Baseline threshold θ for ret_1 sign baseline.",
    )
    parser.add_argument(
        "--model-version",
        default=os.environ.get("MODEL_VERSION"),
        help="Stored in predictions_15m (default: MODEL_VERSION).",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def default_model_version() -> str:
    return datetime.now(tz=timezone.utc).strftime("v%Y%m%d_%H%M%S")


def load_training_data(engine, *, coin_id: int) -> pd.DataFrame:
    cols = ["ts", *FEATURE_COLUMNS, "target_class"]
    query = f"""
        SELECT {", ".join(cols)}
        FROM features_15m
        WHERE coin_id = %(coin_id)s
          AND target_class IS NOT NULL
        ORDER BY ts ASC
    """
    df = pd.read_sql_query(query, engine, params={"coin_id": coin_id})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 500:
        raise ValueError(f"Not enough rows to train (need ~500+, got {n}).")
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def baseline_majority(y_train: np.ndarray, n: int) -> np.ndarray:
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]
    return np.full(n, majority)


def baseline_sign_ret_1(ret_1: np.ndarray, *, theta: float) -> np.ndarray:
    return np.where(ret_1 >= theta, 1, np.where(ret_1 <= -theta, -1, 0))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
    }


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


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_classes(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = to_tensor(X, device)
    logits = model(xb)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    return np.vectorize(INDEX_TO_CLASS.get)(preds)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = to_tensor(X, device)
    logits = model(xb)
    proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    device = require_cuda()
    torch.manual_seed(42)
    np.random.seed(42)

    model_version = args.model_version or default_model_version()
    engine = get_engine()
    coin_ids = ensure_coin_ids(engine, [args.symbol])
    coin_id = coin_ids[args.symbol]

    df = load_training_data(engine, coin_id=coin_id)
    if df.empty:
        logger.error("No labeled rows in features_15m for %s; run features.py first.", args.symbol)
        return 2

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_class"])
    train_df, val_df, test_df = split_time(df)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype="float64")
    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype="float64")
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype="float64")

    y_train = train_df["target_class"].to_numpy(dtype="int64")
    y_val = val_df["target_class"].to_numpy(dtype="int64")
    y_test = test_df["target_class"].to_numpy(dtype="int64")

    metrics: dict[str, Any] = {}
    metrics["baseline_majority_val"] = eval_metrics(y_val, baseline_majority(y_train, len(y_val)))
    metrics["baseline_majority_test"] = eval_metrics(y_test, baseline_majority(y_train, len(y_test)))
    metrics["baseline_ret_sign_val"] = eval_metrics(
        y_val, baseline_sign_ret_1(val_df["ret_1"].to_numpy(dtype="float64"), theta=args.theta)
    )
    metrics["baseline_ret_sign_test"] = eval_metrics(
        y_test, baseline_sign_ret_1(test_df["ret_1"].to_numpy(dtype="float64"), theta=args.theta)
    )

    y_train_idx = np.vectorize(CLASS_TO_INDEX.get)(y_train)
    y_val_idx = np.vectorize(CLASS_TO_INDEX.get)(y_val)
    y_test_idx = np.vectorize(CLASS_TO_INDEX.get)(y_test)

    hidden_dims = [256, 128]
    dropout = 0.2

    mean, std = standardize_fit(X_train)
    X_train = standardize_transform(X_train, mean, std)
    X_val = standardize_transform(X_val, mean, std)
    X_test = standardize_transform(X_test, mean, std)

    model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    counts = np.bincount(y_train_idx, minlength=3).astype("float64")
    weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = torch.utils.data.TensorDataset(
        to_tensor(X_train, device), torch.tensor(y_train_idx, device=device, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    best_val_f1 = -1.0
    best_state = None
    patience = 5
    patience_left = patience

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_pred = predict_classes(model, X_val, device)
        val_f1 = f1_score(y_val, val_pred, average="macro")

        logger.info("epoch=%s loss=%.5f val_macro_f1=%.4f", epoch + 1, loss, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = predict_classes(model, X_test, device)
    metrics["mlp_test"] = eval_metrics(y_test, test_pred)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.symbol.lower()}_{model_version}.pt"
    meta_path = out_dir / f"{args.symbol.lower()}_{model_version}.json"

    torch.save(model.state_dict(), model_path)
    meta = TrainMetadata(
        symbol=args.symbol,
        coin_id=coin_id,
        model_version=model_version,
        theta=float(args.theta),
        trained_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        feature_columns=FEATURE_COLUMNS,
        class_to_index=CLASS_TO_INDEX,
        metrics=metrics,
        model_params={
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "scaler_mean": mean.tolist(),
            "scaler_std": std.tolist(),
            "class_weights": weights.tolist(),
        },
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))

    logger.info("Saved model: %s", model_path)
    logger.info("Saved metadata: %s", meta_path)
    logger.info("Test metrics: %s", json.dumps({"mlp_test": metrics["mlp_test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
