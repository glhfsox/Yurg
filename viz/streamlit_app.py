from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from sqlalchemy import create_engine
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLUMNS  # noqa: E402


def get_engine():
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set (see .env)")
    return create_engine(db_url, pool_pre_ping=True)


def load_coin_id(engine, symbol: str) -> int:
    df = pd.read_sql_query(
        "SELECT coin_id FROM coin WHERE symbol = %(symbol)s",
        engine,
        params={"symbol": symbol},
    )
    if df.empty:
        raise RuntimeError(f"Symbol not found in coin table: {symbol}")
    return int(df.iloc[0]["coin_id"])


def list_model_versions(engine, coin_id: int, symbol: str) -> list[str]:
    local_versions: dict[str, pd.Timestamp] = {}
    models_dir = Path("models")
    if models_dir.exists():
        for meta_path in models_dir.glob(f"{symbol.lower()}_*.json"):
            try:
                meta = json.loads(meta_path.read_text())
                mv = str(meta.get("model_version", ""))
                trained_at = meta.get("trained_at_utc")
                if not mv or not trained_at:
                    continue
                ts = pd.to_datetime(trained_at, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                local_versions[mv] = ts
            except Exception:
                continue

    df = pd.read_sql_query(
        """
        SELECT model_version, MAX(created_at) AS last_created_at
        FROM predictions_15m
        WHERE coin_id = %(coin_id)s
        GROUP BY model_version
        ORDER BY last_created_at DESC
        """,
        engine,
        params={"coin_id": coin_id},
    )
    db_versions: dict[str, pd.Timestamp] = {}
    if not df.empty:
        for _, row in df.iterrows():
            mv = str(row["model_version"])
            ts = pd.to_datetime(row["last_created_at"], utc=True, errors="coerce")
            if mv and not pd.isna(ts):
                db_versions[mv] = ts

    # Prefer "latest model" by trained_at_utc if available; otherwise fall back to DB recency.
    all_versions = set(db_versions) | set(local_versions)

    def sort_key(mv: str) -> pd.Timestamp:
        return local_versions.get(mv) or db_versions.get(mv) or pd.Timestamp.min

    return sorted(all_versions, key=sort_key, reverse=True)


def load_price(engine, coin_id: int, bars: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT ts, open, high, low, close, volume
        FROM ohlcv_15m
        WHERE coin_id = %(coin_id)s
        ORDER BY ts DESC
        LIMIT %(bars)s
        """,
        engine,
        params={"coin_id": coin_id, "bars": int(bars)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def load_features_labels(engine, coin_id: int, bars: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT ts, target_class
        FROM features_15m
        WHERE coin_id = %(coin_id)s
        ORDER BY ts DESC
        LIMIT %(bars)s
        """,
        engine,
        params={"coin_id": coin_id, "bars": int(bars)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def load_predictions(engine, coin_id: int, model_version: str, bars: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT ts, predicted_class, prob_down, prob_flat, prob_up
        FROM predictions_15m
        WHERE coin_id = %(coin_id)s
          AND model_version = %(model_version)s
        ORDER BY ts DESC
        LIMIT %(bars)s
        """,
        engine,
        params={"coin_id": coin_id, "model_version": model_version, "bars": int(bars)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def load_feature_rows(engine, coin_id: int, rows: int) -> pd.DataFrame:
    cols = ["ts", *FEATURE_COLUMNS, "target_class"]
    df = pd.read_sql_query(
        f"""
        SELECT {", ".join(cols)}
        FROM features_15m
        WHERE coin_id = %(coin_id)s
        ORDER BY ts DESC
        LIMIT %(rows)s
        """,
        engine,
        params={"coin_id": coin_id, "rows": int(rows)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def load_labeled_features(engine, coin_id: int, rows: int) -> pd.DataFrame:
    cols = ["ts", "target_class", *FEATURE_COLUMNS]
    df = pd.read_sql_query(
        f"""
        SELECT {", ".join(cols)}
        FROM features_15m
        WHERE coin_id = %(coin_id)s
          AND target_class IS NOT NULL
        ORDER BY ts DESC
        LIMIT %(rows)s
        """,
        engine,
        params={"coin_id": coin_id, "rows": int(rows)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


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


def _parse_utc_ts(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.Timestamp(0, tz="UTC")
    ts = pd.to_datetime(str(value), utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp(0, tz="UTC")
    return ts  # type: ignore[return-value]


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


@st.cache_data
def load_local_model_metadatas(symbol: str) -> list[dict[str, Any]]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    metas: list[dict[str, Any]] = []
    for meta_path in models_dir.glob(f"{symbol.lower()}_*.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if not isinstance(meta, dict):
                continue
            mv = meta.get("model_version")
            if not mv:
                # Fallback to filename: btcusdt_v4.json -> v4
                mv = meta_path.stem.split("_", 1)[-1]
                meta["model_version"] = mv
            meta["_meta_path"] = str(meta_path)
            metas.append(meta)
        except Exception:
            continue

    def sort_key(m: dict[str, Any]) -> pd.Timestamp:
        return _parse_utc_ts(m.get("trained_at_utc"))

    return sorted(metas, key=sort_key, reverse=True)


def summarize_model_metas(metas: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for meta in metas:
        metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
        wf = metrics.get("walk_forward", {}) if isinstance(metrics.get("walk_forward"), dict) else {}
        wf_cfg = wf.get("config", {}) if isinstance(wf.get("config"), dict) else {}
        wf_folds = wf.get("folds", []) if isinstance(wf.get("folds"), list) else []

        def metric_block(name: str) -> dict[str, Any]:
            blk = metrics.get(name, {})
            if not isinstance(blk, dict):
                return {}
            return {
                f"{name}.acc": _coerce_float(blk.get("accuracy")),
                f"{name}.macro_f1": _coerce_float(blk.get("macro_f1")),
            }

        row: dict[str, Any] = {
            "model_version": str(meta.get("model_version", "")),
            "trained_at_utc": _parse_utc_ts(meta.get("trained_at_utc")),
            "theta": _coerce_float(meta.get("theta")),
            "wf.folds": len(wf_folds) if wf_folds else None,
            "wf.mean_acc": _coerce_float(wf.get("mean_accuracy")),
            "wf.std_acc": _coerce_float(wf.get("std_accuracy")),
            "wf.mean_macro_f1": _coerce_float(wf.get("mean_macro_f1")),
            "wf.std_macro_f1": _coerce_float(wf.get("std_macro_f1")),
            "wf.cfg.min_train_frac": _coerce_float(wf_cfg.get("min_train_frac")),
            "wf.cfg.val_frac": _coerce_float(wf_cfg.get("val_frac")),
            "_meta_path": meta.get("_meta_path"),
        }
        row.update(metric_block("mlp_val"))
        row.update(metric_block("mlp_test"))
        row.update(metric_block("baseline_majority_val"))
        row.update(metric_block("baseline_majority_test"))
        row.update(metric_block("baseline_ret_sign_val"))
        row.update(metric_block("baseline_ret_sign_test"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("trained_at_utc", ascending=False, na_position="last")
    # Display-friendly formatting (keep raw numeric types for sorting/plots).
    return df


def load_walk_forward_folds(meta: dict[str, Any]) -> pd.DataFrame:
    metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
    wf = metrics.get("walk_forward", {}) if isinstance(metrics.get("walk_forward"), dict) else {}
    folds = wf.get("folds", []) if isinstance(wf.get("folds"), list) else []
    rows: list[dict[str, Any]] = []
    for f in folds:
        if not isinstance(f, dict):
            continue
        vm = f.get("val_metrics", {}) if isinstance(f.get("val_metrics"), dict) else {}
        rows.append(
            {
                "fold": _coerce_int(f.get("fold")),
                "train_rows": _coerce_int(f.get("train_rows")),
                "val_rows": _coerce_int(f.get("val_rows")),
                "best_epoch": _coerce_int(f.get("best_epoch")),
                "val_accuracy": _coerce_float(vm.get("accuracy")),
                "val_macro_f1": _coerce_float(vm.get("macro_f1")),
                "confusion_matrix": vm.get("confusion_matrix"),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["fold"])
    if df.empty:
        return df
    return df.sort_values("fold")


def make_cv_fold_chart(df_folds: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_folds["fold"],
            y=df_folds["val_accuracy"],
            mode="lines+markers",
            name="Fold val accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_folds["fold"],
            y=df_folds["val_macro_f1"],
            mode="lines+markers",
            name="Fold val macro-F1",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis=dict(title="Fold #", tickmode="linear", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title="Walk-forward CV scores by fold",
    )
    return fig


@st.cache_resource
def load_local_model(symbol: str, model_version: str) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    meta_path = Path("models") / f"{symbol.lower()}_{model_version}.json"
    model_path = Path("models") / f"{symbol.lower()}_{model_version}.pt"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError("Model files not found under ./models")

    meta = json.loads(meta_path.read_text())
    hidden_dims = list(meta.get("model_params", {}).get("hidden_dims", [256, 128]))
    dropout = float(meta.get("model_params", {}).get("dropout", 0.2))
    scaler_mean = np.array(meta.get("model_params", {}).get("scaler_mean", []), dtype="float64")
    scaler_std = np.array(meta.get("model_params", {}).get("scaler_std", []), dtype="float64")
    if scaler_mean.size == 0 or scaler_std.size == 0:
        raise RuntimeError("Scaler parameters not found in model metadata.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for on-the-fly evaluation but is not available.")
    device = torch.device("cuda")

    model = MLP(input_dim=len(FEATURE_COLUMNS), hidden_dims=hidden_dims, dropout=dropout).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler_mean, scaler_std


@torch.no_grad()
def predict_on_the_fly(
    model: nn.Module, X: np.ndarray, scaler_mean: np.ndarray, scaler_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Returns (pred_class, proba) where classes are [-1,0,1] and proba columns are [down, flat, up]
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)
    Xs = (X - scaler_mean) / scaler_std
    xb = torch.tensor(Xs, dtype=torch.float32, device=torch.device("cuda"))
    logits = model(xb)
    proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred_idx = np.argmax(proba, axis=1)
    idx_to_class = {0: -1, 1: 0, 2: 1}
    pred_class = np.vectorize(idx_to_class.get)(pred_idx).astype(int)
    return pred_class, proba


def load_eval(engine, coin_id: int, model_version: str, rows: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
          f.ts,
          f.target_class,
          p.predicted_class,
          p.prob_down,
          p.prob_flat,
          p.prob_up
        FROM predictions_15m p
        JOIN features_15m f
          ON f.coin_id = p.coin_id
         AND f.ts = p.ts
        WHERE p.coin_id = %(coin_id)s
          AND p.model_version = %(model_version)s
          AND f.target_class IS NOT NULL
        ORDER BY p.ts DESC
        LIMIT %(rows)s
        """,
        engine,
        params={"coin_id": coin_id, "model_version": model_version, "rows": int(rows)},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def make_price_chart(df_price: pd.DataFrame, df_labels: pd.DataFrame, df_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_price["ts"],
            y=df_price["close"],
            mode="lines",
            name="Close",
            line=dict(width=2),
        )
    )

    if not df_labels.empty:
        label_colors = {1: "#2ecc71", 0: "#f1c40f", -1: "#e74c3c"}
        colors = df_labels["target_class"].map(label_colors).fillna("#95a5a6")
        fig.add_trace(
            go.Scatter(
                x=df_labels["ts"],
                y=df_price.set_index("ts").reindex(df_labels["ts"])["close"],
                mode="markers",
                name="Target (-1/0/+1)",
                marker=dict(size=6, color=colors),
            )
        )

    if not df_pred.empty:
        pred_colors = {1: "#27ae60", 0: "#f39c12", -1: "#c0392b"}
        colors = df_pred["predicted_class"].map(pred_colors).fillna("#7f8c8d")
        fig.add_trace(
            go.Scatter(
                x=df_pred["ts"],
                y=df_price.set_index("ts").reindex(df_pred["ts"])["close"],
                mode="markers",
                name="Predicted (-1/0/+1)",
                marker=dict(size=9, symbol="x", color=colors),
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="UTC time",
        yaxis_title="Price",
    )
    return fig


def make_prob_chart(df_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred["ts"], y=df_pred["prob_down"], name="P(down)", mode="lines"))
    fig.add_trace(go.Scatter(x=df_pred["ts"], y=df_pred["prob_flat"], name="P(flat)", mode="lines"))
    fig.add_trace(go.Scatter(x=df_pred["ts"], y=df_pred["prob_up"], name="P(up)", mode="lines"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="UTC time",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    # cm is 3x3 indexed by true class (rows) and pred class (cols)
    cm = np.asarray(cm, dtype="float64")
    f1s: list[float] = []
    for i in range(3):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / 3.0)


def rolling_metrics(y_true: list[int], y_pred: list[int], window: int) -> tuple[list[float], list[float]]:
    # Efficient rolling accuracy + macro-F1 via sliding 3x3 confusion counts
    # Class order: [-1, 0, +1]
    cls_to_idx = {-1: 0, 0: 1, 1: 2}
    pairs = [(cls_to_idx[int(t)], cls_to_idx[int(p)]) for t, p in zip(y_true, y_pred)]

    c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    correct = 0
    acc: list[float] = []
    f1: list[float] = []

    for i, (ti, pi) in enumerate(pairs):
        c[ti][pi] += 1
        if ti == pi:
            correct += 1

        if i >= window:
            old_ti, old_pi = pairs[i - window]
            c[old_ti][old_pi] -= 1
            if old_ti == old_pi:
                correct -= 1

        denom = min(i + 1, window)
        acc.append(correct / denom)

        f1.append(macro_f1_from_confusion(np.asarray(c, dtype="float64")))

    return acc, f1


def make_confusion_heatmap(y_true: list[int], y_pred: list[int]) -> go.Figure:
    labels = [-1, 0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"pred {l}" for l in labels],
            y=[f"true {l}" for l in labels],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Confusion matrix")
    return fig


def make_rolling_chart(ts: pd.Series, acc: list[float], f1: list[float], window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=acc, mode="lines", name=f"Rolling accuracy ({window})"))
    fig.add_trace(go.Scatter(x=ts, y=f1, mode="lines", name=f"Rolling macro-F1 ({window})"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 1]),
        xaxis_title="UTC time",
    )
    return fig


def main():
    st.set_page_config(page_title="Yurg Dashboard", layout="wide")
    st.title("Yurg — BTC/ETH 15m Dashboard")

    engine = get_engine()

    with st.sidebar:
        symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
        bars = st.slider("Bars (15m)", min_value=200, max_value=8000, value=2000, step=200)
        coin_id = load_coin_id(engine, symbol)
        versions = list_model_versions(engine, coin_id, symbol)
        auto_latest = st.checkbox("Auto-select latest model", value=True)
        if auto_latest:
            model_version = versions[0] if versions else "(none)"
            st.caption(f"Model version: `{model_version}`")
        else:
            model_version = st.selectbox(
                "Model version (predictions_15m)",
                options=(["(none)"] + versions),
                index=0,
            )
        eval_rows = st.slider(
            "Eval rows (labeled + predicted)",
            min_value=200,
            max_value=20000,
            value=5000,
            step=200,
        )
        eval_window = st.slider(
            "Rolling window",
            min_value=50,
            max_value=2000,
            value=200,
            step=50,
        )

    tab_dashboard, tab_cv = st.tabs(["Dashboard", "CV report"])

    with tab_dashboard:
        df_price = load_price(engine, coin_id, bars)
        if df_price.empty:
            st.error("No OHLCV data found. Run ingestion first.")
            st.stop()

        df_labels = load_features_labels(engine, coin_id, bars)
        df_pred = pd.DataFrame()
        df_eval = pd.DataFrame()
        if model_version != "(none)":
            df_pred = load_predictions(engine, coin_id, model_version, bars)
            df_eval = load_eval(engine, coin_id, model_version, eval_rows)

            # If DB has no predictions for this model_version yet, compute on-the-fly predictions
            # so charts/Latest/probabilities still work for the newest trained model.
            if df_pred.empty:
                try:
                    df_feat = load_feature_rows(engine, coin_id, bars).dropna(subset=FEATURE_COLUMNS)
                    if not df_feat.empty:
                        model, scaler_mean, scaler_std = load_local_model(symbol, model_version)
                        X = df_feat[FEATURE_COLUMNS].to_numpy(dtype="float64")
                        pred_class, proba = predict_on_the_fly(model, X, scaler_mean, scaler_std)
                        df_pred = pd.DataFrame(
                            {
                                "ts": df_feat["ts"].values,
                                "predicted_class": pred_class,
                                "prob_down": proba[:, 0],
                                "prob_flat": proba[:, 1],
                                "prob_up": proba[:, 2],
                            }
                        )
                except Exception:
                    # Metrics block will show the error if needed; charts can still render without predictions.
                    pass

        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.subheader("Price + targets + predictions")
            st.plotly_chart(make_price_chart(df_price, df_labels, df_pred), use_container_width=True)

        with col2:
            st.subheader("Latest")
            latest = df_price.iloc[-1]
            st.metric("Last close", f"{latest['close']:.2f}", help=str(latest["ts"]))
            st.metric("Last volume", f"{latest['volume']:.2f}")
            if not df_pred.empty:
                last_p = df_pred.iloc[-1]
                st.write(f"Model version: `{model_version}`")
                st.write(f"Predicted class: `{int(last_p['predicted_class'])}`")
                st.write(
                    {
                        "prob_down": float(last_p["prob_down"]),
                        "prob_flat": float(last_p["prob_flat"]),
                        "prob_up": float(last_p["prob_up"]),
                    }
                )

        if not df_pred.empty:
            st.subheader("Prediction probabilities")
            st.plotly_chart(make_prob_chart(df_pred), use_container_width=True)

        if model_version != "(none)":
            st.subheader("Model metrics (where target exists)")
            if df_eval.empty:
                st.info(
                    "No overlapping labeled targets and DB-stored predictions for this model yet. "
                    "Computing metrics on-the-fly from local model files (not persisted to DB)..."
                )
                try:
                    df_lab = load_labeled_features(engine, coin_id, eval_rows).dropna(
                        subset=FEATURE_COLUMNS
                    )
                    if df_lab.empty:
                        st.info("No labeled feature rows available yet.")
                        return
                    model, scaler_mean, scaler_std = load_local_model(symbol, model_version)
                    X = df_lab[FEATURE_COLUMNS].to_numpy(dtype="float64")
                    pred_class, proba = predict_on_the_fly(model, X, scaler_mean, scaler_std)
                    df_eval = pd.DataFrame(
                        {
                            "ts": df_lab["ts"].values,
                            "target_class": df_lab["target_class"].astype(int).values,
                            "predicted_class": pred_class,
                            "prob_down": proba[:, 0],
                            "prob_flat": proba[:, 1],
                            "prob_up": proba[:, 2],
                        }
                    )
                except Exception as exc:
                    st.error(f"On-the-fly evaluation failed: {exc}")
                    st.stop()

            y_true = df_eval["target_class"].astype(int).tolist()
            y_pred = df_eval["predicted_class"].astype(int).tolist()

            overall_acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
            overall_f1 = macro_f1_from_confusion(confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]))

            m1, m2, m3 = st.columns(3)
            m1.metric("Eval rows", str(len(df_eval)), help="Number of labeled rows used for evaluation.")
            m2.metric("Accuracy", f"{overall_acc:.3f}", help="Fraction of correct predictions.")
            m3.metric(
                "Macro-F1",
                f"{overall_f1:.3f}",
                help="Average F1 across classes (-1, 0, +1). Treats all classes equally.",
            )

            st.plotly_chart(make_confusion_heatmap(y_true, y_pred), use_container_width=True)

            window = min(eval_window, len(df_eval))
            acc, f1 = rolling_metrics(y_true, y_pred, window=window)
            st.plotly_chart(make_rolling_chart(df_eval["ts"], acc, f1, window), use_container_width=True)

    with tab_cv:
        st.subheader("Cross-validation report (walk-forward)")
        st.caption(
            "This tab reads per-model training metadata from `models/*.json` and shows time-series CV "
            "results. Walk-forward CV is an expanding-window validation designed for time-ordered data "
            "(no shuffling)."
        )

        metas = load_local_model_metadatas(symbol)
        if not metas:
            st.info(
                "No local model metadata found under `models/`. Train a model first "
                "(`python3 -m model.train_model ...`)."
            )
            return

        df_summary = summarize_model_metas(metas)
        if df_summary.empty:
            st.info("No readable model metadata found.")
            return

        st.markdown("**Model summary**")
        st.dataframe(
            df_summary.drop(columns=["_meta_path"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("How to read the table"):
            st.markdown(
                "- **`mlp_val.*`**: metrics on a single validation split (recent history).\n"
                "- **`mlp_test.*`**: metrics on a held-out future test split (best proxy for live).\n"
                "- **`wf.*`**: walk-forward CV results across multiple time-ordered folds "
                "(mean/std show average performance + stability).\n"
                "- **`baseline_*`**: simple baselines (useful sanity checks).\n"
                "- **Accuracy**: share of correct class predictions.\n"
                "- **Macro-F1**: average F1 over classes (-1/0/+1), so rare classes matter."
            )

        version_options = [str(m.get("model_version")) for m in metas if m.get("model_version")]
        if not version_options:
            st.info("No `model_version` found in local metadata files.")
            st.stop()
        default_version = model_version if model_version in version_options else version_options[0]
        inspect_version = st.selectbox(
            "Inspect model version",
            options=version_options,
            index=version_options.index(default_version),
            help="Shows per-fold walk-forward validation results saved during training.",
        )

        meta = next(m for m in metas if str(m.get("model_version")) == inspect_version)
        metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
        wf = metrics.get("walk_forward") if isinstance(metrics.get("walk_forward"), dict) else None

        st.markdown("**Walk-forward CV**")
        if not wf:
            st.info(
                "This model metadata does not include walk-forward CV results. "
                "Re-train with `--cv walk_forward` (default)."
            )
        else:
            wf_cfg = wf.get("config", {}) if isinstance(wf.get("config"), dict) else {}
            c1, c2, c3, c4 = st.columns(4)
            mean_acc = _coerce_float(wf.get("mean_accuracy"))
            mean_f1 = _coerce_float(wf.get("mean_macro_f1"))
            std_f1 = _coerce_float(wf.get("std_macro_f1"))
            c1.metric(
                "Folds",
                str(_coerce_int(wf_cfg.get("folds")) or len(wf.get("folds", []))),
                help="How many sequential validation splits were evaluated.",
            )
            c2.metric(
                "Mean accuracy",
                f"{mean_acc:.3f}" if mean_acc is not None else "n/a",
                help="Average validation accuracy across folds.",
            )
            c3.metric(
                "Mean macro-F1",
                f"{mean_f1:.3f}" if mean_f1 is not None else "n/a",
                help="Average validation macro-F1 across folds (better for class imbalance).",
            )
            c4.metric(
                "Stability (std macro-F1)",
                f"{std_f1:.3f}" if std_f1 is not None else "n/a",
                help="Standard deviation of macro-F1 across folds (lower = more stable).",
            )

            st.caption(
                f"Config: `min_train_frac={wf_cfg.get('min_train_frac')}`, "
                f"`val_frac={wf_cfg.get('val_frac')}`. "
                "Each fold trains on earlier history and validates on the next contiguous chunk."
            )

            df_folds = load_walk_forward_folds(meta)
            if df_folds.empty:
                st.info("No fold details found in metadata.")
            else:
                st.plotly_chart(make_cv_fold_chart(df_folds), use_container_width=True)
                st.dataframe(
                    df_folds.drop(columns=["confusion_matrix"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("What do these terms mean?"):
                    st.markdown(
                        "- **Fold**: one time-ordered train → validate split.\n"
                        "- **Train rows / Val rows**: how many labeled candles were used in that fold.\n"
                        "- **Accuracy**: share of correctly predicted classes.\n"
                        "- **Macro-F1**: average F1 for each class (-1/0/+1), so rare classes matter.\n"
                        "- **Best epoch**: epoch chosen by early stopping (best validation macro-F1)."
                    )


if __name__ == "__main__":
    main()
