from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from sqlalchemy import create_engine


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


def list_model_versions(engine, coin_id: int) -> list[str]:
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
    return [str(x) for x in df["model_version"].tolist()]


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


def macro_f1_from_confusion(conf: pd.DataFrame) -> float:
    # conf is 3x3 indexed by true class (rows) and pred class (cols)
    f1s: list[float] = []
    for i in range(3):
        tp = float(conf.iat[i, i])
        fp = float(conf.iloc[:, i].sum() - tp)
        fn = float(conf.iloc[i, :].sum() - tp)
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

        conf = pd.DataFrame(c)
        f1.append(macro_f1_from_confusion(conf))

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
        versions = list_model_versions(engine, coin_id)
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
            st.info("No overlapping labeled targets and predictions yet. Wait for future candles to become labeled.")
        else:
            y_true = df_eval["target_class"].astype(int).tolist()
            y_pred = df_eval["predicted_class"].astype(int).tolist()

            overall_acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
            overall_f1 = macro_f1_from_confusion(
                pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]))
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Eval rows", str(len(df_eval)))
            m2.metric("Accuracy", f"{overall_acc:.3f}")
            m3.metric("Macro-F1", f"{overall_f1:.3f}")

            st.plotly_chart(make_confusion_heatmap(y_true, y_pred), use_container_width=True)

            window = min(eval_window, len(df_eval))
            acc, f1 = rolling_metrics(y_true, y_pred, window=window)
            st.plotly_chart(make_rolling_chart(df_eval["ts"], acc, f1, window), use_container_width=True)


if __name__ == "__main__":
    main()
