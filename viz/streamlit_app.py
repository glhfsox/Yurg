from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
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
        SELECT DISTINCT model_version
        FROM predictions_15m
        WHERE coin_id = %(coin_id)s
        ORDER BY model_version DESC
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


def main():
    st.set_page_config(page_title="Yurg Dashboard", layout="wide")
    st.title("Yurg — BTC/ETH 15m Dashboard")

    engine = get_engine()

    with st.sidebar:
        symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
        bars = st.slider("Bars (15m)", min_value=200, max_value=8000, value=2000, step=200)
        coin_id = load_coin_id(engine, symbol)
        versions = list_model_versions(engine, coin_id)
        model_version = st.selectbox(
            "Model version (predictions_15m)",
            options=(["(none)"] + versions),
            index=0,
        )

    df_price = load_price(engine, coin_id, bars)
    if df_price.empty:
        st.error("No OHLCV data found. Run ingestion first.")
        st.stop()

    df_labels = load_features_labels(engine, coin_id, bars)
    df_pred = pd.DataFrame()
    if model_version != "(none)":
        df_pred = load_predictions(engine, coin_id, model_version, bars)

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


if __name__ == "__main__":
    main()

