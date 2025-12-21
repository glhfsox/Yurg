from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from src.db import ensure_coin_ids, get_engine


logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "vol_ret_6",
    "vol_ret_24",
    "high_low_range_6",
    "close_to_high",
    "close_to_low",
    "ma_8",
    "ma_32",
    "ma_96",
    "close_ma8_ratio",
    "close_ma32_ratio",
    "ma8_ma32_ratio",
    "ma32_ma96_ratio",
    "ma8_diff_4",
    "ma32_diff_8",
    "vol_1",
    "vol_8_mean",
    "vol_32_mean",
    "vol_8_std",
    "vol_zscore_8",
    "hour_ret_1",
    "hour_ret_4",
    "hour_ret_24",
    "hour_vol_ret_24",
    "hour_vol_zscore_24",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "other_ret_1",
    "other_ret_6",
    "other_close_ma32_ratio",
    "diff_ret_1",
    "diff_close_ma32_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and upsert 15m features into Postgres.")
    parser.add_argument(
        "--theta",
        type=float,
        default=float(os.environ.get("FEATURES_THETA", "0.001")),
        help="Label threshold θ for next-bar log return.",
    )
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=6000,
        help="How many most recent 15m rows to upsert per coin.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Recompute + upsert all history.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
    )
    return parser.parse_args()


def load_ohlcv(*, engine, table: str, coin_id: int, limit: int | None) -> pd.DataFrame:
    if limit is None:
        query = """
            SELECT ts, open, high, low, close, volume
            FROM {table}
            WHERE coin_id = %(coin_id)s
            ORDER BY ts ASC
        """.format(table=table)
        params = {"coin_id": coin_id}
    else:
        query = """
            SELECT ts, open, high, low, close, volume
            FROM (
              SELECT ts, open, high, low, close, volume
              FROM {table}
              WHERE coin_id = %(coin_id)s
              ORDER BY ts DESC
              LIMIT %(limit)s
            ) t
            ORDER BY ts ASC
        """.format(table=table)
        params = {"coin_id": coin_id, "limit": int(limit)}

    df = pd.read_sql_query(query, engine, params=params)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.where(denom != 0)
    return numer / denom


def compute_hour_context(df_1h: pd.DataFrame) -> pd.DataFrame:
    out = df_1h[["ts"]].copy()

    log_close = pd.Series(np.log(df_1h["close"].to_numpy()), index=df_1h.index)
    hour_ret_1 = log_close.diff(1)
    out["hour_ret_1"] = hour_ret_1
    out["hour_ret_4"] = hour_ret_1.rolling(4).sum()
    out["hour_ret_24"] = hour_ret_1.rolling(24).sum()
    out["hour_vol_ret_24"] = hour_ret_1.rolling(24).std()

    vol_24_mean = df_1h["volume"].rolling(24).mean()
    vol_24_std = df_1h["volume"].rolling(24).std()
    out["hour_vol_zscore_24"] = safe_div(df_1h["volume"] - vol_24_mean, vol_24_std)
    return out


def compute_15m_features(df_15m: pd.DataFrame, *, theta: float) -> pd.DataFrame:
    df = df_15m.copy()

    log_close = pd.Series(np.log(df["close"].to_numpy()), index=df.index)
    df["ret_1"] = log_close.diff(1)
    df["ret_3"] = df["ret_1"].rolling(3).sum()
    df["ret_6"] = df["ret_1"].rolling(6).sum()
    df["ret_12"] = df["ret_1"].rolling(12).sum()
    df["ret_24"] = df["ret_1"].rolling(24).sum()
    df["vol_ret_6"] = df["ret_1"].rolling(6).std()
    df["vol_ret_24"] = df["ret_1"].rolling(24).std()

    roll_max_high_6 = df["high"].rolling(6).max()
    roll_min_low_6 = df["low"].rolling(6).min()
    df["high_low_range_6"] = roll_max_high_6 - roll_min_low_6

    high_low = df["high"] - df["low"]
    df["close_to_high"] = safe_div(df["close"] - df["low"], high_low)
    df["close_to_low"] = safe_div(df["high"] - df["close"], high_low)

    df["ma_8"] = df["close"].rolling(8).mean()
    df["ma_32"] = df["close"].rolling(32).mean()
    df["ma_96"] = df["close"].rolling(96).mean()

    df["close_ma8_ratio"] = safe_div(df["close"], df["ma_8"])
    df["close_ma32_ratio"] = safe_div(df["close"], df["ma_32"])
    df["ma8_ma32_ratio"] = safe_div(df["ma_8"], df["ma_32"])
    df["ma32_ma96_ratio"] = safe_div(df["ma_32"], df["ma_96"])

    df["ma8_diff_4"] = df["ma_8"] - df["ma_8"].shift(4)
    df["ma32_diff_8"] = df["ma_32"] - df["ma_32"].shift(8)

    df["vol_1"] = df["volume"]
    df["vol_8_mean"] = df["volume"].rolling(8).mean()
    df["vol_32_mean"] = df["volume"].rolling(32).mean()
    df["vol_8_std"] = df["volume"].rolling(8).std()
    df["vol_zscore_8"] = safe_div(df["vol_1"] - df["vol_8_mean"], df["vol_8_std"])

    r_next = log_close.shift(-1) - log_close
    target = np.where(r_next >= theta, 1, np.where(r_next <= -theta, -1, 0)).astype("float64")
    target = pd.Series(target, index=df.index).where(r_next.notna(), np.nan)
    df["target_class"] = target.astype("Int64")

    ts = pd.to_datetime(df["ts"], utc=True)
    hour = ts.dt.hour.astype("int16")
    df["hour_of_day"] = hour
    df["day_of_week"] = ts.dt.dayofweek.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int16")
    df["sin_hour"] = np.sin(2 * math.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * math.pi * hour / 24.0)

    return df


def with_cross_coin_features(df_main: pd.DataFrame, *, df_other: pd.DataFrame) -> pd.DataFrame:
    other = df_other[["ts", "ret_1", "ret_6", "close_ma32_ratio"]].rename(
        columns={
            "ret_1": "other_ret_1",
            "ret_6": "other_ret_6",
            "close_ma32_ratio": "other_close_ma32_ratio",
        }
    )
    merged = df_main.merge(other, on="ts", how="inner")
    merged["diff_ret_1"] = merged["ret_1"] - merged["other_ret_1"]
    merged["diff_close_ma32_ratio"] = merged["close_ma32_ratio"] - merged["other_close_ma32_ratio"]
    return merged


def df_to_rows(df: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for rec in df[columns].itertuples(index=False, name=None):
        row: list[Any] = []
        for v in rec:
            if pd.isna(v):
                row.append(None)
            elif isinstance(v, pd.Timestamp):
                row.append(v.to_pydatetime())
            elif hasattr(v, "item"):
                row.append(v.item())
            else:
                row.append(v)
        rows.append(tuple(row))
    return rows


def upsert_features(engine, df_features: pd.DataFrame) -> None:
    cols = ["coin_id", "ts", *FEATURE_COLUMNS, "target_class"]
    update_cols = [c for c in cols if c not in {"coin_id", "ts"}]
    set_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])

    sql = (
        f"INSERT INTO features_15m ({', '.join(cols)}) VALUES %s "
        "ON CONFLICT (coin_id, ts) DO UPDATE SET "
        f"{set_clause}"
    )

    rows = df_to_rows(df_features, cols)
    if not rows:
        return

    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=5000)
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    engine = get_engine()
    coin_ids = ensure_coin_ids(engine, ["BTCUSDT", "ETHUSDT"])

    max_rolling = 96
    extra_for_target = 2
    limit_15m = None if args.full else max(args.lookback_bars + max_rolling + extra_for_target, 0)

    df_15m_btc = load_ohlcv(engine=engine, table="ohlcv_15m", coin_id=coin_ids["BTCUSDT"], limit=limit_15m)
    df_15m_eth = load_ohlcv(engine=engine, table="ohlcv_15m", coin_id=coin_ids["ETHUSDT"], limit=limit_15m)
    if df_15m_btc.empty or df_15m_eth.empty:
        logger.error("Need both BTCUSDT and ETHUSDT 15m data to compute cross-coin features.")
        return 2

    df_1h_btc = load_ohlcv(engine=engine, table="ohlcv_1h", coin_id=coin_ids["BTCUSDT"], limit=None)
    df_1h_eth = load_ohlcv(engine=engine, table="ohlcv_1h", coin_id=coin_ids["ETHUSDT"], limit=None)
    if df_1h_btc.empty or df_1h_eth.empty:
        logger.error("Need both BTCUSDT and ETHUSDT 1h data to compute 1h context features.")
        return 2

    df_btc = compute_15m_features(df_15m_btc, theta=args.theta)
    df_eth = compute_15m_features(df_15m_eth, theta=args.theta)

    btc_hour = compute_hour_context(df_1h_btc)
    df_btc = pd.merge_asof(
        df_btc.sort_values("ts"),
        btc_hour.sort_values("ts"),
        on="ts",
        direction="backward",
    )

    eth_hour = compute_hour_context(df_1h_eth)
    df_eth = pd.merge_asof(
        df_eth.sort_values("ts"),
        eth_hour.sort_values("ts"),
        on="ts",
        direction="backward",
    )

    df_btc = with_cross_coin_features(df_btc, df_other=df_eth)
    df_eth = with_cross_coin_features(df_eth, df_other=df_btc)

    df_btc.insert(0, "coin_id", coin_ids["BTCUSDT"])
    df_eth.insert(0, "coin_id", coin_ids["ETHUSDT"])

    df_btc = df_btc.dropna(subset=FEATURE_COLUMNS)
    df_eth = df_eth.dropna(subset=FEATURE_COLUMNS)

    if not args.full:
        df_btc = df_btc.tail(args.lookback_bars)
        df_eth = df_eth.tail(args.lookback_bars)

    upsert_features(engine, df_btc)
    upsert_features(engine, df_eth)

    logger.info("Upserted features: BTC=%s rows, ETH=%s rows", len(df_btc), len(df_eth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
