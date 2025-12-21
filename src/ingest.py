from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from typing import Iterable

import requests
from psycopg2.extras import execute_values

from src.binance import filter_closed_klines, get_server_time_ms, iter_klines
from src.db import ensure_coin_ids, get_engine, get_latest_ts


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Binance OHLCV into Postgres.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"], choices=["15m", "1h"])
    parser.add_argument(
        "--lookback-days-15m",
        type=int,
        default=int(os.environ.get("INGEST_LOOKBACK_DAYS_15M", "730")),
    )
    parser.add_argument(
        "--lookback-days-1h",
        type=int,
        default=int(os.environ.get("INGEST_LOOKBACK_DAYS_1H", "730")),
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--base-url", default=os.environ.get("BINANCE_BASE_URL", "https://api.binance.com"))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("Expected timezone-aware datetime")
    return int(round(dt.timestamp() * 1000))


def insert_ohlcv_rows(*, engine, table: str, rows: list[tuple]) -> None:
    if not rows:
        return
    sql = (
        f"INSERT INTO {table} (coin_id, ts, open, high, low, close, volume) "
        "VALUES %s "
        "ON CONFLICT (coin_id, ts) DO NOTHING"
    )
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    finally:
        conn.close()


def iter_closed_rows(
    *,
    session: requests.Session,
    base_url: str,
    symbol: str,
    interval: str,
    start_time_ms: int,
    server_time_ms: int,
    limit: int,
    sleep_s: float,
) -> list[tuple]:
    klines = iter_klines(
        session,
        base_url=base_url,
        symbol=symbol,
        interval=interval,
        start_time_ms=start_time_ms,
        limit=limit,
        sleep_s=sleep_s,
    )
    closed = filter_closed_klines(klines, server_time_ms=server_time_ms)

    rows: list[tuple] = []
    for k in closed:
        ts = datetime.fromtimestamp(k.close_time_ms / 1000.0, tz=timezone.utc)
        rows.append((ts, k.open, k.high, k.low, k.close, k.volume))
    return rows


def chunked(items: list[tuple], chunk_size: int) -> Iterable[list[tuple]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    engine = get_engine()
    coin_ids = ensure_coin_ids(engine, args.symbols)

    session = requests.Session()
    server_time_ms = get_server_time_ms(session, args.base_url)
    server_dt = datetime.fromtimestamp(server_time_ms / 1000.0, tz=timezone.utc)
    logger.info("Binance server time: %s", server_dt.isoformat())

    timeframe_cfg = {
        "15m": ("ohlcv_15m", args.lookback_days_15m),
        "1h": ("ohlcv_1h", args.lookback_days_1h),
    }

    for interval in args.timeframes:
        table, lookback_days = timeframe_cfg[interval]
        lookback_ms = int(lookback_days * 24 * 60 * 60 * 1000)

        for symbol in args.symbols:
            coin_id = coin_ids[symbol]
            last_ts = get_latest_ts(engine, table=table, coin_id=coin_id)

            if last_ts is None:
                start_ms = server_time_ms - lookback_ms
                logger.info(
                    "%s %s: no data, backfilling lookback_days=%s",
                    symbol,
                    interval,
                    lookback_days,
                )
            else:
                start_ms = epoch_ms(last_ts) + 1
                logger.info("%s %s: incremental from %s", symbol, interval, last_ts.isoformat())

            rows_no_coin = iter_closed_rows(
                session=session,
                base_url=args.base_url,
                symbol=symbol,
                interval=interval,
                start_time_ms=start_ms,
                server_time_ms=server_time_ms,
                limit=args.limit,
                sleep_s=args.sleep,
            )

            if not rows_no_coin:
                logger.info("%s %s: no new closed candles", symbol, interval)
                continue

            rows = [(coin_id, *r) for r in rows_no_coin]
            for chunk in chunked(rows, 10_000):
                insert_ohlcv_rows(engine=engine, table=table, rows=chunk)

            logger.info("%s %s: ingested %s rows", symbol, interval, len(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
