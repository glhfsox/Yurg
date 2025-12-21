from __future__ import annotations
import os
from typing import Iterable
from dotenv import load_dotenv
from sqlalchemy import create_engine , text
from sqlalchemy.engine import Engine

def get_engine() -> Engine : 
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    
    if not db_url:
        raise RuntimeError("database url is not set.")
    return create_engine(db_url , pool_pre_ping=True)

def ensure_coin_ids(engine: Engine , symbols : Iterable[str]) -> dict[str, int]:
    symbols = list(symbols)
    coin_ids : dict[str , int] = {}

    with engine.begin() as conn:
        for symbol in symbols:
            conn.execute(
                text(
                    "INSERT INTO coin(symbol) VALUES (:symbol)" 
                    "ON CONFLICT (symbol) DO NOTHING"
                ),
                {"symbol" : symbol},
            )
            coin_id = conn.execute(
                text("SELECT coin_id FROM coin WHERE symbol = :symbol"),
                {"symbol": symbol},
            ).scalar_one()
            coin_ids[symbol] = int(coin_id)
        return coin_ids


def get_latest_ts(engine : Engine , * , table : str , coin_id : int):
    allowed = {"ohlcv_15m", "ohlcv_1h" , "features_15m", "predictions_15m"}
    if table not in allowed:
        raise ValueError(f"Unexpected table {table!r}")
    with engine.begin() as conn :
        return conn.execute(
            text(f"SELECT MAX(ts) FROM {table} WHERE coin_id = :coin_id"),
            {"coin_id" : coin_id},
        ).scalar()