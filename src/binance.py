from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator
import requests


class BinanceError(RuntimeError):
    pass


def interval_to_milliseconds(interval : str) -> int:
    if interval.endswith("m"):
        return int(interval.removesuffix("m")) * 60_000
    if interval.endswith("h"):
        return int(interval.removesuffix("h")) * 3_600_000
    raise ValueError(f"unsupported interval : {interval!r}")

def request_json (
        session: requests.Session,
        url:str , 
        *,
        params: dict[str , Any] | None = None,
        timeout_s : float = 30.0,
        max_retries : int = 6,
        min_backoff_s : float = 1.0,
) -> Any :
    last_exc : Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url,params=params , timeout=timeout_s)
        except requests.RequestException as exc : 
            last_exc = exc 
            time.sleep(min_backoff_s * (2**attempt))
            continue
        
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in {418, 429, 500, 502, 503, 504}:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = min_backoff_s * (2**attempt)
            else:
                sleep_s = min_backoff_s * (2**attempt)
            time.sleep(min(60.0 , sleep_s))
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc : 
            raise BinanceError(str(exc)) from exc
    raise BinanceError(f"Binance request failed after {max_retries} retries: {last_exc}")


def get_server_time_ms(session : requests.Session , base_url : str) -> int :
    payload = request_json(session, f"{base_url}/api/v3/time")
    try:
        return int(payload["serverTime"])
    except Exception as exc : 
        raise BinanceError(f"Unexpected /time payload: {payload!r}") from exc


@dataclass(frozen=True)
class Kline : 
    open_time_ms: int
    close_time_ms: int
    open: str
    high: str
    low: str
    close: str
    volume: str

def iter_klines (
    session: requests.Session,
    *,
    base_url: str,
    symbol: str,
    interval: str,
    start_time_ms: int | None,
    end_time_ms: int | None = None,
    limit: int = 1000,
    sleep_s: float = 0.15,
) -> Iterator[Kline]:
    if limit <= 0 or limit > 1000 : 
        raise ValueError("Binance limit must be from 1 to 1000")
    interval_ms = interval_to_milliseconds(interval)
    next_start = start_time_ms

    while True: 
        params : dict[str , Any] = {"symbol" : symbol , "interval" : interval , "limit": limit}
        if next_start is not None:
            params["startTime"] = int (next_start)
        if end_time_ms is not None :
            params["endTime"] = int(end_time_ms)
        
        payload = request_json(session, f"{base_url}/api/v3/klines", params=params)
        if not isinstance(payload , list) : 
            raise BinanceError(f"Unexpected /klines payload : {payload!r}")
        if not payload:
            break


        for row in payload:
            yield Kline(
                open_time_ms=int(row[0]),
                open=str(row[1]),
                high=str(row[2]),
                low=str(row[3]),
                close=str(row[4]),
                volume=str(row[5]),
                close_time_ms=int(row[6]),
            )

        last_open_ms = int(payload[-1][0])
        next_start = last_open_ms + interval_ms

        if sleep_s > 0 : 
            time.sleep(sleep_s)
        


def filter_closed_klines(klines: Iterable[Kline], * , server_time_ms : int) -> list[Kline]:
    return[k for k in klines if k.close_time_ms <=server_time_ms]
