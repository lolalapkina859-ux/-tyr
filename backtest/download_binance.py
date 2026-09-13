from __future__ import annotations

import io
import zipfile
from datetime import timedelta

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/futures/um"
COLUMNS = [
    "time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _read_zip(url: str, session: requests.Session) -> pd.DataFrame | None:
    r = session.get(url, timeout=45)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV inside {url}")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, header=None)
    if df.shape[1] < 6:
        raise RuntimeError(f"Unexpected Binance kline format: {url}")
    df = df.iloc[:, : min(df.shape[1], len(COLUMNS))]
    df.columns = COLUMNS[: df.shape[1]]
    return df[["time", "open", "high", "low", "close", "volume"]]


def _normalize(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    raw_time = pd.to_numeric(df["time"], errors="coerce")
    # Binance Spot switched to microseconds in 2025; futures archives are
    # normally milliseconds, but handle either representation defensively.
    raw_time = raw_time.where(raw_time < 10**14, raw_time // 1000)
    df = df.copy()
    df["time"] = raw_time
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df = df[(df["time"] >= start_ms) & (df["time"] < end_ms)]
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    return df


def download_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    symbol = symbol.upper()
    interval = {"15": "15m", "240": "4h"}.get(str(interval), str(interval))
    start_ts, end_ts = _utc(start), _utc(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "TradeVision24-7-Backtest"})

    # Full completed months use Binance's compact monthly archive.
    month = start_ts.to_period("M").start_time.tz_localize("UTC")
    last_month = end_ts.to_period("M").start_time.tz_localize("UTC")
    while month <= last_month:
        next_month = (month + pd.offsets.MonthBegin(1)).to_pydatetime()
        next_month = pd.Timestamp(next_month).tz_convert("UTC")
        overlap_start = max(start_ts, month)
        overlap_end = min(end_ts, next_month)
        if overlap_start >= overlap_end:
            month = next_month
            continue

        name = f"{symbol}-{interval}-{month.year:04d}-{month.month:02d}.zip"
        url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{name}"
        df = _read_zip(url, session)

        if df is not None:
            frames.append(df)
            print(f"[BINANCE] monthly {month:%Y-%m}: {len(df)} rows")
        else:
            # Recent month may not have a monthly archive yet. Fall back to
            # daily files only for this missing month.
            day = overlap_start.floor("D")
            while day < overlap_end:
                daily_name = f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
                daily_url = f"{BASE_URL}/daily/klines/{symbol}/{interval}/{daily_name}"
                daily = _read_zip(daily_url, session)
                if daily is not None:
                    frames.append(daily)
                    print(f"[BINANCE] daily {day:%Y-%m-%d}: {len(daily)} rows")
                day += timedelta(days=1)
        month = next_month

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df = _normalize(df, start_ms, end_ms)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    interval_minutes = {"15m": 15, "4h": 240}.get(interval)
    if interval_minutes:
        expected = int((end_ts - start_ts).total_seconds() // (interval_minutes * 60))
        coverage = 100.0 * len(df) / expected if expected else 0.0
        print(
            f"[BINANCE] COMPLETE {symbol} {interval}: unique={len(df)} "
            f"expected~={expected} coverage={coverage:.2f}%"
        )
        if expected >= 1000 and len(df) < expected * 0.98:
            raise RuntimeError(
                f"Binance historical coverage too low: {len(df)}/{expected} "
                f"({coverage:.2f}%)"
            )
    return df
