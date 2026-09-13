from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from exchange.bybit import BINGX_BASE_URL, INTERVAL_MAP, _format_symbol


MAX_LIMIT = 1440


def _interval_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "2h": 2 * 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "6h": 6 * 60 * 60_000,
        "8h": 8 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval for historical downloader: {interval}")
    return mapping[interval]


def download_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    pause_seconds: float = 1.05,
) -> pd.DataFrame:
    bingx_symbol = _format_symbol(symbol)
    bingx_interval = INTERVAL_MAP.get(str(interval), str(interval))
    step_ms = _interval_ms(bingx_interval)

    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")

    cursor = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    url = f"{BINGX_BASE_URL}/openApi/swap/v3/quote/klines"
    headers = {"User-Agent": "TradeVision24-7-Backtest"}
    rows_out: list[list] = []

    while cursor < end_ms:
        page_end = min(end_ms, cursor + step_ms * MAX_LIMIT)
        params = {
            "symbol": bingx_symbol,
            "interval": bingx_interval,
            "startTime": cursor,
            "endTime": page_end,
            "limit": MAX_LIMIT,
        }

        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") not in (0, "0", None):
            raise RuntimeError(f"BingX error {payload.get('code')}: {payload.get('msg')}")

        rows = payload.get("data", []) or []
        if not rows:
            cursor = page_end
            time.sleep(pause_seconds)
            continue

        last_open = None
        for row in rows:
            if isinstance(row, list) and len(row) >= 6:
                open_time = int(row[0])
                rows_out.append([open_time, row[1], row[2], row[3], row[4], row[5]])
                last_open = max(last_open or open_time, open_time)
            elif isinstance(row, dict):
                open_time = int(row.get("time") or row.get("openTime"))
                rows_out.append([
                    open_time,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                ])
                last_open = max(last_open or open_time, open_time)

        if last_open is None:
            cursor = page_end
        else:
            cursor = max(cursor + step_ms, last_open + step_ms)

        print(f"[BACKTEST DOWNLOAD] {symbol} {bingx_interval}: {len(rows_out)} candles")
        time.sleep(pause_seconds)

    df = pd.DataFrame(rows_out, columns=["time", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df

    df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BingX OHLCV history for Trade Vision backtests")
    parser.add_argument("--symbol", required=True, help="Example: BTCUSDT")
    parser.add_argument("--start", required=True, help="UTC date/time, example: 2026-01-01")
    parser.add_argument("--end", required=True, help="UTC date/time, example: 2026-09-01")
    parser.add_argument("--interval", default="15", help="Default: 15 (15m)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    interval_name = INTERVAL_MAP.get(str(args.interval), str(args.interval))
    out = Path(args.out or f"data_{args.symbol.upper()}_{interval_name}.csv")

    df = download_klines(args.symbol.upper(), args.interval, args.start, args.end)
    df.to_csv(out, index=False)
    print(f"saved: {out.resolve()} ({len(df)} candles)")


if __name__ == "__main__":
    main()
