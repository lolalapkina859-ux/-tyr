from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from exchange.bybit import BINGX_BASE_URL, INTERVAL_MAP, _format_symbol


# BingX historical kline endpoint can return fewer rows than the requested
# limit. Use small deterministic time windows so every part of the requested
# range is queried instead of advancing from the last row returned by BingX.
PAGE_BARS = 900


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


def _utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def download_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    pause_seconds: float = 0.35,
) -> pd.DataFrame:
    bingx_symbol = _format_symbol(symbol)
    bingx_interval = INTERVAL_MAP.get(str(interval), str(interval))
    step_ms = _interval_ms(bingx_interval)

    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    window_ms = PAGE_BARS * step_ms

    url = f"{BINGX_BASE_URL}/openApi/swap/v3/quote/klines"
    headers = {"User-Agent": "TradeVision24-7-Backtest"}
    rows_out: list[list] = []

    # Query every fixed time window. Crucially, advance to page_end rather
    # than to BingX's last returned candle. This prevents a short/truncated
    # response from silently skipping the rest of the requested history.
    cursor = start_ms
    page = 0
    while cursor < end_ms:
        page += 1
        page_end = min(end_ms, cursor + window_ms)
        params = {
            "symbol": bingx_symbol,
            "interval": bingx_interval,
            "startTime": cursor,
            "endTime": page_end - 1,
            "limit": PAGE_BARS,
        }

        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") not in (0, "0", None):
            raise RuntimeError(f"BingX error {payload.get('code')}: {payload.get('msg')}")

        rows = payload.get("data", []) or []
        accepted = 0
        for row in rows:
            if isinstance(row, list) and len(row) >= 6:
                open_time = int(row[0])
                values = [open_time, row[1], row[2], row[3], row[4], row[5]]
            elif isinstance(row, dict):
                raw_time = row.get("time") or row.get("openTime")
                if raw_time is None:
                    continue
                open_time = int(raw_time)
                values = [
                    open_time,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                ]
            else:
                continue

            # BingX responses are not assumed to be perfectly bounded or
            # ordered; keep only candles belonging to this requested window.
            if cursor <= open_time < page_end and start_ms <= open_time < end_ms:
                rows_out.append(values)
                accepted += 1

        print(
            f"[BACKTEST DOWNLOAD] page={page} "
            f"{pd.to_datetime(cursor, unit='ms', utc=True)} -> "
            f"{pd.to_datetime(page_end, unit='ms', utc=True)} | "
            f"received={len(rows)} accepted={accepted} total_raw={len(rows_out)}"
        )

        cursor = page_end
        if cursor < end_ms:
            time.sleep(pause_seconds)

    df = pd.DataFrame(
        rows_out,
        columns=["time", "open", "high", "low", "close", "volume"],
    )
    if df.empty:
        return df

    df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    expected = max(0, (end_ms - start_ms) // step_ms)
    coverage = (100.0 * len(df) / expected) if expected else 0.0
    print(
        f"[BACKTEST DOWNLOAD] COMPLETE {symbol} {bingx_interval}: "
        f"unique={len(df)} expected~={expected} coverage={coverage:.2f}%"
    )

    # A backtest over heavily incomplete history is misleading. Fail loudly
    # rather than silently producing statistics from a small fragment.
    if expected >= 1000 and len(df) < expected * 0.90:
        raise RuntimeError(
            f"Historical coverage too low: got {len(df)} of ~{expected} "
            f"{bingx_interval} candles ({coverage:.2f}%)."
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BingX OHLCV history for Trade Vision backtests"
    )
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
