from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from backtest.download import download_klines_range
from backtest.runner import run_backtest


def _parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trade Vision historical backtest for BTCUSDT")
    parser.add_argument("--start", default="2026-06-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-09-01T00:00:00+00:00")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    print(f"[BACKTEST] downloading {args.symbol} 4H...")
    df4h = download_klines_range(args.symbol, "240", start, end)
    print(f"[BACKTEST] downloading {args.symbol} 15M...")
    df15 = download_klines_range(args.symbol, "15", start, end)

    if df4h is None or df4h.empty:
        raise SystemExit("No 4H data")
    if df15 is None or df15.empty:
        raise SystemExit("No 15M data")

    print(f"[BACKTEST] 4H candles: {len(df4h)}")
    print(f"[BACKTEST] 15M candles: {len(df15)}")

    result = run_backtest(
        symbol=args.symbol,
        df4h=df4h,
        df15=df15,
    )

    print("\n========== TRADE VISION BTC BACKTEST ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
