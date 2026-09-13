from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest.download import download_klines
from backtest.runner import run_backtest, summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BTCUSDT history from BingX and run Trade Vision backtest"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--warmup", type=int, default=6400)
    parser.add_argument("--out", default="backtest_btc_results.csv")
    args = parser.parse_args()

    symbol = args.symbol.upper()

    print(f"[BACKTEST] Downloading {symbol} 15M history: {args.start} -> {args.end}")
    df15 = download_klines(
        symbol=symbol,
        interval="15",
        start=args.start,
        end=args.end,
    )

    if df15.empty:
        raise SystemExit("No 15M data received from BingX")

    print(f"[BACKTEST] 15M candles: {len(df15)}")
    print("[BACKTEST] Replaying live Trade Vision logic candle-by-candle...")

    trades = run_backtest(
        symbol=symbol,
        df15=df15,
        warmup_15m=args.warmup,
    )

    summary = summarize(trades)

    out = Path(args.out)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(out, index=False)

    print("\n========== TRADE VISION BTC BACKTEST ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"results: {out.resolve()}")


if __name__ == "__main__":
    main()
