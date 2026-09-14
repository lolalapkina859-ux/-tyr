from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest.download_binance import download_klines
from backtest.ob_quality import measure_ob_quality, passes_ob_quality
from backtest.runner import run_backtest, summarize


def _slice_at_signal(df15: pd.DataFrame, signal_time: str, max_15m_bars: int = 500) -> pd.DataFrame:
    ts = pd.to_datetime(signal_time, utc=True)
    times = pd.to_datetime(df15["time"], utc=True)
    matches = times[times <= ts]
    if matches.empty:
        return pd.DataFrame()
    idx = int(matches.index[-1])
    return df15.iloc[max(0, idx - max_15m_bars + 1): idx + 1].copy().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental BTC backtest: Trade Vision + SMC-style OB volume quality filter"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2026-03-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-09-01T00:00:00+00:00")
    parser.add_argument("--warmup", type=int, default=3200)
    parser.add_argument("--min-ob-volume-ratio", type=float, default=1.20)
    parser.add_argument("--min-ob-balance", type=float, default=0.0)
    parser.add_argument("--out", default="backtest_btc_ob_quality.csv")
    parser.add_argument("--filtered-out", default="backtest_btc_ob_filtered.csv")
    args = parser.parse_args()

    symbol = args.symbol.upper()

    print(
        f"[OB TEST] Downloading Binance USD-M {symbol} 15M history: "
        f"{args.start} -> {args.end}"
    )
    df15 = download_klines(
        symbol=symbol,
        interval="15",
        start=args.start,
        end=args.end,
    )

    if df15.empty:
        raise SystemExit("No 15M data received from Binance public archives")

    effective_warmup = min(args.warmup, max(1600, len(df15) - 1000))
    if len(df15) <= effective_warmup + 2:
        raise SystemExit(
            f"Not enough history to backtest: {len(df15)} candles, "
            f"effective warmup {effective_warmup}"
        )

    print(f"[OB TEST] 15M candles: {len(df15)}")
    print(f"[OB TEST] Effective warmup: {effective_warmup}")
    print("[OB TEST] Running unchanged baseline Trade Vision replay...")

    trades = run_backtest(
        symbol=symbol,
        df15=df15,
        warmup_15m=effective_warmup,
    )

    rows: list[dict] = []
    filtered_trades = []

    ob_total = 0
    ob_passed = 0

    for trade in trades:
        row = asdict(trade)
        row["ob_volume"] = None
        row["ob_volume_ratio"] = None
        row["ob_balance_pct"] = None
        row["ob_quality_pass"] = True

        # Test only the weak historical branch: pure ORDER BLOCK 50% entries.
        # FVG + ORDER BLOCK confluence remains untouched.
        is_pure_ob = "ORDER BLOCK 50%" in str(trade.entry_type).upper()

        if is_pure_ob:
            ob_total += 1
            live15 = _slice_at_signal(df15, trade.signal_time)
            metrics = measure_ob_quality(live15, trade.side) if not live15.empty else None
            passed = passes_ob_quality(
                metrics,
                min_volume_ratio=args.min_ob_volume_ratio,
                min_balance_pct=args.min_ob_balance,
            )

            row["ob_quality_pass"] = passed
            if metrics is not None:
                row["ob_volume"] = round(float(metrics["ob_volume"]), 8)
                row["ob_volume_ratio"] = round(float(metrics["volume_ratio"]), 4)
                row["ob_balance_pct"] = round(float(metrics["balance_pct"]), 2)

            if passed:
                ob_passed += 1
                filtered_trades.append(trade)
        else:
            filtered_trades.append(trade)

        rows.append(row)

    out = Path(args.out)
    filtered_out = Path(args.filtered_out)

    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame([asdict(t) for t in filtered_trades]).to_csv(filtered_out, index=False)

    print("\n========== BASELINE ==========")
    for key, value in summarize(trades).items():
        print(f"{key}: {value}")

    print("\n========== OB QUALITY FILTER ==========")
    print(f"min_ob_volume_ratio: {args.min_ob_volume_ratio}")
    print(f"min_ob_balance_pct: {args.min_ob_balance}")
    print(f"pure_ob_setups: {ob_total}")
    print(f"pure_ob_passed: {ob_passed}")
    print(f"pure_ob_rejected: {ob_total - ob_passed}")
    for key, value in summarize(filtered_trades).items():
        print(f"{key}: {value}")

    print(f"diagnostics: {out.resolve()}")
    print(f"filtered_results: {filtered_out.resolve()}")
    print("\nNOTE: filtered summary is a same-signal-set experiment. Rejected OB trades are removed; the runner does not search for replacement signals while those baseline trades were active.")


if __name__ == "__main__":
    main()
