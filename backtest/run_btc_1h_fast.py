from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import backtest.runner as runner
from backtest.download_binance import download_klines
from backtest.run_btc_market_fast import bar_hits, realized_r, summarize as summarize_market

START = "2026-03-01T00:00:00+00:00"
END = "2026-09-01T00:00:00+00:00"
SYMBOL = "BTCUSDT"


def run_market_from_same_signals(base: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    candles = candles.copy()
    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    candles = candles.sort_values("time").reset_index(drop=True)

    base = base.copy()
    base["signal_time"] = pd.to_datetime(base["signal_time"], utc=True, errors="coerce")

    rows: list[dict] = []
    for _, trade in base.iterrows():
        signal_time = trade.get("signal_time")
        if pd.isna(signal_time):
            continue

        pos = candles["time"].searchsorted(signal_time)
        if pos >= len(candles):
            continue

        side = str(trade["side"])
        signal_bar = candles.iloc[pos]
        market_entry = float(signal_bar["close"])
        sl = float(trade["sl"])
        targets = [float(trade[c]) for c in ("tp1", "tp2", "tp3", "tp4") if pd.notna(trade.get(c))]
        if not targets:
            continue

        highest_tp = 0
        status = "OPEN"
        exit_time = None

        # Start on next 1H candle so signal candle movement is not reused.
        for j in range(pos + 1, len(candles)):
            row = candles.iloc[j]
            stop = market_entry if highest_tp >= 1 else sl
            sl_hit, tp_hit = bar_hits(side, row, stop, targets)
            new_highest = max(highest_tp, tp_hit)

            if sl_hit and new_highest > highest_tp:
                status = "AMBIGUOUS"
                exit_time = row["time"]
                break

            highest_tp = new_highest
            if highest_tp >= len(targets):
                status = "CLOSED"
                exit_time = row["time"]
                break
            if sl_hit:
                status = "CLOSED"
                exit_time = row["time"]
                break

        if status == "OPEN":
            continue

        r = 0.0 if status == "AMBIGUOUS" else realized_r(side, market_entry, sl, targets, highest_tp)
        out = trade.to_dict()
        out["limit_entry_original"] = float(trade["entry"])
        out["entry"] = market_entry
        out["entry_time"] = signal_time
        out["exit_time"] = exit_time
        out["highest_tp"] = highest_tp
        out["status"] = status
        out["realized_r"] = round(r, 4)
        out["entry_mode"] = "MARKET_AT_1H_SIGNAL_CLOSE"
        rows.append(out)

    return pd.DataFrame(rows)


def main() -> None:
    print(f"[1H FAST] Downloading Binance {SYMBOL} 1H: {START} -> {END}")
    df1h = download_klines(SYMBOL, "1h", START, END)
    if df1h.empty:
        raise SystemExit("No 1H data downloaded")

    # Original 15M setup expires after 48 bars = 12 hours.
    # On 1H we keep the same real-time expiry using 12 bars.
    runner.PENDING_EXPIRY_BARS = 12

    print(f"[1H FAST] candles: {len(df1h)}")
    print("[1H FAST] Running 4H context -> 1H confirmation replay...")
    trades = runner.run_backtest(
        symbol=SYMBOL,
        df15=df1h,
        warmup_15m=400,       # 400 x 1H = 100 x 4H bars
        max_4h_bars=400,
        max_15m_bars=500,     # here this is the 1H execution/confirmation frame
    )

    limit_df = pd.DataFrame([asdict(t) for t in trades])
    limit_path = Path("backtest_btc_1h_limit.csv")
    limit_df.to_csv(limit_path, index=False)

    market_df = run_market_from_same_signals(limit_df, df1h)
    market_path = Path("backtest_btc_1h_market.csv")
    market_df.to_csv(market_path, index=False)

    print("\n=== BTC 1H LIMIT ===")
    print(summarize_market(limit_df))
    print("\n=== BTC 1H MARKET ===")
    print(summarize_market(market_df))
    print(f"LIMIT CSV: {limit_path.resolve()}")
    print(f"MARKET CSV: {market_path.resolve()}")


if __name__ == "__main__":
    main()
