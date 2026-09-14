from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest.download_binance import download_klines

TP_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def rr(side: str, entry: float, sl: float, target: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    if side == "LONG":
        return max(0.0, (target - entry) / risk)
    return max(0.0, (entry - target) / risk)


def realized_r(side: str, entry: float, sl: float, targets: list[float], highest_tp: int) -> float:
    if highest_tp <= 0:
        return -1.0
    total = 0.0
    for idx in range(min(highest_tp, len(targets), len(TP_WEIGHTS))):
        total += TP_WEIGHTS[idx] * rr(side, entry, sl, targets[idx])
    return total


def bar_hits(side: str, row, stop: float, targets: list[float]) -> tuple[bool, int]:
    high = float(row["high"])
    low = float(row["low"])
    if side == "LONG":
        sl_hit = low <= stop
        tp_hit = sum(1 for t in targets if high >= t)
    else:
        sl_hit = high >= stop
        tp_hit = sum(1 for t in targets if low <= t)
    return sl_hit, tp_hit


def summarize(df: pd.DataFrame) -> dict:
    closed = df[df["status"] == "CLOSED"].copy()
    wins = closed[closed["highest_tp"] >= 1]
    losses = closed[closed["highest_tp"] == 0]
    return {
        "closed": int(len(closed)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100.0 * len(wins) / len(closed), 2) if len(closed) else 0.0,
        "total_r": round(float(closed["realized_r"].sum()), 4) if len(closed) else 0.0,
        "avg_r": round(float(closed["realized_r"].mean()), 4) if len(closed) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast LIMIT vs MARKET comparison using saved signal rows")
    parser.add_argument("--baseline", default="backtest_btc_ob_filtered_1_10.csv")
    parser.add_argument("--start", default="2026-03-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-09-01T00:00:00+00:00")
    parser.add_argument("--out", default="backtest_btc_market_fast.csv")
    args = parser.parse_args()

    base = pd.read_csv(args.baseline)
    if base.empty:
        raise SystemExit("Baseline CSV is empty")

    for col in ("signal_time", "entry_time", "exit_time"):
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")

    print(f"[FAST MARKET] Loading Binance 15M: {args.start} -> {args.end}")
    candles = download_klines("BTCUSDT", "15", args.start, args.end)
    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    candles = candles.sort_values("time").reset_index(drop=True)

    rows = []

    for _, trade in base.iterrows():
        signal_time = trade.get("signal_time")
        if pd.isna(signal_time):
            continue

        pos = candles["time"].searchsorted(signal_time)
        if pos >= len(candles):
            continue

        signal_bar = candles.iloc[pos]
        market_entry = float(signal_bar["close"])
        sl = float(trade["sl"])
        side = str(trade["side"])

        targets = []
        for name in ("tp1", "tp2", "tp3", "tp4"):
            value = trade.get(name)
            if pd.notna(value):
                targets.append(float(value))

        if not targets:
            continue

        status = "OPEN"
        highest_tp = 0
        exit_time = None

        # Start from the NEXT 15m candle to avoid ambiguous pre-signal movement
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
        out["entry_mode"] = "MARKET_AT_SIGNAL_CLOSE"
        rows.append(out)

    market = pd.DataFrame(rows)
    market.to_csv(args.out, index=False)

    print("\n=== LIMIT BASELINE ===")
    print(summarize(base))
    print("\n=== MARKET AT SIGNAL CLOSE ===")
    print(summarize(market))
    print(f"results: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
