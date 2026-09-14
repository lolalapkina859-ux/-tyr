from __future__ import annotations

import argparse
import pandas as pd

from backtest.download_binance import download_klines


def summarize(df: pd.DataFrame) -> dict:
    closed = df[df["status"] == "CLOSED"].copy()
    wins = closed[closed["highest_tp"] >= 1]
    losses = closed[closed["highest_tp"] == 0]
    return {
        "setups": int(len(df)),
        "closed": int(len(closed)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100.0 * len(wins) / len(closed), 2) if len(closed) else 0.0,
        "total_r": round(float(closed["realized_r"].sum()), 4) if len(closed) else 0.0,
        "avg_r": round(float(closed["realized_r"].mean()), 4) if len(closed) else 0.0,
    }


def confirmed_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    highs, lows = [], []
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    for i in range(left, len(df) - right):
        if h[i] >= max(h[i-left:i]) and h[i] > max(h[i+1:i+1+right]):
            highs.append((i, h[i]))
        if l[i] <= min(l[i-left:i]) and l[i] < min(l[i+1:i+1+right]):
            lows.append((i, l[i]))
    return highs, lows


def latest_equal_pool(pivots, current_i: int, lookback: int, tolerance_pct: float):
    pts = [(i, p) for i, p in pivots if current_i - lookback <= i < current_i]
    best = None
    for a in range(len(pts)):
        ia, pa = pts[a]
        for b in range(a + 1, len(pts)):
            ib, pb = pts[b]
            mid = (pa + pb) / 2.0
            if mid <= 0:
                continue
            if abs(pa - pb) / mid <= tolerance_pct:
                candidate = (max(ia, ib), mid)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best


def swept_before_signal(df: pd.DataFrame, side: str, signal_i: int, pool, sweep_window: int = 24):
    if pool is None:
        return False
    pool_i, level = pool
    start = max(pool_i + 1, signal_i - sweep_window)
    if start >= signal_i:
        return False
    part = df.iloc[start:signal_i]
    if part.empty:
        return False
    # SHORT wants equal highs taken and reclaimed; LONG mirrored.
    if side == "SHORT":
        return bool(((part["high"] > level) & (part["close"] < level)).any())
    return bool(((part["low"] < level) & (part["close"] > level)).any())


def main() -> None:
    p = argparse.ArgumentParser(description="Fast equal-high/equal-low liquidity post-filter")
    p.add_argument("--baseline", default="backtest/data/backtest_btc_ob_filtered_1_10.csv")
    p.add_argument("--start", default="2026-03-01T00:00:00+00:00")
    p.add_argument("--end", default="2026-09-01T00:00:00+00:00")
    p.add_argument("--tolerance", type=float, default=0.0010, help="0.001 = 0.10 percent")
    p.add_argument("--pool-lookback", type=int, default=192, help="15M bars, default 48h")
    p.add_argument("--sweep-window", type=int, default=24, help="15M bars, default 6h")
    p.add_argument("--out", default="backtest_btc_equal_liquidity.csv")
    args = p.parse_args()

    base = pd.read_csv(args.baseline)
    base["signal_time"] = pd.to_datetime(base["signal_time"], utc=True, errors="coerce")

    candles = download_klines("BTCUSDT", "15m", args.start, args.end)
    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    candles = candles.sort_values("time").reset_index(drop=True)

    highs, lows = confirmed_pivots(candles, 3, 3)
    rows = []

    for _, trade in base.iterrows():
        t = trade["signal_time"]
        if pd.isna(t):
            continue
        signal_i = int(candles["time"].searchsorted(t))
        if signal_i <= 10 or signal_i >= len(candles):
            continue

        side = str(trade["side"])
        pivots = highs if side == "SHORT" else lows
        pool = latest_equal_pool(pivots, signal_i, args.pool_lookback, args.tolerance)
        hit = swept_before_signal(candles, side, signal_i, pool, args.sweep_window)

        out = trade.to_dict()
        out["equal_liquidity_side"] = "EQH" if side == "SHORT" else "EQL"
        out["equal_liquidity_level"] = pool[1] if pool else None
        out["equal_liquidity_sweep"] = bool(hit)
        rows.append(out)

    annotated = pd.DataFrame(rows)
    annotated.to_csv(args.out, index=False)

    passed = annotated[annotated["equal_liquidity_sweep"] == True].copy()
    failed = annotated[annotated["equal_liquidity_sweep"] == False].copy()

    print("\n=== BASELINE OB 1.10 ===")
    print(summarize(base))
    print("\n=== WITH RECENT EQH/EQL SWEEP ===")
    print(summarize(passed))
    print("\n=== WITHOUT EQH/EQL SWEEP ===")
    print(summarize(failed))

    if not passed.empty:
        for side in ("LONG", "SHORT"):
            print("PASS", side, summarize(passed[passed["side"] == side]))
    if not failed.empty:
        for side in ("LONG", "SHORT"):
            print("FAIL", side, summarize(failed[failed["side"] == side]))

    print("results:", args.out)


if __name__ == "__main__":
    main()
