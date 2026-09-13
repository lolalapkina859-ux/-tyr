from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.engine_hybrid import analyze
from state.tracker import build_scenario_key


PENDING_EXPIRY_BARS = 48
TP_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    score: int
    signal_time: str
    entry_time: str | None
    exit_time: str | None
    entry: float
    sl: float
    tp1: float | None
    tp2: float | None
    tp3: float | None
    tp4: float | None
    highest_tp: int
    status: str
    realized_r: float
    entry_type: str
    scenario_key: str


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return (
        out.dropna()
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def _resample_4h(df15: pd.DataFrame) -> pd.DataFrame:
    indexed = df15.set_index("time")
    h4 = indexed.resample("4h", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return h4.dropna().reset_index()


def _target_prices(sig) -> list[float]:
    prices: list[float] = []
    for item in getattr(sig, "targets", []) or []:
        try:
            prices.append(float(item[0]))
        except (TypeError, ValueError, IndexError):
            continue
    return prices[:4]


def _rr(side: str, entry: float, sl: float, target: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    if side == "LONG":
        return max(0.0, (target - entry) / risk)
    return max(0.0, (entry - target) / risk)


def _realized_r(side: str, entry: float, sl: float, targets: list[float], highest_tp: int) -> float:
    if highest_tp <= 0:
        return -1.0

    total = 0.0
    for idx in range(min(highest_tp, len(targets), len(TP_WEIGHTS))):
        total += TP_WEIGHTS[idx] * _rr(side, entry, sl, targets[idx])
    return total


def _touches_entry(side: str, bar, entry: float) -> bool:
    return float(bar.low) <= entry <= float(bar.high)


def _bar_hits(side: str, bar, sl_price: float, targets: list[float]) -> tuple[bool, int]:
    high = float(bar.high)
    low = float(bar.low)

    if side == "LONG":
        sl_hit = low <= sl_price
        highest_tp = sum(1 for target in targets if high >= target)
    else:
        sl_hit = high >= sl_price
        highest_tp = sum(1 for target in targets if low <= target)

    return sl_hit, highest_tp


def run_backtest(
    symbol: str,
    df15: pd.DataFrame,
    *,
    warmup_15m: int = 6400,
    max_4h_bars: int = 400,
    max_15m_bars: int = 500,
) -> list[BacktestTrade]:
    """
    Candle-by-candle historical replay using the same live hybrid analyzer.

    The last candle in every slice is treated as the currently forming candle,
    matching the live engine's use of the previous closed candle. This avoids
    feeding future candles into signal generation.
    """
    df15 = _prepare(df15)
    if len(df15) <= warmup_15m + 2:
        raise ValueError(
            f"Need more than {warmup_15m + 2} 15m candles; received {len(df15)}"
        )

    results: list[BacktestTrade] = []
    active: dict | None = None

    for i in range(warmup_15m, len(df15)):
        live15 = df15.iloc[max(0, i - max_15m_bars + 1): i + 1].copy()
        full_context = df15.iloc[: i + 1].copy()
        live4 = _resample_4h(full_context).tail(max_4h_bars).reset_index(drop=True)

        if len(live15) < 100 or len(live4) < 100:
            continue

        bar = live15.iloc[-1]
        now = bar["time"]

        if active is not None:
            active["age"] += 1

            if active["status"] == "PENDING":
                if _touches_entry(active["side"], bar, active["entry"]):
                    active["status"] = "OPEN"
                    active["entry_time"] = now

                    sl_hit, tp_hit = _bar_hits(
                        active["side"],
                        bar,
                        active["sl"],
                        active["targets"],
                    )
                    if sl_hit or tp_hit > 0:
                        active["status"] = "AMBIGUOUS"
                        active["exit_time"] = now
                elif active["age"] >= PENDING_EXPIRY_BARS:
                    active["status"] = "EXPIRED"
                    active["exit_time"] = now

            elif active["status"] == "OPEN":
                stop = active["entry"] if active["highest_tp"] >= 1 else active["sl"]
                sl_hit, tp_hit = _bar_hits(active["side"], bar, stop, active["targets"])

                new_highest = max(active["highest_tp"], tp_hit)

                # Candle OHLC cannot tell whether SL/BE or a new TP happened first.
                if sl_hit and new_highest > active["highest_tp"]:
                    active["status"] = "AMBIGUOUS"
                    active["exit_time"] = now
                else:
                    active["highest_tp"] = new_highest

                    if active["highest_tp"] >= len(active["targets"]) and active["targets"]:
                        active["status"] = "CLOSED"
                        active["exit_time"] = now
                    elif sl_hit:
                        active["status"] = "CLOSED"
                        active["exit_time"] = now

            if active["status"] in {"CLOSED", "EXPIRED", "AMBIGUOUS"}:
                if active["status"] == "EXPIRED":
                    realized_r = 0.0
                elif active["status"] == "AMBIGUOUS":
                    realized_r = 0.0
                else:
                    realized_r = _realized_r(
                        active["side"],
                        active["entry"],
                        active["sl"],
                        active["targets"],
                        active["highest_tp"],
                    )

                padded = active["targets"] + [None] * (4 - len(active["targets"]))
                results.append(
                    BacktestTrade(
                        symbol=symbol,
                        side=active["side"],
                        score=active["score"],
                        signal_time=str(active["signal_time"]),
                        entry_time=str(active["entry_time"]) if active["entry_time"] is not None else None,
                        exit_time=str(active["exit_time"]) if active["exit_time"] is not None else None,
                        entry=active["entry"],
                        sl=active["sl"],
                        tp1=padded[0],
                        tp2=padded[1],
                        tp3=padded[2],
                        tp4=padded[3],
                        highest_tp=active["highest_tp"],
                        status=active["status"],
                        realized_r=round(realized_r, 4),
                        entry_type=active["entry_type"],
                        scenario_key=active["scenario_key"],
                    )
                )
                active = None

        if active is not None:
            continue

        sig = analyze(symbol, live4, live15, emit_watch=False)
        if sig is None:
            continue

        targets = _target_prices(sig)
        if not targets:
            continue

        reasons = getattr(sig, "reasons", []) or []
        scenario_key = build_scenario_key(symbol, sig.side, reasons)

        active = {
            "side": str(sig.side),
            "score": int(getattr(sig, "score", 0)),
            "signal_time": now,
            "entry_time": None,
            "exit_time": None,
            "entry": float(sig.entry),
            "sl": float(sig.sl),
            "targets": targets,
            "highest_tp": 0,
            "status": "PENDING",
            "age": 0,
            "entry_type": str(getattr(sig, "entry_type", "UNKNOWN")),
            "scenario_key": scenario_key,
        }

    return results


def summarize(trades: Iterable[BacktestTrade]) -> dict:
    trades = list(trades)
    filled = [t for t in trades if t.entry_time is not None and t.status != "AMBIGUOUS"]
    closed = [t for t in filled if t.status == "CLOSED"]
    wins = [t for t in closed if t.highest_tp >= 1]
    losses = [t for t in closed if t.highest_tp == 0]
    expired = [t for t in trades if t.status == "EXPIRED"]
    ambiguous = [t for t in trades if t.status == "AMBIGUOUS"]

    avg_r = sum(t.realized_r for t in closed) / len(closed) if closed else 0.0
    win_rate = 100.0 * len(wins) / len(closed) if closed else 0.0

    return {
        "total_setups": len(trades),
        "filled": len(filled),
        "closed": len(closed),
        "expired": len(expired),
        "ambiguous": len(ambiguous),
        "wins_tp1_plus": len(wins),
        "losses_before_tp1": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "avg_realized_r": round(avg_r, 4),
        "tp1": sum(t.highest_tp >= 1 for t in closed),
        "tp2": sum(t.highest_tp >= 2 for t in closed),
        "tp3": sum(t.highest_tp >= 3 for t in closed),
        "tp4": sum(t.highest_tp >= 4 for t in closed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trade Vision 24/7 historical replay")
    parser.add_argument("--symbol", required=True, help="Example: BTCUSDT")
    parser.add_argument("--csv", required=True, help="15m OHLCV CSV with time/open/high/low/close/volume")
    parser.add_argument("--out", default="backtest_results.csv")
    parser.add_argument("--warmup", type=int, default=6400, help="15m warmup candles; 6400 ~= 400 x 4H")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    trades = run_backtest(args.symbol.upper(), df, warmup_15m=args.warmup)

    out = Path(args.out)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(out, index=False)

    print("\n=== TRADE VISION BACKTEST ===")
    for key, value in summarize(trades).items():
        print(f"{key}: {value}")
    print(f"results: {out.resolve()}")


if __name__ == "__main__":
    main()
