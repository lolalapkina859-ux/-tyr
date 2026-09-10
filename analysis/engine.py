from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from config import (
    ATR_LENGTH, WT_CHANNEL, WT_AVG, WT_SIGNAL, WT_EXTREME,
    MF_SMOOTH_1, MF_SMOOTH_2, MF_WEIGHT, MF_SCALE,
    SWING_LEFT, SWING_RIGHT
)
from indicators.core import atr, wavetrend, money_flow
from analysis.liquidity import (
    previous_period_levels, session_levels, pivot_levels,
    detect_event, nearest_targets
)
from analysis.structure import local_structure_shift

@dataclass
class Signal:
    symbol: str
    side: str
    score: int
    entry: float
    sl: float
    targets: list[tuple[float, str]]
    reasons: list[str]
    invalidation: str

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, ATR_LENGTH)
    wt = wavetrend(out, WT_CHANNEL, WT_AVG, WT_SIGNAL)
    out["wt1"] = wt["wt1"]
    out["wt2"] = wt["wt2"]
    out["wt_cross_up"] = wt["cross_up"]
    out["wt_cross_down"] = wt["cross_down"]
    out["mf"] = money_flow(out, MF_SMOOTH_1, MF_SMOOTH_2, MF_WEIGHT, MF_SCALE)
    return out

def _latest_closed(df: pd.DataFrame):
    return df.iloc[-2]

def analyze(symbol: str, df4h: pd.DataFrame, df15: pd.DataFrame) -> Signal | None:
    h4 = enrich(df4h)
    m15 = enrich(df15)

    levels4 = previous_period_levels(h4) | pivot_levels(h4, SWING_LEFT, SWING_RIGHT)
    levels15 = previous_period_levels(m15) | session_levels(m15) | pivot_levels(m15, 12, 3)

    ev4 = detect_event(h4, levels4, lookback=2)
    ev15 = detect_event(m15, levels15, lookback=3)

    last4 = _latest_closed(h4)
    last15 = _latest_closed(m15)
    struct = local_structure_shift(m15)

    low_sweeps4 = [e for e in ev4 if e["type"] == "sweep_low"]
    high_sweeps4 = [e for e in ev4 if e["type"] == "sweep_high"]
    low_sweeps15 = [e for e in ev15 if e["type"] == "sweep_low"]
    high_sweeps15 = [e for e in ev15 if e["type"] == "sweep_high"]

    long_score = 0
    short_score = 0
    long_reasons, short_reasons = [], []

    if low_sweeps4:
        long_score += 25
        e = low_sweeps4[-1]
        long_reasons.append(f"4H liquidity sweep: {e['level']} {e['price']:.8g}")
    if high_sweeps4:
        short_score += 25
        e = high_sweeps4[-1]
        short_reasons.append(f"4H liquidity sweep: {e['level']} {e['price']:.8g}")

    if last4["wt2"] <= -WT_EXTREME:
        long_score += 10
        long_reasons.append(f"4H WaveTrend oversold ({last4['wt2']:.1f})")
    if last4["wt2"] >= WT_EXTREME:
        short_score += 10
        short_reasons.append(f"4H WaveTrend overbought ({last4['wt2']:.1f})")

    if bool(last4["wt_cross_up"]):
        long_score += 10
        long_reasons.append("4H WaveTrend bullish cross")
    if bool(last4["wt_cross_down"]):
        short_score += 10
        short_reasons.append("4H WaveTrend bearish cross")

    # Money flow direction rather than just sign.
    if h4["mf"].iloc[-2] > h4["mf"].iloc[-3]:
        long_score += 10
        long_reasons.append("4H Money Flow rising")
    if h4["mf"].iloc[-2] < h4["mf"].iloc[-3]:
        short_score += 10
        short_reasons.append("4H Money Flow falling")

    if low_sweeps15:
        long_score += 15
        e = low_sweeps15[-1]
        long_reasons.append(f"15M liquidity sweep: {e['level']} {e['price']:.8g}")
    if high_sweeps15:
        short_score += 15
        e = high_sweeps15[-1]
        short_reasons.append(f"15M liquidity sweep: {e['level']} {e['price']:.8g}")

    if bool(last15["wt_cross_up"]):
        long_score += 10
        long_reasons.append("15M WaveTrend bullish cross")
    if bool(last15["wt_cross_down"]):
        short_score += 10
        short_reasons.append("15M WaveTrend bearish cross")

    if m15["mf"].iloc[-2] > m15["mf"].iloc[-3]:
        long_score += 10
        long_reasons.append("15M Money Flow rising")
    if m15["mf"].iloc[-2] < m15["mf"].iloc[-3]:
        short_score += 10
        short_reasons.append("15M Money Flow falling")

    if struct.get("bullish"):
        long_score += 10
        long_reasons.append("15M bullish structure shift")
    if struct.get("bearish"):
        short_score += 10
        short_reasons.append("15M bearish structure shift")

    side = "LONG" if long_score >= short_score else "SHORT"
    score = max(long_score, short_score)
    reasons = long_reasons if side == "LONG" else short_reasons

    # Don't signal without an actual liquidity event.
    if side == "LONG" and not (low_sweeps4 or low_sweeps15):
        return None
    if side == "SHORT" and not (high_sweeps4 or high_sweeps15):
        return None

    current = float(last15["close"])
    atr15 = float(last15["atr"])

    if side == "LONG":
        source_event = (low_sweeps15 or low_sweeps4)[-1]
        reclaim = float(source_event["price"])
        entry = min(current, reclaim + 0.10 * atr15)
        recent_low = float(m15.iloc[-8:-1]["low"].min())
        sl = min(recent_low, reclaim) - 0.25 * atr15
        invalidation = f"15M close below {sl:.8g}"
    else:
        source_event = (high_sweeps15 or high_sweeps4)[-1]
        reclaim = float(source_event["price"])
        entry = max(current, reclaim - 0.10 * atr15)
        recent_high = float(m15.iloc[-8:-1]["high"].max())
        sl = max(recent_high, reclaim) + 0.25 * atr15
        invalidation = f"15M close above {sl:.8g}"

    all_levels = levels4 | levels15
    targets = nearest_targets(all_levels, entry, side, 4)

    # At least one structural target is required.
    if not targets:
        return None

    return Signal(
        symbol=symbol,
        side=side,
        score=min(score, 100),
        entry=entry,
        sl=sl,
        targets=targets,
        reasons=reasons,
        invalidation=invalidation,
    )
