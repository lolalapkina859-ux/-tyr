from __future__ import annotations

from config import SWING_LEFT, SWING_RIGHT
from analysis.engine import Signal, analyze as base_analyze, enrich
from analysis.liquidity import (
    previous_period_levels,
    session_levels,
    pivot_levels,
    nearest_targets,
)
from analysis.targets import build_hybrid_targets
from analysis.volume_profile import build_htf_volume_profiles


def analyze(symbol, df4h, df15) -> Signal | None:
    sig = base_analyze(symbol, df4h, df15)
    if sig is None:
        return None

    h4 = enrich(df4h)
    m15 = enrich(df15)
    atr15 = float(m15.iloc[-2]["atr"])

    levels4 = (
        previous_period_levels(h4)
        | pivot_levels(h4, SWING_LEFT, SWING_RIGHT)
    )
    levels15 = (
        previous_period_levels(m15)
        | session_levels(m15)
        | pivot_levels(m15, 12, 3)
    )
    all_levels = levels4 | levels15

    liquidity_targets = nearest_targets(
        all_levels,
        float(sig.entry),
        sig.side,
        30,
    )

    vp_profiles = build_htf_volume_profiles(
        h4,
        sig.side,
    )

    targets = build_hybrid_targets(
        side=sig.side,
        entry=float(sig.entry),
        sl=float(sig.sl),
        atr15=atr15,
        df15=m15,
        df4h=h4,
        liquidity_targets=liquidity_targets,
        vp_profiles=vp_profiles,
        limit=4,
    )

    if targets:
        sig.targets = targets
        sig.reasons.append(
            "Hybrid targets: liquidity + FVG/OB + HTF VP"
        )

    return sig
