from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from config import (
    ATR_LENGTH,
    WT_CHANNEL,
    WT_AVG,
    WT_SIGNAL,
    WT_EXTREME,
    MF_SMOOTH_1,
    MF_SMOOTH_2,
    MF_WEIGHT,
    MF_SCALE,
    SWING_LEFT,
    SWING_RIGHT,
)

from indicators.core import atr, wavetrend, money_flow

from analysis.liquidity import (
    previous_period_levels,
    session_levels,
    pivot_levels,
    detect_event,
    nearest_targets,
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

    wt = wavetrend(
        out,
        WT_CHANNEL,
        WT_AVG,
        WT_SIGNAL,
    )

    out["wt1"] = wt["wt1"]
    out["wt2"] = wt["wt2"]

    out["wt_cross_up"] = wt["cross_up"]
    out["wt_cross_down"] = wt["cross_down"]

    out["mf"] = money_flow(
        out,
        MF_SMOOTH_1,
        MF_SMOOTH_2,
        MF_WEIGHT,
        MF_SCALE,
    )

    return out


def _latest_closed(df: pd.DataFrame):
    return df.iloc[-2]


def _level_weight(level: str) -> int:

    # Higher timeframe liquidity gets more importance.

    weights = {
        "PMH": 15,
        "PML": 15,

        "PWH": 14,
        "PWL": 14,

        "PDH": 12,
        "PDL": 12,

        "PSH": 9,
        "PSL": 9,

        "NYH": 7,
        "NYL": 7,

        "LONH": 6,
        "LONL": 6,

        "ASIAH": 5,
        "ASIAL": 5,
    }

    return weights.get(level, 5)


def _best_event(events: list[dict]) -> dict | None:

    if not events:
        return None

    return max(
        events,
        key=lambda e: _level_weight(e["level"])
    )


def _mf_rising(df: pd.DataFrame) -> bool:

    return (
        df["mf"].iloc[-2] >
        df["mf"].iloc[-3] >
        df["mf"].iloc[-4]
    )


def _mf_falling(df: pd.DataFrame) -> bool:

    return (
        df["mf"].iloc[-2] <
        df["mf"].iloc[-3] <
        df["mf"].iloc[-4]
    )


def _wt_turning_up(df: pd.DataFrame) -> bool:

    return (
        df["wt1"].iloc[-2] >
        df["wt1"].iloc[-3]
    )


def _wt_turning_down(df: pd.DataFrame) -> bool:

    return (
        df["wt1"].iloc[-2] <
        df["wt1"].iloc[-3]
    )


def analyze(
    symbol: str,
    df4h: pd.DataFrame,
    df15: pd.DataFrame,
) -> Signal | None:

    h4 = enrich(df4h)
    m15 = enrich(df15)

    # =====================================================
    # LEVELS
    # =====================================================

    levels4 = (
        previous_period_levels(h4)
        | pivot_levels(
            h4,
            SWING_LEFT,
            SWING_RIGHT,
        )
    )

    levels15 = (
        previous_period_levels(m15)
        | session_levels(m15)
        | pivot_levels(m15, 12, 3)
    )

    # =====================================================
    # LIQUIDITY EVENTS
    # =====================================================

    ev4 = detect_event(
        h4,
        levels4,
        lookback=3,
    )

    ev15 = detect_event(
        m15,
        levels15,
        lookback=4,
    )

    low_sweeps4 = [
        e for e in ev4
        if e["type"] == "sweep_low"
    ]

    high_sweeps4 = [
        e for e in ev4
        if e["type"] == "sweep_high"
    ]

    low_sweeps15 = [
        e for e in ev15
        if e["type"] == "sweep_low"
    ]

    high_sweeps15 = [
        e for e in ev15
        if e["type"] == "sweep_high"
    ]

    best_long_4h = _best_event(low_sweeps4)
    best_short_4h = _best_event(high_sweeps4)

    best_long_15 = _best_event(low_sweeps15)
    best_short_15 = _best_event(high_sweeps15)

    last4 = _latest_closed(h4)
    last15 = _latest_closed(m15)

    struct = local_structure_shift(m15)

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # =====================================================
    # 4H LIQUIDITY — MAX ~30
    # =====================================================

    if best_long_4h:

        points = 15 + min(
            _level_weight(best_long_4h["level"]),
            15,
        )

        long_score += points

        long_reasons.append(
            f"4H {best_long_4h['level']} liquidity sweep"
        )

    if best_short_4h:

        points = 15 + min(
            _level_weight(best_short_4h["level"]),
            15,
        )

        short_score += points

        short_reasons.append(
            f"4H {best_short_4h['level']} liquidity sweep"
        )

    # =====================================================
    # 4H MOMENTUM — MAX 15
    # =====================================================

    if last4["wt2"] <= -WT_EXTREME:

        long_score += 7

        long_reasons.append(
            f"4H WaveTrend oversold "
            f"({last4['wt2']:.1f})"
        )

    if last4["wt2"] >= WT_EXTREME:

        short_score += 7

        short_reasons.append(
            f"4H WaveTrend overbought "
            f"({last4['wt2']:.1f})"
        )

    if _wt_turning_up(h4):

        long_score += 4

        long_reasons.append(
            "4H momentum turning bullish"
        )

    if _wt_turning_down(h4):

        short_score += 4

        short_reasons.append(
            "4H momentum turning bearish"
        )

    if _mf_rising(h4):

        long_score += 4

        long_reasons.append(
            "4H Money Flow rising"
        )

    if _mf_falling(h4):

        short_score += 4

        short_reasons.append(
            "4H Money Flow falling"
        )

    # =====================================================
    # 15M LIQUIDITY — MAX ~20
    # =====================================================

    if best_long_15:

        points = 10 + min(
            _level_weight(best_long_15["level"]),
            10,
        )

        long_score += points

        long_reasons.append(
            f"15M {best_long_15['level']} liquidity sweep"
        )

    if best_short_15:

        points = 10 + min(
            _level_weight(best_short_15["level"]),
            10,
        )

        short_score += points

        short_reasons.append(
            f"15M {best_short_15['level']} liquidity sweep"
        )

    # =====================================================
    # 15M STRUCTURE — VERY IMPORTANT
    # =====================================================

    if struct.get("bullish"):

        long_score += 15

        long_reasons.append(
            "15M bullish BOS / structure shift"
        )

    if struct.get("bearish"):

        short_score += 15

        short_reasons.append(
            "15M bearish BOS / structure shift"
        )

    # =====================================================
    # 15M MOMENTUM
    # =====================================================

    if bool(last15["wt_cross_up"]):

        long_score += 8

        long_reasons.append(
            "15M WaveTrend bullish cross"
        )

    elif _wt_turning_up(m15):

        long_score += 4

        long_reasons.append(
            "15M momentum turning bullish"
        )

    if bool(last15["wt_cross_down"]):

        short_score += 8

        short_reasons.append(
            "15M WaveTrend bearish cross"
        )

    elif _wt_turning_down(m15):

        short_score += 4

        short_reasons.append(
            "15M momentum turning bearish"
        )

    if _mf_rising(m15):

        long_score += 7

        long_reasons.append(
            "15M Money Flow rising"
        )

    if _mf_falling(m15):

        short_score += 7

        short_reasons.append(
            "15M Money Flow falling"
        )

    # =====================================================
    # CHOOSE SIDE
    # =====================================================

    side = (
        "LONG"
        if long_score > short_score
        else "SHORT"
    )

    score = (
        long_score
        if side == "LONG"
        else short_score
    )

    reasons = (
        long_reasons
        if side == "LONG"
        else short_reasons
    )

    # =====================================================
    # REQUIRE LIQUIDITY
    # =====================================================

    if side == "LONG":

        if not (
            best_long_4h
            or best_long_15
        ):
            return None

    else:

        if not (
            best_short_4h
            or best_short_15
        ):
            return None

    # =====================================================
    # REQUIRE 15M CONFIRMATION FOR STRONG SIGNALS
    # =====================================================

    if side == "LONG":

        confirmation = (
            struct.get("bullish")
            or bool(last15["wt_cross_up"])
            or (
                _wt_turning_up(m15)
                and _mf_rising(m15)
            )
        )

    else:

        confirmation = (
            struct.get("bearish")
            or bool(last15["wt_cross_down"])
            or (
                _wt_turning_down(m15)
                and _mf_falling(m15)
            )
        )

    if not confirmation:

        score = max(
            score - 15,
            0,
        )

        reasons.append(
            "15M confirmation still weak"
        )

    # =====================================================
    # ENTRY / STOP
    # =====================================================

    current = float(
        last15["close"]
    )

    atr15 = float(
        last15["atr"]
    )

    if (
        pd.isna(atr15)
        or atr15 <= 0
    ):
        return None

    if side == "LONG":

        source_event = (
            best_long_15
            or best_long_4h
        )

        reclaim = float(
            source_event["price"]
        )

        entry = min(
            current,
            reclaim + 0.10 * atr15,
        )

        recent_low = float(
            m15.iloc[-10:-1]["low"].min()
        )

        sl = (
            min(
                recent_low,
                reclaim,
            )
            - 0.20 * atr15
        )

        invalidation = (
            f"15M close below "
            f"{sl:.8g}"
        )

    else:

        source_event = (
            best_short_15
            or best_short_4h
        )

        reclaim = float(
            source_event["price"]
        )

        entry = max(
            current,
            reclaim - 0.10 * atr15,
        )

        recent_high = float(
            m15.iloc[-10:-1]["high"].max()
        )

        sl = (
            max(
                recent_high,
                reclaim,
            )
            + 0.20 * atr15
        )

        invalidation = (
            f"15M close above "
            f"{sl:.8g}"
        )

    # =====================================================
    # SMART TARGETS
    # =====================================================

    all_levels = (
        levels4
        | levels15
    )

    risk = abs(
        entry - sl
    )

    if risk <= 0:
        return None

    raw_targets = nearest_targets(
        all_levels,
        entry,
        side,
        20,
    )

    if not raw_targets:
        return None

    rr_steps = [
        0.70,
        1.30,
        2.00,
        3.00,
    ]

    targets = []
    used_prices = set()

    for minimum_rr in rr_steps:

        selected = None

        for price, name in raw_targets:

            price = float(
                price
            )

            reward = abs(
                price - entry
            )

            rr = (
                reward / risk
            )

            rounded_price = round(
                price,
                12
            )

            if (
                rr >= minimum_rr
                and rounded_price not in used_prices
            ):

                selected = (
                    price,
                    name,
                )

                break

        if selected:

            targets.append(
                selected
            )

            used_prices.add(
                round(
                    selected[0],
                    12
                )
            )

    if len(targets) < 4:

        for price, name in raw_targets:

            if len(targets) >= 4:
                break

            price = float(
                price
            )

            rounded_price = round(
                price,
                12
            )

            if (
                rounded_price
                in used_prices
            ):
                continue

            reward = abs(
                price - entry
            )

            rr = (
                reward / risk
            )

            if rr < 0.70:
                continue

            targets.append(
                (
                    price,
                    name,
                )
            )

            used_prices.add(
                rounded_price
            )

    if not targets:
        return None

    # =====================================================
    # RISK / REWARD BONUS
    # =====================================================

    first_target = float(
        targets[0][0]
    )

    reward = abs(
        first_target - entry
    )

    rr1 = (
        reward / risk
    )

    if rr1 >= 1.0:

        score += 5

        reasons.append(
            f"TP1 R:R {rr1:.2f}"
        )

    if len(targets) >= 2:

        reward2 = abs(
            float(
                targets[1][0]
            )
            - entry
        )

        rr2 = (
            reward2 / risk
        )

        if rr2 >= 2.0:

            score += 5

            reasons.append(
                f"TP2 R:R {rr2:.2f}"
            )

    # =====================================================
    # FINAL FILTER
    # =====================================================

    score = min(
        int(score),
        100,
    )

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        entry=entry,
        sl=sl,
        targets=targets,
        reasons=reasons,
        invalidation=invalidation,
    )
