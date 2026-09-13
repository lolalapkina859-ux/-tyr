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
from analysis.entries import build_retrace_entry
from analysis.volume_profile import (
    build_htf_volume_profiles,
    volume_confluence,
)


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

    entry_status: str
    entry_type: str

    zone_low: float
    zone_high: float

    current_price: float

    # Higher-timeframe Fixed Range Volume Profile diagnostics.
    vp_1d_poc: float | None = None
    vp_1d_val: float | None = None
    vp_1d_vah: float | None = None

    vp_4h_poc: float | None = None
    vp_4h_val: float | None = None
    vp_4h_vah: float | None = None

    vp_confluence: bool = False
    vp_score_bonus: int = 0


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



# =========================================================
# SMART STRUCTURAL STOP LOSS
# =========================================================

_LONG_PROTECT_LEVELS = {
    "PML",
    "PWL",
    "PDL",
    "PSL",
    "ASIAL",
    "LONL",
    "NYL",
}

_SHORT_PROTECT_LEVELS = {
    "PMH",
    "PWH",
    "PDH",
    "PSH",
    "ASIAH",
    "LONH",
    "NYH",
}


def _level_price(value) -> float | None:
    """
    previous_period_levels/session_levels/pivot_levels normally
    return numeric prices. This helper also tolerates dict values.
    """
    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if isinstance(
        value,
        dict,
    ):
        for key in (
            "price",
            "value",
            "level",
        ):
            if key in value:
                try:
                    return float(
                        value[key]
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def _smart_structural_sl(
    side: str,
    entry: float,
    zone_low: float,
    zone_high: float,
    reclaim: float,
    atr15: float,
    m15: pd.DataFrame,
    all_levels: dict,
    retrace: dict | None,
) -> tuple[float, str]:
    """
    Smart SL logic.

    LONG:
        stop below the protected structure:
        OB/FVG low, recent swing low, liquidity/reclaim level,
        and the nearest relevant PDL/PWL/PSL/session-low cluster.

    SHORT:
        mirrored logic above structure.

    The stop also receives a volatility buffer and a minimum
    distance from entry, so it is not placed directly in obvious
    liquidity where a normal wick can remove the position.
    """

    entry = float(entry)
    zone_low = float(zone_low)
    zone_high = float(zone_high)
    reclaim = float(reclaim)
    atr15 = float(atr15)

    # We only use completed 15M candles for structural anchors.
    closed15 = (
        m15.iloc[:-1]
        .copy()
    )

    recent = (
        closed15.tail(16)
        if len(closed15) >= 16
        else closed15
    )

    # Structural levels too far away should not make the SL absurdly wide.
    max_level_distance = max(
        5.0 * atr15,
        0.020 * entry,
    )

    # Buffer behind the invalidation structure.
    # The percentage floor helps high-priced/low-ATR instruments such as BTC.
    buffer = max(
        0.35 * atr15,
        0.0015 * entry,
    )

    # Avoid microscopic stops even when the nearest OB is extremely tight.
    min_risk = max(
        0.65 * atr15,
        0.0025 * entry,
    )

    ob = (
        retrace.get("order_block")
        if retrace
        else None
    )

    fvg = (
        retrace.get("fvg")
        if retrace
        else None
    )

    if side == "LONG":

        candidates: list[
            tuple[float, str]
        ] = []

        if not recent.empty:
            recent_low = float(
                recent["low"].min()
            )

            if (
                recent_low < entry
                and entry - recent_low
                <= max_level_distance
            ):
                candidates.append(
                    (
                        recent_low,
                        "15M swing low",
                    )
                )

        for price, label in (
            (
                zone_low,
                "entry zone low",
            ),
            (
                reclaim,
                "liquidity reclaim",
            ),
        ):
            if (
                price < entry
                and entry - price
                <= max_level_distance
            ):
                candidates.append(
                    (
                        float(price),
                        label,
                    )
                )

        if ob:
            ob_low = float(
                ob["low"]
            )

            if (
                ob_low < entry
                and entry - ob_low
                <= max_level_distance
            ):
                candidates.append(
                    (
                        ob_low,
                        "Order Block low",
                    )
                )

        if fvg:
            fvg_low = float(
                fvg["low"]
            )

            if (
                fvg_low < entry
                and entry - fvg_low
                <= max_level_distance
            ):
                candidates.append(
                    (
                        fvg_low,
                        "FVG low",
                    )
                )

        structural_levels = []

        for name, value in (
            all_levels.items()
        ):
            if (
                name
                not in _LONG_PROTECT_LEVELS
            ):
                continue

            price = _level_price(
                value
            )

            if price is None:
                continue

            if (
                price < entry
                and entry - price
                <= max_level_distance
            ):
                structural_levels.append(
                    (
                        price,
                        name,
                    )
                )

        # Nearest support under entry.
        # If several levels form a tight cluster, protect below
        # the deepest level of that cluster.
        if structural_levels:
            structural_levels.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            nearest_price = (
                structural_levels[0][0]
            )

            cluster = [
                item
                for item in structural_levels
                if (
                    nearest_price
                    - item[0]
                    <= 0.50 * atr15
                )
            ]

            cluster_price = min(
                item[0]
                for item in cluster
            )

            cluster_names = "/".join(
                item[1]
                for item in cluster
            )

            candidates.append(
                (
                    cluster_price,
                    f"HTF/LTF liquidity {cluster_names}",
                )
            )

        if not candidates:
            base = (
                entry - min_risk
            )
            base_reason = (
                "minimum volatility distance"
            )
        else:
            # Stop must be behind every relevant nearby structure,
            # not just behind the entry-zone boundary.
            base, base_reason = min(
                candidates,
                key=lambda x: x[0],
            )

        sl = (
            float(base)
            - buffer
        )

        # Hard anti-tight-stop guard.
        if (
            entry - sl
            < min_risk
        ):
            sl = (
                entry
                - min_risk
            )

            base_reason += (
                " + minimum risk guard"
            )

        reason = (
            f"Smart SL below {base_reason}; "
            f"buffer={buffer:.8g}"
        )

        return (
            float(sl),
            reason,
        )

    # =====================================================
    # SHORT
    # =====================================================

    candidates = []

    if not recent.empty:
        recent_high = float(
            recent["high"].max()
        )

        if (
            recent_high > entry
            and recent_high - entry
            <= max_level_distance
        ):
            candidates.append(
                (
                    recent_high,
                    "15M swing high",
                )
            )

    for price, label in (
        (
            zone_high,
            "entry zone high",
        ),
        (
            reclaim,
            "liquidity reclaim",
        ),
    ):
        if (
            price > entry
            and price - entry
            <= max_level_distance
        ):
            candidates.append(
                (
                    float(price),
                    label,
                )
            )

    if ob:
        ob_high = float(
            ob["high"]
        )

        if (
            ob_high > entry
            and ob_high - entry
            <= max_level_distance
        ):
            candidates.append(
                (
                    ob_high,
                    "Order Block high",
                )
            )

    if fvg:
        fvg_high = float(
            fvg["high"]
        )

        if (
            fvg_high > entry
            and fvg_high - entry
            <= max_level_distance
        ):
            candidates.append(
                (
                    fvg_high,
                    "FVG high",
                )
            )

    structural_levels = []

    for name, value in (
        all_levels.items()
    ):
        if (
            name
            not in _SHORT_PROTECT_LEVELS
        ):
            continue

        price = _level_price(
            value
        )

        if price is None:
            continue

        if (
            price > entry
            and price - entry
            <= max_level_distance
        ):
            structural_levels.append(
                (
                    price,
                    name,
                )
            )

    if structural_levels:
        structural_levels.sort(
            key=lambda x: x[0]
        )

        nearest_price = (
            structural_levels[0][0]
        )

        cluster = [
            item
            for item in structural_levels
            if (
                item[0]
                - nearest_price
                <= 0.50 * atr15
            )
        ]

        cluster_price = max(
            item[0]
            for item in cluster
        )

        cluster_names = "/".join(
            item[1]
            for item in cluster
        )

        candidates.append(
            (
                cluster_price,
                f"HTF/LTF liquidity {cluster_names}",
            )
        )

    if not candidates:
        base = (
            entry + min_risk
        )
        base_reason = (
            "minimum volatility distance"
        )
    else:
        base, base_reason = max(
            candidates,
            key=lambda x: x[0],
        )

    sl = (
        float(base)
        + buffer
    )

    if (
        sl - entry
        < min_risk
    ):
        sl = (
            entry
            + min_risk
        )

        base_reason += (
            " + minimum risk guard"
        )

    reason = (
        f"Smart SL above {base_reason}; "
        f"buffer={buffer:.8g}"
    )

    return (
        float(sl),
        reason,
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
    # REQUIRE 15M STRUCTURE CONFIRMATION
    # =====================================================

    # Liquidity creates the setup.
    # BOS / structure shift on 15M gives permission
    # to treat it as a confirmed trading signal.

    if side == "LONG":

        structure_confirmed = bool(
            struct.get("bullish")
        )

        momentum_confirmed = (
            bool(last15["wt_cross_up"])
            or _wt_turning_up(m15)
        )

        money_flow_confirmed = (
            _mf_rising(m15)
        )

    else:

        structure_confirmed = bool(
            struct.get("bearish")
        )

        momentum_confirmed = (
            bool(last15["wt_cross_down"])
            or _wt_turning_down(m15)
        )

        money_flow_confirmed = (
            _mf_falling(m15)
        )

    # -----------------------------------------------------
    # NO BOS / STRUCTURE SHIFT = WATCH ONLY
    # -----------------------------------------------------

    if not structure_confirmed:

        # Не позволяем такому сетапу стать сигналом 70+
        # даже если liquidity + momentum набрали много баллов.
        score = min(
            score,
            69,
        )

        reasons.append(
            "15M structure NOT confirmed — WATCH only"
        )

    # -----------------------------------------------------
    # STRUCTURE EXISTS, BUT MOMENTUM IS WEAK
    # -----------------------------------------------------

    elif not momentum_confirmed:

        score = max(
            score - 10,
            0,
        )

        reasons.append(
            "15M structure confirmed but momentum weak"
        )

    # -----------------------------------------------------
    # FULL 15M CONFIRMATION
    # -----------------------------------------------------

    elif (
        structure_confirmed
        and momentum_confirmed
        and money_flow_confirmed
    ):

        score += 5

        reasons.append(
            "15M full confirmation: "
            "structure + momentum + money flow"
        )

    # =====================================================
    # ENTRY / STOP — FVG / ORDER BLOCK RETRACE
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

    # -----------------------------------------------------
    # SOURCE LIQUIDITY EVENT
    # -----------------------------------------------------

    if side == "LONG":
        source_event = (
            best_long_15
            or best_long_4h
        )
    else:
        source_event = (
            best_short_15
            or best_short_4h
        )

    reclaim = float(
        source_event["price"]
    )

    # -----------------------------------------------------
    # SEARCH 15M RETRACE ENTRY
    # -----------------------------------------------------

    retrace = build_retrace_entry(
        m15,
        side,
    )

    # =====================================================
    # FVG / OB FOUND
    # =====================================================

    if retrace:
        entry = float(
            retrace["entry"]
        )

        zone_low = float(
            retrace["zone_low"]
        )

        zone_high = float(
            retrace["zone_high"]
        )

        entry_type = str(
            retrace["entry_type"]
        )

        if (
            zone_low
            <= current
            <= zone_high
        ):
            entry_status = "ENTER_NOW"
            reasons.append(
                f"15M {entry_type} entry zone active"
            )
        else:
            entry_status = "WAIT_FOR_RETRACE"
            reasons.append(
                f"WAIT FOR RETRACE to {entry_type}"
            )

        if side == "LONG":
            recent_low = float(
                m15.iloc[-12:-1]["low"].min()
            )

            sl = (
                min(
                    recent_low,
                    zone_low,
                    reclaim,
                )
                - 0.20 * atr15
            )

            invalidation = (
                f"15M close below "
                f"{sl:.8g}"
            )
        else:
            recent_high = float(
                m15.iloc[-12:-1]["high"].max()
            )

            sl = (
                max(
                    recent_high,
                    zone_high,
                    reclaim,
                )
                + 0.20 * atr15
            )

            invalidation = (
                f"15M close above "
                f"{sl:.8g}"
            )

    # =====================================================
    # NO FVG / OB FOUND — FALLBACK TO LIQUIDITY RETEST
    # =====================================================

    else:
        entry_type = "LIQUIDITY RETEST"

        if side == "LONG":
            entry = min(
                current,
                reclaim + 0.10 * atr15,
            )

            zone_low = (
                entry - 0.10 * atr15
            )

            zone_high = (
                entry + 0.10 * atr15
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
            entry = max(
                current,
                reclaim - 0.10 * atr15,
            )

            zone_low = (
                entry - 0.10 * atr15
            )

            zone_high = (
                entry + 0.10 * atr15
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

        distance = abs(
            current - entry
        )

        if distance <= 0.25 * atr15:
            entry_status = "ENTER_NOW"
        else:
            entry_status = "WAIT_FOR_RETRACE"
            reasons.append(
                "Price moved away from entry — wait for retest"
            )


    # =====================================================
    # HTF FIXED RANGE VOLUME PROFILE — 1D + 4H
    # =====================================================
    #
    # Volume Profile does NOT create a signal by itself.
    # It only improves an already-confirmed liquidity + BOS setup.
    #
    # 1D profile is reconstructed from 4H candles and is therefore
    # a higher-resolution approximation of a daily FRVP.
    # =====================================================

    vp_profiles = build_htf_volume_profiles(
        h4,
        side,
    )

    vp_result = volume_confluence(
        vp_profiles,
        zone_low,
        zone_high,
        entry,
    )

    vp_1d = vp_profiles.get(
        "1D"
    )
    vp_4h = vp_profiles.get(
        "4H"
    )

    vp_bonus = int(
        vp_result.get(
            "score_bonus",
            0,
        )
    )

    vp_confluence = bool(
        vp_result.get(
            "has_confluence",
            False,
        )
    )

    if vp_confluence:
        old_entry = float(
            entry
        )

        preferred_entry = float(
            vp_result.get(
                "preferred_entry",
                entry,
            )
        )

        # Safety: Volume Profile may refine an entry,
        # but may never pull it outside the original FVG/OB zone.
        if (
            zone_low
            <= preferred_entry
            <= zone_high
        ):
            entry = preferred_entry

        score += vp_bonus

        if "HTF VP" not in entry_type:
            entry_type = (
                f"{entry_type} + HTF VP"
            )

        for match in vp_result.get(
            "matches",
            [],
        ):
            reasons.append(
                f"HTF Volume Profile: {match}"
            )

        if abs(
            entry - old_entry
        ) > 1e-12:
            reasons.append(
                f"HTF Volume Profile refined entry "
                f"{old_entry:.8g} -> {entry:.8g}"
            )
    else:
        reasons.append(
            "HTF Volume Profile: no strong POC/HVN confluence"
        )

    # =====================================================
    # SMART TARGETS
    # =====================================================

    all_levels = (
        levels4
        | levels15
    )

    # =====================================================
    # SMART STRUCTURAL SL
    # =====================================================
    #
    # Recalculate SL AFTER Volume Profile has had a chance
    # to refine the final entry price.
    #
    # The old SL above is only a provisional fallback.
    # From here onward risk/R:R/targets use the Smart SL.
    # =====================================================

    sl, smart_sl_reason = (
        _smart_structural_sl(
            side=side,
            entry=entry,
            zone_low=zone_low,
            zone_high=zone_high,
            reclaim=reclaim,
            atr15=atr15,
            m15=m15,
            all_levels=all_levels,
            retrace=retrace,
        )
    )

    reasons.append(
        smart_sl_reason
    )

    if side == "LONG":
        invalidation = (
            f"15M close below "
            f"{sl:.8g}"
        )
    else:
        invalidation = (
            f"15M close above "
            f"{sl:.8g}"
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
    # SIGNAL FRESHNESS
    # =====================================================

    first_target_price = float(
        targets[0][0]
    )

    # If price already reached TP1 before the alert,
    # the setup is considered missed and must not be sent.
    if side == "LONG":
        if current >= first_target_price:
            print(
                f"{symbol}: MISSED — "
                f"price already reached TP1 "
                f"before alert"
            )
            return None
    else:
        if current <= first_target_price:
            print(
                f"{symbol}: MISSED — "
                f"price already reached TP1 "
                f"before alert"
            )
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

    # Если 15M структура не подтверждена,
    # сигнал НИКОГДА не может стать 70+,
    # даже после бонусов за R:R.
    if not structure_confirmed:
        score = min(
            int(score),
            69,
        )

    else:
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

        entry_status=entry_status,
        entry_type=entry_type,

        zone_low=zone_low,
        zone_high=zone_high,

        current_price=current,

        vp_1d_poc=(
            float(vp_1d["poc"])
            if vp_1d
            else None
        ),
        vp_1d_val=(
            float(vp_1d["val"])
            if vp_1d
            else None
        ),
        vp_1d_vah=(
            float(vp_1d["vah"])
            if vp_1d
            else None
        ),

        vp_4h_poc=(
            float(vp_4h["poc"])
            if vp_4h
            else None
        ),
        vp_4h_val=(
            float(vp_4h["val"])
            if vp_4h
            else None
        ),
        vp_4h_vah=(
            float(vp_4h["vah"])
            if vp_4h
            else None
        ),

        vp_confluence=vp_confluence,
        vp_score_bonus=vp_bonus,
    )
