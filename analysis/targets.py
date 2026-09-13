from __future__ import annotations

from typing import Iterable
import math

import pandas as pd

from analysis.entries import (
    find_fvgs,
    find_order_block,
)


# =========================================================
# HYBRID TARGET ENGINE
# =========================================================
#
# Target sources:
#   1. Classical liquidity levels (PDH/PWH/PMH/session/swing)
#   2. Opposing 15M / 4H FVG boundaries
#   3. Opposing 15M / 4H Order Blocks
#   4. 4H / 1D Volume Profile POC / VA boundaries / HVNs
#
# Important:
#   - targets NEVER alter Entry or SL;
#   - targets are directional obstacles/magnets ahead of Entry;
#   - near-duplicate prices are merged;
#   - TP1..TP4 are selected by minimum R:R steps.
# =========================================================


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _is_ahead(
    side: str,
    entry: float,
    price: float,
) -> bool:
    if side == "LONG":
        return price > entry

    return price < entry


def _distance(
    side: str,
    entry: float,
    price: float,
) -> float:
    if side == "LONG":
        return price - entry

    return entry - price


def _append_candidate(
    out: list[tuple[float, str, int]],
    side: str,
    entry: float,
    price,
    name: str,
    priority: int,
) -> None:
    price = _finite(price)

    if price is None:
        return

    if not _is_ahead(
        side,
        entry,
        price,
    ):
        return

    out.append(
        (
            price,
            name,
            int(priority),
        )
    )


def _opposite_side(side: str) -> str:
    return (
        "SHORT"
        if side == "LONG"
        else "LONG"
    )


def _fvg_target_price(
    side: str,
    fvg: dict,
) -> float:
    """
    Use the FIRST boundary price is expected to meet.

    LONG -> lower boundary of bearish FVG.
    SHORT -> upper boundary of bullish FVG.
    """
    if side == "LONG":
        return float(
            fvg["low"]
        )

    return float(
        fvg["high"]
    )


def _ob_target_price(
    side: str,
    ob: dict,
) -> float:
    """
    Use the proximal / first-touch boundary.

    LONG into supply -> OB low.
    SHORT into demand -> OB high.
    """
    if side == "LONG":
        return float(
            ob["low"]
        )

    return float(
        ob["high"]
    )


def _collect_fvg_targets(
    out: list[tuple[float, str, int]],
    df: pd.DataFrame,
    timeframe: str,
    side: str,
    entry: float,
    lookback: int,
) -> None:
    opposite = _opposite_side(
        side
    )

    try:
        fvgs = find_fvgs(
            df,
            lookback=lookback,
            auto_threshold=True,
        )
    except Exception:
        return

    for fvg in fvgs:
        if (
            str(
                fvg.get(
                    "side",
                    "",
                )
            ).upper()
            != opposite
        ):
            continue

        try:
            price = _fvg_target_price(
                side,
                fvg,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        _append_candidate(
            out,
            side,
            entry,
            price,
            f"{timeframe}_FVG",
            82 if timeframe == "4H" else 72,
        )


def _collect_ob_target(
    out: list[tuple[float, str, int]],
    df: pd.DataFrame,
    timeframe: str,
    side: str,
    entry: float,
    lookback: int,
    structure_window: int,
) -> None:
    opposite = _opposite_side(
        side
    )

    try:
        ob = find_order_block(
            df,
            opposite,
            lookback=lookback,
            structure_window=structure_window,
        )
    except Exception:
        return

    if not ob:
        return

    try:
        price = _ob_target_price(
            side,
            ob,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return

    _append_candidate(
        out,
        side,
        entry,
        price,
        f"{timeframe}_OB",
        88 if timeframe == "4H" else 78,
    )


def _collect_volume_profile_targets(
    out: list[tuple[float, str, int]],
    vp_profiles: dict,
    side: str,
    entry: float,
) -> None:
    for timeframe, priority in (
        ("1D", 90),
        ("4H", 84),
    ):
        profile = (
            vp_profiles.get(
                timeframe
            )
            if vp_profiles
            else None
        )

        if not profile:
            continue

        # POC is a strong volume magnet.
        _append_candidate(
            out,
            side,
            entry,
            profile.get(
                "poc"
            ),
            f"{timeframe}_POC",
            priority,
        )

        # For LONG the upside VAH is generally the useful forward boundary.
        # For SHORT the downside VAL is generally the useful forward boundary.
        va_key = (
            "vah"
            if side == "LONG"
            else "val"
        )

        _append_candidate(
            out,
            side,
            entry,
            profile.get(
                va_key
            ),
            f"{timeframe}_{va_key.upper()}",
            priority - 4,
        )

        # HVNs are meaningful intermediate magnets.
        for row in (
            profile.get(
                "hvn_rows",
                []
            )
            or []
        ):
            _append_candidate(
                out,
                side,
                entry,
                row.get(
                    "mid"
                ),
                f"{timeframe}_HVN",
                priority - 8,
            )


def _merge_near_duplicates(
    candidates: list[tuple[float, str, int]],
    side: str,
    entry: float,
    atr15: float,
) -> list[tuple[float, str]]:
    if not candidates:
        return []

    # Merge levels that are essentially the same obstacle.
    tolerance = max(
        abs(float(atr15)) * 0.12,
        abs(float(entry)) * 0.0008,
        1e-12,
    )

    candidates = sorted(
        candidates,
        key=lambda item: (
            _distance(
                side,
                entry,
                item[0],
            ),
            -item[2],
        ),
    )

    clusters: list[
        list[tuple[float, str, int]]
    ] = []

    for item in candidates:
        if not clusters:
            clusters.append(
                [item]
            )
            continue

        last_cluster = (
            clusters[-1]
        )

        anchor_price = float(
            last_cluster[0][0]
        )

        if (
            abs(
                float(item[0])
                - anchor_price
            )
            <= tolerance
        ):
            last_cluster.append(
                item
            )
        else:
            clusters.append(
                [item]
            )

    merged: list[
        tuple[float, str]
    ] = []

    for cluster in clusters:
        if side == "LONG":
            price = min(
                item[0]
                for item in cluster
            )
        else:
            price = max(
                item[0]
                for item in cluster
            )

        labels = []

        for _, label, _ in sorted(
            cluster,
            key=lambda item: item[2],
            reverse=True,
        ):
            if (
                label
                not in labels
            ):
                labels.append(
                    label
                )

        # Keep Telegram compact.
        label = (
            labels[0]
            if len(labels) == 1
            else "+".join(
                labels[:2]
            )
        )

        merged.append(
            (
                float(price),
                label,
            )
        )

    return sorted(
        merged,
        key=lambda item: _distance(
            side,
            entry,
            item[0],
        ),
    )


def build_hybrid_targets(
    *,
    side: str,
    entry: float,
    sl: float,
    atr15: float,
    df15: pd.DataFrame,
    df4h: pd.DataFrame,
    liquidity_targets: Iterable[
        tuple[float, str]
    ],
    vp_profiles: dict,
    limit: int = 4,
) -> list[tuple[float, str]]:
    """
    Build TP1..TP4 from multiple market-structure sources.

    Selection thresholds:
        TP1 >= 0.70R
        TP2 >= 1.30R
        TP3 >= 2.00R
        TP4 >= 3.00R

    A farther HTF liquidity level is retained as runner only when
    closer structural/volume targets do not already fill the slots.
    """

    entry = float(entry)
    sl = float(sl)
    atr15 = float(atr15)

    risk = abs(
        entry - sl
    )

    if (
        risk <= 0
        or not math.isfinite(
            risk
        )
    ):
        return []

    candidates: list[
        tuple[float, str, int]
    ] = []

    # -----------------------------------------------------
    # 1) CLASSICAL LIQUIDITY
    # -----------------------------------------------------

    liquidity_priority = {
        "PMH": 98,
        "PML": 98,
        "PWH": 96,
        "PWL": 96,
        "PDH": 94,
        "PDL": 94,
        "PSH": 90,
        "PSL": 90,
        "NYH": 80,
        "NYL": 80,
        "LONH": 76,
        "LONL": 76,
        "ASIAH": 72,
        "ASIAL": 72,
    }

    for price, name in (
        liquidity_targets
        or []
    ):
        _append_candidate(
            candidates,
            side,
            entry,
            price,
            str(name),
            liquidity_priority.get(
                str(name),
                70,
            ),
        )

    # -----------------------------------------------------
    # 2) OPPOSING FVG / ORDER BLOCK
    # -----------------------------------------------------

    _collect_fvg_targets(
        candidates,
        df15,
        "15M",
        side,
        entry,
        lookback=120,
    )

    _collect_fvg_targets(
        candidates,
        df4h,
        "4H",
        side,
        entry,
        lookback=100,
    )

    _collect_ob_target(
        candidates,
        df15,
        "15M",
        side,
        entry,
        lookback=120,
        structure_window=12,
    )

    _collect_ob_target(
        candidates,
        df4h,
        "4H",
        side,
        entry,
        lookback=100,
        structure_window=8,
    )

    # -----------------------------------------------------
    # 3) VOLUME PROFILE
    # -----------------------------------------------------

    _collect_volume_profile_targets(
        candidates,
        vp_profiles,
        side,
        entry,
    )

    raw = _merge_near_duplicates(
        candidates,
        side,
        entry,
        atr15,
    )

    if not raw:
        return []

    rr_steps = [
        0.70,
        1.30,
        2.00,
        3.00,
    ]

    chosen: list[
        tuple[float, str]
    ] = []

    used_prices = set()

    for minimum_rr in rr_steps[
        :limit
    ]:
        selected = None

        for price, name in raw:
            rounded = round(
                float(price),
                12,
            )

            if (
                rounded
                in used_prices
            ):
                continue

            rr = (
                abs(
                    float(price)
                    - entry
                )
                / risk
            )

            if rr < minimum_rr:
                continue

            selected = (
                float(price),
                str(name),
            )
            break

        if selected:
            chosen.append(
                selected
            )

            used_prices.add(
                round(
                    selected[0],
                    12,
                )
            )

    # Fill missing slots with remaining meaningful levels >= 0.70R.
    if len(chosen) < limit:
        for price, name in raw:
            if len(chosen) >= limit:
                break

            rounded = round(
                float(price),
                12,
            )

            if (
                rounded
                in used_prices
            ):
                continue

            rr = (
                abs(
                    float(price)
                    - entry
                )
                / risk
            )

            if rr < 0.70:
                continue

            chosen.append(
                (
                    float(price),
                    str(name),
                )
            )

            used_prices.add(
                rounded
            )

    return chosen
