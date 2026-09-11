from __future__ import annotations

import pandas as pd


# =========================================================
# HELPERS
# =========================================================

def _closed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Работаем только с закрытыми свечами.
    Последняя строка считается текущей незакрытой свечой.
    """
    if len(df) < 2:
        return df.copy()

    return (
        df.iloc[:-1]
        .copy()
        .reset_index(drop=True)
    )


# =========================================================
# FAIR VALUE GAPS
# =========================================================

def find_fvgs(
    df: pd.DataFrame,
    lookback: int = 60,
) -> list[dict]:

    """
    Bullish FVG:
        high candle[i-2] < low candle[i]

        zone:
        high[i-2] -> low[i]

    Bearish FVG:
        low candle[i-2] > high candle[i]

        zone:
        high[i] -> low[i-2]
    """

    d = _closed(df)

    if len(d) < 3:
        return []

    start = max(
        2,
        len(d) - lookback,
    )

    fvgs = []

    for i in range(
        start,
        len(d),
    ):

        left = d.iloc[i - 2]
        middle = d.iloc[i - 1]
        right = d.iloc[i]

        # =============================================
        # BULLISH FVG
        # =============================================

        if float(left["high"]) < float(right["low"]):

            low = float(
                left["high"]
            )

            high = float(
                right["low"]
            )

            fvgs.append({
                "side": "LONG",
                "type": "FVG",
                "low": low,
                "high": high,
                "mid": (
                    low + high
                ) / 2.0,
                "time": right["time"],
                "index": i,
                "displacement_close":
                    float(
                        middle["close"]
                    ),
            })

        # =============================================
        # BEARISH FVG
        # =============================================

        if float(left["low"]) > float(right["high"]):

            low = float(
                right["high"]
            )

            high = float(
                left["low"]
            )

            fvgs.append({
                "side": "SHORT",
                "type": "FVG",
                "low": low,
                "high": high,
                "mid": (
                    low + high
                ) / 2.0,
                "time": right["time"],
                "index": i,
                "displacement_close":
                    float(
                        middle["close"]
                    ),
            })

    return fvgs


def latest_fvg(
    df: pd.DataFrame,
    side: str,
    current_price: float,
    lookback: int = 60,
) -> dict | None:

    fvgs = find_fvgs(
        df,
        lookback,
    )

    matching = [
        fvg
        for fvg in fvgs
        if fvg["side"] == side
    ]

    if not matching:
        return None

    # Берём самый свежий FVG,
    # который находится со стороны нормального retrace.
    for fvg in reversed(
        matching
    ):

        if side == "LONG":

            # Для LONG FVG должен находиться
            # ниже либо вокруг текущей цены.
            if fvg["low"] <= current_price:
                return fvg

        else:

            # Для SHORT FVG должен находиться
            # выше либо вокруг текущей цены.
            if fvg["high"] >= current_price:
                return fvg

    return None


# =========================================================
# ORDER BLOCK
# =========================================================

def find_order_block(
    df: pd.DataFrame,
    side: str,
    lookback: int = 30,
) -> dict | None:

    """
    Простая SMC-логика:

    LONG:
        последняя bearish candle
        перед сильным bullish displacement.

    SHORT:
        последняя bullish candle
        перед сильным bearish displacement.

    Пока используем полный диапазон свечи как OB.
    Позже можно сузить до body / 50%.
    """

    d = _closed(df)

    if len(d) < 5:
        return None

    start = max(
        1,
        len(d) - lookback,
    )

    # Идём с конца — нужен самый свежий OB.
    for i in range(
        len(d) - 2,
        start - 1,
        -1,
    ):

        candle = d.iloc[i]
        nxt = d.iloc[i + 1]

        candle_open = float(
            candle["open"]
        )

        candle_close = float(
            candle["close"]
        )

        candle_high = float(
            candle["high"]
        )

        candle_low = float(
            candle["low"]
        )

        next_open = float(
            nxt["open"]
        )

        next_close = float(
            nxt["close"]
        )

        next_range = abs(
            next_close - next_open
        )

        current_range = max(
            abs(
                candle_close
                - candle_open
            ),
            1e-12,
        )

        # Нужен хотя бы заметный displacement.
        displacement = (
            next_range
            >= current_range * 1.25
        )

        if not displacement:
            continue

        # =============================================
        # BULLISH ORDER BLOCK
        # =============================================

        if (
            side == "LONG"
            and candle_close < candle_open
            and next_close > next_open
        ):

            return {
                "side": "LONG",
                "type": "ORDER_BLOCK",
                "low": candle_low,
                "high": candle_high,
                "mid": (
                    candle_low
                    + candle_high
                ) / 2.0,
                "time": candle["time"],
                "index": i,
            }

        # =============================================
        # BEARISH ORDER BLOCK
        # =============================================

        if (
            side == "SHORT"
            and candle_close > candle_open
            and next_close < next_open
        ):

            return {
                "side": "SHORT",
                "type": "ORDER_BLOCK",
                "low": candle_low,
                "high": candle_high,
                "mid": (
                    candle_low
                    + candle_high
                ) / 2.0,
                "time": candle["time"],
                "index": i,
            }

    return None


# =========================================================
# OVERLAP / CONFLUENCE
# =========================================================

def overlap_zone(
    fvg: dict | None,
    ob: dict | None,
) -> dict | None:

    """
    Если FVG и Order Block пересекаются,
    это наша premium retrace zone.
    """

    if not fvg or not ob:
        return None

    low = max(
        float(fvg["low"]),
        float(ob["low"]),
    )

    high = min(
        float(fvg["high"]),
        float(ob["high"]),
    )

    if low > high:
        return None

    return {
        "type": "FVG_OB",
        "side": fvg["side"],
        "low": low,
        "high": high,
        "mid": (
            low + high
        ) / 2.0,
    }


# =========================================================
# BUILD RETRACE ENTRY
# =========================================================

def build_retrace_entry(
    df15: pd.DataFrame,
    side: str,
) -> dict | None:

    """
    Приоритет:

    1. FVG + Order Block overlap
    2. FVG 50%
    3. Order Block 50%

    Возвращает готовую retrace zone.
    """

    if df15 is None or len(df15) < 20:
        return None

    current_price = float(
        df15.iloc[-2]["close"]
    )

    fvg = latest_fvg(
        df15,
        side,
        current_price,
    )

    ob = find_order_block(
        df15,
        side,
    )

    overlap = overlap_zone(
        fvg,
        ob,
    )

    # =============================================
    # BEST CASE: FVG + OB
    # =============================================

    if overlap:

        return {
            "side": side,
            "entry_type": "FVG + ORDER BLOCK",
            "zone_low":
                overlap["low"],
            "zone_high":
                overlap["high"],
            "entry":
                overlap["mid"],
            "current_price":
                current_price,
            "fvg": fvg,
            "order_block": ob,
        }

    # =============================================
    # FVG
    # =============================================

    if fvg:

        return {
            "side": side,
            "entry_type": "FVG 50%",
            "zone_low":
                float(fvg["low"]),
            "zone_high":
                float(fvg["high"]),
            "entry":
                float(fvg["mid"]),
            "current_price":
                current_price,
            "fvg": fvg,
            "order_block": ob,
        }

    # =============================================
    # ORDER BLOCK
    # =============================================

    if ob:

        return {
            "side": side,
            "entry_type": "ORDER BLOCK 50%",
            "zone_low":
                float(ob["low"]),
            "zone_high":
                float(ob["high"]),
            "entry":
                float(ob["mid"]),
            "current_price":
                current_price,
            "fvg": None,
            "order_block": ob,
        }

    return None
