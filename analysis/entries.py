from __future__ import annotations

import math
import pandas as pd


# =========================================================
# HELPERS
# =========================================================

def _closed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Work only with closed candles.
    The last dataframe row is treated as the live candle.
    """
    if df is None or len(df) < 2:
        return pd.DataFrame() if df is None else df.copy()

    return (
        df.iloc[:-1]
        .copy()
        .reset_index(drop=True)
    )


def _atr_series(df: pd.DataFrame, length: int = 55) -> pd.Series:
    """
    Use pre-calculated ATR when available.
    Otherwise calculate Wilder-like rolling TR mean locally.
    """
    if "atr" in df.columns:
        atr_values = pd.to_numeric(
            df["atr"],
            errors="coerce",
        )
        if atr_values.notna().any():
            return atr_values

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        length,
        min_periods=max(5, length // 4),
    ).mean()


def _zone_mid(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0


# =========================================================
# LUXALGO-STYLE FAIR VALUE GAPS
# =========================================================

def find_fvgs(
    df: pd.DataFrame,
    lookback: int = 80,
    auto_threshold: bool = True,
) -> list[dict]:
    """
    LuxAlgo-style FVG logic adapted for our 15M dataframe.

    Bullish:
        current low > high[2]
        previous candle close > high[2]
        previous candle body/displacement > dynamic threshold

    Bearish:
        current high < low[2]
        previous candle close < low[2]
        negative previous candle displacement > threshold

    Old / already invalidated FVGs are discarded.
    """

    d = _closed(df)

    if len(d) < 5:
        return []

    open_ = pd.to_numeric(d["open"], errors="coerce")
    close = pd.to_numeric(d["close"], errors="coerce")

    # Relative body delta. The original LuxAlgo threshold is based on a
    # cumulative average of absolute candle delta and multiplied by 2.
    body_delta = (
        (close - open_)
        / open_.replace(0, math.nan)
    )

    if auto_threshold:
        threshold = (
            body_delta.abs()
            .expanding(min_periods=3)
            .mean()
            * 2.0
        )
    else:
        threshold = pd.Series(
            0.0,
            index=d.index,
        )

    start = max(
        2,
        len(d) - lookback,
    )

    fvgs: list[dict] = []

    for i in range(start, len(d)):
        left = d.iloc[i - 2]
        middle = d.iloc[i - 1]
        right = d.iloc[i]

        left_high = float(left["high"])
        left_low = float(left["low"])

        middle_close = float(middle["close"])
        middle_open = float(middle["open"])

        right_high = float(right["high"])
        right_low = float(right["low"])

        if middle_open == 0:
            continue

        delta = (
            middle_close - middle_open
        ) / middle_open

        dynamic_threshold = float(
            threshold.iloc[i - 1]
        )

        if math.isnan(dynamic_threshold):
            dynamic_threshold = 0.0

        # -------------------------------------------------
        # BULLISH FVG
        # -------------------------------------------------

        bullish = (
            right_low > left_high
            and middle_close > left_high
            and (
                (not auto_threshold)
                or delta > dynamic_threshold
            )
        )

        if bullish:
            low = left_high
            high = right_low

            # LuxAlgo removes bullish FVG when price trades
            # below its bottom.
            future = d.iloc[i + 1:]

            invalidated = (
                not future.empty
                and float(future["low"].min()) < low
            )

            if not invalidated:
                fvgs.append(
                    {
                        "side": "LONG",
                        "type": "FVG",
                        "low": low,
                        "high": high,
                        "mid": _zone_mid(low, high),
                        "time": right["time"],
                        "index": i,
                        "displacement": delta,
                        "threshold": dynamic_threshold,
                    }
                )

        # -------------------------------------------------
        # BEARISH FVG
        # -------------------------------------------------

        bearish = (
            right_high < left_low
            and middle_close < left_low
            and (
                (not auto_threshold)
                or (-delta) > dynamic_threshold
            )
        )

        if bearish:
            low = right_high
            high = left_low

            future = d.iloc[i + 1:]

            invalidated = (
                not future.empty
                and float(future["high"].max()) > high
            )

            if not invalidated:
                fvgs.append(
                    {
                        "side": "SHORT",
                        "type": "FVG",
                        "low": low,
                        "high": high,
                        "mid": _zone_mid(low, high),
                        "time": right["time"],
                        "index": i,
                        "displacement": -delta,
                        "threshold": dynamic_threshold,
                    }
                )

    return fvgs


def latest_fvg(
    df: pd.DataFrame,
    side: str,
    current_price: float,
    lookback: int = 80,
) -> dict | None:

    matching = [
        item
        for item in find_fvgs(
            df,
            lookback=lookback,
            auto_threshold=True,
        )
        if item["side"] == side
    ]

    if not matching:
        return None

    for fvg in reversed(matching):

        if side == "LONG":
            # A long retrace FVG should normally be at/below price.
            if float(fvg["low"]) <= current_price:
                return fvg

        else:
            # A short retrace FVG should normally be at/above price.
            if float(fvg["high"]) >= current_price:
                return fvg

    return None


# =========================================================
# STRUCTURE / BOS-LINKED ORDER BLOCKS
# =========================================================

def _bos_events(
    df: pd.DataFrame,
    lookback: int = 80,
    structure_window: int = 12,
) -> list[dict]:
    """
    Detect simple confirmed BOS events using closed candles only.

    Bullish BOS:
        close breaks the highest high of previous structure_window candles.

    Bearish BOS:
        close breaks the lowest low of previous structure_window candles.
    """

    d = _closed(df)

    if len(d) < structure_window + 3:
        return []

    start = max(
        structure_window,
        len(d) - lookback,
    )

    events: list[dict] = []

    for i in range(start, len(d)):
        previous = d.iloc[
            i - structure_window:i
        ]

        if previous.empty:
            continue

        prev_high = float(
            previous["high"].max()
        )

        prev_low = float(
            previous["low"].min()
        )

        candle = d.iloc[i]
        close = float(candle["close"])

        if close > prev_high:
            events.append(
                {
                    "side": "LONG",
                    "type": "BOS",
                    "index": i,
                    "time": candle["time"],
                    "broken_level": prev_high,
                }
            )

        elif close < prev_low:
            events.append(
                {
                    "side": "SHORT",
                    "type": "BOS",
                    "index": i,
                    "time": candle["time"],
                    "broken_level": prev_low,
                }
            )

    return events


def _refine_order_block(
    candle: pd.Series,
    atr_value: float,
    side: str,
    mode: str = "DEFENSIVE",
) -> tuple[float, float]:
    """
    Refine a raw order block using its size relative to ATR.

    This follows the same idea as the TradingFinder defensive/aggressive
    refinement: wide blocks are narrowed, while small blocks are kept intact.

    Returns:
        zone_low, zone_high
    """

    low = float(candle["low"])
    high = float(candle["high"])
    open_ = float(candle["open"])
    close = float(candle["close"])

    raw_range = max(
        high - low,
        1e-12,
    )

    if (
        mode.upper() == "AGGRESSIVE"
        or not math.isfinite(atr_value)
        or atr_value <= 0
    ):
        return low, high

    # Defensive refinement:
    # progressively narrow very large blocks.
    ratio = raw_range / atr_value

    if ratio >= 3.0:
        keep = 0.30
    elif ratio >= 2.0:
        keep = 0.40
    elif ratio >= 1.6:
        keep = 0.50
    elif ratio > 1.0:
        keep = 0.75
    else:
        keep = 1.00

    body_low = min(open_, close)
    body_high = max(open_, close)

    if side == "LONG":
        # Keep the deeper / discount part of bullish demand OB.
        refined_high = low + raw_range * keep

        # Do not refine beyond the candle body in an unrealistic way.
        refined_high = min(
            max(refined_high, body_low),
            high,
        )

        return low, refined_high

    # SHORT: keep upper / premium part of supply OB.
    refined_low = high - raw_range * keep

    refined_low = max(
        min(refined_low, body_high),
        low,
    )

    return refined_low, high


def find_order_block(
    df: pd.DataFrame,
    side: str,
    lookback: int = 80,
    structure_window: int = 12,
    origin_search: int = 8,
    refine_mode: str = "DEFENSIVE",
) -> dict | None:
    """
    Order Block is linked to a confirmed BOS.

    LONG:
        find latest bullish BOS,
        then locate the last bearish origin candle before the BOS.

    SHORT:
        find latest bearish BOS,
        then locate the last bullish origin candle before the BOS.

    Old / mitigated order blocks are discarded.
    """

    d = _closed(df)

    if len(d) < structure_window + 5:
        return None

    atr_values = _atr_series(d, length=55)

    events = [
        event
        for event in _bos_events(
            df,
            lookback=lookback,
            structure_window=structure_window,
        )
        if event["side"] == side
    ]

    if not events:
        return None

    for bos in reversed(events):
        bos_index = int(
            bos["index"]
        )

        search_start = max(
            0,
            bos_index - origin_search,
        )

        origin_index = None

        for i in range(
            bos_index - 1,
            search_start - 1,
            -1,
        ):
            candle = d.iloc[i]

            candle_open = float(
                candle["open"]
            )

            candle_close = float(
                candle["close"]
            )

            if (
                side == "LONG"
                and candle_close < candle_open
            ):
                origin_index = i
                break

            if (
                side == "SHORT"
                and candle_close > candle_open
            ):
                origin_index = i
                break

        if origin_index is None:
            continue

        candle = d.iloc[
            origin_index
        ]

        atr_value = float(
            atr_values.iloc[
                origin_index
            ]
        )

        zone_low, zone_high = (
            _refine_order_block(
                candle,
                atr_value,
                side,
                mode=refine_mode,
            )
        )

        future = d.iloc[
            bos_index + 1:
        ]

        # TradingFinder/LuxAlgo-style mitigation:
        # bullish OB is dead below distal;
        # bearish OB is dead above distal.
        if side == "LONG":

            invalidated = (
                not future.empty
                and float(
                    future["low"].min()
                ) < zone_low
            )

        else:

            invalidated = (
                not future.empty
                and float(
                    future["high"].max()
                ) > zone_high
            )

        if invalidated:
            continue

        return {
            "side": side,
            "type": "ORDER_BLOCK",
            "low": float(zone_low),
            "high": float(zone_high),
            "mid": _zone_mid(
                zone_low,
                zone_high,
            ),
            "time": candle["time"],
            "index": origin_index,
            "bos_time": bos["time"],
            "bos_index": bos_index,
            "broken_level": float(
                bos["broken_level"]
            ),
            "atr": atr_value,
            "refine_mode": refine_mode.upper(),
        }

    return None


# =========================================================
# OVERLAP / CONFLUENCE
# =========================================================

def overlap_zone(
    fvg: dict | None,
    ob: dict | None,
    min_overlap_ratio: float = 0.05,
) -> dict | None:
    """
    True overlap only.

    A single touching price is NOT enough to call it FVG + OB.
    The overlap must have real width and be at least a small
    fraction of the FVG width.
    """

    if not fvg or not ob:
        return None

    if fvg["side"] != ob["side"]:
        return None

    low = max(
        float(fvg["low"]),
        float(ob["low"]),
    )

    high = min(
        float(fvg["high"]),
        float(ob["high"]),
    )

    overlap_width = (
        high - low
    )

    fvg_width = max(
        float(fvg["high"])
        - float(fvg["low"]),
        1e-12,
    )

    if overlap_width <= 0:
        return None

    if (
        overlap_width / fvg_width
        < min_overlap_ratio
    ):
        return None

    return {
        "type": "FVG_OB",
        "side": fvg["side"],
        "low": low,
        "high": high,
        "mid": _zone_mid(
            low,
            high,
        ),
        "width": overlap_width,
    }


# =========================================================
# BUILD RETRACE ENTRY
# =========================================================

def build_retrace_entry(
    df15: pd.DataFrame,
    side: str,
) -> dict | None:
    """
    Priority:

    1. Valid FVG + BOS-linked refined Order Block overlap
    2. Valid LuxAlgo-style FVG 50%
    3. Valid BOS-linked refined Order Block 50%

    Keeps the same return interface used by engine.py.
    """

    if (
        df15 is None
        or len(df15) < 25
    ):
        return None

    current_price = float(
        df15.iloc[-2]["close"]
    )

    fvg = latest_fvg(
        df15,
        side,
        current_price,
        lookback=80,
    )

    ob = find_order_block(
        df15,
        side,
        lookback=80,
        structure_window=12,
        origin_search=8,
        refine_mode="DEFENSIVE",
    )

    overlap = overlap_zone(
        fvg,
        ob,
    )

    # -------------------------------------------------
    # BEST CASE: FVG + OB
    # -------------------------------------------------

    if overlap:
        return {
            "side": side,
            "entry_type": "FVG + ORDER BLOCK",

            # Show the REAL FVG range in Telegram.
            # Entry itself still comes from the actual FVG/OB overlap.
            "zone_low": float(
                fvg["low"]
            ),
            "zone_high": float(
                fvg["high"]
            ),
            "entry": float(
                overlap["mid"]
            ),
            "current_price": current_price,

            # Extra diagnostics for later notifier/tracker upgrades.
            "entry_zone_low": float(
                overlap["low"]
            ),
            "entry_zone_high": float(
                overlap["high"]
            ),
            "fvg_low": float(
                fvg["low"]
            ),
            "fvg_high": float(
                fvg["high"]
            ),
            "ob_low": float(
                ob["low"]
            ),
            "ob_high": float(
                ob["high"]
            ),

            "fvg": fvg,
            "order_block": ob,
        }

    # -------------------------------------------------
    # FVG 50%
    # -------------------------------------------------

    if fvg:
        return {
            "side": side,
            "entry_type": "FVG 50%",
            "zone_low": float(
                fvg["low"]
            ),
            "zone_high": float(
                fvg["high"]
            ),
            "entry": float(
                fvg["mid"]
            ),
            "current_price": current_price,
            "fvg": fvg,
            "order_block": ob,
        }

    # -------------------------------------------------
    # ORDER BLOCK 50%
    # -------------------------------------------------

    if ob:
        return {
            "side": side,
            "entry_type": "ORDER BLOCK 50%",
            "zone_low": float(
                ob["low"]
            ),
            "zone_high": float(
                ob["high"]
            ),
            "entry": float(
                ob["mid"]
            ),
            "current_price": current_price,
            "fvg": None,
            "order_block": ob,
        }

    return None
