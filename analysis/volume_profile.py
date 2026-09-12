from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PROFILE_ROWS = 24
VALUE_AREA_PCT = 0.70


def _closed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat the last row as the current/live candle.
    Volume Profile should be built from completed information only.
    """
    if df is None or len(df) < 2:
        return pd.DataFrame() if df is None else df.copy()

    return (
        df.iloc[:-1]
        .copy()
        .reset_index(drop=True)
    )


def _volume_column(df: pd.DataFrame) -> str | None:
    """
    Support the common names used by exchange adapters.
    """
    for name in (
        "volume",
        "vol",
        "quote_volume",
        "turnover",
    ):
        if name in df.columns:
            return name

    return None


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = _closed(df)

    if d.empty:
        return d

    vol_col = _volume_column(d)

    if vol_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "time": pd.to_datetime(
            d["time"],
            utc=True,
            errors="coerce",
        ),
        "open": pd.to_numeric(
            d["open"],
            errors="coerce",
        ),
        "high": pd.to_numeric(
            d["high"],
            errors="coerce",
        ),
        "low": pd.to_numeric(
            d["low"],
            errors="coerce",
        ),
        "close": pd.to_numeric(
            d["close"],
            errors="coerce",
        ),
        "volume": pd.to_numeric(
            d[vol_col],
            errors="coerce",
        ),
    })

    out = (
        out.dropna()
        .reset_index(drop=True)
    )

    out = out[
        (out["high"] >= out["low"])
        & (out["volume"] >= 0)
    ].reset_index(drop=True)

    return out


def resample_daily_from_4h(
    df4h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a 1D structure from 4H candles.

    This is useful for a higher-timeframe FRVP while preserving
    more intraday information than requesting only one daily candle.
    """
    d = _prepare_ohlcv(df4h)

    if d.empty:
        return d

    d = d.set_index("time")

    daily = d.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    return (
        daily.dropna()
        .reset_index()
    )


def _select_structural_range(
    df: pd.DataFrame,
    side: str,
    lookback: int,
    min_bars: int,
) -> pd.DataFrame:
    """
    Approximate a manually anchored Fixed Range Volume Profile.

    LONG:
        anchor at the lowest structural point inside the lookback.

    SHORT:
        anchor at the highest structural point inside the lookback.

    If the selected swing is too recent, fall back to a stable
    fixed window so the profile is not built from only 1-2 candles.
    """
    if df is None or len(df) < min_bars:
        return pd.DataFrame()

    d = df.tail(
        min(
            lookback,
            len(df),
        )
    ).copy().reset_index(drop=True)

    if len(d) < min_bars:
        return pd.DataFrame()

    if side == "LONG":
        anchor = int(
            d["low"].idxmin()
        )
    else:
        anchor = int(
            d["high"].idxmax()
        )

    selected = (
        d.iloc[anchor:]
        .copy()
        .reset_index(drop=True)
    )

    if len(selected) < min_bars:
        selected = (
            d.tail(min_bars)
            .copy()
            .reset_index(drop=True)
        )

    return selected


def fixed_range_profile(
    df: pd.DataFrame,
    rows: int = PROFILE_ROWS,
    value_area_pct: float = VALUE_AREA_PCT,
) -> dict[str, Any] | None:
    """
    FRVP-like approximation using OHLCV candles.

    TradingView has access to lower-timeframe/intrabar distribution.
    Here each candle's volume is distributed across the price rows
    it actually overlaps, proportional to overlap width.

    Returns:
        POC, VAH, VAL, row volumes, range boundaries.
    """
    if (
        df is None
        or len(df) < 2
        or rows < 4
    ):
        return None

    d = df.copy()

    for col in (
        "high",
        "low",
        "close",
        "volume",
    ):
        d[col] = pd.to_numeric(
            d[col],
            errors="coerce",
        )

    d = d.dropna(
        subset=[
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    if len(d) < 2:
        return None

    price_low = float(
        d["low"].min()
    )
    price_high = float(
        d["high"].max()
    )

    if (
        not math.isfinite(price_low)
        or not math.isfinite(price_high)
        or price_high <= price_low
    ):
        return None

    edges = np.linspace(
        price_low,
        price_high,
        rows + 1,
    )

    volumes = np.zeros(
        rows,
        dtype=float,
    )

    for _, candle in d.iterrows():
        low = float(
            candle["low"]
        )
        high = float(
            candle["high"]
        )
        close = float(
            candle["close"]
        )
        volume = max(
            float(candle["volume"]),
            0.0,
        )

        if volume <= 0:
            continue

        candle_range = (
            high - low
        )

        if candle_range <= 1e-12:
            idx = int(
                np.searchsorted(
                    edges,
                    close,
                    side="right",
                ) - 1
            )
            idx = max(
                0,
                min(
                    rows - 1,
                    idx,
                ),
            )
            volumes[idx] += volume
            continue

        overlaps = np.zeros(
            rows,
            dtype=float,
        )

        for i in range(rows):
            row_low = float(
                edges[i]
            )
            row_high = float(
                edges[i + 1]
            )

            overlap = max(
                0.0,
                min(
                    high,
                    row_high,
                )
                - max(
                    low,
                    row_low,
                ),
            )

            overlaps[i] = overlap

        total_overlap = float(
            overlaps.sum()
        )

        if total_overlap <= 0:
            idx = int(
                np.searchsorted(
                    edges,
                    close,
                    side="right",
                ) - 1
            )
            idx = max(
                0,
                min(
                    rows - 1,
                    idx,
                ),
            )
            volumes[idx] += volume
        else:
            volumes += (
                volume
                * overlaps
                / total_overlap
            )

    total_volume = float(
        volumes.sum()
    )

    if total_volume <= 0:
        return None

    poc_idx = int(
        np.argmax(volumes)
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    target_volume = (
        total_volume
        * float(value_area_pct)
    )

    included = {
        poc_idx
    }

    cumulative = float(
        volumes[poc_idx]
    )

    left = (
        poc_idx - 1
    )
    right = (
        poc_idx + 1
    )

    while (
        cumulative < target_volume
        and (
            left >= 0
            or right < rows
        )
    ):
        left_volume = (
            float(volumes[left])
            if left >= 0
            else -1.0
        )

        right_volume = (
            float(volumes[right])
            if right < rows
            else -1.0
        )

        if right_volume > left_volume:
            included.add(
                right
            )
            cumulative += max(
                right_volume,
                0.0,
            )
            right += 1
        else:
            included.add(
                left
            )
            cumulative += max(
                left_volume,
                0.0,
            )
            left -= 1

    va_low_idx = min(
        included
    )
    va_high_idx = max(
        included
    )

    # HVN approximation: rows in the top quartile by volume.
    positive = volumes[
        volumes > 0
    ]

    hvn_threshold = (
        float(
            np.quantile(
                positive,
                0.75,
            )
        )
        if len(positive)
        else 0.0
    )

    hvn_rows = [
        {
            "low": float(
                edges[i]
            ),
            "high": float(
                edges[i + 1]
            ),
            "mid": float(
                centers[i]
            ),
            "volume": float(
                volumes[i]
            ),
        }
        for i in range(rows)
        if volumes[i] >= hvn_threshold
        and volumes[i] > 0
    ]

    return {
        "rows": rows,
        "value_area_pct": float(
            value_area_pct
        ),
        "range_low": price_low,
        "range_high": price_high,
        "poc": float(
            centers[poc_idx]
        ),
        "poc_low": float(
            edges[poc_idx]
        ),
        "poc_high": float(
            edges[poc_idx + 1]
        ),
        "val": float(
            edges[va_low_idx]
        ),
        "vah": float(
            edges[va_high_idx + 1]
        ),
        "total_volume": total_volume,
        "hvn_rows": hvn_rows,
        "start_time": (
            d["time"].iloc[0]
            if "time" in d.columns
            else None
        ),
        "end_time": (
            d["time"].iloc[-1]
            if "time" in d.columns
            else None
        ),
    }


def build_htf_volume_profiles(
    df4h: pd.DataFrame,
    side: str,
) -> dict[str, dict[str, Any] | None]:
    """
    Build two higher-timeframe profiles:

    4H profile:
        recent 4H structural swing -> current closed 4H area.

    1D profile:
        resampled daily structure from the same 4H source,
        anchored from the recent daily swing.
    """
    h4 = _prepare_ohlcv(
        df4h
    )

    if h4.empty:
        return {
            "4H": None,
            "1D": None,
        }

    h4_range = _select_structural_range(
        h4,
        side,
        lookback=42,
        min_bars=12,
    )

    daily = resample_daily_from_4h(
        df4h
    )

    daily_range = _select_structural_range(
        daily,
        side,
        lookback=45,
        min_bars=8,
    )

    return {
        "4H": fixed_range_profile(
            h4_range,
            rows=PROFILE_ROWS,
            value_area_pct=VALUE_AREA_PCT,
        ),
        "1D": fixed_range_profile(
            daily_range,
            rows=PROFILE_ROWS,
            value_area_pct=VALUE_AREA_PCT,
        ),
    }


def _overlap_ratio(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> float:
    width_a = max(
        high_a - low_a,
        1e-12,
    )

    overlap = max(
        0.0,
        min(
            high_a,
            high_b,
        )
        - max(
            low_a,
            low_b,
        ),
    )

    return (
        overlap
        / width_a
    )


def volume_confluence(
    profiles: dict[str, dict[str, Any] | None],
    zone_low: float,
    zone_high: float,
    entry: float,
) -> dict[str, Any]:
    """
    Evaluate whether a 15M FVG/OB entry zone is confirmed by
    the higher-timeframe Volume Profile.

    Important:
    - We NEVER move entry outside the original FVG/OB zone.
    - 1D has more weight than 4H.
    - POC inside the zone is the strongest confirmation.
    """
    zone_low = float(
        min(
            zone_low,
            zone_high,
        )
    )
    zone_high = float(
        max(
            zone_low,
            zone_high,
        )
    )

    score_bonus = 0
    matches = []
    preferred_candidates = []

    for timeframe, weight in (
        ("1D", 5),
        ("4H", 4),
    ):
        profile = profiles.get(
            timeframe
        )

        if not profile:
            continue

        poc = float(
            profile["poc"]
        )
        val = float(
            profile["val"]
        )
        vah = float(
            profile["vah"]
        )

        poc_in_zone = (
            zone_low
            <= poc
            <= zone_high
        )

        va_overlap = _overlap_ratio(
            zone_low,
            zone_high,
            val,
            vah,
        )

        hvn_overlap = False

        for row in profile.get(
            "hvn_rows",
            [],
        ):
            if (
                min(
                    zone_high,
                    float(row["high"]),
                )
                > max(
                    zone_low,
                    float(row["low"]),
                )
            ):
                hvn_overlap = True
                break

        if poc_in_zone:
            score_bonus += weight
            preferred_candidates.append(
                (
                    timeframe,
                    poc,
                    weight,
                )
            )
            matches.append(
                f"{timeframe} POC inside entry zone"
            )
        elif (
            hvn_overlap
            and va_overlap >= 0.25
        ):
            score_bonus += 2
            matches.append(
                f"{timeframe} HVN/Value Area overlaps entry zone"
            )
        elif va_overlap >= 0.50:
            score_bonus += 1
            matches.append(
                f"{timeframe} Value Area overlaps entry zone"
            )

    # Cap the bonus. VP is confirmation, not a standalone setup generator.
    score_bonus = min(
        score_bonus,
        8,
    )

    preferred_entry = float(
        entry
    )
    preferred_tf = None

    if preferred_candidates:
        # Prefer 1D POC over 4H POC.
        preferred_candidates.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        preferred_tf, candidate, _ = (
            preferred_candidates[0]
        )

        if (
            zone_low
            <= candidate
            <= zone_high
        ):
            preferred_entry = float(
                candidate
            )

    return {
        "score_bonus": int(
            score_bonus
        ),
        "matches": matches,
        "preferred_entry": preferred_entry,
        "preferred_timeframe": preferred_tf,
        "has_confluence": bool(
            score_bonus > 0
        ),
    }
