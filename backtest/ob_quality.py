from __future__ import annotations

import math
import pandas as pd

from analysis.entries import find_order_block


def measure_ob_quality(df15: pd.DataFrame, side: str, baseline_bars: int = 20) -> dict | None:
    """Measure the current Trade Vision 15M order block with SMC-style volume diagnostics.

    Important: this does NOT change live entry logic. It is used only by the
    experimental historical backtest.

    Metrics:
      - ob_volume: BOS candle volume + previous two candle volumes
      - volume_ratio: that 3-candle sum divided by 3x median prior volume
      - balance_pct: min(volume_group_a, volume_group_b) / max(...) * 100,
        mirroring joshyattridge/smart-money-concepts
    """
    if df15 is None or len(df15) < 30:
        return None

    ob = find_order_block(
        df15,
        side,
        lookback=80,
        structure_window=12,
        origin_search=8,
        refine_mode="DEFENSIVE",
    )
    if not ob:
        return None

    # find_order_block works on closed candles internally. Mirror that here.
    closed = df15.iloc[:-1].copy().reset_index(drop=True)
    bos_index = int(ob.get("bos_index", -1))
    if bos_index < 2 or bos_index >= len(closed):
        return None

    volumes = pd.to_numeric(closed["volume"], errors="coerce")
    v0 = float(volumes.iloc[bos_index])
    v1 = float(volumes.iloc[bos_index - 1])
    v2 = float(volumes.iloc[bos_index - 2])

    if not all(math.isfinite(v) and v >= 0 for v in (v0, v1, v2)):
        return None

    ob_volume = v0 + v1 + v2

    # Compare the 3-candle structural-break volume against the typical volume
    # immediately before it, without using any future candles.
    baseline_end = max(0, bos_index - 2)
    baseline_start = max(0, baseline_end - baseline_bars)
    baseline = volumes.iloc[baseline_start:baseline_end].dropna()

    if baseline.empty:
        volume_ratio = 0.0
    else:
        median_volume = float(baseline.median())
        volume_ratio = ob_volume / (3.0 * median_volume) if median_volume > 0 else 0.0

    # Mirror the original library's bullish/bearish grouping.
    if side.upper() == "LONG":
        group_a = v2
        group_b = v0 + v1
    else:
        group_a = v0 + v1
        group_b = v2

    maximum = max(group_a, group_b)
    balance_pct = (min(group_a, group_b) / maximum * 100.0) if maximum > 0 else 100.0

    return {
        "ob_volume": ob_volume,
        "volume_ratio": volume_ratio,
        "balance_pct": balance_pct,
        "bos_index": bos_index,
        "broken_level": float(ob.get("broken_level", 0.0)),
        "ob_low": float(ob.get("low", 0.0)),
        "ob_high": float(ob.get("high", 0.0)),
    }


def passes_ob_quality(
    metrics: dict | None,
    *,
    min_volume_ratio: float = 1.20,
    min_balance_pct: float = 0.0,
) -> bool:
    if metrics is None:
        return False

    return (
        float(metrics.get("volume_ratio", 0.0)) >= float(min_volume_ratio)
        and float(metrics.get("balance_pct", 0.0)) >= float(min_balance_pct)
    )
