import numpy as np
import pandas as pd

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rma(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=1/length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)

def wavetrend(
    df: pd.DataFrame,
    channel: int = 9,
    avg: int = 12,
    signal: int = 3,
) -> pd.DataFrame:
    # Mirrors the Pine logic:
    # esa = ema(hlc3, 9)
    # dev = ema(abs(hlc3-esa), 9)
    # ci  = (hlc3-esa)/(0.015*dev)
    # wt1 = ema(ci, 12)
    # wt2 = sma(wt1, 3)
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ema(src, channel)
    dev = ema((src - esa).abs(), channel)
    ci = (src - esa) / (0.015 * dev.replace(0, np.nan))
    ci = ci.fillna(0.0)
    wt1 = ema(ci, avg)
    wt2 = wt1.rolling(signal, min_periods=1).mean()

    out = pd.DataFrame(index=df.index)
    out["wt1"] = wt1
    out["wt2"] = wt2
    out["cross_up"] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    out["cross_down"] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    return out

def money_flow(
    df: pd.DataFrame,
    smooth1: int = 21,
    smooth2: int = 9,
    weight: float = 0.50,
    scale: float = 150.0,
) -> pd.Series:
    # Mirrors the Pine EMA defaults.
    rng = df["high"] - df["low"]
    body_pos = ((df["close"] - df["open"]) / rng.replace(0, np.nan)).fillna(0.0)
    delta_mult = (
        ((df["close"] - df["low"]) - (df["high"] - df["close"]))
        / rng.replace(0, np.nan)
    ).fillna(0.0)

    raw = body_pos * (1.0 - weight) + delta_mult * weight
    one = ema(raw, smooth1)
    two = ema(one, max(smooth2, 2))
    return (one if smooth2 <= 1 else two) * scale
