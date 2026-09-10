import pandas as pd

def local_structure_shift(df: pd.DataFrame, window: int = 12) -> dict:
    closed = df.iloc[:-1].copy()
    if len(closed) < window + 3:
        return {"bullish": False, "bearish": False}

    prev = closed.iloc[-window-1:-1]
    last = closed.iloc[-1]

    prev_high = float(prev["high"].max())
    prev_low = float(prev["low"].min())

    return {
        "bullish": float(last["close"]) > prev_high,
        "bearish": float(last["close"]) < prev_low,
        "prev_high": prev_high,
        "prev_low": prev_low,
    }
