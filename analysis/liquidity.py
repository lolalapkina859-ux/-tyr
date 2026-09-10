from __future__ import annotations
import pandas as pd

def previous_period_levels(df: pd.DataFrame) -> dict[str, float]:
    d = df.set_index("time").copy()

    def prev(resample_rule: str, high_name: str, low_name: str):
        x = d.resample(resample_rule).agg({"high":"max", "low":"min"}).dropna()
        if len(x) < 2:
            return {}
        row = x.iloc[-2]
        return {high_name: float(row["high"]), low_name: float(row["low"])}

    out = {}
    out |= prev("1D", "PDH", "PDL")
    out |= prev("W-MON", "PWH", "PWL")
    out |= prev("MS", "PMH", "PML")
    return out

def session_levels(df: pd.DataFrame) -> dict[str, float]:
    # Approximate cash-session ranges in their local timezones.
    out = {}
    specs = [
        ("ASIA", "Asia/Tokyo", "09:00", "15:00"),
        ("LON", "Europe/London", "08:00", "16:00"),
        ("NY", "America/New_York", "09:30", "16:00"),
    ]

    for prefix, tz, start, end in specs:
        local = df.copy()
        local["local_time"] = local["time"].dt.tz_convert(tz)
        local["local_date"] = local["local_time"].dt.date
        # Use the most recent completed session, or today's partial session if active.
        dates = list(local["local_date"].drop_duplicates())
        if not dates:
            continue

        chosen = None
        for day in reversed(dates):
            day_df = local[local["local_date"] == day].set_index("local_time")
            ses = day_df.between_time(start, end, inclusive="left")
            if len(ses):
                chosen = ses
                break

        if chosen is not None and len(chosen):
            out[f"{prefix}H"] = float(chosen["high"].max())
            out[f"{prefix}L"] = float(chosen["low"].min())

    return out

def pivot_levels(df: pd.DataFrame, left: int = 30, right: int = 3) -> dict[str, float]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    ph, pl = [], []

    for i in range(left, len(df) - right):
        if highs[i] >= highs[i-left:i+right+1].max():
            ph.append(highs[i])
        if lows[i] <= lows[i-left:i+right+1].min():
            pl.append(lows[i])

    out = {}
    if ph:
        out["PSH"] = float(ph[-1])
    if pl:
        out["PSL"] = float(pl[-1])
    return out

def detect_event(df: pd.DataFrame, levels: dict[str, float], lookback: int = 3):
    """
    Sweep high: wick > level, close back below.
    Sweep low:  wick < level, close back above.
    Run: close remains beyond the level.
    Only closed candles are inspected.
    """
    closed = df.iloc[:-1] if len(df) > 2 else df
    recent = closed.iloc[-lookback:]

    events = []
    for name, level in levels.items():
        is_high = name.endswith("H")
        for idx, row in recent.iterrows():
            if is_high:
                if row["high"] > level and row["close"] < level:
                    events.append({"type":"sweep_high", "level":name, "price":level, "bar":idx})
                elif row["close"] > level:
                    events.append({"type":"run_high", "level":name, "price":level, "bar":idx})
            else:
                if row["low"] < level and row["close"] > level:
                    events.append({"type":"sweep_low", "level":name, "price":level, "bar":idx})
                elif row["close"] < level:
                    events.append({"type":"run_low", "level":name, "price":level, "bar":idx})
    return events

def nearest_targets(levels: dict[str, float], entry: float, side: str, max_targets: int = 4):
    vals = []
    for name, price in levels.items():
        if side == "LONG" and price > entry:
            vals.append((price, name))
        elif side == "SHORT" and price < entry:
            vals.append((price, name))

    vals.sort(key=lambda x: x[0], reverse=(side == "SHORT"))
    return vals[:max_targets]
