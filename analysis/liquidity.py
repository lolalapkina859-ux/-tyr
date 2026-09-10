from __future__ import annotations

import pandas as pd


# =========================================================
# PREVIOUS PERIOD LEVELS
# =========================================================

def previous_period_levels(
    df: pd.DataFrame
) -> dict[str, float]:

    d = (
        df
        .set_index("time")
        .copy()
        .sort_index()
    )

    # Используем только данные с нормальным UTC index.
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")

    out = {}

    # =====================================================
    # PREVIOUS DAY
    # =====================================================

    daily = (
        d
        .resample(
            "1D",
            label="left",
            closed="left",
        )
        .agg({
            "high": "max",
            "low": "min",
        })
        .dropna()
    )

    if len(daily) >= 2:

        prev_day = daily.iloc[-2]

        out["PDH"] = float(
            prev_day["high"]
        )

        out["PDL"] = float(
            prev_day["low"]
        )

    # =====================================================
    # PREVIOUS WEEK
    #
    # Crypto week:
    # Monday 00:00 UTC → Sunday 23:59 UTC
    #
    # W-SUN = week ending Sunday.
    # =====================================================

    weekly = (
        d
        .resample(
            "W-SUN",
            label="right",
            closed="right",
        )
        .agg({
            "high": "max",
            "low": "min",
        })
        .dropna()
    )

    if len(weekly) >= 2:

        prev_week = weekly.iloc[-2]

        out["PWH"] = float(
            prev_week["high"]
        )

        out["PWL"] = float(
            prev_week["low"]
        )

    # =====================================================
    # PREVIOUS MONTH
    # =====================================================

    monthly = (
        d
        .resample(
            "MS",
            label="left",
            closed="left",
        )
        .agg({
            "high": "max",
            "low": "min",
        })
        .dropna()
    )

    if len(monthly) >= 2:

        prev_month = monthly.iloc[-2]

        out["PMH"] = float(
            prev_month["high"]
        )

        out["PML"] = float(
            prev_month["low"]
        )

    return out


# =========================================================
# SESSION LEVELS
# =========================================================

def session_levels(
    df: pd.DataFrame
) -> dict[str, float]:

    out = {}

    specs = [
        (
            "ASIA",
            "Asia/Tokyo",
            "09:00",
            "15:00",
        ),
        (
            "LON",
            "Europe/London",
            "08:00",
            "16:00",
        ),
        (
            "NY",
            "America/New_York",
            "09:30",
            "16:00",
        ),
    ]

    for (
        prefix,
        tz,
        start,
        end,
    ) in specs:

        local = df.copy()

        local["local_time"] = (
            local["time"]
            .dt
            .tz_convert(tz)
        )

        local["local_date"] = (
            local["local_time"]
            .dt
            .date
        )

        dates = list(
            local["local_date"]
            .drop_duplicates()
        )

        if not dates:
            continue

        chosen = None

        for day in reversed(
            dates
        ):

            day_df = (
                local[
                    local["local_date"]
                    == day
                ]
                .set_index(
                    "local_time"
                )
            )

            ses = (
                day_df
                .between_time(
                    start,
                    end,
                    inclusive="left",
                )
            )

            if len(ses):
                chosen = ses
                break

        if (
            chosen is not None
            and len(chosen)
        ):

            out[
                f"{prefix}H"
            ] = float(
                chosen["high"].max()
            )

            out[
                f"{prefix}L"
            ] = float(
                chosen["low"].min()
            )

    return out


# =========================================================
# PIVOT / SWING LEVELS
# =========================================================

def pivot_levels(
    df: pd.DataFrame,
    left: int = 30,
    right: int = 3,
) -> dict[str, float]:

    highs = (
        df["high"]
        .to_numpy()
    )

    lows = (
        df["low"]
        .to_numpy()
    )

    ph = []
    pl = []

    for i in range(
        left,
        len(df) - right,
    ):

        high_window = highs[
            i - left:
            i + right + 1
        ]

        low_window = lows[
            i - left:
            i + right + 1
        ]

        if (
            highs[i]
            >= high_window.max()
        ):
            ph.append(
                highs[i]
            )

        if (
            lows[i]
            <= low_window.min()
        ):
            pl.append(
                lows[i]
            )

    out = {}

    if ph:
        out["PSH"] = float(
            ph[-1]
        )

    if pl:
        out["PSL"] = float(
            pl[-1]
        )

    return out


# =========================================================
# LIQUIDITY EVENT DETECTION
# =========================================================

def detect_event(
    df: pd.DataFrame,
    levels: dict[str, float],
    lookback: int = 3,
):

    """
    sweep_high:
        wick above liquidity,
        candle closes back below.

    sweep_low:
        wick below liquidity,
        candle closes back above.

    run_high:
        candle closes above liquidity.

    run_low:
        candle closes below liquidity.

    Only CLOSED candles are inspected.
    """

    if len(df) < 3:
        return []

    closed = df.iloc[:-1]

    recent = (
        closed
        .iloc[-lookback:]
    )

    events = []

    for (
        name,
        level,
    ) in levels.items():

        level = float(
            level
        )

        is_high = (
            name.endswith("H")
        )

        for (
            idx,
            row,
        ) in recent.iterrows():

            if is_high:

                # LIQUIDITY SWEEP HIGH
                if (
                    row["high"] > level
                    and row["close"] < level
                ):

                    events.append({
                        "type": "sweep_high",
                        "level": name,
                        "price": level,
                        "bar": idx,
                    })

                # LIQUIDITY RUN HIGH
                elif (
                    row["close"] > level
                ):

                    events.append({
                        "type": "run_high",
                        "level": name,
                        "price": level,
                        "bar": idx,
                    })

            else:

                # LIQUIDITY SWEEP LOW
                if (
                    row["low"] < level
                    and row["close"] > level
                ):

                    events.append({
                        "type": "sweep_low",
                        "level": name,
                        "price": level,
                        "bar": idx,
                    })

                # LIQUIDITY RUN LOW
                elif (
                    row["close"] < level
                ):

                    events.append({
                        "type": "run_low",
                        "level": name,
                        "price": level,
                        "bar": idx,
                    })

    return events


# =========================================================
# TARGETS
# =========================================================

def nearest_targets(
    levels: dict[str, float],
    entry: float,
    side: str,
    max_targets: int = 4,
):

    vals = []

    for (
        name,
        price,
    ) in levels.items():

        price = float(
            price
        )

        if (
            side == "LONG"
            and price > entry
        ):

            vals.append(
                (
                    price,
                    name,
                )
            )

        elif (
            side == "SHORT"
            and price < entry
        ):

            vals.append(
                (
                    price,
                    name,
                )
            )

    vals.sort(
        key=lambda x: x[0],
        reverse=(
            side == "SHORT"
        ),
    )

    return vals[
        :max_targets
    ]
