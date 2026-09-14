from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from backtest.download_binance import download_klines


START = "2026-03-01"
END = "2026-09-01"

SIGNALS_PATH = Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")

LONG_MIN_SCORE = 70
SHORT_MIN_SCORE = 88

ATR_LEN = 55
FVG_LOOKBACK = 80

THRESHOLDS = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def atr_series(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
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


def find_active_fvgs(df: pd.DataFrame, lookback: int = FVG_LOOKBACK) -> list[dict]:
    """
    Mirrors the current Trade Vision FVG logic closely enough for
    metadata/post-filter research.

    IMPORTANT:
    df contains only candles available at signal time.
    All rows in df are treated as closed candles here.
    """
    d = df.copy().reset_index(drop=True)

    if len(d) < 5:
        return []

    open_ = pd.to_numeric(d["open"], errors="coerce")
    close = pd.to_numeric(d["close"], errors="coerce")

    body_delta = (close - open_) / open_.replace(0, math.nan)

    threshold = (
        body_delta.abs()
        .expanding(min_periods=3)
        .mean()
        * 2.0
    )

    start = max(2, len(d) - lookback)

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

        delta = (middle_close - middle_open) / middle_open
        dyn = float(threshold.iloc[i - 1])

        if math.isnan(dyn):
            dyn = 0.0

        bullish = (
            right_low > left_high
            and middle_close > left_high
            and delta > dyn
        )

        if bullish:
            low = left_high
            high = right_low
            future = d.iloc[i + 1:]

            invalidated = (
                not future.empty
                and float(future["low"].min()) < low
            )

            if not invalidated:
                fvgs.append(
                    {
                        "side": "LONG",
                        "low": low,
                        "high": high,
                        "mid": (low + high) / 2.0,
                        "index": i,
                        "time": pd.Timestamp(right["time"]),
                    }
                )

        bearish = (
            right_high < left_low
            and middle_close < left_low
            and (-delta) > dyn
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
                        "low": low,
                        "high": high,
                        "mid": (low + high) / 2.0,
                        "index": i,
                        "time": pd.Timestamp(right["time"]),
                    }
                )

    return fvgs


def choose_fvg_for_entry(
    signal_row: pd.Series,
    candles_to_signal: pd.DataFrame,
) -> dict | None:
    side = str(signal_row["side"])
    entry = float(signal_row["entry"])

    fvgs = [
        fvg
        for fvg in find_active_fvgs(candles_to_signal)
        if fvg["side"] == side
    ]

    if not fvgs:
        return None

    # Current FVG 50% entry is the midpoint, so the FVG whose midpoint is
    # closest to the backtest entry is the most defensible association.
    fvgs.sort(key=lambda x: abs(float(x["mid"]) - entry))
    return fvgs[0]


def mitigation_state_before_entry(
    fvg: dict,
    candles: pd.DataFrame,
    signal_time: pd.Timestamp,
    entry_time: pd.Timestamp | pd.NaT,
) -> str:
    if pd.isna(entry_time):
        end_time = signal_time + pd.Timedelta(hours=12)
    else:
        end_time = entry_time

    future = candles[
        (candles["time"] > signal_time)
        & (candles["time"] <= end_time)
    ]

    if future.empty:
        return "UNTOUCHED"

    low = float(fvg["low"])
    high = float(fvg["high"])
    mid = float(fvg["mid"])
    side = fvg["side"]

    if side == "LONG":
        min_low = float(future["low"].min())

        if min_low < low:
            return "FULL_FILL"
        if min_low <= mid:
            return "HALF_FILL"
        if min_low <= high:
            return "TOUCH"
        return "UNTOUCHED"

    max_high = float(future["high"].max())

    if max_high > high:
        return "FULL_FILL"
    if max_high >= mid:
        return "HALF_FILL"
    if max_high >= low:
        return "TOUCH"
    return "UNTOUCHED"


def production_score_ok(row: pd.Series) -> bool:
    side = str(row["side"])
    score = int(row["score"])

    if side == "SHORT":
        return score >= SHORT_MIN_SCORE

    return score >= LONG_MIN_SCORE


def summarize(df: pd.DataFrame, label: str) -> dict:
    closed = df[df["status"] == "CLOSED"].copy()

    if closed.empty:
        return {
            "filter": label,
            "setups": len(df),
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
        }

    wins = int((closed["realized_r"] > 0).sum())
    losses = int((closed["realized_r"] < 0).sum())
    total_r = float(closed["realized_r"].sum())

    return {
        "filter": label,
        "setups": len(df),
        "closed": len(closed),
        "wins": wins,
        "losses": losses,
        "winrate": 100.0 * wins / len(closed),
        "total_r": total_r,
        "avg_r": total_r / len(closed),
    }


def main() -> None:
    if not SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SIGNALS_PATH}. Run from repository root."
        )

    print("Downloading BTCUSDT 15M Binance archive...")
    candles = download_klines(
        "BTCUSDT",
        "15m",
        START,
        END,
    ).copy()

    if candles.empty:
        raise RuntimeError("No BTCUSDT historical candles downloaded.")

    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    candles["atr55"] = atr_series(candles)

    signals = pd.read_csv(SIGNALS_PATH)

    signals["signal_time"] = pd.to_datetime(
        signals["signal_time"],
        utc=True,
        errors="coerce",
    )
    signals["entry_time"] = pd.to_datetime(
        signals["entry_time"],
        utc=True,
        errors="coerce",
    )

    for col in ("score", "entry", "sl", "realized_r"):
        signals[col] = pd.to_numeric(
            signals[col],
            errors="coerce",
        )

    # Match CURRENT production score gate:
    # LONG >=70 / SHORT >=88
    signals = signals[
        signals.apply(production_score_ok, axis=1)
    ].copy()

    fvg_mask = signals["entry_type"].astype(str).str.contains(
        "FVG",
        regex=False,
        na=False,
    )

    fvg_signals = signals[fvg_mask].copy()

    print()
    print(
        f"Production-gated setups: {len(signals)} | "
        f"FVG-related setups: {len(fvg_signals)}"
    )

    rows: list[dict] = []

    for idx, sig in fvg_signals.iterrows():
        signal_time = sig["signal_time"]

        # Candles available by the signal timestamp.
        hist = candles[candles["time"] <= signal_time].copy()

        if len(hist) < 60:
            continue

        fvg = choose_fvg_for_entry(sig, hist)

        if fvg is None:
            continue

        atr_candidates = hist[
            hist["time"] <= pd.Timestamp(fvg["time"])
        ]

        if atr_candidates.empty:
            continue

        atr = float(atr_candidates.iloc[-1]["atr55"])

        if not math.isfinite(atr) or atr <= 0:
            continue

        size = float(fvg["high"]) - float(fvg["low"])
        ratio = size / atr

        state = mitigation_state_before_entry(
            fvg=fvg,
            candles=candles,
            signal_time=signal_time,
            entry_time=sig["entry_time"],
        )

        rows.append(
            {
                "source_index": idx,
                "side": sig["side"],
                "score": int(sig["score"]),
                "signal_time": signal_time,
                "entry_time": sig["entry_time"],
                "entry_type": sig["entry_type"],
                "status": sig["status"],
                "realized_r": float(sig["realized_r"]),
                "fvg_low": float(fvg["low"]),
                "fvg_high": float(fvg["high"]),
                "fvg_mid": float(fvg["mid"]),
                "fvg_size": size,
                "atr55": atr,
                "fvg_atr_ratio": ratio,
                "mitigation_state": state,
                "entry_mid_error_pct": (
                    abs(float(fvg["mid"]) - float(sig["entry"]))
                    / float(sig["entry"])
                    * 100.0
                ),
            }
        )

    meta = pd.DataFrame(rows)

    if meta.empty:
        raise RuntimeError(
            "No FVG metadata could be matched to backtest signals."
        )

    # Association sanity check. If nearest FVG midpoint is too far from
    # recorded FVG entry, do not use it in threshold research.
    matched = meta[
        meta["entry_mid_error_pct"] <= 0.35
    ].copy()

    print(
        f"Matched FVG entries: {len(matched)}/{len(meta)} "
        f"(midpoint error <= 0.35%)"
    )

    out_dir = Path("backtest/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "btc_fvg_quality_metadata.csv"
    matched.to_csv(meta_path, index=False)

    summaries: list[dict] = []

    # Existing strategy baseline under production score gate.
    summaries.append(
        summarize(
            signals,
            "CURRENT PRODUCTION GATE",
        )
    )

    summaries.append(
        summarize(
            fvg_signals,
            "ALL FVG-RELATED",
        )
    )

    # FVG ATR threshold sweep.
    for threshold in THRESHOLDS:
        ids = set(
            matched.loc[
                matched["fvg_atr_ratio"] >= threshold,
                "source_index",
            ].tolist()
        )

        subset = fvg_signals[
            fvg_signals.index.isin(ids)
        ]

        summaries.append(
            summarize(
                subset,
                f"FVG/ATR >= {threshold:.2f}",
            )
        )

    # Mitigation state study.
    for state in [
        "UNTOUCHED",
        "TOUCH",
        "HALF_FILL",
        "FULL_FILL",
    ]:
        ids = set(
            matched.loc[
                matched["mitigation_state"] == state,
                "source_index",
            ].tolist()
        )

        subset = fvg_signals[
            fvg_signals.index.isin(ids)
        ]

        summaries.append(
            summarize(
                subset,
                f"STATE = {state}",
            )
        )

    summary_df = pd.DataFrame(summaries)

    summary_path = out_dir / "btc_fvg_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print()
    print("=== BTC FVG QUALITY TEST ===")
    print(summary_df.to_string(index=False))

    print()
    print("FVG/ATR distribution:")
    print(
        matched["fvg_atr_ratio"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_string()
    )

    print()
    print("Saved:")
    print(meta_path)
    print(summary_path)


if __name__ == "__main__":
    main()
