from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines

START = "2026-03-01"
END = "2026-09-01"
PIVOT_LEN = 5
RMA_LEN = 20
SCORE_WINDOW = 40

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

BIAS_THRESHOLDS = [0.00, -0.005, -0.01, -0.02, -0.03, -0.05]
SCORE_THRESHOLDS = [0, -1, -2, -3, -5]


def production_gate(row: pd.Series) -> bool:
    side = str(row["side"])
    score = int(row["score"])
    return score >= (88 if side == "SHORT" else 70)


def _rma(values: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return values.ewm(alpha=alpha, adjust=False).mean()


def build_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    n = len(d)
    bull_event = [0] * n
    bear_event = [0] * n
    event_weight = [0] * n

    last_hi = None
    last_lo = None
    os = 0  # 1 bullish structure, -1 bearish structure

    highs = d["high"].tolist()
    lows = d["low"].tolist()
    closes = d["close"].tolist()

    for i in range(n):
        # Confirm pivot at j only after PIVOT_LEN bars to the right exist.
        j = i - PIVOT_LEN
        if j >= PIVOT_LEN:
            lo = j - PIVOT_LEN
            hi = j + PIVOT_LEN + 1
            if hi <= n:
                hwin = highs[lo:hi]
                lwin = lows[lo:hi]
                if highs[j] == max(hwin):
                    last_hi = highs[j]
                if lows[j] == min(lwin):
                    last_lo = lows[j]

        prev_close = closes[i - 1] if i > 0 else closes[i]
        bull = last_hi is not None and closes[i] > last_hi and prev_close <= last_hi
        bear = last_lo is not None and closes[i] < last_lo and prev_close >= last_lo

        if bull:
            bull_event[i] = 1
            # Zeiierman logic: first reversal event is ChoCH=1, continuation BOS=3.
            w = 1 if os == -1 else 3
            event_weight[i] = w
            os = 1
        elif bear:
            bear_event[i] = 1
            w = 1 if os == 1 else 3
            event_weight[i] = -w
            os = -1

    imp = pd.Series([b - s for b, s in zip(bull_event, bear_event)], dtype=float)
    weighted = pd.Series(event_weight, dtype=float)

    d["bos_imp"] = imp
    d["bos_activity"] = _rma(imp.abs(), RMA_LEN)
    d["bos_bias"] = _rma(imp, RMA_LEN)
    d["structure_score_40"] = weighted.rolling(SCORE_WINDOW, min_periods=1).sum()
    d["bull_events_40"] = pd.Series(bull_event).rolling(SCORE_WINDOW, min_periods=1).sum()
    d["bear_events_40"] = pd.Series(bear_event).rolling(SCORE_WINDOW, min_periods=1).sum()
    return d


def attach_features(signals: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    sig = signals.copy().sort_values("signal_time")
    feat = features[[
        "time", "bos_bias", "bos_activity", "structure_score_40",
        "bull_events_40", "bear_events_40"
    ]].copy().sort_values("time")

    # Most recent fully available 15M state at or before signal timestamp.
    return pd.merge_asof(
        sig,
        feat,
        left_on="signal_time",
        right_on="time",
        direction="backward",
    )


def summarize(df: pd.DataFrame, asset: str, label: str) -> dict:
    closed = df[df["status"] == "CLOSED"].copy()
    wins = int((closed["realized_r"] > 0).sum()) if len(closed) else 0
    losses = int((closed["realized_r"] < 0).sum()) if len(closed) else 0
    total_r = float(closed["realized_r"].sum()) if len(closed) else 0.0
    return {
        "asset": asset,
        "filter": label,
        "setups": len(df),
        "closed": len(closed),
        "wins": wins,
        "losses": losses,
        "winrate": (100.0 * wins / len(closed)) if len(closed) else 0.0,
        "total_r": total_r,
        "avg_r": (total_r / len(closed)) if len(closed) else 0.0,
        "short_closed": int(((closed["side"] == "SHORT")).sum()) if len(closed) else 0,
        "short_total_r": float(closed.loc[closed["side"] == "SHORT", "realized_r"].sum()) if len(closed) else 0.0,
    }


def run_asset(asset: str, symbol: str, path: Path):
    signals = pd.read_csv(path)
    signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True, errors="coerce")
    for c in ("score", "realized_r"):
        signals[c] = pd.to_numeric(signals[c], errors="coerce")
    signals = signals[signals.apply(production_gate, axis=1)].copy()

    candles = download_klines(symbol, "15m", START, END)
    features = build_structure_features(candles)
    x = attach_features(signals, features)
    x["asset"] = asset

    rows = [summarize(x, asset, "BASELINE")]

    for th in BIAS_THRESHOLDS:
        keep = (x["side"] != "SHORT") | (x["bos_bias"] <= th)
        rows.append(summarize(x[keep], asset, f"SHORT bos_bias <= {th:.3f}"))

    for th in SCORE_THRESHOLDS:
        keep = (x["side"] != "SHORT") | (x["structure_score_40"] <= th)
        rows.append(summarize(x[keep], asset, f"SHORT structure_score_40 <= {th}"))

    # Combined condition candidates: bearish bias + net bearish event score.
    combos = [
        (-0.005, 0),
        (-0.01, -1),
        (-0.02, -2),
        (-0.03, -3),
    ]
    for bth, sth in combos:
        keep_short = (x["bos_bias"] <= bth) & (x["structure_score_40"] <= sth)
        keep = (x["side"] != "SHORT") | keep_short
        rows.append(summarize(x[keep], asset, f"SHORT bias<={bth:.3f} & score<={sth}"))

    return x, pd.DataFrame(rows)


def main():
    out = Path("backtest/data")
    out.mkdir(parents=True, exist_ok=True)

    details = []
    summaries = []
    for asset, (symbol, path) in ASSETS.items():
        detail, summary = run_asset(asset, symbol, path)
        details.append(detail)
        summaries.append(summary)

    detail_all = pd.concat(details, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)

    # Build cross-asset summary by applying each exact filter to the combined detail.
    labels = summary_all["filter"].drop_duplicates().tolist()
    combined_rows = []
    for label in labels:
        if label == "BASELINE":
            sub = detail_all
        elif label.startswith("SHORT bos_bias <="):
            th = float(label.split("<=")[1].strip())
            sub = detail_all[(detail_all["side"] != "SHORT") | (detail_all["bos_bias"] <= th)]
        elif label.startswith("SHORT structure_score_40 <="):
            th = float(label.split("<=")[1].strip())
            sub = detail_all[(detail_all["side"] != "SHORT") | (detail_all["structure_score_40"] <= th)]
        else:
            left, right = label.replace("SHORT ", "").split(" & ")
            bth = float(left.split("<=")[1])
            sth = float(right.split("<=")[1])
            keep_short = (detail_all["bos_bias"] <= bth) & (detail_all["structure_score_40"] <= sth)
            sub = detail_all[(detail_all["side"] != "SHORT") | keep_short]
        combined_rows.append(summarize(sub, "ALL", label))

    final = pd.concat([summary_all, pd.DataFrame(combined_rows)], ignore_index=True)
    base = final[(final["asset"] == "ALL") & (final["filter"] == "BASELINE")].iloc[0]
    final["delta_total_r_vs_all_base"] = final["total_r"] - float(base["total_r"])

    detail_all.to_csv(out / "structure_strength_short_filter_detail.csv", index=False)
    final.to_csv(out / "structure_strength_short_filter_summary.csv", index=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print("\n=== STRUCTURE STRENGTH SHORT FILTER TEST ===")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
