from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

from backtest.download_binance import download_klines

START = "2026-03-01"
END = "2026-09-01"

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

TARGET_LEN = 3
MEMORY = 150
TOP_N = 20
THRESHOLDS = (-0.25, 0.0, 0.25, 0.50)


def production_gate(row: pd.Series) -> bool:
    score = int(row["score"])
    return score >= (88 if str(row["side"]) == "SHORT" else 70)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def norm100(x: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi == lo:
        return pd.Series(50.0, index=x.index)
    return (((x - lo) / (hi - lo)) * 100.0).clip(0, 100)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    c = d["close"].astype(float)
    a = atr(d, 14).replace(0, np.nan)
    ef = ema(c, 20)
    es = ema(c, 100)
    rr = rsi(c, 14)

    trend_spread = (ef - es) / a
    trend_slope = (ef - ef.shift(1)) / a
    trend_score = norm100(trend_spread + trend_slope, -2.5, 2.5)

    basis = c.rolling(20).mean()
    dev = c.rolling(20).std().replace(0, np.nan)
    z = (c - basis) / dev
    mean_score = (100 - rr) * 0.5 + norm100(-z, -2.5, 2.5) * 0.5

    roc = c / c.shift(20) - 1.0
    momentum_score = (
        norm100(roc, -0.05, 0.05) * 0.45
        + rr * 0.35
        + norm100((ef - ef.shift(1)) / a, -0.5, 0.5) * 0.20
    )

    d["trend_f"] = (trend_score - 50.0) / 50.0
    d["mean_f"] = (mean_score - 50.0) / 50.0
    d["mom_f"] = (momentum_score - 50.0) / 50.0
    d["atr_pct"] = (a / c).replace([np.inf, -np.inf], np.nan)

    fwd = c.shift(-TARGET_LEN) / c - 1.0
    d["target_dir"] = np.sign(fwd)
    d["sample_strength"] = (fwd.abs() / d["atr_pct"].clip(lower=1e-8)).replace([np.inf, -np.inf], np.nan)
    return d


def causal_prediction(feat: pd.DataFrame, i: int) -> float | None:
    # Only samples whose TARGET_LEN-bar outcome is already known at bar i.
    end = i - TARGET_LEN
    if end <= 100:
        return None

    start = max(0, end - MEMORY + 1)
    hist = feat.iloc[start:end + 1].copy()
    hist = hist.dropna(subset=["trend_f", "mean_f", "mom_f", "target_dir", "sample_strength"])
    if len(hist) < 20:
        return None

    hist = hist.nlargest(min(TOP_N, len(hist)), "sample_strength")
    X = hist[["trend_f", "mean_f", "mom_f"]].to_numpy(dtype=float)
    y = hist["target_dir"].to_numpy(dtype=float)

    X1 = np.column_stack([X, np.ones(len(X))])
    # Small ridge stabilization; bias not penalized materially.
    reg = np.diag([0.05, 0.05, 0.05, 0.001])
    try:
        beta = np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)
    except np.linalg.LinAlgError:
        return None

    cur = feat.iloc[i]
    if cur[["trend_f", "mean_f", "mom_f"]].isna().any():
        return None

    x = np.array([cur["trend_f"], cur["mean_f"], cur["mom_f"], 1.0], dtype=float)
    return float(np.tanh(x @ beta))


def summarize(df: pd.DataFrame, asset: str, label: str) -> dict:
    closed = df[df["status"] == "CLOSED"].copy()
    wins = int((closed["realized_r"] > 0).sum()) if len(closed) else 0
    losses = int((closed["realized_r"] < 0).sum()) if len(closed) else 0
    total_r = float(closed["realized_r"].sum()) if len(closed) else 0.0
    shorts = closed[closed["side"] == "SHORT"]
    longs = closed[closed["side"] == "LONG"]
    return {
        "asset": asset,
        "filter": label,
        "closed": len(closed),
        "wins": wins,
        "losses": losses,
        "winrate": round(100*wins/len(closed), 2) if len(closed) else 0.0,
        "total_r": round(total_r, 4),
        "avg_r": round(total_r/len(closed), 4) if len(closed) else 0.0,
        "long_closed": len(longs),
        "long_r": round(float(longs["realized_r"].sum()), 4) if len(longs) else 0.0,
        "short_closed": len(shorts),
        "short_r": round(float(shorts["realized_r"].sum()), 4) if len(shorts) else 0.0,
    }


def period_label(ts: pd.Timestamp) -> str:
    return "MAR_MAY" if ts.month <= 5 else "JUN_AUG"


def run_asset(asset: str, symbol: str, path: Path):
    candles = download_klines(symbol, "15m", START, END).copy()
    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    feat = build_features(candles)

    sig = pd.read_csv(path)
    sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True, errors="coerce")
    sig["realized_r"] = pd.to_numeric(sig["realized_r"], errors="coerce")
    sig["score"] = pd.to_numeric(sig["score"], errors="coerce")
    sig = sig[sig.apply(production_gate, axis=1)].copy()

    preds = []
    for _, row in sig.iterrows():
        t = row["signal_time"]
        if pd.isna(t):
            preds.append(np.nan)
            continue
        idxs = candles.index[candles["time"] <= t]
        if len(idxs) == 0:
            preds.append(np.nan)
            continue
        i = int(idxs[-1])
        preds.append(causal_prediction(feat, i))

    sig["adaptive_pred"] = preds
    sig["period"] = sig["signal_time"].apply(period_label)
    sig["asset"] = asset

    rows = [summarize(sig, asset, "BASELINE")]

    for th in THRESHOLDS:
        keep = sig.copy()
        aligned = np.where(keep["side"] == "LONG", keep["adaptive_pred"], -keep["adaptive_pred"])
        keep = keep[(pd.Series(aligned, index=keep.index) >= th) | keep["adaptive_pred"].isna()].copy()
        rows.append(summarize(keep, asset, f"ADAPTIVE >= {th:+.2f}"))

    # SHORT-only version: leave LONG untouched, filter only SHORT.
    for th in (0.0, 0.25, 0.50):
        aligned_short = -sig["adaptive_pred"]
        mask = (sig["side"] != "SHORT") | (aligned_short >= th) | sig["adaptive_pred"].isna()
        keep = sig[mask].copy()
        rows.append(summarize(keep, asset, f"SHORT ADAPTIVE >= {th:+.2f}"))

    return sig, pd.DataFrame(rows)


def main():
    out = Path("backtest/data")
    out.mkdir(parents=True, exist_ok=True)
    details = []
    sums = []

    for asset, (symbol, path) in ASSETS.items():
        d, s = run_asset(asset, symbol, path)
        details.append(d)
        sums.append(s)

    detail = pd.concat(details, ignore_index=True)
    summary = pd.concat(sums, ignore_index=True)

    # Combined summaries by same filter.
    combined = []
    for label in summary["filter"].unique():
        # Recreate filter on combined detail.
        if label == "BASELINE":
            sub = detail.copy()
        elif label.startswith("SHORT ADAPTIVE"):
            th = float(label.split()[-1])
            mask = (detail["side"] != "SHORT") | (-detail["adaptive_pred"] >= th) | detail["adaptive_pred"].isna()
            sub = detail[mask].copy()
        else:
            th = float(label.split()[-1])
            aligned = np.where(detail["side"] == "LONG", detail["adaptive_pred"], -detail["adaptive_pred"])
            sub = detail[(pd.Series(aligned, index=detail.index) >= th) | detail["adaptive_pred"].isna()].copy()
        combined.append(summarize(sub, "ALL", label))

    summary = pd.concat([summary, pd.DataFrame(combined)], ignore_index=True)

    # Split-sample rows for ALL.
    split_rows = []
    for period in ("MAR_MAY", "JUN_AUG"):
        p = detail[detail["period"] == period].copy()
        split_rows.append(summarize(p, f"ALL_{period}", "BASELINE"))
        for th in (0.0, 0.25, 0.50):
            aligned = np.where(p["side"] == "LONG", p["adaptive_pred"], -p["adaptive_pred"])
            sub = p[(pd.Series(aligned, index=p.index) >= th) | p["adaptive_pred"].isna()].copy()
            split_rows.append(summarize(sub, f"ALL_{period}", f"ADAPTIVE >= {th:+.2f}"))
            mask = (p["side"] != "SHORT") | (-p["adaptive_pred"] >= th) | p["adaptive_pred"].isna()
            split_rows.append(summarize(p[mask], f"ALL_{period}", f"SHORT ADAPTIVE >= {th:+.2f}"))

    summary = pd.concat([summary, pd.DataFrame(split_rows)], ignore_index=True)
    summary.to_csv(out / "adaptive_momentum_filter_summary.csv", index=False)
    detail.to_csv(out / "adaptive_momentum_filter_detail.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
