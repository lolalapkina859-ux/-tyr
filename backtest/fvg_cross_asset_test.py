from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines
from backtest.fvg_quality_test import (
    atr_series,
    choose_fvg_for_entry,
    mitigation_state_before_entry,
    production_score_ok,
    summarize,
)

START = "2026-03-01"
END = "2026-09-01"

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

THRESHOLDS = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def run_asset(asset: str, symbol: str, signals_path: Path):
    if not signals_path.exists():
        raise FileNotFoundError(f"Missing baseline: {signals_path}")

    print(f"\n=== {asset}: downloading {symbol} 15M ===")
    candles = download_klines(symbol, "15m", START, END).copy()
    if candles.empty:
        raise RuntimeError(f"No candles for {symbol}")

    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    candles["atr55"] = atr_series(candles)

    signals = pd.read_csv(signals_path)
    signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True, errors="coerce")
    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=True, errors="coerce")

    for col in ("score", "entry", "sl", "realized_r"):
        signals[col] = pd.to_numeric(signals[col], errors="coerce")

    signals = signals[signals.apply(production_score_ok, axis=1)].copy()

    fvg_mask = signals["entry_type"].astype(str).str.contains("FVG", regex=False, na=False)
    fvg_signals = signals[fvg_mask].copy()

    rows = []

    for idx, sig in fvg_signals.iterrows():
        hist = candles[candles["time"] <= sig["signal_time"]].copy()
        if len(hist) < 60:
            continue

        fvg = choose_fvg_for_entry(sig, hist)
        if fvg is None:
            continue

        atr_candidates = hist[hist["time"] <= pd.Timestamp(fvg["time"])]
        if atr_candidates.empty:
            continue

        atr = float(atr_candidates.iloc[-1]["atr55"])
        if pd.isna(atr) or atr <= 0:
            continue

        size = float(fvg["high"]) - float(fvg["low"])
        ratio = size / atr

        state = mitigation_state_before_entry(
            fvg=fvg,
            candles=candles,
            signal_time=sig["signal_time"],
            entry_time=sig["entry_time"],
        )

        rows.append({
            "asset": asset,
            "source_index": idx,
            "side": sig["side"],
            "score": int(sig["score"]),
            "signal_time": sig["signal_time"],
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
            "entry_mid_error_pct": abs(float(fvg["mid"]) - float(sig["entry"])) / float(sig["entry"]) * 100.0,
        })

    meta = pd.DataFrame(rows)
    matched = meta[meta["entry_mid_error_pct"] <= 0.35].copy() if not meta.empty else meta.copy()

    summaries = []
    s = summarize(signals, "CURRENT PRODUCTION GATE")
    s["asset"] = asset
    summaries.append(s)

    s = summarize(fvg_signals, "ALL FVG-RELATED")
    s["asset"] = asset
    summaries.append(s)

    for threshold in THRESHOLDS:
        ids = set(matched.loc[matched["fvg_atr_ratio"] >= threshold, "source_index"].tolist())
        subset = fvg_signals[fvg_signals.index.isin(ids)]
        s = summarize(subset, f"FVG/ATR >= {threshold:.2f}")
        s["asset"] = asset
        summaries.append(s)

    for state in ["UNTOUCHED", "TOUCH", "HALF_FILL", "FULL_FILL"]:
        ids = set(matched.loc[matched["mitigation_state"] == state, "source_index"].tolist())
        subset = fvg_signals[fvg_signals.index.isin(ids)]
        s = summarize(subset, f"STATE = {state}")
        s["asset"] = asset
        summaries.append(s)

    print(f"{asset}: production setups={len(signals)}, FVG setups={len(fvg_signals)}, matched={len(matched)}/{len(meta)}")
    return signals, fvg_signals, matched, pd.DataFrame(summaries)


def combined_summary(all_fvg, all_meta):
    rows = []

    def summarize_combined(df, label):
        closed = df[df["status"] == "CLOSED"].copy()
        wins = int((closed["realized_r"] > 0).sum()) if len(closed) else 0
        losses = int((closed["realized_r"] < 0).sum()) if len(closed) else 0
        total_r = float(closed["realized_r"].sum()) if len(closed) else 0.0
        return {
            "asset": "ALL",
            "filter": label,
            "setups": len(df),
            "closed": len(closed),
            "wins": wins,
            "losses": losses,
            "winrate": (100.0 * wins / len(closed)) if len(closed) else 0.0,
            "total_r": total_r,
            "avg_r": (total_r / len(closed)) if len(closed) else 0.0,
        }

    rows.append(summarize_combined(all_fvg, "ALL FVG-RELATED"))

    for threshold in THRESHOLDS:
        keys = set(
            zip(
                all_meta.loc[all_meta["fvg_atr_ratio"] >= threshold, "asset"],
                all_meta.loc[all_meta["fvg_atr_ratio"] >= threshold, "source_index"],
            )
        )
        mask = [(a, i) in keys for a, i in zip(all_fvg["asset"], all_fvg["source_index"])]
        rows.append(summarize_combined(all_fvg[mask], f"FVG/ATR >= {threshold:.2f}"))

    for state in ["UNTOUCHED", "TOUCH", "HALF_FILL", "FULL_FILL"]:
        keys = set(
            zip(
                all_meta.loc[all_meta["mitigation_state"] == state, "asset"],
                all_meta.loc[all_meta["mitigation_state"] == state, "source_index"],
            )
        )
        mask = [(a, i) in keys for a, i in zip(all_fvg["asset"], all_fvg["source_index"])]
        rows.append(summarize_combined(all_fvg[mask], f"STATE = {state}"))

    return pd.DataFrame(rows)


def main():
    out_dir = Path("backtest/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    all_summaries = []
    all_fvg = []

    for asset, (symbol, path) in ASSETS.items():
        signals, fvg_signals, meta, summary = run_asset(asset, symbol, path)

        if not meta.empty:
            meta.to_csv(out_dir / f"{asset.lower()}_fvg_quality_metadata.csv", index=False)
            all_meta.append(meta)

        fvg_copy = fvg_signals.copy()
        fvg_copy["asset"] = asset
        fvg_copy["source_index"] = fvg_copy.index
        all_fvg.append(fvg_copy)
        all_summaries.append(summary)

    meta_all = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    fvg_all = pd.concat(all_fvg, ignore_index=True)
    per_asset = pd.concat(all_summaries, ignore_index=True)

    combined = combined_summary(fvg_all, meta_all)

    final = pd.concat([per_asset, combined], ignore_index=True)
    final = final[["asset", "filter", "setups", "closed", "wins", "losses", "winrate", "total_r", "avg_r"]]

    final.to_csv(out_dir / "cross_asset_fvg_quality_summary.csv", index=False)
    meta_all.to_csv(out_dir / "cross_asset_fvg_quality_metadata.csv", index=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)

    print("\n=== CROSS-ASSET FVG QUALITY ===")
    print(final.to_string(index=False))
    print("\nSaved:")
    print(out_dir / "cross_asset_fvg_quality_summary.csv")
    print(out_dir / "cross_asset_fvg_quality_metadata.csv")


if __name__ == "__main__":
    main()
