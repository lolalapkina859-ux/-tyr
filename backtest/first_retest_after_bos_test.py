from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from backtest.download_binance import download_klines
from backtest.fvg_quality_test import choose_fvg_for_entry, production_score_ok
from backtest.fvg_depth_cross_asset_test import _simulate_trade, _targets, _entry_for_depth

START = "2026-03-01"
END = "2026-09-01"
STRUCTURE_WINDOW = 12
BOS_SEARCH_BARS = 64

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}


def _latest_directional_bos(hist: pd.DataFrame, side: str) -> dict | None:
    d = hist.copy().reset_index(drop=True)
    if len(d) < STRUCTURE_WINDOW + 2:
        return None

    start = max(STRUCTURE_WINDOW, len(d) - BOS_SEARCH_BARS)
    found = None

    for i in range(start, len(d)):
        prev = d.iloc[i - STRUCTURE_WINDOW:i]
        if prev.empty:
            continue

        c = d.iloc[i]
        close = float(c["close"])
        prev_close = float(d.iloc[i - 1]["close"])
        prev_high = float(prev["high"].max())
        prev_low = float(prev["low"].min())

        if side == "LONG":
            trigger = close > prev_high and prev_close <= prev_high
            level = prev_high
        else:
            trigger = close < prev_low and prev_close >= prev_low
            level = prev_low

        if trigger:
            found = {
                "index": i,
                "time": c["time"],
                "level": level,
            }

    return found


def _bos_linked_fvg(hist: pd.DataFrame, side: str, bos: dict) -> dict | None:
    """Find a fresh three-candle FVG formed by the BOS impulse itself or within
    the next three CLOSED candles. Uses only data available at signal time.
    """
    d = hist.copy().reset_index(drop=True)
    bi = int(bos["index"])
    end = min(len(d) - 1, bi + 3)

    candidates: list[dict] = []
    for i in range(max(2, bi), end + 1):
        left = d.iloc[i - 2]
        mid = d.iloc[i - 1]
        right = d.iloc[i]

        if side == "LONG":
            valid = (
                float(right["low"]) > float(left["high"])
                and float(mid["close"]) > float(left["high"])
            )
            if valid:
                low = float(left["high"])
                high = float(right["low"])
                future = d.iloc[i + 1:]
                invalid = (not future.empty and float(future["low"].min()) < low)
                if not invalid:
                    candidates.append({"low": low, "high": high, "index": i, "time": right["time"]})
        else:
            valid = (
                float(right["high"]) < float(left["low"])
                and float(mid["close"]) < float(left["low"])
            )
            if valid:
                low = float(right["high"])
                high = float(left["low"])
                future = d.iloc[i + 1:]
                invalid = (not future.empty and float(future["high"].max()) > high)
                if not invalid:
                    candidates.append({"low": low, "high": high, "index": i, "time": right["time"]})

    return candidates[-1] if candidates else None


def _summary(df: pd.DataFrame, asset: str, mode: str) -> dict:
    sub = df[df["mode"] == mode].copy()
    filled = sub[sub["entry_time"].notna() & (sub["status"] != "AMBIGUOUS")]
    closed = sub[sub["status"] == "CLOSED"]
    wins = closed[closed["highest_tp"] >= 1]
    losses = closed[closed["highest_tp"] == 0]
    total_r = float(closed["realized_r"].sum()) if not closed.empty else 0.0

    return {
        "asset": asset,
        "mode": mode,
        "setups": len(sub),
        "filled": len(filled),
        "closed": len(closed),
        "expired": int((sub["status"] == "EXPIRED").sum()),
        "ambiguous": int((sub["status"] == "AMBIGUOUS").sum()),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(100.0 * len(wins) / len(closed), 2) if len(closed) else 0.0,
        "total_r": round(total_r, 4),
        "avg_r": round(total_r / len(closed), 4) if len(closed) else 0.0,
        "tp1": int((closed["highest_tp"] >= 1).sum()) if len(closed) else 0,
        "tp2": int((closed["highest_tp"] >= 2).sum()) if len(closed) else 0,
        "tp3": int((closed["highest_tp"] >= 3).sum()) if len(closed) else 0,
        "tp4": int((closed["highest_tp"] >= 4).sum()) if len(closed) else 0,
    }


def run_asset(asset: str, symbol: str, baseline_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candles = download_klines(symbol, "15m", START, END).copy()
    if candles.empty:
        raise RuntimeError(f"No candles for {symbol}")
    candles["time"] = pd.to_datetime(candles["time"], utc=True)

    sigs = pd.read_csv(baseline_path)
    sigs["signal_time"] = pd.to_datetime(sigs["signal_time"], utc=True, errors="coerce")
    for col in ("score", "entry", "sl", "tp1", "tp2", "tp3", "tp4", "realized_r"):
        if col in sigs.columns:
            sigs[col] = pd.to_numeric(sigs[col], errors="coerce")

    sigs = sigs[sigs.apply(production_score_ok, axis=1)].copy()
    sigs = sigs[sigs["entry_type"].astype(str).str.contains("FVG", regex=False, na=False)].copy()

    rows: list[dict] = []

    for source_index, sig in sigs.iterrows():
        st = sig["signal_time"]
        if pd.isna(st):
            continue

        # Closed candles only at signal time; do not use the forming signal candle.
        hist = candles[candles["time"] < st].copy()
        if len(hist) < 80:
            continue

        side = str(sig["side"])
        sl = float(sig["sl"])
        targets = _targets(sig)
        if not targets:
            continue

        # Current tested clean-FVG candidate: 75% of the FVG associated with the setup.
        current_fvg = choose_fvg_for_entry(sig, hist)
        bos = _latest_directional_bos(hist, side)
        bos_fvg = _bos_linked_fvg(hist, side, bos) if bos else None

        variants: list[tuple[str, float, dict]] = []

        if current_fvg is not None:
            variants.append(("CURRENT_FVG75", _entry_for_depth(side, current_fvg, 75), current_fvg))

        if bos is not None:
            variants.append(("BOS_LEVEL_RETEST", float(bos["level"]), bos))

        if bos_fvg is not None:
            variants.append(("BOS_FVG50", _entry_for_depth(side, bos_fvg, 50), bos_fvg))
            variants.append(("BOS_FVG75", _entry_for_depth(side, bos_fvg, 75), bos_fvg))

        for mode, entry, meta in variants:
            if not math.isfinite(entry):
                continue
            if side == "LONG" and not (sl < entry):
                continue
            if side == "SHORT" and not (sl > entry):
                continue

            sim = _simulate_trade(
                candles,
                side=side,
                signal_time=st,
                entry=entry,
                sl=sl,
                targets=targets,
            )

            rows.append({
                "asset": asset,
                "symbol": symbol,
                "source_index": source_index,
                "side": side,
                "score": int(sig["score"]),
                "signal_time": st,
                "mode": mode,
                "entry": entry,
                "sl": sl,
                "entry_type": sig["entry_type"],
                "bos_time": bos["time"] if bos else pd.NaT,
                "bos_level": float(bos["level"]) if bos else math.nan,
                "zone_low": float(meta.get("low", math.nan)) if isinstance(meta, dict) else math.nan,
                "zone_high": float(meta.get("high", math.nan)) if isinstance(meta, dict) else math.nan,
                "tp1": targets[0] if len(targets) > 0 else math.nan,
                "tp2": targets[1] if len(targets) > 1 else math.nan,
                "tp3": targets[2] if len(targets) > 2 else math.nan,
                "tp4": targets[3] if len(targets) > 3 else math.nan,
                **sim,
            })

    detail = pd.DataFrame(rows)
    modes = ["CURRENT_FVG75", "BOS_LEVEL_RETEST", "BOS_FVG50", "BOS_FVG75"]
    summary = pd.DataFrame([_summary(detail, asset, m) for m in modes])
    return detail, summary


def main() -> None:
    out = Path("backtest/data")
    out.mkdir(parents=True, exist_ok=True)

    details = []
    summaries = []

    for asset, (symbol, path) in ASSETS.items():
        d, s = run_asset(asset, symbol, path)
        d.to_csv(out / f"{asset.lower()}_first_retest_after_bos_detail.csv", index=False)
        details.append(d)
        summaries.append(s)

    detail_all = pd.concat(details, ignore_index=True)
    per_asset = pd.concat(summaries, ignore_index=True)

    modes = ["CURRENT_FVG75", "BOS_LEVEL_RETEST", "BOS_FVG50", "BOS_FVG75"]
    combined = pd.DataFrame([_summary(detail_all, "ALL", m) for m in modes])
    final = pd.concat([per_asset, combined], ignore_index=True)

    base = final[final["mode"] == "CURRENT_FVG75"][["asset", "total_r", "avg_r", "winrate", "filled"]].rename(columns={
        "total_r": "base_total_r",
        "avg_r": "base_avg_r",
        "winrate": "base_winrate",
        "filled": "base_filled",
    })
    final = final.merge(base, on="asset", how="left")
    final["delta_total_r_vs_fvg75"] = final["total_r"] - final["base_total_r"]
    final["delta_avg_r_vs_fvg75"] = final["avg_r"] - final["base_avg_r"]
    final["delta_wr_vs_fvg75"] = final["winrate"] - final["base_winrate"]
    final["delta_filled_vs_fvg75"] = final["filled"] - final["base_filled"]

    detail_all.to_csv(out / "first_retest_after_bos_detail.csv", index=False)
    final.to_csv(out / "first_retest_after_bos_summary.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n=== FIRST RETEST AFTER BOS vs CURRENT FVG75 ===")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
