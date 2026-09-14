from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from backtest.download_binance import download_klines
from backtest.fvg_quality_test import choose_fvg_for_entry, production_score_ok

START = "2026-03-01"
END = "2026-09-01"
PENDING_EXPIRY_BARS = 48
DEPTHS = (25, 50, 75, 100)
TP_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}


def _targets(row: pd.Series) -> list[float]:
    out: list[float] = []
    for col in ("tp1", "tp2", "tp3", "tp4"):
        value = row.get(col)
        if pd.notna(value):
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                pass
    return out


def _entry_for_depth(side: str, fvg: dict, depth: int) -> float:
    low = float(fvg["low"])
    high = float(fvg["high"])
    width = high - low
    frac = float(depth) / 100.0

    if side == "LONG":
        # 0% = proximal/top edge, 100% = distal/bottom edge.
        return high - width * frac

    # SHORT: 0% = proximal/bottom edge, 100% = distal/top edge.
    return low + width * frac


def _touches_entry(bar: pd.Series, entry: float) -> bool:
    return float(bar["low"]) <= entry <= float(bar["high"])


def _bar_hits(side: str, bar: pd.Series, stop: float, targets: list[float]) -> tuple[bool, int]:
    high = float(bar["high"])
    low = float(bar["low"])

    if side == "LONG":
        sl_hit = low <= stop
        highest_tp = sum(1 for target in targets if high >= target)
    else:
        sl_hit = high >= stop
        highest_tp = sum(1 for target in targets if low <= target)

    return sl_hit, highest_tp


def _rr(side: str, entry: float, sl: float, target: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    if side == "LONG":
        return max(0.0, (target - entry) / risk)
    return max(0.0, (entry - target) / risk)


def _realized_r(side: str, entry: float, sl: float, targets: list[float], highest_tp: int) -> float:
    if highest_tp <= 0:
        return -1.0

    total = 0.0
    for idx in range(min(highest_tp, len(targets), len(TP_WEIGHTS))):
        total += TP_WEIGHTS[idx] * _rr(side, entry, sl, targets[idx])
    return total


def _simulate_trade(
    candles: pd.DataFrame,
    *,
    side: str,
    signal_time: pd.Timestamp,
    entry: float,
    sl: float,
    targets: list[float],
) -> dict:
    future = candles[candles["time"] > signal_time].copy()
    if future.empty:
        return {
            "status": "EXPIRED",
            "entry_time": pd.NaT,
            "exit_time": pd.NaT,
            "highest_tp": 0,
            "realized_r": 0.0,
        }

    pending = future.head(PENDING_EXPIRY_BARS)
    fill_pos = None

    for pos, (_, bar) in enumerate(pending.iterrows()):
        if _touches_entry(bar, entry):
            fill_pos = pos
            break

    if fill_pos is None:
        exit_time = pending.iloc[-1]["time"] if not pending.empty else pd.NaT
        return {
            "status": "EXPIRED",
            "entry_time": pd.NaT,
            "exit_time": exit_time,
            "highest_tp": 0,
            "realized_r": 0.0,
        }

    active = future.iloc[fill_pos:].copy()
    fill_bar = active.iloc[0]
    entry_time = fill_bar["time"]

    # Mirror the production backtester: if the fill candle also touches
    # SL or any TP, intrabar ordering is unknown -> AMBIGUOUS.
    sl_hit, tp_hit = _bar_hits(side, fill_bar, sl, targets)
    if sl_hit or tp_hit > 0:
        return {
            "status": "AMBIGUOUS",
            "entry_time": entry_time,
            "exit_time": entry_time,
            "highest_tp": 0,
            "realized_r": 0.0,
        }

    highest_tp = 0

    for _, bar in active.iloc[1:].iterrows():
        stop = entry if highest_tp >= 1 else sl
        sl_hit, tp_hit = _bar_hits(side, bar, stop, targets)
        new_highest = max(highest_tp, tp_hit)

        if sl_hit and new_highest > highest_tp:
            return {
                "status": "AMBIGUOUS",
                "entry_time": entry_time,
                "exit_time": bar["time"],
                "highest_tp": highest_tp,
                "realized_r": 0.0,
            }

        highest_tp = new_highest

        if targets and highest_tp >= len(targets):
            return {
                "status": "CLOSED",
                "entry_time": entry_time,
                "exit_time": bar["time"],
                "highest_tp": highest_tp,
                "realized_r": round(_realized_r(side, entry, sl, targets, highest_tp), 4),
            }

        if sl_hit:
            return {
                "status": "CLOSED",
                "entry_time": entry_time,
                "exit_time": bar["time"],
                "highest_tp": highest_tp,
                "realized_r": round(_realized_r(side, entry, sl, targets, highest_tp), 4),
            }

    return {
        "status": "OPEN",
        "entry_time": entry_time,
        "exit_time": pd.NaT,
        "highest_tp": highest_tp,
        "realized_r": 0.0,
    }


def _summary(df: pd.DataFrame, asset: str, depth: int) -> dict:
    filled = df[df["entry_time"].notna() & (df["status"] != "AMBIGUOUS")]
    closed = df[df["status"] == "CLOSED"]
    wins = closed[closed["highest_tp"] >= 1]
    losses = closed[closed["highest_tp"] == 0]
    expired = df[df["status"] == "EXPIRED"]
    ambiguous = df[df["status"] == "AMBIGUOUS"]

    total_r = float(closed["realized_r"].sum()) if not closed.empty else 0.0
    winrate = 100.0 * len(wins) / len(closed) if len(closed) else 0.0
    avg_r = total_r / len(closed) if len(closed) else 0.0

    return {
        "asset": asset,
        "depth_pct": depth,
        "setups": len(df),
        "filled": len(filled),
        "closed": len(closed),
        "expired": len(expired),
        "ambiguous": len(ambiguous),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(winrate, 2),
        "total_r": round(total_r, 4),
        "avg_r": round(avg_r, 4),
        "tp1": int((closed["highest_tp"] >= 1).sum()) if len(closed) else 0,
        "tp2": int((closed["highest_tp"] >= 2).sum()) if len(closed) else 0,
        "tp3": int((closed["highest_tp"] >= 3).sum()) if len(closed) else 0,
        "tp4": int((closed["highest_tp"] >= 4).sum()) if len(closed) else 0,
    }


def run_asset(asset: str, symbol: str, baseline_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline: {baseline_path}")

    print(f"\n=== {asset}: {symbol} FVG DEPTH TEST ===")
    candles = download_klines(symbol, "15m", START, END).copy()
    if candles.empty:
        raise RuntimeError(f"No candles downloaded for {symbol}")

    candles["time"] = pd.to_datetime(candles["time"], utc=True)

    signals = pd.read_csv(baseline_path)
    signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True, errors="coerce")
    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=True, errors="coerce")

    for col in ("score", "entry", "sl", "tp1", "tp2", "tp3", "tp4", "realized_r"):
        if col in signals.columns:
            signals[col] = pd.to_numeric(signals[col], errors="coerce")

    signals = signals[signals.apply(production_score_ok, axis=1)].copy()
    signals = signals[
        signals["entry_type"].astype(str).str.contains("FVG", regex=False, na=False)
    ].copy()

    detail_rows: list[dict] = []

    for source_index, sig in signals.iterrows():
        hist = candles[candles["time"] <= sig["signal_time"]].copy()
        if len(hist) < 60:
            continue

        fvg = choose_fvg_for_entry(sig, hist)
        if fvg is None:
            continue

        fvg_low = float(fvg["low"])
        fvg_high = float(fvg["high"])
        width = fvg_high - fvg_low
        if not math.isfinite(width) or width <= 0:
            continue

        midpoint_error_pct = (
            abs(float(fvg["mid"]) - float(sig["entry"]))
            / abs(float(sig["entry"]))
            * 100.0
        )

        # Same association sanity gate as the prior FVG quality test.
        if midpoint_error_pct > 0.35:
            continue

        side = str(sig["side"])
        sl = float(sig["sl"])
        targets = _targets(sig)
        if not targets:
            continue

        for depth in DEPTHS:
            entry = _entry_for_depth(side, fvg, depth)

            # Structural SL must remain beyond the hypothetical entry.
            if side == "LONG" and not (sl < entry):
                continue
            if side == "SHORT" and not (sl > entry):
                continue

            sim = _simulate_trade(
                candles,
                side=side,
                signal_time=sig["signal_time"],
                entry=entry,
                sl=sl,
                targets=targets,
            )

            detail_rows.append({
                "asset": asset,
                "symbol": symbol,
                "source_index": source_index,
                "side": side,
                "score": int(sig["score"]),
                "signal_time": sig["signal_time"],
                "depth_pct": depth,
                "entry": entry,
                "original_entry": float(sig["entry"]),
                "sl": sl,
                "fvg_low": fvg_low,
                "fvg_high": fvg_high,
                "fvg_mid": float(fvg["mid"]),
                "midpoint_error_pct": midpoint_error_pct,
                "entry_type": sig["entry_type"],
                "tp1": targets[0] if len(targets) > 0 else math.nan,
                "tp2": targets[1] if len(targets) > 1 else math.nan,
                "tp3": targets[2] if len(targets) > 2 else math.nan,
                "tp4": targets[3] if len(targets) > 3 else math.nan,
                **sim,
            })

    detail = pd.DataFrame(detail_rows)
    summaries = []

    for depth in DEPTHS:
        subset = detail[detail["depth_pct"] == depth].copy()
        summaries.append(_summary(subset, asset, depth))

    summary = pd.DataFrame(summaries)
    print(summary.to_string(index=False))
    return detail, summary


def main() -> None:
    out_dir = Path("backtest/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detail: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for asset, (symbol, baseline_path) in ASSETS.items():
        detail, summary = run_asset(asset, symbol, baseline_path)
        detail.to_csv(out_dir / f"{asset.lower()}_fvg_depth_detail.csv", index=False)
        all_detail.append(detail)
        all_summary.append(summary)

    detail_all = pd.concat(all_detail, ignore_index=True)
    per_asset = pd.concat(all_summary, ignore_index=True)

    combined_rows = []
    for depth in DEPTHS:
        subset = detail_all[detail_all["depth_pct"] == depth].copy()
        combined_rows.append(_summary(subset, "ALL", depth))

    combined = pd.DataFrame(combined_rows)
    final = pd.concat([per_asset, combined], ignore_index=True)

    # Delta vs 50% entry, useful for quick decision-making.
    final["delta_total_r_vs_50"] = 0.0
    final["delta_avg_r_vs_50"] = 0.0
    final["delta_wr_vs_50"] = 0.0

    for asset in final["asset"].unique():
        base = final[(final["asset"] == asset) & (final["depth_pct"] == 50)]
        if base.empty:
            continue
        b = base.iloc[0]
        mask = final["asset"] == asset
        final.loc[mask, "delta_total_r_vs_50"] = final.loc[mask, "total_r"] - float(b["total_r"])
        final.loc[mask, "delta_avg_r_vs_50"] = final.loc[mask, "avg_r"] - float(b["avg_r"])
        final.loc[mask, "delta_wr_vs_50"] = final.loc[mask, "winrate"] - float(b["winrate"])

    final.to_csv(out_dir / "cross_asset_fvg_depth_summary.csv", index=False)
    detail_all.to_csv(out_dir / "cross_asset_fvg_depth_detail.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n=== CROSS-ASSET FVG DEPTH 25/50/75/100 ===")
    print(final.to_string(index=False))
    print("\nSaved:")
    print(out_dir / "cross_asset_fvg_depth_summary.csv")
    print(out_dir / "cross_asset_fvg_depth_detail.csv")


if __name__ == "__main__":
    main()
