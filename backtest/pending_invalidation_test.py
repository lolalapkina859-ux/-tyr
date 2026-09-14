from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines

START = "2026-03-01"
END = "2026-09-01"
PENDING_BARS = 48

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

RULES = [
    "BASELINE_48",
    "AGE_16",
    "AGE_24",
    "AGE_32",
    "TP1_BEFORE_ENTRY",
    "OPP_BOS_8",
    "OPP_BOS_12",
    "OPP_BOS_16",
    "BOS12_OR_TP1",
    "BOS12_OR_TP1_OR_AGE24",
]


def production_gate(row: pd.Series) -> bool:
    side = str(row["side"])
    score = int(row["score"])
    return score >= (88 if side == "SHORT" else 70)


def opposite_bos(df: pd.DataFrame, i: int, side: str, lookback: int) -> bool:
    if i < lookback:
        return False
    prev = df.iloc[i-lookback:i]
    close = float(df.iloc[i]["close"])
    if side == "LONG":
        return close < float(prev["low"].min())
    return close > float(prev["high"].max())


def tp1_reached(bar: pd.Series, side: str, tp1: float) -> bool:
    if side == "LONG":
        return float(bar["high"]) >= tp1
    return float(bar["low"]) <= tp1


def first_cancel_event(
    row: pd.Series,
    candles: pd.DataFrame,
    *,
    max_age: int | None = None,
    bos_lookback: int | None = None,
    cancel_on_tp1: bool = False,
) -> dict:
    signal_time = row["signal_time"]
    entry_time = row["entry_time"]
    side = str(row["side"])
    tp1 = pd.to_numeric(row.get("tp1"), errors="coerce")

    future = candles[candles["time"] > signal_time].head(PENDING_BARS).reset_index(drop=True)
    if future.empty:
        return {"cancelled": False, "reason": "NONE", "bars": None, "time": pd.NaT}

    for i, bar in future.iterrows():
        # If historical fill happens on this bar, the setup survived until entry.
        if pd.notna(entry_time) and pd.Timestamp(bar["time"]) >= entry_time:
            return {"cancelled": False, "reason": "FILLED_FIRST", "bars": i+1, "time": pd.NaT}

        age = i + 1

        if max_age is not None and age >= max_age:
            return {"cancelled": True, "reason": f"AGE_{max_age}", "bars": age, "time": bar["time"]}

        if cancel_on_tp1 and pd.notna(tp1) and tp1_reached(bar, side, float(tp1)):
            return {"cancelled": True, "reason": "TP1_BEFORE_ENTRY", "bars": age, "time": bar["time"]}

        if bos_lookback is not None and opposite_bos(future, i, side, bos_lookback):
            return {"cancelled": True, "reason": f"OPP_BOS_{bos_lookback}", "bars": age, "time": bar["time"]}

    return {"cancelled": False, "reason": "NONE", "bars": len(future), "time": pd.NaT}


def apply_rule(row: pd.Series, candles: pd.DataFrame, rule: str) -> dict:
    if rule == "BASELINE_48":
        return {"cancelled": False, "reason": "BASELINE", "bars": None, "time": pd.NaT}
    if rule.startswith("AGE_"):
        return first_cancel_event(row, candles, max_age=int(rule.split("_")[1]))
    if rule == "TP1_BEFORE_ENTRY":
        return first_cancel_event(row, candles, cancel_on_tp1=True)
    if rule.startswith("OPP_BOS_"):
        return first_cancel_event(row, candles, bos_lookback=int(rule.split("_")[-1]))
    if rule == "BOS12_OR_TP1":
        return first_cancel_event(row, candles, bos_lookback=12, cancel_on_tp1=True)
    if rule == "BOS12_OR_TP1_OR_AGE24":
        return first_cancel_event(row, candles, max_age=24, bos_lookback=12, cancel_on_tp1=True)
    raise ValueError(rule)


def summarize(df: pd.DataFrame, asset: str, rule: str) -> dict:
    cancelled = df[df["cancelled"]]
    kept = df[~df["cancelled"]]

    closed = kept[kept["status"] == "CLOSED"]
    wins = closed[closed["realized_r"] > 0]
    losses = closed[closed["realized_r"] < 0]

    removed_closed = cancelled[cancelled["status"] == "CLOSED"]
    removed_wins = removed_closed[removed_closed["realized_r"] > 0]
    removed_losses = removed_closed[removed_closed["realized_r"] < 0]

    total_r = float(closed["realized_r"].sum()) if len(closed) else 0.0
    winrate = 100.0 * len(wins) / len(closed) if len(closed) else 0.0
    avg_r = total_r / len(closed) if len(closed) else 0.0

    return {
        "asset": asset,
        "rule": rule,
        "setups": len(df),
        "cancelled": len(cancelled),
        "kept": len(kept),
        "closed_kept": len(closed),
        "wins_kept": len(wins),
        "losses_kept": len(losses),
        "winrate_kept": round(winrate, 2),
        "total_r_kept": round(total_r, 4),
        "avg_r_kept": round(avg_r, 4),
        "removed_closed": len(removed_closed),
        "removed_wins": len(removed_wins),
        "removed_losses": len(removed_losses),
        "removed_r": round(float(removed_closed["realized_r"].sum()) if len(removed_closed) else 0.0, 4),
        "expired_cancelled": int(((cancelled["status"] == "EXPIRED")).sum()),
        "avg_cancel_bar": round(float(cancelled["cancel_bars"].dropna().mean()), 2) if len(cancelled) else 0.0,
    }


def main() -> None:
    out_dir = Path("backtest/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detail = []
    summaries = []

    for asset, (symbol, path) in ASSETS.items():
        signals = pd.read_csv(path)
        signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True, errors="coerce")
        signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=True, errors="coerce")
        for col in ("score", "realized_r", "tp1"):
            if col in signals.columns:
                signals[col] = pd.to_numeric(signals[col], errors="coerce")

        signals = signals[signals.apply(production_gate, axis=1)].copy()

        candles = download_klines(symbol, "15m", START, END).copy()
        candles["time"] = pd.to_datetime(candles["time"], utc=True)

        for rule in RULES:
            rows = []
            for idx, row in signals.iterrows():
                event = apply_rule(row, candles, rule)
                item = row.to_dict()
                item.update({
                    "asset": asset,
                    "source_index": idx,
                    "rule": rule,
                    "cancelled": bool(event["cancelled"]),
                    "cancel_reason": event["reason"],
                    "cancel_bars": event["bars"],
                    "cancel_time": event["time"],
                })
                rows.append(item)

            detail = pd.DataFrame(rows)
            all_detail.append(detail)
            summaries.append(summarize(detail, asset, rule))

    detail_all = pd.concat(all_detail, ignore_index=True)

    for rule in RULES:
        sub = detail_all[detail_all["rule"] == rule].copy()
        summaries.append(summarize(sub, "ALL", rule))

    summary = pd.DataFrame(summaries)

    base = summary[summary["rule"] == "BASELINE_48"][["asset", "total_r_kept", "avg_r_kept", "winrate_kept"]].rename(columns={
        "total_r_kept": "base_total_r",
        "avg_r_kept": "base_avg_r",
        "winrate_kept": "base_winrate",
    })
    summary = summary.merge(base, on="asset", how="left")
    summary["delta_total_r"] = summary["total_r_kept"] - summary["base_total_r"]
    summary["delta_avg_r"] = summary["avg_r_kept"] - summary["base_avg_r"]
    summary["delta_winrate"] = summary["winrate_kept"] - summary["base_winrate"]

    detail_all.to_csv(out_dir / "pending_invalidation_detail.csv", index=False)
    summary.to_csv(out_dir / "pending_invalidation_summary.csv", index=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)
    print("\n=== PENDING INVALIDATION TEST ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
