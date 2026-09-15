from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START = "2026-03-01"
END = "2026-09-01"
MAX_HOLD_BARS = 7 * 24 * 4
TP_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}


def gate(row):
    score = int(row["score"])
    return score >= (88 if str(row["side"]) == "SHORT" else 70)


def targets(row):
    out = []
    for c in ("tp1", "tp2", "tp3", "tp4"):
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v):
            out.append(float(v))
    return out


def rr(side, entry, sl, price):
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return (price - entry) / risk if side == "LONG" else (entry - price) / risk


def target_rr(side, entry, sl, target):
    return max(0.0, rr(side, entry, sl, target))


def simulate(row, candles, pc, mode):
    side = str(row["side"])
    entry = float(row["entry"])
    sl = float(row["sl"])
    tps = targets(row)
    if pd.isna(row["entry_time"]) or not tps:
        return "SKIP", 0.0, 0, False, None, None

    idx = candles.index[candles["time"] >= row["entry_time"]]
    if len(idx) == 0:
        return "SKIP", 0.0, 0, False, None, None

    start = int(idx[0])
    end = min(start + MAX_HOLD_BARS, len(candles))
    hi_tp = 0
    realized = 0.0
    remaining = 1.0

    for i in range(start, end):
        bar = candles.iloc[i]
        high, low, close = float(bar.high), float(bar.low), float(bar.close)
        stop_hit = low <= sl if side == "LONG" else high >= sl

        new_hi = hi_tp
        for j, t in enumerate(tps, 1):
            if high >= t if side == "LONG" else low <= t:
                new_hi = max(new_hi, j)

        opposite = False
        if mode == "PC_OPPOSITE_NORMAL":
            opposite = bool(pc.iloc[i].pc_sell) if side == "LONG" else bool(pc.iloc[i].pc_buy)
        elif mode == "PC_OPPOSITE_STRONG":
            opposite = bool(pc.iloc[i].pc_strong_sell) if side == "LONG" else bool(pc.iloc[i].pc_strong_buy)

        # Same candle has conflicting unknown intrabar order: exclude it.
        if stop_hit and (new_hi > hi_tp or opposite):
            return "AMBIGUOUS", 0.0, hi_tp, False, None, None
        if opposite and new_hi > hi_tp:
            return "AMBIGUOUS", 0.0, hi_tp, False, None, None

        if new_hi > hi_tp:
            for n in range(hi_tp + 1, new_hi + 1):
                if n <= len(TP_WEIGHTS):
                    w = TP_WEIGHTS[n - 1]
                    realized += w * target_rr(side, entry, sl, tps[n - 1])
                    remaining -= w
            hi_tp = new_hi
            if hi_tp >= len(tps):
                return "CLOSED_TP", realized, hi_tp, False, None, None

        if opposite:
            exit_r = rr(side, entry, sl, close)
            realized += remaining * exit_r
            return "CLOSED_PC", realized, hi_tp, True, bar.time, close

        if stop_hit:
            realized -= remaining
            return "CLOSED_SL", realized, hi_tp, False, None, None

    return "TIMEOUT", realized, hi_tp, False, None, None


def summarize(df, asset, mode):
    v = df[~df.status_new.isin(["SKIP", "AMBIGUOUS", "TIMEOUT"])].copy()
    total = float(v.realized_r_new.sum()) if len(v) else 0.0
    return {
        "asset": asset,
        "mode": mode,
        "trades": len(v),
        "wins": int((v.realized_r_new > 0).sum()),
        "losses": int((v.realized_r_new < 0).sum()),
        "winrate": round(100 * (v.realized_r_new > 0).sum() / len(v), 2) if len(v) else 0.0,
        "total_r": round(total, 4),
        "avg_r": round(total / len(v), 4) if len(v) else 0.0,
        "pc_exits": int((v.status_new == "CLOSED_PC").sum()),
        "pc_saved_losses": int(((v.status_new == "CLOSED_PC") & (v.realized_r_new > -1.0)).sum()),
        "pc_profitable_exits": int(((v.status_new == "CLOSED_PC") & (v.realized_r_new > 0)).sum()),
        "tp2_plus": int((v.highest_tp_new >= 2).sum()),
        "tp3_plus": int((v.highest_tp_new >= 3).sum()),
    }


def main():
    out = Path("backtest/data")
    out.mkdir(parents=True, exist_ok=True)
    details = []
    summaries = []

    for asset, (symbol, path) in ASSETS.items():
        sig = pd.read_csv(path)
        sig["entry_time"] = pd.to_datetime(sig["entry_time"], utc=True, errors="coerce")
        for c in ("score", "entry", "sl", "realized_r", "tp1", "tp2", "tp3", "tp4"):
            if c in sig.columns:
                sig[c] = pd.to_numeric(sig[c], errors="coerce")
        sig = sig[sig.apply(gate, axis=1) & sig.entry_time.notna()].copy()

        candles = download_klines(symbol, "15m", START, END).reset_index(drop=True)
        candles["time"] = pd.to_datetime(candles["time"], utc=True)
        pc = purple_cloud(candles)

        for mode in ("PC_OPPOSITE_NORMAL", "PC_OPPOSITE_STRONG"):
            rows = []
            for idx, row in sig.iterrows():
                st, r, h, used, exit_time, exit_price = simulate(row, candles, pc, mode)
                rows.append({
                    **row.to_dict(), "asset": asset, "source_index": idx, "mode": mode,
                    "status_new": st, "realized_r_new": r, "highest_tp_new": h,
                    "pc_exit_used": used, "pc_exit_time": exit_time, "pc_exit_price": exit_price,
                })
            d = pd.DataFrame(rows)
            details.append(d)
            summaries.append(summarize(d, asset, mode))

    detail = pd.concat(details, ignore_index=True)
    for mode in ("PC_OPPOSITE_NORMAL", "PC_OPPOSITE_STRONG"):
        summaries.append(summarize(detail[detail.mode == mode], "ALL", mode))

    # Existing production result from the source CSVs for an apples-to-apples reference.
    base_rows = []
    for asset, (_, path) in ASSETS.items():
        s = pd.read_csv(path)
        s["score"] = pd.to_numeric(s["score"], errors="coerce")
        s["realized_r"] = pd.to_numeric(s["realized_r"], errors="coerce")
        s = s[s.apply(gate, axis=1) & (s.status == "CLOSED")].copy()
        base_rows.append({"asset": asset, "mode": "SOURCE_BASELINE", "trades": len(s),
                          "wins": int((s.realized_r > 0).sum()), "losses": int((s.realized_r < 0).sum()),
                          "winrate": round(100*(s.realized_r > 0).sum()/len(s),2) if len(s) else 0,
                          "total_r": round(float(s.realized_r.sum()),4),
                          "avg_r": round(float(s.realized_r.mean()),4) if len(s) else 0,
                          "pc_exits": 0, "pc_saved_losses": 0, "pc_profitable_exits": 0,
                          "tp2_plus": int((pd.to_numeric(s.highest_tp, errors="coerce") >= 2).sum()) if "highest_tp" in s else 0,
                          "tp3_plus": int((pd.to_numeric(s.highest_tp, errors="coerce") >= 3).sum()) if "highest_tp" in s else 0})
    b = pd.DataFrame(base_rows)
    base_rows.append({"asset":"ALL", "mode":"SOURCE_BASELINE", "trades":int(b.trades.sum()),
                      "wins":int(b.wins.sum()), "losses":int(b.losses.sum()),
                      "winrate":round(100*b.wins.sum()/b.trades.sum(),2) if b.trades.sum() else 0,
                      "total_r":round(float(b.total_r.sum()),4),
                      "avg_r":round(float(b.total_r.sum()/b.trades.sum()),4) if b.trades.sum() else 0,
                      "pc_exits":0,"pc_saved_losses":0,"pc_profitable_exits":0,
                      "tp2_plus":int(b.tp2_plus.sum()),"tp3_plus":int(b.tp3_plus.sum())})

    summary = pd.concat([pd.DataFrame(base_rows), pd.DataFrame(summaries)], ignore_index=True)
    detail.to_csv(out / "purple_cloud_opposite_exit_detail.csv", index=False)
    summary.to_csv(out / "purple_cloud_opposite_exit_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
