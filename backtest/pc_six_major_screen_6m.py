from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START = "2026-03-01"
END = "2026-09-01"
TF = "30m"
NOTIONAL = 60.0
SYMBOLS = ["BTCUSDT", "SOLUSDT", "ARBUSDT", "WIFUSDT", "1000PEPEUSDT", "LINKUSDT"]
PERIODS = {
    "MAR_JUN": (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC")),
    "JUN_SEP": (pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-09-01", tz="UTC")),
}
OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)


def trade_return(side, entry, exit_):
    return (exit_ / entry - 1.0) if side == "LONG" else (entry / exit_ - 1.0)


def test_period(df, start, end, symbol, period_name):
    d = df[(df.time >= start) & (df.time < end)].reset_index(drop=True)
    pos = None
    trades = []
    for _, r in d.iterrows():
        buy, sell, px, tm = bool(r.pc_buy), bool(r.pc_sell), float(r.close), r.time
        if pos and ((pos["side"] == "LONG" and sell) or (pos["side"] == "SHORT" and buy)):
            ret = trade_return(pos["side"], pos["entry"], px)
            trades.append({"symbol": symbol, "period": period_name, "side": pos["side"], "entry_time": pos["time"], "exit_time": tm, "entry": pos["entry"], "exit": px, "return_pct": ret * 100, "pnl_60_usdt": NOTIONAL * ret, "exit_reason": "OPPOSITE"})
            pos = None
        if pos is None and (buy or sell):
            pos = {"side": "LONG" if buy else "SHORT", "entry": px, "time": tm}
    if pos is not None and len(d):
        r = d.iloc[-1]
        px = float(r.close)
        ret = trade_return(pos["side"], pos["entry"], px)
        trades.append({"symbol": symbol, "period": period_name, "side": pos["side"], "entry_time": pos["time"], "exit_time": r.time, "entry": pos["entry"], "exit": px, "return_pct": ret * 100, "pnl_60_usdt": NOTIONAL * ret, "exit_reason": "PERIOD_END"})
    return pd.DataFrame(trades)


def main():
    summaries, all_trades, failures = [], [], []
    for symbol in SYMBOLS:
        try:
            raw = download_klines(symbol, TF, START, END).reset_index(drop=True)
            raw["time"] = pd.to_datetime(raw["time"], utc=True)
            d = purple_cloud(raw)
            d["time"] = raw["time"]
            period_results = {}
            symbol_trades = []
            for pname, (pstart, pend) in PERIODS.items():
                t = test_period(d, pstart, pend, symbol, pname)
                symbol_trades.append(t)
                pct = float(t.return_pct.sum()) if len(t) else 0.0
                pnl = float(t.pnl_60_usdt.sum()) if len(t) else 0.0
                wins = int((t.return_pct > 0).sum()) if len(t) else 0
                period_results[pname] = {"pct": pct, "pnl": pnl, "trades": len(t), "winrate": 100 * wins / len(t) if len(t) else 0.0}
            tt = pd.concat(symbol_trades, ignore_index=True)
            all_trades.append(tt)
            a, b = period_results["MAR_JUN"], period_results["JUN_SEP"]
            summaries.append({
                "symbol": symbol,
                "MAR_JUN_pct": a["pct"], "MAR_JUN_pnl60": a["pnl"], "MAR_JUN_trades": a["trades"], "MAR_JUN_winrate": a["winrate"],
                "JUN_SEP_pct": b["pct"], "JUN_SEP_pnl60": b["pnl"], "JUN_SEP_trades": b["trades"], "JUN_SEP_winrate": b["winrate"],
                "both_positive": a["pct"] > 0 and b["pct"] > 0,
                "six_month_sum_pct": a["pct"] + b["pct"],
                "six_month_pnl60": a["pnl"] + b["pnl"],
                "worst_period_pct": min(a["pct"], b["pct"]),
                "total_trades": a["trades"] + b["trades"],
            })
            print("DONE", symbol)
        except Exception as e:
            failures.append({"symbol": symbol, "error": str(e)})
            print("FAIL", symbol, e)

    s = pd.DataFrame(summaries)
    if len(s):
        s = s.sort_values(["both_positive", "worst_period_pct", "six_month_sum_pct"], ascending=[False, False, False]).reset_index(drop=True)
    s.to_csv(OUT / "pc_six_major_screen_6m_summary.csv", index=False)
    (pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()).to_csv(OUT / "pc_six_major_screen_6m_trades.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "pc_six_major_screen_6m_failures.csv", index=False)
    print("\nSUMMARY\n", s.to_string(index=False))


if __name__ == "__main__":
    main()
