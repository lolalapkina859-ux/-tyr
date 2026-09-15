from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START = "2026-03-01"
END = "2026-09-01"
TIMEFRAME = "30m"
ATR_LEN = 20
ATR_MULTS = (1.0, 1.5, 2.0)
MAX_HOLD_BARS = 7 * 24 * 2
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "ZEC": "ZECUSDT"}


def atr(df, n=20):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def run(asset, symbol, mode, atr_mult):
    d = download_klines(symbol, TIMEFRAME, START, END).reset_index(drop=True)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["atr"] = atr(d, ATR_LEN)
    pc = purple_cloud(d)
    buy_col = "pc_buy" if mode == "NORMAL" else "pc_strong_buy"
    sell_col = "pc_sell" if mode == "NORMAL" else "pc_strong_sell"
    rows = []
    pos = None

    for i in range(len(d)):
        bar = d.iloc[i]
        buy = bool(pc.iloc[i][buy_col])
        sell = bool(pc.iloc[i][sell_col])

        if pos is not None:
            stop_hit = float(bar["low"]) <= pos["sl"] if pos["side"] == "LONG" else float(bar["high"]) >= pos["sl"]
            opposite = sell if pos["side"] == "LONG" else buy
            timeout = i - pos["i"] >= MAX_HOLD_BARS

            # If stop and opposite signal occur on the same 30m candle, intrabar
            # ordering is unknown. Count the conservative stop first.
            if stop_hit:
                exit_price = pos["sl"]
                exit_reason = "SL"
            elif opposite:
                exit_price = float(bar["close"])
                exit_reason = "OPPOSITE_PC"
            elif timeout:
                exit_price = float(bar["close"])
                exit_reason = "TIMEOUT"
            else:
                exit_price = None
                exit_reason = None

            if exit_price is not None:
                r = ((exit_price - pos["entry"]) / pos["risk"] if pos["side"] == "LONG"
                     else (pos["entry"] - exit_price) / pos["risk"])
                rows.append({
                    "asset": asset, "mode": mode, "atr_mult": atr_mult,
                    "side": pos["side"], "entry_time": pos["time"], "exit_time": bar["time"],
                    "entry": pos["entry"], "sl": pos["sl"], "exit": exit_price,
                    "risk": pos["risk"], "realized_r": r, "bars": i-pos["i"],
                    "exit_reason": exit_reason,
                })
                pos = None

        if pos is None and (buy or sell):
            a = float(bar["atr"])
            if np.isfinite(a) and a > 0:
                side = "LONG" if buy else "SHORT"
                entry = float(bar["close"])
                risk = a * atr_mult
                sl = entry - risk if side == "LONG" else entry + risk
                pos = {"side": side, "entry": entry, "sl": sl, "risk": risk, "i": i, "time": bar["time"]}

    if pos is not None and len(d):
        bar = d.iloc[-1]
        exit_price = float(bar["close"])
        r = ((exit_price-pos["entry"])/pos["risk"] if pos["side"] == "LONG"
             else (pos["entry"]-exit_price)/pos["risk"])
        rows.append({"asset":asset,"mode":mode,"atr_mult":atr_mult,"side":pos["side"],
                     "entry_time":pos["time"],"exit_time":bar["time"],"entry":pos["entry"],
                     "sl":pos["sl"],"exit":exit_price,"risk":pos["risk"],"realized_r":r,
                     "bars":len(d)-1-pos["i"],"exit_reason":"END_OF_DATA"})
    return pd.DataFrame(rows)


def summarize(x, asset, mode, mult):
    if x.empty:
        return {"asset":asset,"mode":mode,"atr_mult":mult,"trades":0,"wins":0,"losses":0,
                "winrate":0,"total_r":0,"avg_r":0,"median_r":0,"max_dd_r":0,"sl_exits":0,
                "pc_exits":0,"avg_bars":0}
    eq=x["realized_r"].cumsum(); peak=eq.cummax(); dd=eq-peak
    wins=int((x["realized_r"]>0).sum()); losses=int((x["realized_r"]<0).sum())
    return {"asset":asset,"mode":mode,"atr_mult":mult,"trades":len(x),"wins":wins,"losses":losses,
            "winrate":round(100*wins/len(x),2),"total_r":round(float(x["realized_r"].sum()),4),
            "avg_r":round(float(x["realized_r"].mean()),4),"median_r":round(float(x["realized_r"].median()),4),
            "max_dd_r":round(float(dd.min()),4),"sl_exits":int((x["exit_reason"]=="SL").sum()),
            "pc_exits":int((x["exit_reason"]=="OPPOSITE_PC").sum()),"avg_bars":round(float(x["bars"].mean()),2)}


def main():
    out=Path("backtest/data"); out.mkdir(parents=True,exist_ok=True)
    details=[]; sums=[]
    for asset,symbol in ASSETS.items():
        for mode in ("NORMAL","STRONG"):
            for mult in ATR_MULTS:
                x=run(asset,symbol,mode,mult)
                details.append(x); sums.append(summarize(x,asset,mode,mult))
    detail=pd.concat([x for x in details if not x.empty],ignore_index=True)
    for mode in ("NORMAL","STRONG"):
        for mult in ATR_MULTS:
            x=detail[(detail["mode"]==mode)&(detail["atr_mult"]==mult)]
            sums.append(summarize(x,"ALL",mode,mult))
    summary=pd.DataFrame(sums)
    detail.to_csv(out/"purple_cloud_30m_real_sl_detail.csv",index=False)
    summary.to_csv(out/"purple_cloud_30m_real_sl_summary.csv",index=False)
    print("=== PURPLE CLOUD 30M + REAL ATR STOP ===")
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
