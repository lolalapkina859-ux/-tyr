from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START = "2026-03-01"
END = "2026-09-01"
MAX_HOLD_BARS = 7 * 24 * 4
ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","ZEC":"ZECUSDT"}


def run_mode(asset, symbol, mode):
    d = download_klines(symbol, "15m", START, END).reset_index(drop=True)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    pc = purple_cloud(d)
    rows=[]
    pos=None

    for i in range(len(d)):
        bar=d.iloc[i]
        buy = bool(pc.iloc[i].pc_buy if mode=="NORMAL" else pc.iloc[i].pc_strong_buy)
        sell = bool(pc.iloc[i].pc_sell if mode=="NORMAL" else pc.iloc[i].pc_strong_sell)

        if pos is not None:
            opposite = sell if pos["side"]=="LONG" else buy
            timeout = i-pos["i"] >= MAX_HOLD_BARS
            if opposite or timeout:
                exit_price=float(bar.close)
                raw_r=(exit_price-pos["entry"])/pos["risk"] if pos["side"]=="LONG" else (pos["entry"]-exit_price)/pos["risk"]
                rows.append({"asset":asset,"mode":mode,"side":pos["side"],"entry_time":pos["time"],"exit_time":bar.time,"entry":pos["entry"],"exit":exit_price,"initial_risk":pos["risk"],"realized_r":raw_r,"bars":i-pos["i"],"exit_reason":"OPPOSITE" if opposite else "TIMEOUT"})
                pos=None

        if pos is None and (buy or sell):
            side="LONG" if buy else "SHORT"
            entry=float(bar.close)
            # Standalone PC has no native SL. ATR is used only as a normalization unit for comparable R statistics.
            risk=float((d.iloc[max(0,i-20):i+1].high-d.iloc[max(0,i-20):i+1].low).mean())
            if risk>0:
                pos={"side":side,"entry":entry,"risk":risk,"i":i,"time":bar.time}

    return pd.DataFrame(rows)


def summarize(x, asset, mode):
    if x.empty:
        return {"asset":asset,"mode":mode,"trades":0,"wins":0,"losses":0,"winrate":0,"total_r":0,"avg_r":0,"median_r":0,"avg_bars":0}
    wins=int((x.realized_r>0).sum())
    return {"asset":asset,"mode":mode,"trades":len(x),"wins":wins,"losses":int((x.realized_r<0).sum()),"winrate":round(100*wins/len(x),2),"total_r":round(float(x.realized_r.sum()),4),"avg_r":round(float(x.realized_r.mean()),4),"median_r":round(float(x.realized_r.median()),4),"avg_bars":round(float(x.bars.mean()),2)}


def main():
    out=Path("backtest/data"); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; sums=[]
    for asset,symbol in ASSETS.items():
        for mode in ("NORMAL","STRONG"):
            x=run_mode(asset,symbol,mode)
            all_rows.append(x)
            sums.append(summarize(x,asset,mode))
    detail=pd.concat(all_rows,ignore_index=True)
    for mode in ("NORMAL","STRONG"):
        sums.append(summarize(detail[detail.mode==mode],"ALL",mode))
    summary=pd.DataFrame(sums)
    detail.to_csv(out/"purple_cloud_standalone_detail.csv",index=False)
    summary.to_csv(out/"purple_cloud_standalone_summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
