from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines

START = "2026-03-01"
END = "2026-09-01"
MAX_HOLD_BARS = 7 * 24 * 4
TP_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
LOOKBACKS = (6, 8, 12)

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

def gate(row):
    score = int(row["score"])
    return score >= (88 if str(row["side"]) == "SHORT" else 70)

def get_targets(row):
    out = []
    for c in ("tp1","tp2","tp3","tp4"):
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v):
            out.append(float(v))
    return out

def rr(side, entry, sl, target):
    risk = abs(entry-sl)
    if risk <= 0:
        return 0.0
    return max(0.0, (target-entry)/risk) if side=="LONG" else max(0.0, (entry-target)/risk)

def bos(df, i, side, lb):
    if i < lb:
        return False
    prev = df.iloc[i-lb:i]
    c = float(df.iloc[i]["close"])
    return c > float(prev["high"].max()) if side=="LONG" else c < float(prev["low"].min())

def simulate(row, candles, mode, lb=0):
    side = str(row["side"])
    entry = float(row["entry"])
    sl = float(row["sl"])
    targets = get_targets(row)
    if pd.isna(row["entry_time"]) or not targets:
        return ("SKIP",0.0,0,False)

    future = candles[candles["time"] >= row["entry_time"]].head(MAX_HOLD_BARS).reset_index(drop=True)
    if future.empty:
        return ("SKIP",0.0,0,False)

    hi_tp = 0
    be = False
    realized = 0.0
    remaining = 1.0

    for i, bar in future.iterrows():
        high, low = float(bar["high"]), float(bar["low"])
        stop = entry if be else sl
        stop_hit = low <= stop if side=="LONG" else high >= stop

        new_hi = hi_tp
        for j,t in enumerate(targets,1):
            if (high >= t if side=="LONG" else low <= t):
                new_hi = max(new_hi,j)

        bos_now = mode=="BOS_BE" and (not be) and bos(future,i,side,lb)

        if stop_hit and (new_hi > hi_tp or bos_now):
            return ("AMBIGUOUS",0.0,hi_tp,be)

        if new_hi > hi_tp:
            for n in range(hi_tp+1,new_hi+1):
                if n <= len(TP_WEIGHTS):
                    w = TP_WEIGHTS[n-1]
                    realized += w*rr(side,entry,sl,targets[n-1])
                    remaining -= w
            hi_tp = new_hi

            if hi_tp >= len(targets):
                return ("CLOSED_TP",realized,hi_tp,be)

            if mode=="TP1_BE" and hi_tp >= 1:
                be = True

        if bos_now:
            be = True

        if stop_hit:
            if stop == entry:
                return ("CLOSED_BE",realized,hi_tp,be)
            realized -= max(0.0,remaining)
            return ("CLOSED_SL",realized,hi_tp,be)

    return ("TIMEOUT",realized,hi_tp,be)

def summarize(df, asset, mode, lb):
    v = df[~df["status_new"].isin(["SKIP","AMBIGUOUS","TIMEOUT"])].copy()
    total = float(v["realized_r_new"].sum()) if len(v) else 0.0
    wins = int((v["realized_r_new"]>0).sum()) if len(v) else 0
    return {
        "asset":asset, "mode":mode, "bos_lookback":lb,
        "trades":len(v), "wins":wins,
        "losses":int((v["realized_r_new"]<0).sum()) if len(v) else 0,
        "winrate":100*wins/len(v) if len(v) else 0.0,
        "total_r":total, "avg_r":total/len(v) if len(v) else 0.0,
        "tp2_plus":int((v["highest_tp_new"]>=2).sum()) if len(v) else 0,
        "tp3_plus":int((v["highest_tp_new"]>=3).sum()) if len(v) else 0,
        "tp4":int((v["highest_tp_new"]>=4).sum()) if len(v) else 0,
        "be_exits":int((v["status_new"]=="CLOSED_BE").sum()) if len(v) else 0,
        "sl_exits":int((v["status_new"]=="CLOSED_SL").sum()) if len(v) else 0,
    }

def main():
    out = Path("backtest/data")
    out.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []

    for asset,(symbol,path) in ASSETS.items():
        s = pd.read_csv(path)
        s["entry_time"] = pd.to_datetime(s["entry_time"],utc=True,errors="coerce")
        s["signal_time"] = pd.to_datetime(s["signal_time"],utc=True,errors="coerce")
        for c in ("score","entry","sl","realized_r","tp1","tp2","tp3","tp4"):
            if c in s.columns:
                s[c] = pd.to_numeric(s[c],errors="coerce")
        s = s[s.apply(gate,axis=1) & s["entry_time"].notna()].copy()

        candles = download_klines(symbol,"15m",START,END)
        candles["time"] = pd.to_datetime(candles["time"],utc=True)

        for mode,lb in [("TP1_BE",0),("BOS_BE",6),("BOS_BE",8),("BOS_BE",12)]:
            rows=[]
            for idx,row in s.iterrows():
                st,r,h,b = simulate(row,candles,mode,lb)
                rows.append({
                    **row.to_dict(),"asset":asset,"source_index":idx,
                    "mode":mode,"bos_lookback":lb,
                    "status_new":st,"realized_r_new":r,
                    "highest_tp_new":h,"be_armed_new":b,
                })
            d=pd.DataFrame(rows)
            all_rows.append(d)
            summaries.append(summarize(d,asset,mode,lb))

    detail=pd.concat(all_rows,ignore_index=True)

    # Combined rows.
    for mode,lb in [("TP1_BE",0),("BOS_BE",6),("BOS_BE",8),("BOS_BE",12)]:
        sub=detail[(detail["mode"]==mode)&(detail["bos_lookback"]==lb)]
        summaries.append(summarize(sub,"ALL",mode,lb))

    summary=pd.DataFrame(summaries)
    base=summary[summary["mode"]=="TP1_BE"][["asset","total_r","avg_r","tp2_plus","tp3_plus","be_exits"]].rename(columns={
        "total_r":"base_total_r","avg_r":"base_avg_r","tp2_plus":"base_tp2_plus",
        "tp3_plus":"base_tp3_plus","be_exits":"base_be_exits"})
    summary=summary.merge(base,on="asset",how="left")
    summary["delta_total_r"]=summary["total_r"]-summary["base_total_r"]
    summary["delta_avg_r"]=summary["avg_r"]-summary["base_avg_r"]
    summary["delta_tp2_plus"]=summary["tp2_plus"]-summary["base_tp2_plus"]
    summary["delta_tp3_plus"]=summary["tp3_plus"]-summary["base_tp3_plus"]
    summary["delta_be_exits"]=summary["be_exits"]-summary["base_be_exits"]

    detail.to_csv(out/"be_structural_bos_detail.csv",index=False)
    summary.to_csv(out/"be_structural_bos_summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
