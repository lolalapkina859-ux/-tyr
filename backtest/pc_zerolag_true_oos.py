from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import purple_cloud, atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START="2025-09-01"; END="2026-03-01"
START_BALANCE=145.58352658; NOTIONAL=60.
CASES=[("BASE15",None,None),("ZL70_1.0",70,1.0),("ZL70_1.4",70,1.4)]

def zl(d,n,m):
    lag=(n-1)//2; s=d.close
    z=(s+(s-s.shift(lag))).ewm(span=n,adjust=False).mean()
    v=atr(d,n).rolling(n*3).max()*m
    t=np.zeros(len(d),dtype=int)
    for i in range(1,len(d)):
        t[i]=t[i-1]
        if np.isfinite(v.iloc[i]) and np.isfinite(v.iloc[i-1]):
            if s.iloc[i]>z.iloc[i]+v.iloc[i] and s.iloc[i-1]<=z.iloc[i-1]+v.iloc[i-1]: t[i]=1
            if s.iloc[i]<z.iloc[i]-v.iloc[i] and s.iloc[i-1]>=z.iloc[i-1]-v.iloc[i-1]: t[i]=-1
    return t

def main():
    raw={}
    for s in SYMBOLS:
        d=eng.download_klines(s,"30m",START,END).reset_index(drop=True)
        d["time"]=pd.to_datetime(d["time"],utc=True)
        raw[s]=purple_cloud(d)
    oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
    eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
    rows=[]; allby=[]; alltr=[]
    try:
        for name,n,m in CASES:
            data={}
            for s,d0 in raw.items():
                d=d0.copy()
                if n:
                    t=zl(d,n,m)
                    d["pc_buy"]=d.pc_buy&(t==1)
                    d["pc_sell"]=d.pc_sell&(t==-1)
                data[s]=d[["time","high","low","close","pc_buy","pc_sell"]]
            r,tr,b=eng.run(data,name,SYMBOLS)
            r["test_period"]=f"{START} to {END}"; rows.append(r)
            if len(b): allby.append(b.assign(case=name))
            if len(tr): alltr.append(tr.assign(case=name))
            print("\n",r)
    finally:
        eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
    pd.DataFrame(rows).to_csv("backtest/data/pc_zerolag_true_oos_summary.csv",index=False)
    if allby: pd.concat(allby,ignore_index=True).to_csv("backtest/data/pc_zerolag_true_oos_by_symbol.csv",index=False)
    if alltr: pd.concat(alltr,ignore_index=True).to_csv("backtest/data/pc_zerolag_true_oos_trades.csv",index=False)
    print("\nTRUE OOS 2025-09-01 -> 2026-03-01\n",pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
