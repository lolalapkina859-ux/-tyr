from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=145.58352658
NOTIONAL=60.
N=70
MULTS=[1.0,1.4]
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]

def zero_lag_signals(d,mult):
    s=d.close
    lag=(N-1)//2
    z=(s+(s-s.shift(lag))).ewm(span=N,adjust=False).mean()
    vol=atr(d,N).rolling(N*3).max()*mult
    trend=np.zeros(len(d),dtype=int)
    buy=np.zeros(len(d),dtype=bool); sell=np.zeros(len(d),dtype=bool)
    for i in range(1,len(d)):
        trend[i]=trend[i-1]
        if not (np.isfinite(z.iloc[i]) and np.isfinite(z.iloc[i-1]) and np.isfinite(vol.iloc[i]) and np.isfinite(vol.iloc[i-1])):
            continue
        up_now=z.iloc[i]+vol.iloc[i]; up_prev=z.iloc[i-1]+vol.iloc[i-1]
        dn_now=z.iloc[i]-vol.iloc[i]; dn_prev=z.iloc[i-1]-vol.iloc[i-1]
        if s.iloc[i]>up_now and s.iloc[i-1]<=up_prev:
            if trend[i-1] != 1: buy[i]=True
            trend[i]=1
        elif s.iloc[i]<dn_now and s.iloc[i-1]>=dn_prev:
            if trend[i-1] != -1: sell[i]=True
            trend[i]=-1
    x=d.copy()
    x["pc_buy"]=buy
    x["pc_sell"]=sell
    return x

def main():
    rows=[]; bys=[]
    oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
    eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
    try:
        for pname,start,end in PERIODS:
            raw={}
            for sym in SYMBOLS:
                d=eng.download_klines(sym,"30m",start,end).reset_index(drop=True)
                d["time"]=pd.to_datetime(d["time"],utc=True)
                raw[sym]=d
            for mult in MULTS:
                data={}
                for sym,d in raw.items():
                    x=zero_lag_signals(d,mult)
                    data[sym]=x[["time","high","low","close","pc_buy","pc_sell"]]
                label=f"{pname}_ZERO_LAG_70_{mult}_PURE"
                r,tr,by=eng.run(data,label,SYMBOLS)
                r["period"]=pname; r["mode"]=f"ZL70/{mult}_PURE_NEXT_OPPOSITE"
                rows.append(r)
                if len(by): bys.append(by.assign(period=pname,mode=r["mode"]))
                print("\n",r)
    finally:
        eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
    out=pd.DataFrame(rows)
    out.to_csv("backtest/data/zero_lag_pure_next_signal_summary.csv",index=False)
    if bys:
        pd.concat(bys,ignore_index=True).to_csv("backtest/data/zero_lag_pure_next_signal_by_symbol.csv",index=False)
    print("\nPURE ZERO LAG RESULTS\n",out.to_string(index=False))

if __name__=="__main__":
    main()
