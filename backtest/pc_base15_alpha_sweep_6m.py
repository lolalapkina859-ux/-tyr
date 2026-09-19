from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import atr, vwma, rma, supertrend_direction

BASE15 = list(eng.BASE14) + ["FLOWUSDT"]
ALPHAS = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2.0]
START_BALANCE = float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL = 60.0
PERIOD=20; BPT=.2; SPT=.2; ATR_LEN=20; ST_FACTOR=2.0
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def purple_cloud_alpha(df, alpha):
    d=df.copy().reset_index(drop=True)
    n1=int(np.ceil(PERIOD/4)); n2=int(np.ceil(PERIOD/2))
    x2=atr(d,PERIOD)*alpha
    xh=d.close+x2; xl=d.close-x2
    hl2=(d.high+d.low)/2
    a1=vwma(hl2*d.volume,d.volume,n1)/vwma(d.volume,d.volume,n1)
    a2=vwma(hl2*d.volume,d.volume,n2)/vwma(d.volume,d.volume,n2)
    a3=2*a1-a2
    a4=vwma(a3,d.volume,PERIOD)
    b1=rma(d.close,PERIOD)
    a5=2*a4*b1/(a4+b1)
    buy=(a5<=xl)&(d.close>b1*(1+BPT*.01))
    sell=(a5>=xh)&(d.close<b1*(1-SPT*.01))
    xs=np.zeros(len(d),dtype=int)
    for i in range(1,len(d)):
        xs[i]=1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
    changed=pd.Series(xs).ne(pd.Series(xs).shift(1))
    direction=supertrend_direction(d,ATR_LEN,ST_FACTOR)
    d["pc_buy"]=buy&changed
    d["pc_sell"]=sell&changed
    d["pc_strong_buy"]=d.pc_buy&(direction<0)
    d["pc_strong_sell"]=d.pc_sell&(direction>0)
    return d

def load_raw(s):
    d=eng.download_klines(s,"30m",eng.START,eng.END).reset_index(drop=True)
    d["time"]=pd.to_datetime(d["time"],utc=True)
    return d

def main():
    print("Loading",len(BASE15),"symbols...")
    raw={s:load_raw(s) for s in BASE15}
    old_balance,old_notional=eng.START_BALANCE,eng.NOTIONAL
    rows=[]; trades=[]
    try:
        eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
        for alpha in ALPHAS:
            data={}
            for s,d in raw.items():
                pc=purple_cloud_alpha(d,alpha)
                data[s]=pc[["time","high","low","close","pc_buy","pc_sell"]]
            name=f"BASE15_ALPHA_{alpha:.1f}"
            r,t,_=eng.run(data,name,BASE15)
            r["alpha"]=alpha; r["position_notional_usdt"]=NOTIONAL
            rows.append(r); t["alpha"]=alpha; trades.append(t)
            print(r)
    finally:
        eng.START_BALANCE=old_balance; eng.NOTIONAL=old_notional
    summary=pd.DataFrame(rows).sort_values("final_balance",ascending=False)
    summary.to_csv(OUT/"pc_base15_alpha_sweep_summary.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_alpha_sweep_trades.csv",index=False)
    print("\nALPHA SWEEP\n",summary.to_string(index=False))

if __name__=="__main__": main()
