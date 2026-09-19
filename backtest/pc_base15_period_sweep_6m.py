from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import atr, vwma, rma, supertrend_direction

BASE15=list(eng.BASE14)+["FLOWUSDT"]
PERIODS=[15,18,20,22,25,30]
ALPHA=1.5; BPT=.2; SPT=.2; ATR_LEN=20; ST_FACTOR=2.0
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.0
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def pc_period(df, period):
 d=df.copy().reset_index(drop=True)
 n1=int(np.ceil(period/4)); n2=int(np.ceil(period/2))
 x2=atr(d,period)*ALPHA; xh=d.close+x2; xl=d.close-x2
 hl2=(d.high+d.low)/2
 a1=vwma(hl2*d.volume,d.volume,n1)/vwma(d.volume,d.volume,n1)
 a2=vwma(hl2*d.volume,d.volume,n2)/vwma(d.volume,d.volume,n2)
 a3=2*a1-a2; a4=vwma(a3,d.volume,period); b1=rma(d.close,period)
 a5=2*a4*b1/(a4+b1)
 buy=(a5<=xl)&(d.close>b1*(1+BPT*.01))
 sell=(a5>=xh)&(d.close<b1*(1-SPT*.01))
 xs=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)):
  xs[i]=1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
 changed=pd.Series(xs).ne(pd.Series(xs).shift(1))
 direction=supertrend_direction(d,ATR_LEN,ST_FACTOR)
 d["pc_buy"]=buy&changed; d["pc_sell"]=sell&changed
 d["pc_strong_buy"]=d.pc_buy&(direction<0); d["pc_strong_sell"]=d.pc_sell&(direction>0)
 return d

def raw(s):
 d=eng.download_klines(s,"30m",eng.START,eng.END).reset_index(drop=True)
 d["time"]=pd.to_datetime(d["time"],utc=True); return d

def main():
 print("Loading",len(BASE15),"symbols...")
 source={s:raw(s) for s in BASE15}; rows=[]; trades=[]
 oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
 try:
  eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
  for period in PERIODS:
   data={}
   for s,d in source.items():
    p=pc_period(d,period)
    data[s]=p[["time","high","low","close","pc_buy","pc_sell"]]
   name=f"BASE15_PERIOD_{period}"
   r,t,_=eng.run(data,name,BASE15)
   r["period"]=period; r["alpha"]=ALPHA; r["position_notional_usdt"]=NOTIONAL
   rows.append(r); t["period"]=period; trades.append(t); print(r)
 finally:
  eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
 summary=pd.DataFrame(rows).sort_values("final_balance",ascending=False)
 summary.to_csv(OUT/"pc_base15_period_sweep_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_period_sweep_trades.csv",index=False)
 print("\nPERIOD SWEEP\n",summary.to_string(index=False))
if __name__=="__main__": main()
