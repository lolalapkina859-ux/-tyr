from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import purple_cloud, atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=145.58352658; NOTIONAL=60.
LENGTHS=[50,60,70,80,90]; MULTS=[1.0,1.2,1.4]

def zl(d,n,m):
 lag=int(np.floor((n-1)/2)); src=d.close
 z= (src+(src-src.shift(lag))).ewm(span=n,adjust=False).mean()
 v=atr(d,n).rolling(n*3).max()*m
 t=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)):
  t[i]=t[i-1]
  if np.isfinite(v.iloc[i]) and np.isfinite(v.iloc[i-1]):
   up=z.iloc[i]+v.iloc[i]; pu=z.iloc[i-1]+v.iloc[i-1]
   lo=z.iloc[i]-v.iloc[i]; pl=z.iloc[i-1]-v.iloc[i-1]
   if src.iloc[i]>up and src.iloc[i-1]<=pu: t[i]=1
   if src.iloc[i]<lo and src.iloc[i-1]>=pl: t[i]=-1
 return t

def main():
 raw={}
 for s in SYMBOLS:
  d=eng.download_klines(s,"30m",eng.START,eng.END).reset_index(drop=True)
  d["time"]=pd.to_datetime(d["time"],utc=True); raw[s]=purple_cloud(d)
 rows=[]; allby=[]
 oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
 eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
 try:
  for n in LENGTHS:
   for m in MULTS:
    data={}
    for s,d0 in raw.items():
     d=d0.copy(); t=zl(d,n,m)
     d["pc_buy"]=d.pc_buy & (t==1); d["pc_sell"]=d.pc_sell & (t==-1)
     data[s]=d[["time","high","low","close","pc_buy","pc_sell"]]
    name=f"ZL{n}_{m}"
    r,tr,b=eng.run(data,name,SYMBOLS); r["zl_length"]=n; r["zl_mult"]=m; rows.append(r)
    if len(b): allby.append(b.assign(zl_length=n,zl_mult=m))
    print(r)
 finally:
  eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
 out=pd.DataFrame(rows).sort_values("net_profit_usdt",ascending=False)
 out.to_csv("backtest/data/pc_zerolag_sweep_summary.csv",index=False)
 if allby: pd.concat(allby,ignore_index=True).to_csv("backtest/data/pc_zerolag_sweep_by_symbol.csv",index=False)
 print("\nZERO LAG SWEEP RANKING\n",out.to_string(index=False))

if __name__=="__main__": main()
