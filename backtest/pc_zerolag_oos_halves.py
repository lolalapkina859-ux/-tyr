from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import purple_cloud, atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=145.58352658; NOTIONAL=60.
PARAMS=[(70,1.0),(70,1.4)]
PERIODS=[("MAR_JUN","2026-03-01","2026-06-01"),("JUN_SEP","2026-06-01","2026-09-01")]

def zl(d,n,m):
 lag=(n-1)//2; src=d.close
 z=(src+(src-src.shift(lag))).ewm(span=n,adjust=False).mean()
 v=atr(d,n).rolling(n*3).max()*m
 t=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)):
  t[i]=t[i-1]
  if np.isfinite(v.iloc[i]) and np.isfinite(v.iloc[i-1]):
   if src.iloc[i]>z.iloc[i]+v.iloc[i] and src.iloc[i-1]<=z.iloc[i-1]+v.iloc[i-1]: t[i]=1
   if src.iloc[i]<z.iloc[i]-v.iloc[i] and src.iloc[i-1]>=z.iloc[i-1]-v.iloc[i-1]: t[i]=-1
 return t

def main():
 rows=[]; allby=[]
 old_start,old_end,oldb,oldn=eng.START,eng.END,eng.START_BALANCE,eng.NOTIONAL
 eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
 try:
  for pname,start,end in PERIODS:
   eng.START=start; eng.END=end
   raw={}
   for s in SYMBOLS:
    d=eng.download_klines(s,"30m",start,end).reset_index(drop=True)
    d["time"]=pd.to_datetime(d["time"],utc=True); raw[s]=purple_cloud(d)
   for n,m in PARAMS:
    data={}
    for s,d0 in raw.items():
     d=d0.copy(); t=zl(d,n,m)
     d["pc_buy"]=d.pc_buy & (t==1); d["pc_sell"]=d.pc_sell & (t==-1)
     data[s]=d[["time","high","low","close","pc_buy","pc_sell"]]
    name=f"{pname}_ZL{n}_{m}"
    r,tr,b=eng.run(data,name,SYMBOLS)
    r["period"]=pname; r["start"]=start; r["end"]=end; r["zl_length"]=n; r["zl_mult"]=m
    rows.append(r)
    if len(b): allby.append(b.assign(period=pname,zl_length=n,zl_mult=m))
    print("\n",r)
 finally:
  eng.START=old_start; eng.END=old_end; eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
 out=pd.DataFrame(rows)
 out.to_csv("backtest/data/pc_zerolag_oos_halves_summary.csv",index=False)
 if allby: pd.concat(allby,ignore_index=True).to_csv("backtest/data/pc_zerolag_oos_halves_by_symbol.csv",index=False)
 print("\nOOS HALF-PERIOD RESULTS\n",out.to_string(index=False))

if __name__=="__main__": main()
