from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import purple_cloud, atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]; START_BALANCE=145.58352658; NOTIONAL=60.
CASES=[("BASE15",None,None),("ZL70_1.0",70,1.0),("ZL70_1.4",70,1.4)]
CUT=pd.Timestamp("2026-06-01",tz="UTC")

def zl(d,n,m):
 lag=(n-1)//2; s=d.close
 z=(s+(s-s.shift(lag))).ewm(span=n,adjust=False).mean()
 v=atr(d,n).rolling(n*3).max()*m; t=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)):
  t[i]=t[i-1]
  if np.isfinite(v.iloc[i]) and np.isfinite(v.iloc[i-1]):
   if s.iloc[i]>z.iloc[i]+v.iloc[i] and s.iloc[i-1]<=z.iloc[i-1]+v.iloc[i-1]: t[i]=1
   if s.iloc[i]<z.iloc[i]-v.iloc[i] and s.iloc[i-1]>=z.iloc[i-1]-v.iloc[i-1]: t[i]=-1
 return t

def main():
 raw={}
 for s in SYMBOLS:
  d=eng.download_klines(s,"30m","2026-03-01","2026-09-01").reset_index(drop=True)
  d["time"]=pd.to_datetime(d["time"],utc=True); raw[s]=purple_cloud(d)
 oldb,oldn=eng.START_BALANCE,eng.NOTIONAL; eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
 rows=[]; halfrows=[]; alltr=[]
 try:
  for name,n,m in CASES:
   data={}
   for s,d0 in raw.items():
    d=d0.copy()
    if n:
     t=zl(d,n,m); d["pc_buy"]=d.pc_buy&(t==1); d["pc_sell"]=d.pc_sell&(t==-1)
    data[s]=d[["time","high","low","close","pc_buy","pc_sell"]]
   r,tr,b=eng.run(data,name,SYMBOLS); rows.append(r)
   tr=tr.copy(); tr["case"]=name; alltr.append(tr)
   # No capital reset: split realized trades only for attribution; full-run DD stays authoritative.
   if len(tr):
    tc=None
    for c in ("exit_time","close_time","time","entry_time"):
     if c in tr.columns: tc=c; break
    if tc:
     tt=pd.to_datetime(tr[tc],utc=True)
     for label,mask in [("MAR_JUN",tt<CUT),("JUN_SEP",tt>=CUT)]:
      x=tr.loc[mask]
      pnlcol=next((c for c in ("pnl","pnl_usdt","net_pnl","profit") if c in x.columns),None)
      halfrows.append({"case":name,"period":label,"trades":len(x),"realized_pnl":float(x[pnlcol].sum()) if pnlcol else np.nan})
 finally:
  eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
 pd.DataFrame(rows).to_csv("backtest/data/pc_walkforward_compare_summary.csv",index=False)
 pd.DataFrame(halfrows).to_csv("backtest/data/pc_walkforward_compare_halves.csv",index=False)
 pd.concat(alltr,ignore_index=True).to_csv("backtest/data/pc_walkforward_compare_trades.csv",index=False)
 print("\nFULL CONTINUOUS COMPARISON\n",pd.DataFrame(rows).to_string(index=False))
 print("\nHALF ATTRIBUTION (NO CAPITAL RESET)\n",pd.DataFrame(halfrows).to_string(index=False))

if __name__=="__main__": main()
