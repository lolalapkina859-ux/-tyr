from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import purple_cloud, atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=145.58352658; NOTIONAL=60.; N=70; M=1.0
PERIODS=[("IS_MAR_SEP","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]

def zl(d):
 s=d.close; lag=(N-1)//2
 z=(s+(s-s.shift(lag))).ewm(span=N,adjust=False).mean()
 v=atr(d,N).rolling(N*3).max()*M
 t=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)):
  t[i]=t[i-1]
  if np.isfinite(v.iloc[i]) and np.isfinite(v.iloc[i-1]):
   if s.iloc[i]>z.iloc[i]+v.iloc[i] and s.iloc[i-1]<=z.iloc[i-1]+v.iloc[i-1]: t[i]=1
   if s.iloc[i]<z.iloc[i]-v.iloc[i] and s.iloc[i-1]>=z.iloc[i-1]-v.iloc[i-1]: t[i]=-1
 return pd.Series(t,index=d.index)

def armed_signals(d):
 # A blocked opposite PC signal is remembered. It executes when ZL later confirms.
 # A new opposite PC signal cancels/replaces the pending direction.
 z=zl(d); buy=d.pc_buy.to_numpy(bool); sell=d.pc_sell.to_numpy(bool)
 outb=np.zeros(len(d),bool); outs=np.zeros(len(d),bool)
 pos=0; armed=0
 for i in range(len(d)):
  # New PC direction is authoritative for pending intent.
  if buy[i]:
   if z.iloc[i]==1:
    if pos!=1: outb[i]=True; pos=1
    armed=0
   else:
    armed=1
  elif sell[i]:
   if z.iloc[i]==-1:
    if pos!=-1: outs[i]=True; pos=-1
    armed=0
   else:
    armed=-1
  # If an earlier blocked PC signal is armed, execute on later ZL confirmation.
  if not outb[i] and not outs[i]:
   if armed==1 and z.iloc[i]==1:
    if pos!=1: outb[i]=True; pos=1
    armed=0
   elif armed==-1 and z.iloc[i]==-1:
    if pos!=-1: outs[i]=True; pos=-1
    armed=0
 return outb,outs

def filtered(d):
 z=zl(d); x=d.copy()
 x["pc_buy"]=x.pc_buy&(z==1); x["pc_sell"]=x.pc_sell&(z==-1); return x

def main():
 rows=[]; bys=[]
 oldb,oldn=eng.START_BALANCE,eng.NOTIONAL; eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
 try:
  for pname,start,end in PERIODS:
   raw={}
   for s in SYMBOLS:
    d=eng.download_klines(s,"30m",start,end).reset_index(drop=True)
    d["time"]=pd.to_datetime(d["time"],utc=True); raw[s]=purple_cloud(d)
   for mode in ("BASE15","ZL70_1.0","ARMED_ZL70_1.0"):
    data={}
    for s,d0 in raw.items():
     d=d0.copy()
     if mode=="ZL70_1.0": d=filtered(d)
     elif mode=="ARMED_ZL70_1.0":
      b,se=armed_signals(d); d["pc_buy"]=b; d["pc_sell"]=se
     data[s]=d[["time","high","low","close","pc_buy","pc_sell"]]
    r,tr,by=eng.run(data,f"{pname}_{mode}",SYMBOLS)
    r["period"]=pname; r["mode"]=mode; rows.append(r)
    if len(by): bys.append(by.assign(period=pname,mode=mode))
    print("\n",r)
 finally:
  eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
 out=pd.DataFrame(rows)
 out.to_csv("backtest/data/pc_zerolag_armed_compare_summary.csv",index=False)
 if bys: pd.concat(bys,ignore_index=True).to_csv("backtest/data/pc_zerolag_armed_compare_by_symbol.csv",index=False)
 print("\nARMED COMPARISON\n",out.to_string(index=False))

if __name__=="__main__": main()
