from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; MAX_HOLD_15M=7*24*4
W=(.25,.25,.25,.25)
ASSETS={
 "BTC":("BTCUSDT",Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
 "ETH":("ETHUSDT",Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
 "ZEC":("ZECUSDT",Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}
def gate(r): return int(r.score)>=(88 if str(r.side)=="SHORT" else 70)
def tps(r):
 out=[]
 for c in ("tp1","tp2","tp3","tp4"):
  v=pd.to_numeric(r.get(c),errors="coerce")
  if pd.notna(v): out.append(float(v))
 return out
def rr(side,e,sl,p):
 risk=abs(e-sl)
 return ((p-e)/risk if side=="LONG" else (e-p)/risk) if risk else 0
def hit(side,h,l,p): return h>=p if side=="LONG" else l<=p

def simulate(r,c15,pc30,mode):
 side=str(r.side); e=float(r.entry); sl0=float(r.sl); ts=tps(r)
 idx=c15.index[c15.time>=r.entry_time]
 if not len(idx) or not ts:return ("SKIP",0,0,0,None)
 start=int(idx[0]); end=min(start+MAX_HOLD_15M,len(c15)); sl=sl0; hi=0; realized=0.; rem=1.; pc_used=0
 for i in range(start,end):
  b=c15.iloc[i]; h=float(b.high); l=float(b.low)
  stop=(l<=sl if side=="LONG" else h>=sl)
  new=hi
  for j,t in enumerate(ts,1):
   if hit(side,h,l,t):new=max(new,j)
  pcopp=False; pcprice=None
  if mode=="PC30_NO_BE":
   closed=pc30[pc30.time+pd.Timedelta(minutes=30)<=b.time+pd.Timedelta(minutes=15)]
   if len(closed):
    q=closed.iloc[-1]
    pcopp=bool(q.pc_strong_sell) if side=="LONG" else bool(q.pc_strong_buy)
    pcprice=float(b.close)
  if stop and (new>hi or pcopp): return ("AMBIGUOUS",0,hi,pc_used,None)
  if pcopp and new>hi:return ("AMBIGUOUS",0,hi,pc_used,None)
  if new>hi:
   for n in range(hi+1,new+1):
    if n<=4:
     realized+=W[n-1]*max(0,rr(side,e,sl0,ts[n-1])); rem-=W[n-1]
   hi=new
   if mode=="BASE_TP1_BE" and hi>=1: sl=e
   if hi>=len(ts):return ("TP_FINAL",realized,hi,pc_used,None)
  if mode=="PC30_NO_BE" and pcopp:
   realized+=rem*rr(side,e,sl0,pcprice); pc_used=1
   return ("PC30_EXIT",realized,hi,pc_used,b.time)
  if stop:
   if mode=="BASE_TP1_BE" and hi>=1: return ("BE",realized,hi,pc_used,None)
   realized-=rem
   return ("SL",realized,hi,pc_used,None)
 return ("TIMEOUT",realized,hi,pc_used,None)

def summ(d,asset,mode):
 v=d[~d.status_new.isin(["SKIP","AMBIGUOUS","TIMEOUT"])].copy()
 if not len(v):return {"asset":asset,"mode":mode,"trades":0}
 eq=v.realized_r_new.cumsum(); dd=eq-eq.cummax(); wins=(v.realized_r_new>0)
 return {"asset":asset,"mode":mode,"trades":len(v),"wins":int(wins.sum()),"losses":int((v.realized_r_new<0).sum()),"winrate":round(100*wins.mean(),2),"total_r":round(v.realized_r_new.sum(),4),"avg_r":round(v.realized_r_new.mean(),4),"max_dd_r":round(dd.min(),4),"tp2_plus":int((v.highest_tp_new>=2).sum()),"tp3_plus":int((v.highest_tp_new>=3).sum()),"pc30_exits":int(v.pc30_exit.sum())}

def main():
 out=Path("backtest/data");out.mkdir(parents=True,exist_ok=True); details=[]; sums=[]
 for asset,(symbol,path) in ASSETS.items():
  s=pd.read_csv(path);s["entry_time"]=pd.to_datetime(s.entry_time,utc=True,errors="coerce")
  for c in ("score","entry","sl","tp1","tp2","tp3","tp4"):
   if c in s:s[c]=pd.to_numeric(s[c],errors="coerce")
  s=s[s.apply(gate,axis=1)&s.entry_time.notna()].copy()
  c15=download_klines(symbol,"15m",START,END).reset_index(drop=True);c15["time"]=pd.to_datetime(c15["time"],utc=True)
  c30=download_klines(symbol,"30m",START,END).reset_index(drop=True);c30["time"]=pd.to_datetime(c30["time"],utc=True)
  pc30=purple_cloud(c30).reset_index(drop=True).copy()
  # purple_cloud may already return source OHLCV/time columns. Never concat a second
  # time column: duplicate labels break pandas boolean filtering/reindexing.
  if "time" not in pc30.columns:
   pc30.insert(0,"time",c30["time"].values)
  else:
   pc30["time"]=pd.to_datetime(pc30["time"],utc=True)
  pc30=pc30.loc[:,~pc30.columns.duplicated()].copy()
  for mode in ("BASE_TP1_BE","PC30_NO_BE"):
   rows=[]
   for ix,r in s.iterrows():
    st,val,hi,used,xt=simulate(r,c15,pc30,mode)
    rows.append({**r.to_dict(),"asset":asset,"mode":mode,"status_new":st,"realized_r_new":val,"highest_tp_new":hi,"pc30_exit":used,"pc30_exit_time":xt})
   d=pd.DataFrame(rows);details.append(d);sums.append(summ(d,asset,mode))
 detail=pd.concat(details,ignore_index=True)
 for mode in ("BASE_TP1_BE","PC30_NO_BE"):sums.append(summ(detail[detail.mode==mode],"ALL",mode))
 summary=pd.DataFrame(sums);detail.to_csv(out/"tradevision_pc30_management_detail.csv",index=False);summary.to_csv(out/"tradevision_pc30_management_summary.csv",index=False);print(summary.to_string(index=False))
if __name__=="__main__":main()
