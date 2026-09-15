from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; HORIZONS=(8,16,32)
ASSETS={
 "BTC":("BTCUSDT",Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
 "ETH":("ETHUSDT",Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
 "ZEC":("ZECUSDT",Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}
def gate(r): return int(r.score)>=(88 if str(r.side)=="SHORT" else 70)
def excursion(c,start,side,e,r,n):
 x=c.iloc[start+1:min(start+1+n,len(c))]
 if x.empty or r<=0:return np.nan,np.nan,np.nan
 if side=="LONG":return (x.high.max()-e)/r,(e-x.low.min())/r,(x.close.iloc[-1]-e)/r
 return (e-x.low.min())/r,(x.high.max()-e)/r,(e-x.close.iloc[-1])/r
def metrics(rec,c,j,side,e,r):
 for h in HORIZONS:
  a,b,z=excursion(c,j,side,e,r,h);rec[f"mfe_{h}"]=a;rec[f"mae_{h}"]=b;rec[f"end_{h}"]=z
 return rec

def main():
 out=Path("backtest/data");out.mkdir(parents=True,exist_ok=True);rows=[]
 for asset,(symbol,path) in ASSETS.items():
  c15=download_klines(symbol,"15m",START,END).reset_index(drop=True);c15.time=pd.to_datetime(c15.time,utc=True)
  c30=download_klines(symbol,"30m",START,END).reset_index(drop=True);c30.time=pd.to_datetime(c30.time,utc=True)
  pc=purple_cloud(c30).reset_index(drop=True).copy()
  if "time" not in pc.columns:pc.insert(0,"time",c30.time.values)
  pc=pc.loc[:,~pc.columns.duplicated()].copy();pc.time=pd.to_datetime(pc.time,utc=True)
  # Build non-lookahead PC state from confirmed STRONG signals. State changes only after 30m candle closes.
  ev=[]
  for _,q in pc.iterrows():
   if bool(q.get("pc_strong_buy",False)):ev.append((q.time+pd.Timedelta(minutes=30),"LONG"))
   if bool(q.get("pc_strong_sell",False)):ev.append((q.time+pd.Timedelta(minutes=30),"SHORT"))
  ev=sorted(ev,key=lambda z:z[0])
  tv=pd.read_csv(path);tv.entry_time=pd.to_datetime(tv.entry_time,utc=True,errors="coerce")
  for col in ("score","entry","sl"):tv[col]=pd.to_numeric(tv[col],errors="coerce")
  tv=tv[tv.apply(gate,axis=1)&tv.entry_time.notna()&tv.entry.notna()&tv.sl.notna()].copy()
  for _,r in tv.iterrows():
   side=str(r.side);idx=c15.index[c15.time>=r.entry_time]
   if not len(idx):continue
   j=int(idx[0]);risk=abs(float(r.entry)-float(r.sl))
   if risk<=0:continue
   base=metrics({"asset":asset,"system":"BASE_TV_FVG75","side":side,"entry_time":r.entry_time,"entry":float(r.entry),"risk":risk},c15,j,side,float(r.entry),risk);rows.append(base)
   state=None
   for t,s in ev:
    if t>r.entry_time:break
    state=s
   if state==side:
    filt=metrics({"asset":asset,"system":"PC30_ALIGNED_TV_FVG75","side":side,"entry_time":r.entry_time,"entry":float(r.entry),"risk":risk},c15,j,side,float(r.entry),risk);rows.append(filt)
 d=pd.DataFrame(rows);sums=[]
 for (asset,system),x in list(d.groupby(["asset","system"]))+[(('ALL',s),d[d.system==s]) for s in d.system.unique()]:
  rec={"asset":asset,"system":system,"entries":len(x)}
  for h in HORIZONS:
   rec[f"avg_mfe_{h}"]=round(x[f"mfe_{h}"].mean(),4);rec[f"avg_mae_{h}"]=round(x[f"mae_{h}"].mean(),4);rec[f"positive_close_{h}_pct"]=round(100*(x[f"end_{h}"]>0).mean(),2);rec[f"mfe_ge_1r_{h}_pct"]=round(100*(x[f"mfe_{h}"]>=1).mean(),2);rec[f"mfe_ge_2r_{h}_pct"]=round(100*(x[f"mfe_{h}"]>=2).mean(),2)
  sums.append(rec)
 s=pd.DataFrame(sums);d.to_csv(out/"pc30_state_filter_tv_fvg75_detail.csv",index=False);s.to_csv(out/"pc30_state_filter_tv_fvg75_summary.csv",index=False);print(s.to_string(index=False))
if __name__=="__main__":main()
