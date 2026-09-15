from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; HORIZONS=(8,16,32)
# PC trigger must appear after the TV setup and before its pending setup becomes stale.
WINDOW_BARS_15M=48
ASSETS={
 "BTC":("BTCUSDT",Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
 "ETH":("ETHUSDT",Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
 "ZEC":("ZECUSDT",Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

def gate(r): return int(r.score)>=(88 if str(r.side)=="SHORT" else 70)
def atr(df,n=20):
 prev=df.close.shift(1); tr=pd.concat([df.high-df.low,(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False).mean()
def excursion(c15,start,side,entry,risk,n):
 x=c15.iloc[start+1:min(start+1+n,len(c15))]
 if x.empty or risk<=0:return np.nan,np.nan,np.nan
 if side=="LONG": return (float(x.high.max())-entry)/risk,(entry-float(x.low.min()))/risk,(float(x.close.iloc[-1])-entry)/risk
 return (entry-float(x.low.min()))/risk,(float(x.high.max())-entry)/risk,(entry-float(x.close.iloc[-1]))/risk

def add_metrics(rec,c15,j,side,entry,risk):
 for h in HORIZONS:
  mfe,mae,endr=excursion(c15,j,side,entry,risk,h);rec[f"mfe_{h}"]=mfe;rec[f"mae_{h}"]=mae;rec[f"end_{h}"]=endr
 return rec

def main():
 out=Path("backtest/data");out.mkdir(parents=True,exist_ok=True); rows=[]
 for asset,(symbol,path) in ASSETS.items():
  c15=download_klines(symbol,"15m",START,END).reset_index(drop=True);c15.time=pd.to_datetime(c15.time,utc=True);c15["atr20"]=atr(c15)
  c30=download_klines(symbol,"30m",START,END).reset_index(drop=True);c30.time=pd.to_datetime(c30.time,utc=True)
  pc=purple_cloud(c30).reset_index(drop=True).copy()
  if "time" not in pc.columns:pc.insert(0,"time",c30.time.values)
  pc=pc.loc[:,~pc.columns.duplicated()].copy();pc.time=pd.to_datetime(pc.time,utc=True)
  pc["signal_close"]=pc.time+pd.Timedelta(minutes=30)

  tv=pd.read_csv(path);tv["entry_time"]=pd.to_datetime(tv.entry_time,utc=True,errors="coerce")
  for col in ("score","entry","sl"):
   tv[col]=pd.to_numeric(tv[col],errors="coerce")
  tv=tv[tv.apply(gate,axis=1)&tv.entry_time.notna()&tv.entry.notna()&tv.sl.notna()].copy()

  for _,r in tv.iterrows():
   side=str(r.side); risk_tv=abs(float(r.entry)-float(r.sl)); idx=c15.index[c15.time>=r.entry_time]
   if not len(idx) or risk_tv<=0:continue
   j=int(idx[0])
   rows.append(add_metrics({"asset":asset,"system":"TV_FVG75","side":side,"context_time":r.entry_time,"entry_time":r.entry_time,"entry":float(r.entry),"risk":risk_tv},c15,j,side,float(r.entry),risk_tv))

   deadline=r.entry_time+pd.Timedelta(minutes=15*WINDOW_BARS_15M)
   if side=="LONG": mask=(pc.signal_close>=r.entry_time)&(pc.signal_close<=deadline)&pc.pc_strong_buy.astype(bool)
   else: mask=(pc.signal_close>=r.entry_time)&(pc.signal_close<=deadline)&pc.pc_strong_sell.astype(bool)
   hits=pc[mask]
   if hits.empty:continue
   q=hits.iloc[0]; et=q.signal_close; jj=c15.index[c15.time>=et]
   if not len(jj):continue
   k=int(jj[0]); entry=float(c15.iloc[k].open)
   # Same original TV structural invalidation. If PC fires beyond invalidation, reject it.
   if (side=="LONG" and entry<=float(r.sl)) or (side=="SHORT" and entry>=float(r.sl)):continue
   risk=abs(entry-float(r.sl))
   rows.append(add_metrics({"asset":asset,"system":"TV_CONTEXT_PC30","side":side,"context_time":r.entry_time,"entry_time":et,"entry":entry,"risk":risk},c15,k,side,entry,risk))

 d=pd.DataFrame(rows); sums=[]
 for (asset,system),x in list(d.groupby(["asset","system"]))+[(('ALL',s),d[d.system==s]) for s in d.system.unique()]:
  rec={"asset":asset,"system":system,"entries":len(x)}
  for h in HORIZONS:
   rec[f"avg_mfe_{h}"]=round(x[f"mfe_{h}"].mean(),4);rec[f"avg_mae_{h}"]=round(x[f"mae_{h}"].mean(),4)
   rec[f"positive_close_{h}_pct"]=round(100*(x[f"end_{h}"]>0).mean(),2);rec[f"mfe_ge_1r_{h}_pct"]=round(100*(x[f"mfe_{h}"]>=1).mean(),2);rec[f"mfe_ge_2r_{h}_pct"]=round(100*(x[f"mfe_{h}"]>=2).mean(),2)
  sums.append(rec)
 s=pd.DataFrame(sums);d.to_csv(out/"tv_context_pc30_trigger_detail.csv",index=False);s.to_csv(out/"tv_context_pc30_trigger_summary.csv",index=False);print(s.to_string(index=False))
if __name__=="__main__":main()
