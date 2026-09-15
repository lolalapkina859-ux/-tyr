from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; HORIZONS=(8,16,32) # 2h,4h,8h in 15m bars
ASSETS={
 "BTC":("BTCUSDT",Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
 "ETH":("ETHUSDT",Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
 "ZEC":("ZECUSDT",Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}

def gate(r): return int(r.score)>=(88 if str(r.side)=="SHORT" else 70)

def excursion(c15, start, side, entry, risk, n):
 x=c15.iloc[start+1:min(start+1+n,len(c15))]
 if x.empty or risk<=0:return (np.nan,np.nan,np.nan)
 if side=="LONG":
  mfe=(float(x.high.max())-entry)/risk; mae=(entry-float(x.low.min()))/risk; end=(float(x.close.iloc[-1])-entry)/risk
 else:
  mfe=(entry-float(x.low.min()))/risk; mae=(float(x.high.max())-entry)/risk; end=(entry-float(x.close.iloc[-1]))/risk
 return mfe,mae,end

def atr(df,n=20):
 prev=df.close.shift(1); tr=pd.concat([df.high-df.low,(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False).mean()

def main():
 out=Path("backtest/data");out.mkdir(parents=True,exist_ok=True); rows=[]
 for asset,(symbol,path) in ASSETS.items():
  c15=download_klines(symbol,"15m",START,END).reset_index(drop=True);c15.time=pd.to_datetime(c15.time,utc=True);c15["atr20"]=atr(c15)
  c30=download_klines(symbol,"30m",START,END).reset_index(drop=True);c30.time=pd.to_datetime(c30.time,utc=True)
  pc=purple_cloud(c30).reset_index(drop=True).copy()
  if "time" not in pc.columns:pc.insert(0,"time",c30.time.values)
  pc=pc.loc[:,~pc.columns.duplicated()].copy();pc.time=pd.to_datetime(pc.time,utc=True)

  # Trade Vision: actual LIMIT-filled historical setups only; same production score gate.
  tv=pd.read_csv(path);tv["entry_time"]=pd.to_datetime(tv.entry_time,utc=True,errors="coerce")
  for col in ("score","entry","sl"):
   tv[col]=pd.to_numeric(tv[col],errors="coerce")
  tv=tv[tv.apply(gate,axis=1)&tv.entry_time.notna()&tv.entry.notna()&tv.sl.notna()].copy()
  for _,r in tv.iterrows():
   idx=c15.index[c15.time>=r.entry_time]
   if not len(idx):continue
   i=int(idx[0]); risk=abs(float(r.entry)-float(r.sl))
   rec={"asset":asset,"system":"TRADE_VISION_15M","side":str(r.side),"signal_time":r.entry_time,"entry":float(r.entry),"risk":risk}
   for h in HORIZONS:
    mfe,mae,endr=excursion(c15,i,str(r.side),float(r.entry),risk,h);rec[f"mfe_{h}"]=mfe;rec[f"mae_{h}"]=mae;rec[f"end_{h}"]=endr
   rows.append(rec)

  # Purple Cloud STRONG 30m: enter only after signal candle closes, at its close.
  for i,r in pc.iterrows():
   buy=bool(r.get("pc_strong_buy",False)); sell=bool(r.get("pc_strong_sell",False))
   if not (buy or sell):continue
   side="LONG" if buy else "SHORT"; sigclose=r.time+pd.Timedelta(minutes=30)
   idx=c15.index[c15.time>=sigclose]
   if not len(idx):continue
   j=int(idx[0]); entry=float(c15.iloc[j].open); risk=float(c15.iloc[j].atr20)*2.0
   if not np.isfinite(risk) or risk<=0:continue
   rec={"asset":asset,"system":"PURPLE_CLOUD_30M","side":side,"signal_time":sigclose,"entry":entry,"risk":risk}
   for h in HORIZONS:
    mfe,mae,endr=excursion(c15,j,side,entry,risk,h);rec[f"mfe_{h}"]=mfe;rec[f"mae_{h}"]=mae;rec[f"end_{h}"]=endr
   rows.append(rec)

 d=pd.DataFrame(rows); summaries=[]
 for (asset,system),x in list(d.groupby(["asset","system"]))+[(('ALL',s),d[d.system==s]) for s in d.system.unique()]:
  rec={"asset":asset,"system":system,"entries":len(x)}
  for h in HORIZONS:
   rec[f"avg_mfe_{h}"]=round(x[f"mfe_{h}"].mean(),4);rec[f"median_mfe_{h}"]=round(x[f"mfe_{h}"].median(),4)
   rec[f"avg_mae_{h}"]=round(x[f"mae_{h}"].mean(),4);rec[f"positive_close_{h}_pct"]=round(100*(x[f"end_{h}"]>0).mean(),2)
   rec[f"mfe_ge_1r_{h}_pct"]=round(100*(x[f"mfe_{h}"]>=1).mean(),2);rec[f"mfe_ge_2r_{h}_pct"]=round(100*(x[f"mfe_{h}"]>=2).mean(),2)
  summaries.append(rec)
 s=pd.DataFrame(summaries);d.to_csv(out/"entry_quality_tv15_vs_pc30_detail.csv",index=False);s.to_csv(out/"entry_quality_tv15_vs_pc30_summary.csv",index=False);print(s.to_string(index=False))
if __name__=="__main__":main()
