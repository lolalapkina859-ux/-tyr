from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-06-01"; END="2026-09-01"; TF="30m"
SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","AVAXUSDT","SUIUSDT"]

def test(sym):
 d=download_klines(sym,TF,START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True); pc=purple_cloud(d); rows=[]; pos=None
 for i in range(len(d)):
  b=d.iloc[i]; buy=bool(pc.iloc[i]["pc_buy"]); sell=bool(pc.iloc[i]["pc_sell"])
  if pos is not None and ((pos["side"]=="LONG" and sell) or (pos["side"]=="SHORT" and buy)):
   ex=float(b.close); ret=(ex/pos["entry"]-1) if pos["side"]=="LONG" else (pos["entry"]/ex-1)
   rows.append({"symbol":sym,"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"return_pct":ret*100,"bars":i-pos["i"],"exit_reason":"OPPOSITE"}); pos=None
  if pos is None and (buy or sell): pos={"side":"LONG" if buy else "SHORT","entry":float(b.close),"time":b.time,"i":i}
 if pos:
  b=d.iloc[-1]; ex=float(b.close); ret=(ex/pos["entry"]-1) if pos["side"]=="LONG" else (pos["entry"]/ex-1)
  rows.append({"symbol":sym,"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"return_pct":ret*100,"bars":len(d)-1-pos["i"],"exit_reason":"END_OF_DATA"})
 return pd.DataFrame(rows)

def summarize(x,name):
 w=x[x.return_pct>0]; l=x[x.return_pct<0]
 return {"symbol":name,"trades":len(x),"wins":len(w),"losses":len(l),"winrate":round(100*len(w)/len(x),2) if len(x) else 0,"sum_trade_returns_pct":round(x.return_pct.sum(),4),"avg_return_per_signal_pct":round(x.return_pct.mean(),4),"median_return_pct":round(x.return_pct.median(),4),"avg_winner_pct":round(w.return_pct.mean(),4) if len(w) else 0,"avg_loser_pct":round(l.return_pct.mean(),4) if len(l) else 0,"max_winner_pct":round(x.return_pct.max(),4),"max_loser_pct":round(x.return_pct.min(),4),"avg_hold_hours":round(x.bars.mean()*0.5,2)}

def main():
 allx=[]; summaries=[]
 for s in SYMBOLS:
  try:
   x=test(s); allx.append(x); summaries.append(summarize(x,s)); print(summaries[-1])
  except Exception as e: print(f"SKIP {s}: {e}")
 if not allx: raise RuntimeError("No symbol data")
 x=pd.concat(allx,ignore_index=True); summaries.append(summarize(x,"ALL")); out=Path("backtest/data"); out.mkdir(parents=True,exist_ok=True)
 x.to_csv(out/"purple_cloud_10coins_30m_no_sl_detail.csv",index=False); pd.DataFrame(summaries).to_csv(out/"purple_cloud_10coins_30m_no_sl_summary.csv",index=False); print(pd.DataFrame(summaries).to_string(index=False))
if __name__=="__main__": main()
