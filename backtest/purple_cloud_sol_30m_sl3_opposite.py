from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; TF="30m"; SL_PCT=0.03

def main():
 d=download_klines("SOLUSDT",TF,START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True); pc=purple_cloud(d)
 rows=[]; pos=None
 for i in range(len(d)):
  b=d.iloc[i]; buy=bool(pc.iloc[i]["pc_buy"]); sell=bool(pc.iloc[i]["pc_sell"])
  if pos is not None:
   if pos["side"]=="LONG":
    sl=pos["entry"]*(1-SL_PCT); hit_sl=float(b.low)<=sl; opposite=sell
   else:
    sl=pos["entry"]*(1+SL_PCT); hit_sl=float(b.high)>=sl; opposite=buy
   if hit_sl: ex=sl; reason="SL"
   elif opposite: ex=float(b.close); reason="OPPOSITE"
   else: ex=None; reason=None
   if ex is not None:
    ret=((ex/pos["entry"])-1) if pos["side"]=="LONG" else ((pos["entry"]/ex)-1)
    rows.append({"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"exit_reason":reason,"return_pct":ret*100,"r":ret/SL_PCT,"bars":i-pos["i"]}); pos=None
  if pos is None and (buy or sell): pos={"side":"LONG" if buy else "SHORT","entry":float(b.close),"time":b.time,"i":i}
 if pos is not None:
  b=d.iloc[-1]; ex=float(b.close); ret=((ex/pos["entry"])-1) if pos["side"]=="LONG" else ((pos["entry"]/ex)-1)
  rows.append({"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"exit_reason":"END_OF_DATA","return_pct":ret*100,"r":ret/SL_PCT,"bars":len(d)-1-pos["i"]})
 x=pd.DataFrame(rows); out=Path("backtest/data"); out.mkdir(parents=True,exist_ok=True)
 winners=x[x.return_pct>0]; losers=x[x.return_pct<0]
 s=pd.DataFrame([{"asset":"SOL","timeframe":"30m","mode":"NORMAL","sl_pct":3,"trades":len(x),"wins":len(winners),"losses":len(losers),"winrate":round(100*len(winners)/len(x),2),"total_return_pct_sum":round(x.return_pct.sum(),4),"avg_return_pct_all":round(x.return_pct.mean(),4),"median_return_pct_all":round(x.return_pct.median(),4),"avg_profit_pct_winners":round(winners.return_pct.mean(),4) if len(winners) else 0,"median_profit_pct_winners":round(winners.return_pct.median(),4) if len(winners) else 0,"max_profit_pct":round(x.return_pct.max(),4),"avg_loss_pct_losers":round(losers.return_pct.mean(),4) if len(losers) else 0,"max_loss_pct":round(x.return_pct.min(),4),"total_r":round(x.r.sum(),4),"avg_r":round(x.r.mean(),4),"sl_exits":int((x.exit_reason=="SL").sum()),"opposite_exits":int((x.exit_reason=="OPPOSITE").sum()),"avg_hold_hours":round(x.bars.mean()*0.5,2)}])
 x.to_csv(out/"purple_cloud_sol_30m_sl3_opposite_detail.csv",index=False); s.to_csv(out/"purple_cloud_sol_30m_sl3_opposite_summary.csv",index=False); print(s.to_string(index=False))
if __name__=="__main__": main()
