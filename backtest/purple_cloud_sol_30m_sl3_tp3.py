from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; TF="30m"; SL_PCT=0.03; TP_PCT=0.03

def main():
 d=download_klines("SOLUSDT",TF,START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True); pc=purple_cloud(d)
 rows=[]; pos=None
 for i in range(len(d)):
  b=d.iloc[i]; buy=bool(pc.iloc[i]["pc_buy"]); sell=bool(pc.iloc[i]["pc_sell"])
  if pos is not None:
   if pos["side"]=="LONG": sl=pos["entry"]*(1-SL_PCT); tp=pos["entry"]*(1+TP_PCT); hit_sl=float(b.low)<=sl; hit_tp=float(b.high)>=tp; opposite=sell
   else: sl=pos["entry"]*(1+SL_PCT); tp=pos["entry"]*(1-TP_PCT); hit_sl=float(b.high)>=sl; hit_tp=float(b.low)<=tp; opposite=buy
   # Conservative handling when both SL and TP are touched inside the same 30m candle.
   if hit_sl and hit_tp: ex=sl; reason="AMBIGUOUS_SL_FIRST"
   elif hit_sl: ex=sl; reason="SL"
   elif hit_tp: ex=tp; reason="TP"
   elif opposite: ex=float(b.close); reason="OPPOSITE"
   else: ex=None; reason=None
   if ex is not None:
    pct=((ex/pos["entry"])-1) if pos["side"]=="LONG" else ((pos["entry"]/ex)-1)
    rows.append({"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"exit_reason":reason,"return_pct":pct*100,"r":pct/SL_PCT,"bars":i-pos["i"]}); pos=None
  # Opposite signal closes old trade first; same signal can immediately establish new direction at this candle close.
  if pos is None and (buy or sell):
   pos={"side":"LONG" if buy else "SHORT","entry":float(b.close),"time":b.time,"i":i}
 if pos is not None:
  b=d.iloc[-1]; ex=float(b.close); pct=((ex/pos["entry"])-1) if pos["side"]=="LONG" else ((pos["entry"]/ex)-1)
  rows.append({"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"exit_reason":"END_OF_DATA","return_pct":pct*100,"r":pct/SL_PCT,"bars":len(d)-1-pos["i"]})
 x=pd.DataFrame(rows); out=Path("backtest/data"); out.mkdir(parents=True,exist_ok=True)
 wins=int((x.r>0).sum()); losses=int((x.r<0).sum()); summary=pd.DataFrame([{"asset":"SOL","timeframe":"30m","mode":"NORMAL","sl_pct":3,"tp_pct":3,"trades":len(x),"wins":wins,"losses":losses,"winrate":round(100*wins/len(x),2),"total_r":round(x.r.sum(),4),"avg_r":round(x.r.mean(),4),"max_loss_r":round(x.r.min(),4),"max_loss_pct":round(x.return_pct.min(),4),"tp_exits":int((x.exit_reason=="TP").sum()),"sl_exits":int(x.exit_reason.isin(["SL","AMBIGUOUS_SL_FIRST"]).sum()),"opposite_exits":int((x.exit_reason=="OPPOSITE").sum())}])
 x.to_csv(out/"purple_cloud_sol_30m_sl3_tp3_detail.csv",index=False); summary.to_csv(out/"purple_cloud_sol_30m_sl3_tp3_summary.csv",index=False); print(summary.to_string(index=False))
if __name__=="__main__": main()
