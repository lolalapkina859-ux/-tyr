from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; TIMEFRAME="30m"; MAX_HOLD_BARS=7*24*2

def run(mode):
 d=download_klines("SOLUSDT",TIMEFRAME,START,END).reset_index(drop=True);d["time"]=pd.to_datetime(d["time"],utc=True);pc=purple_cloud(d)
 buy_col="pc_buy" if mode=="NORMAL" else "pc_strong_buy";sell_col="pc_sell" if mode=="NORMAL" else "pc_strong_sell";rows=[];pos=None
 for i in range(len(d)):
  b=d.iloc[i];buy=bool(pc.iloc[i][buy_col]);sell=bool(pc.iloc[i][sell_col])
  if pos:
   opp=sell if pos["side"]=="LONG" else buy;timeout=i-pos["i"]>=MAX_HOLD_BARS
   if opp or timeout:
    ex=float(b.close);rr=(ex-pos["entry"])/pos["risk"] if pos["side"]=="LONG" else (pos["entry"]-ex)/pos["risk"]
    rows.append({"mode":mode,"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"initial_risk":pos["risk"],"realized_r":rr,"bars":i-pos["i"],"exit_reason":"OPPOSITE" if opp else "TIMEOUT"});pos=None
  if pos is None and (buy or sell):
   side="LONG" if buy else "SHORT";entry=float(b.close);w=d.iloc[max(0,i-20):i+1];risk=float((w.high-w.low).mean())
   if pd.notna(risk) and risk>0:pos={"side":side,"entry":entry,"risk":risk,"i":i,"time":b.time}
 if pos:
  b=d.iloc[-1];ex=float(b.close);rr=(ex-pos["entry"])/pos["risk"] if pos["side"]=="LONG" else (pos["entry"]-ex)/pos["risk"]
  rows.append({"mode":mode,"side":pos["side"],"entry_time":pos["time"],"exit_time":b.time,"entry":pos["entry"],"exit":ex,"initial_risk":pos["risk"],"realized_r":rr,"bars":len(d)-1-pos["i"],"exit_reason":"END_OF_DATA"})
 return pd.DataFrame(rows)

def summary(x,mode):
 wins=int((x.realized_r>0).sum());losses=int((x.realized_r<0).sum())
 return {"asset":"SOL","mode":mode,"trades":len(x),"wins":wins,"losses":losses,"winrate":round(100*wins/len(x),2) if len(x) else 0,"total_r":round(x.realized_r.sum(),4) if len(x) else 0,"avg_r":round(x.realized_r.mean(),4) if len(x) else 0,"median_r":round(x.realized_r.median(),4) if len(x) else 0,"avg_bars":round(x.bars.mean(),2) if len(x) else 0}

def main():
 out=Path("backtest/data");out.mkdir(parents=True,exist_ok=True);parts=[];s=[]
 for mode in ("NORMAL","STRONG"):
  x=run(mode);parts.append(x);s.append(summary(x,mode))
 d=pd.concat(parts,ignore_index=True);sm=pd.DataFrame(s);d.to_csv(out/"purple_cloud_sol_30m_detail.csv",index=False);sm.to_csv(out/"purple_cloud_sol_30m_summary.csv",index=False);print(sm.to_string(index=False))
if __name__=="__main__":main()
