from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
BASE15=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT","FLOWUSDT"]
TIMEFRAMES=["30m","1h","4h","1d"]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def load(s,tf):
 d=download_klines(s,tf,START,END).reset_index(drop=True)
 d["time"]=pd.to_datetime(d["time"],utc=True)
 return purple_cloud(d)[["time","high","low","close","pc_buy","pc_sell"]]

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(data,tf):
 ev=[]
 for s,d in data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; tr=[]; skipped=0; maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return NOTIONAL*len(pos)
 def close(s,px,tm):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  tr.append({"timeframe":tf,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"entry_price":p["entry"],"exit_price":px,"pnl_usdt":pnl}); del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); last[s]=px
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  p=pos.get(s)
  if p and sig and sig!=p["side"]: close(s,px,tm)
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time)
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 long=t[t.side=="LONG"] if len(t) else t; short=t[t.side=="SHORT"] if len(t) else t
 return {"timeframe":tf,"symbols":len(data),"start_balance":START_BALANCE,"final_balance":final,
  "net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,
  "closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,
  "avg_pnl_per_trade":t.pnl_usdt.mean() if len(t) else 0,
  "long_trades":len(long),"long_pnl":long.pnl_usdt.sum() if len(long) else 0,
  "short_trades":len(short),"short_pnl":short.pnl_usdt.sum() if len(short) else 0,
  "max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,
  "max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq},t

def main():
 rows=[]; trades=[]
 for tf in TIMEFRAMES:
  print(f"Loading BASE15 {tf}...")
  data={s:load(s,tf) for s in BASE15}
  r,t=run(data,tf); rows.append(r); trades.append(t); print(r)
 summary=pd.DataFrame(rows)
 summary.to_csv(OUT/"pc_base15_timeframe_comparison_6m_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_timeframe_comparison_6m_trades.csv",index=False)
 print("\nTIMEFRAME COMPARISON\n",summary.to_string(index=False))

if __name__=="__main__": main()
