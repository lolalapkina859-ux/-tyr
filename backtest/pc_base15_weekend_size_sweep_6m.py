from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
WEEKDAY_NOTIONAL=60.; WEEKEND_SIZES=[0.,15.,30.,45.,60.]
LEVERAGE=20.; MMR=.005
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(data, weekend_notional):
 name=f"WEEKDAY60_WEEKEND{int(weekend_notional)}"
 ev=[]
 for s,d in data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; tr=[]; skipped=0; blocked=0
 maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def floating(): return sum(p["notional"]*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return sum(p["notional"] for p in pos.values())
 def close(s,px,tm):
  nonlocal bal
  p=pos[s]; pnl=p["notional"]*ret(p["side"],p["entry"],px); bal+=pnl
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,
             "entry_is_weekend":p["time"].weekday()>=5,"notional":p["notional"],"pnl_usdt":pnl})
  del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); last[s]=px
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  p=pos.get(s)
  if p and sig and sig!=p["side"]: close(s,px,tm)
  if sig and s not in pos:
   n=weekend_notional if tm.weekday()>=5 else WEEKDAY_NOTIONAL
   if n<=0: blocked+=1
   elif eq()-gross()/LEVERAGE>=n/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm,"notional":n}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time)
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 return {"variant":name,"weekday_notional":WEEKDAY_NOTIONAL,"weekend_notional":weekend_notional,
 "start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,
 "closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,
 "max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,
 "max_gross_notional":maxgross,"skipped_entries_margin":skipped,"blocked_weekend_entries":blocked,"liquidated":liq},t

def main():
 data={s:eng.load(s) for s in BASE15}; rows=[]; trades=[]
 for n in WEEKEND_SIZES:
  r,t=run(data,n); rows.append(r); trades.append(t); print(r)
 summary=pd.DataFrame(rows).sort_values("weekend_notional")
 summary.to_csv(OUT/"pc_base15_weekend_size_sweep_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_weekend_size_sweep_trades.csv",index=False)
 print("\nWEEKEND SIZE SWEEP\n",summary.to_string(index=False))
if __name__=="__main__": main()
