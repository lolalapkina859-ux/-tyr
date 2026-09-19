from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def load(s):
 d=eng.load(s).copy()
 return d

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(data,name,block_weekend_entries=False):
 ev=[]
 for s,d in data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; tr=[]; skipped=0; blocked=0
 maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return NOTIONAL*len(pos)
 def close(s,px,tm,reason):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,
             "entry_weekday":p["time"].day_name(),"entry_is_weekend":p["time"].weekday()>=5,
             "exit_weekday":tm.day_name(),"pnl_usdt":pnl,"reason":reason})
  del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); last[s]=px
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  p=pos.get(s)
  if p and sig and sig!=p["side"]:
   close(s,px,tm,"opposite_signal")
  if sig and s not in pos:
   if block_weekend_entries and tm.weekday()>=5:
    blocked+=1
   elif eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time,"end")
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 return {"variant":name,"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,
 "closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"max_floating_dd_pct":maxdd,
 "min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,
 "skipped_entries_margin":skipped,"blocked_weekend_entries":blocked,"liquidated":liq},t

def main():
 data={s:load(s) for s in BASE15}
 rows=[]; alltr=[]
 for name,block in [("BASE15_NORMAL",False),("BASE15_NO_WEEKEND_NEW_ENTRIES",True)]:
  r,t=run(data,name,block); rows.append(r); alltr.append(t); print(r)
 normal=alltr[0]
 stats=normal.groupby("entry_is_weekend").agg(trades=("pnl_usdt","size"),pnl_usdt=("pnl_usdt","sum"),
  avg_pnl=("pnl_usdt","mean"),winrate=("pnl_usdt",lambda x:(x>0).mean()*100)).reset_index()
 stats["group"]=stats.entry_is_weekend.map({False:"MON_FRI",True:"SAT_SUN"})
 summary=pd.DataFrame(rows)
 summary.to_csv(OUT/"pc_base15_weekend_filter_summary.csv",index=False)
 pd.concat(alltr,ignore_index=True).to_csv(OUT/"pc_base15_weekend_filter_trades.csv",index=False)
 stats.to_csv(OUT/"pc_base15_weekend_entry_stats.csv",index=False)
 print("\nPORTFOLIO\n",summary.to_string(index=False)); print("\nENTRY DAY STATS\n",stats.to_string(index=False))
if __name__=="__main__": main()
