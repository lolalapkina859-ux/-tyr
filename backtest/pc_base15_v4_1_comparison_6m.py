from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
import backtest.pc_base15_smart_exit_v3_6m as v3

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
MOM_LOSS=.60; SWING=3; MIN_BARS=3; GATE=.20
ATR_LEN=14; ATR_BUFFER=.15; WATCH_BARS=3; CONFIRM_N=2
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def add_indicators(d):
 d=v3.add_v3(d).copy().reset_index(drop=True)
 prev=d.close.shift(1)
 tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
 d["exit_atr"]=tr.ewm(alpha=1/ATR_LEN,adjust=False).mean()
 return d

def run(data, variant):
 name=variant
 ev=[]
 for s,d in data.items():
  for i,r in d.iterrows(): ev.append((r.time,s,i,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; trades=[]; skipped=0
 maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 watch_count=hold_count=smart_exits=0
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return len(pos)*NOTIONAL
 def closepos(s,px,tm,reason):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  trades.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,
   "entry_price":p["entry"],"exit_price":px,"exit_reason":reason,"pnl_usdt":pnl,"peak_return_pct":p["peak_ret"]*100,
   "exit_return_pct":ret(p["side"],p["entry"],px)*100,"bars_held":p["bars"]})
  del pos[s]
 for tm,s,i,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s)
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  if p:
   p["bars"]+=1
   cur=ret(p["side"],p["entry"],px); p["peak_ret"]=max(p["peak_ret"],cur)
   # Gate uses intrabar favorable extreme, exactly like Pine V4.1.
   peak_move=(float(r.high)/p["entry"]-1) if p["side"]=="LONG" else (p["entry"]/float(r.low)-1)
   p["peak_move"]=max(p["peak_move"],peak_move)
   if p["peak_move"]>=GATE: p["gate"]=True
   mom=max(float(r["hist"]),0) if p["side"]=="LONG" else max(-float(r["hist"]),0)
   p["peak_mom"]=max(p["peak_mom"],mom)
   piv=float(r["pivot_low"]) if p["side"]=="LONG" else float(r["pivot_high"])
   if not np.isnan(piv): p["last_pivot"]=piv

   if variant!="NATIVE_PC":
    if not p["armed"] and p["gate"] and p["bars"]>=MIN_BARS and p["peak_mom"]>0 and mom<=p["peak_mom"]*(1-MOM_LOSS) and cur>0 and not np.isnan(p["last_pivot"]):
     p["armed"]=True; p["armed_i"]=i; p["protected"]=p["last_pivot"]

    if variant=="V3_GATE20" and p["armed"] and not np.isnan(piv):
     if p["side"]=="LONG" and piv>p["protected"]: p["protected"]=piv
     if p["side"]=="SHORT" and piv<p["protected"]: p["protected"]=piv

    if variant=="V4_1_GATE20" and p["armed"]:
     # After reclaim/timeout, require a genuinely better newly confirmed swing.
     if p["wait_new"] and not np.isnan(piv):
      better=(p["side"]=="LONG" and piv>p["rejected"]) or (p["side"]=="SHORT" and piv<p["rejected"])
      if better:
       p["protected"]=piv; p["wait_new"]=False; p["rejected"]=np.nan
     elif not p["watch"] and not p["wait_new"] and not np.isnan(piv):
      if p["side"]=="LONG" and piv>p["protected"]: p["protected"]=piv
      if p["side"]=="SHORT" and piv<p["protected"]: p["protected"]=piv

   opposite=bool(sig and sig!=p["side"])
   do_exit=False; reason=""

   if variant=="V3_GATE20" and p["armed"] and i>p["armed_i"] and cur>0:
    broken=(p["side"]=="LONG" and px<p["protected"]) or (p["side"]=="SHORT" and px>p["protected"])
    if broken and not opposite: do_exit=True; reason="V3_EXIT"

   if variant=="V4_1_GATE20" and p["armed"] and not opposite:
    # Start WATCH only after buffered structure break.
    if not p["watch"] and not p["wait_new"] and i>p["armed_i"]:
     atr=float(r["exit_atr"])
     broken=(p["side"]=="LONG" and px < p["protected"]-atr*ATR_BUFFER) or (p["side"]=="SHORT" and px > p["protected"]+atr*ATR_BUFFER)
     if broken:
      p["watch"]=True; p["watch_i"]=i; p["broken"]=p["protected"]; watch_count+=1
    elif p["watch"]:
     age=i-p["watch_i"]
     reclaim=(p["side"]=="LONG" and px>p["broken"]) or (p["side"]=="SHORT" and px<p["broken"])
     if age>=1 and reclaim:
      p["rejected"]=p["broken"]; p["watch"]=False; p["wait_new"]=True; p["broken"]=np.nan; hold_count+=1
     else:
      structure=(p["side"]=="LONG" and px<p["broken"]) or (p["side"]=="SHORT" and px>p["broken"])
      momentum=(p["side"]=="LONG" and float(r["hist"])<0) or (p["side"]=="SHORT" and float(r["hist"])>0)
      priceconf=(p["side"]=="LONG" and px<float(r.open) and px<float(data[s].iloc[i-1].close)) or (p["side"]=="SHORT" and px>float(r.open) and px>float(data[s].iloc[i-1].close))
      score=int(structure)+int(momentum)+int(priceconf)
      if age>=1 and age<=WATCH_BARS and score>=CONFIRM_N and cur>0:
       do_exit=True; reason="V4_1_EXIT"
      elif age>WATCH_BARS:
       p["rejected"]=p["broken"]; p["watch"]=False; p["wait_new"]=True; p["broken"]=np.nan; hold_count+=1

   if do_exit:
    closepos(s,px,tm,reason); smart_exits+=1; p=None
   elif opposite:
    closepos(s,px,tm,"PC_OPPOSITE"); p=None

  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    mom=max(float(r["hist"]),0) if sig=="LONG" else max(-float(r["hist"]),0)
    lp=float(r["pivot_low"]) if sig=="LONG" else float(r["pivot_high"])
    pos[s]={"side":sig,"entry":px,"time":tm,"bars":0,"peak_ret":0.,"peak_move":0.,"gate":False,
      "peak_mom":mom,"last_pivot":lp,"armed":False,"armed_i":-1,"protected":np.nan,
      "watch":False,"watch_i":-1,"broken":np.nan,"wait_new":False,"rejected":np.nan}
    maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   rr=data[s].iloc[-1]; closepos(s,float(rr.close),rr.time,"END")
 t=pd.DataFrame(trades); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 return {"variant":name,"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,
  "closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"max_floating_dd_pct":maxdd,
  "min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,
  "watch_events":watch_count,"hold_reclaims_or_timeouts":hold_count,"smart_exits":smart_exits,"liquidated":liq},t

def main():
 data={s:add_indicators(eng.load(s)) for s in BASE15}; rows=[]; ts=[]
 for variant in ("NATIVE_PC","V3_GATE20","V4_1_GATE20"):
  r,t=run(data,variant); rows.append(r); ts.append(t); print(r)
 summary=pd.DataFrame(rows)
 summary.to_csv(OUT/"pc_base15_v4_1_comparison_summary.csv",index=False)
 pd.concat(ts,ignore_index=True).to_csv(OUT/"pc_base15_v4_1_comparison_trades.csv",index=False)
 print("\nBASE15 V4.1 COMPARISON\n",summary.to_string(index=False))

if __name__=="__main__": main()
