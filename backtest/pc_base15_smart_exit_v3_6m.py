from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
MOM_LOSS=.60; SWING=3; MIN_BARS=3
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def add_v3(d):
 d=d.copy().reset_index(drop=True)
 ema12=d.close.ewm(span=12,adjust=False).mean(); ema26=d.close.ewm(span=26,adjust=False).mean()
 macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean(); d["hist"]=macd-sig
 # Pine pivot is confirmed SWING bars later. Store confirmed pivot value on confirmation bar.
 pl=np.full(len(d),np.nan); ph=np.full(len(d),np.nan)
 lo=d.low.to_numpy(); hi=d.high.to_numpy()
 for i in range(SWING, len(d)-SWING):
  if lo[i] <= lo[i-SWING:i+SWING+1].min(): pl[i+SWING]=lo[i]
  if hi[i] >= hi[i-SWING:i+SWING+1].max(): ph[i+SWING]=hi[i]
 d["pivot_low_confirmed"]=pl; d["pivot_high_confirmed"]=ph
 return d

def run(data, use_v3):
 name="V3_60_SWING3_MIN3" if use_v3 else "NATIVE_PC"
 ev=[]
 for s,d in data.items():
  for i,r in d.iterrows(): ev.append((r.time,s,i,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; trades=[]; skipped=0
 maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return len(pos)*NOTIONAL
 def close(s,px,tm,reason):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  peakret=p["peak_ret"]; exitret=ret(p["side"],p["entry"],px)
  capture=(exitret/peakret*100) if peakret>0 else np.nan
  trades.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,
   "entry_price":p["entry"],"exit_price":px,"exit_reason":reason,"pnl_usdt":pnl,
   "peak_return_pct":peakret*100,"exit_return_pct":exitret*100,"peak_capture_pct":capture,
   "bars_held":p["bars"]})
  del pos[s]
 for tm,s,i,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s)
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  if p:
   p["bars"]+=1
   cur=ret(p["side"],p["entry"],px); p["peak_ret"]=max(p["peak_ret"],cur)
   mom=max(float(r["hist"]),0) if p["side"]=="LONG" else max(-float(r["hist"]),0)
   p["peak_mom"]=max(p["peak_mom"],mom)
   piv=float(r["pivot_low_confirmed"]) if p["side"]=="LONG" else float(r["pivot_high_confirmed"])
   if not np.isnan(piv): p["last_pivot"]=piv
   if use_v3 and not p["armed"] and p["bars"]>=MIN_BARS and p["peak_mom"]>0 and mom<=p["peak_mom"]*(1-MOM_LOSS) and cur>0 and not np.isnan(p["last_pivot"]):
    p["armed"]=True; p["armed_i"]=i; p["protected"]=p["last_pivot"]
   elif use_v3 and p["armed"] and not np.isnan(piv):
    if p["side"]=="LONG" and piv>p["protected"]: p["protected"]=piv
    if p["side"]=="SHORT" and piv<p["protected"]: p["protected"]=piv
   v3exit=use_v3 and p["armed"] and i>p["armed_i"] and cur>0 and ((p["side"]=="LONG" and px<p["protected"]) or (p["side"]=="SHORT" and px>p["protected"]))
   opposite=sig and sig!=p["side"]
   if v3exit and not opposite: close(s,px,tm,"V3_EXIT"); p=None
   elif opposite: close(s,px,tm,"PC_OPPOSITE"); p=None
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    mom=max(float(r["hist"]),0) if sig=="LONG" else max(-float(r["hist"]),0)
    lp=float(r["pivot_low_confirmed"]) if sig=="LONG" else float(r["pivot_high_confirmed"])
    pos[s]={"side":sig,"entry":px,"time":tm,"bars":0,"peak_ret":0.,"peak_mom":mom,
            "last_pivot":lp,"armed":False,"armed_i":-1,"protected":np.nan}
    maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   rr=data[s].iloc[-1]; close(s,float(rr.close),rr.time,"END")
 t=pd.DataFrame(trades); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 v3=t[t.exit_reason=="V3_EXIT"] if len(t) else t
 return {"variant":name,"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,
  "closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"max_floating_dd_pct":maxdd,
  "min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,
  "v3_exits":len(v3),"avg_peak_capture_pct_v3":v3.peak_capture_pct.mean() if len(v3) else np.nan,"liquidated":liq},t

def main():
 data={s:add_v3(eng.load(s)) for s in BASE15}; rows=[]; ts=[]
 for v in (False,True):
  r,t=run(data,v); rows.append(r); ts.append(t); print(r)
 summary=pd.DataFrame(rows); summary.to_csv(OUT/"pc_base15_smart_exit_v3_summary.csv",index=False)
 pd.concat(ts,ignore_index=True).to_csv(OUT/"pc_base15_smart_exit_v3_trades.csv",index=False)
 print("\nSMART EXIT V3\n",summary.to_string(index=False))
if __name__=="__main__": main()
