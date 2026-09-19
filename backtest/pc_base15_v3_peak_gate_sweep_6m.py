from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
MOM_LOSS=.60; SWING=3; MIN_BARS=3
THRESHOLDS=[None,0.10,0.15,0.20,0.25,0.30]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def rr(side,e,x): return x/e-1 if side=="LONG" else e/x-1
def favorable(side,e,hi,lo): return hi/e-1 if side=="LONG" else e/lo-1

def prep(d):
 d=d.copy().reset_index(drop=True)
 e12=d.close.ewm(span=12,adjust=False).mean(); e26=d.close.ewm(span=26,adjust=False).mean()
 m=e12-e26; d["hist"]=m-m.ewm(span=9,adjust=False).mean()
 pl=np.full(len(d),np.nan); ph=np.full(len(d),np.nan); lo=d.low.to_numpy(); hi=d.high.to_numpy()
 for i in range(SWING,len(d)-SWING):
  if lo[i] <= lo[i-SWING:i+SWING+1].min(): pl[i+SWING]=lo[i]
  if hi[i] >= hi[i-SWING:i+SWING+1].max(): ph[i+SWING]=hi[i]
 d["pl"]=pl; d["ph"]=ph
 return d

def run(data,threshold):
 name="NATIVE" if threshold is None else f"V3_FULL_AFTER_PEAK_{int(threshold*100)}PCT"
 ev=sorted([(r.time,s,i,r) for s,d in data.items() for i,r in d.iterrows()],key=lambda x:(x[0],x[1]))
 bal=START; pos={}; last={}; rows=[]; skips=0; peakEq=START; minEq=START; maxdd=0.; maxpos=0; maxgross=0.; liq=False
 def floating(): return sum(NOTIONAL*rr(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return len(pos)*NOTIONAL
 def close(s,px,tm,reason):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*rr(p["side"],p["entry"],px); bal+=pnl
  rows.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"reason":reason,
               "entry_price":p["entry"],"exit_price":px,"peak_favorable_pct":p["peakfav"]*100,"pnl_usdt":pnl})
  del pos[s]
 for tm,s,i,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s)
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  if p:
   p["bars"]+=1
   p["peakfav"]=max(p["peakfav"],favorable(p["side"],p["entry"],float(r.high),float(r.low)))
   cur=rr(p["side"],p["entry"],px)
   mom=max(float(r["hist"]),0) if p["side"]=="LONG" else max(-float(r["hist"]),0); p["peakmom"]=max(p["peakmom"],mom)
   piv=float(r["pl"]) if p["side"]=="LONG" else float(r["ph"])
   if not np.isnan(piv): p["lastpivot"]=piv
   eligible=threshold is not None and p["peakfav"]>=threshold
   if eligible and not p["armed"] and p["bars"]>=MIN_BARS and p["peakmom"]>0 and mom<=p["peakmom"]*(1-MOM_LOSS) and cur>0 and not np.isnan(p["lastpivot"]):
    p["armed"]=True; p["armed_i"]=i; p["protected"]=p["lastpivot"]
   elif p["armed"] and not np.isnan(piv):
    if p["side"]=="LONG" and piv>p["protected"]: p["protected"]=piv
    if p["side"]=="SHORT" and piv<p["protected"]: p["protected"]=piv
   v3=p["armed"] and i>p["armed_i"] and cur>0 and ((p["side"]=="LONG" and px<p["protected"]) or (p["side"]=="SHORT" and px>p["protected"]))
   opp=sig and sig!=p["side"]
   if v3 and not opp: close(s,px,tm,"V3_PROTECTED_EXIT")
   elif opp and s in pos: close(s,px,tm,"PC_OPPOSITE")
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    mom=max(float(r["hist"]),0) if sig=="LONG" else max(-float(r["hist"]),0); lp=float(r["pl"]) if sig=="LONG" else float(r["ph"])
    pos[s]={"side":sig,"entry":px,"time":tm,"bars":0,"peakfav":0.,"peakmom":mom,"lastpivot":lp,"armed":False,"armed_i":-1,"protected":np.nan}
    maxpos=max(maxpos,len(pos))
   else: skips+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); minEq=min(minEq,e); peakEq=max(peakEq,e); maxdd=min(maxdd,(e/peakEq-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time,"END")
 t=pd.DataFrame(rows); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 return {"variant":name,"peak_gate_pct":0 if threshold is None else threshold*100,"final_balance":final,"net_profit_usdt":final-START,
 "closed_trades":len(t),"winrate_pct":wins/len(t)*100 if len(t) else 0,"v3_protected_exits":int((t.reason=="V3_PROTECTED_EXIT").sum()) if len(t) else 0,
 "max_floating_dd_pct":maxdd,"min_equity":minEq,"max_gross_notional":maxgross,"skipped_entries_margin":skips,"liquidated":liq},t

def main():
 data={s:prep(eng.load(s)) for s in BASE15}; summary=[]; trades=[]
 for th in THRESHOLDS:
  r,t=run(data,th); summary.append(r); trades.append(t); print(r)
 pd.DataFrame(summary).to_csv(OUT/"pc_base15_v3_peak_gate_sweep_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_v3_peak_gate_sweep_trades.csv",index=False)
 print(pd.DataFrame(summary).to_string(index=False))
if __name__=="__main__": main()
