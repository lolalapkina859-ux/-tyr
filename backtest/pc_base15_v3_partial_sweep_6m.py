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
PARTIALS=[0.0,0.25,0.33,0.50,1.0]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def rr(side,e,x): return x/e-1 if side=="LONG" else e/x-1

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

def run(data,frac):
 name="NATIVE" if frac==0 else f"V3_PARTIAL_{int(round(frac*100))}"
 ev=sorted([(r.time,s,i,r) for s,d in data.items() for i,r in d.iterrows()],key=lambda x:(x[0],x[1]))
 bal=START; pos={}; last={}; trades=[]; skips=0; peakEq=START; minEq=START; maxdd=0.; maxpos=0; maxgross=0.; liq=False
 def floating(): return sum(p["remaining"]*rr(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return sum(p["remaining"] for p in pos.values())
 def realize(s,px,tm,amt,reason):
  nonlocal bal
  p=pos[s]; pnl=amt*rr(p["side"],p["entry"],px); bal+=pnl
  trades.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"reason":reason,
                 "notional_closed":amt,"entry_price":p["entry"],"exit_price":px,"pnl_usdt":pnl})
  p["remaining"]-=amt
  if p["remaining"]<1e-9: del pos[s]
 for tm,s,i,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s)
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  if p:
   p["bars"]+=1; cur=rr(p["side"],p["entry"],px)
   mom=max(float(r["hist"]),0) if p["side"]=="LONG" else max(-float(r["hist"]),0)
   p["peakmom"]=max(p["peakmom"],mom)
   piv=float(r["pl"]) if p["side"]=="LONG" else float(r["ph"])
   if not np.isnan(piv): p["lastpivot"]=piv
   if frac>0 and not p["armed"] and p["bars"]>=MIN_BARS and p["peakmom"]>0 and mom<=p["peakmom"]*(1-MOM_LOSS) and cur>0 and not np.isnan(p["lastpivot"]):
    p["armed"]=True; p["armed_i"]=i; p["protected"]=p["lastpivot"]
   elif frac>0 and p["armed"] and not np.isnan(piv):
    if p["side"]=="LONG" and piv>p["protected"]: p["protected"]=piv
    if p["side"]=="SHORT" and piv<p["protected"]: p["protected"]=piv
   v3=frac>0 and p["armed"] and not p["partial_done"] and i>p["armed_i"] and cur>0 and ((p["side"]=="LONG" and px<p["protected"]) or (p["side"]=="SHORT" and px>p["protected"]))
   opp=sig and sig!=p["side"]
   if v3 and not opp:
    amt=min(NOTIONAL*frac,p["remaining"]); realize(s,px,tm,amt,"V3_PARTIAL" if frac<1 else "V3_EXIT")
    if s in pos: pos[s]["partial_done"]=True
    p=pos.get(s)
   if opp and s in pos: realize(s,px,tm,pos[s]["remaining"],"PC_OPPOSITE"); p=None
  if sig and s not in pos:
   need=NOTIONAL/LEVERAGE
   if eq()-gross()/LEVERAGE>=need:
    mom=max(float(r["hist"]),0) if sig=="LONG" else max(-float(r["hist"]),0)
    lp=float(r["pl"]) if sig=="LONG" else float(r["ph"])
    pos[s]={"side":sig,"entry":px,"time":tm,"remaining":NOTIONAL,"bars":0,"peakmom":mom,"lastpivot":lp,
            "armed":False,"armed_i":-1,"protected":np.nan,"partial_done":False}
    maxpos=max(maxpos,len(pos))
   else: skips+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); minEq=min(minEq,e); peakEq=max(peakEq,e); maxdd=min(maxdd,(e/peakEq-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; realize(s,float(r.close),r.time,pos[s]["remaining"],"END")
 t=pd.DataFrame(trades); final=bal if not liq else eq()
 # aggregate logical entries by symbol+entry_time so partial close does not inflate trade count
 logical=t.groupby(["symbol","side","entry_time"],as_index=False).pnl_usdt.sum() if len(t) else pd.DataFrame(columns=["pnl_usdt"])
 return {"variant":name,"partial_pct":frac*100,"final_balance":final,"net_profit_usdt":final-START,
         "logical_trades":len(logical),"winrate_pct":(logical.pnl_usdt>0).mean()*100 if len(logical) else 0,
         "v3_partial_events":int((t.reason.str.startswith("V3")).sum()) if len(t) else 0,
         "max_floating_dd_pct":maxdd,"min_equity":minEq,"max_gross_notional":maxgross,
         "skipped_entries_margin":skips,"liquidated":liq},t

def main():
 data={s:prep(eng.load(s)) for s in BASE15}; rows=[]; alltr=[]
 for f in PARTIALS:
  r,t=run(data,f); rows.append(r); alltr.append(t); print(r)
 pd.DataFrame(rows).to_csv(OUT/"pc_base15_v3_partial_sweep_summary.csv",index=False)
 pd.concat(alltr,ignore_index=True).to_csv(OUT/"pc_base15_v3_partial_sweep_trades.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
