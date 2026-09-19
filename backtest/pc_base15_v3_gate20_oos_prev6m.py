from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

BASE15=["ZECUSDT","USELESSUSDT","HYPEUSDT","XRPUSDT","JTOUSDT","FETUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT","FLOWUSDT"]
START_DATE="2025-09-01"; END_DATE="2026-03-01"; TF="30m"
START=float(os.getenv("PC_TEST_START_BALANCE","145.58352658")); NOTIONAL=60.; LEVERAGE=20.; MMR=.005
MOM_LOSS=.60; SWING=3; MIN_BARS=3; GATE=.20
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def rr(side,e,x): return x/e-1 if side=="LONG" else e/x-1
def fav(side,e,h,l): return h/e-1 if side=="LONG" else e/l-1

def load(s):
 d=download_klines(s,TF,START_DATE,END_DATE)
 if d is None or len(d)<100: raise RuntimeError(f"{s}: insufficient data")
 d=purple_cloud(d).reset_index(drop=True)
 e12=d.close.ewm(span=12,adjust=False).mean(); e26=d.close.ewm(span=26,adjust=False).mean(); m=e12-e26
 d["hist"]=m-m.ewm(span=9,adjust=False).mean()
 pl=np.full(len(d),np.nan); ph=np.full(len(d),np.nan); lo=d.low.to_numpy(); hi=d.high.to_numpy()
 for i in range(SWING,len(d)-SWING):
  if lo[i]<=lo[i-SWING:i+SWING+1].min(): pl[i+SWING]=lo[i]
  if hi[i]>=hi[i-SWING:i+SWING+1].max(): ph[i+SWING]=hi[i]
 d["pl"]=pl; d["ph"]=ph
 return d

def run(data,use_v3):
 name="V3_GATE20" if use_v3 else "NATIVE"
 ev=sorted([(r.time,s,i,r) for s,d in data.items() for i,r in d.iterrows()],key=lambda x:(x[0],x[1]))
 bal=START; pos={}; last={}; tr=[]; peak=START; mine=START; dd=0.; skips=0; maxpos=0; maxgross=0.; liq=False
 def floating(): return sum(NOTIONAL*rr(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return len(pos)*NOTIONAL
 def close(s,px,tm,reason):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*rr(p["side"],p["entry"],px); bal+=pnl
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"reason":reason,"entry":p["entry"],"exit":px,"peak_favorable_pct":p["peakfav"]*100,"pnl_usdt":pnl})
  del pos[s]
 for tm,s,i,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s); sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  if p:
   p["bars"]+=1; p["peakfav"]=max(p["peakfav"],fav(p["side"],p["entry"],float(r.high),float(r.low)))
   cur=rr(p["side"],p["entry"],px); mom=max(float(r["hist"]),0) if p["side"]=="LONG" else max(-float(r["hist"]),0); p["peakmom"]=max(p["peakmom"],mom)
   piv=float(r["pl"]) if p["side"]=="LONG" else float(r["ph"])
   if not np.isnan(piv): p["lastpivot"]=piv
   eligible=use_v3 and p["peakfav"]>=GATE
   if eligible and not p["armed"] and p["bars"]>=MIN_BARS and p["peakmom"]>0 and mom<=p["peakmom"]*(1-MOM_LOSS) and cur>0 and not np.isnan(p["lastpivot"]):
    p["armed"]=True; p["armed_i"]=i; p["protected"]=p["lastpivot"]
   elif p["armed"] and not np.isnan(piv):
    if p["side"]=="LONG" and piv>p["protected"]: p["protected"]=piv
    if p["side"]=="SHORT" and piv<p["protected"]: p["protected"]=piv
   vx=p["armed"] and i>p["armed_i"] and cur>0 and ((p["side"]=="LONG" and px<p["protected"]) or (p["side"]=="SHORT" and px>p["protected"]))
   opp=sig and sig!=p["side"]
   if vx and not opp: close(s,px,tm,"V3_PROTECTED_EXIT")
   elif opp and s in pos: close(s,px,tm,"PC_OPPOSITE")
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    mom=max(float(r["hist"]),0) if sig=="LONG" else max(-float(r["hist"]),0); lp=float(r["pl"]) if sig=="LONG" else float(r["ph"])
    pos[s]={"side":sig,"entry":px,"time":tm,"bars":0,"peakfav":0.,"peakmom":mom,"lastpivot":lp,"armed":False,"armed_i":-1,"protected":np.nan}; maxpos=max(maxpos,len(pos))
   else: skips+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); dd=min(dd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time,"END")
 t=pd.DataFrame(tr); final=bal if not liq else eq()
 return {"variant":name,"period":"2025-09-01 to 2026-03-01","final_balance":final,"net_profit_usdt":final-START,"trades":len(t),"winrate_pct":(t.pnl_usdt>0).mean()*100 if len(t) else 0,"v3_exits":int((t.reason=="V3_PROTECTED_EXIT").sum()) if len(t) else 0,"max_floating_dd_pct":dd,"min_equity":mine,"max_gross_notional":maxgross,"skipped_entries_margin":skips,"liquidated":liq},t

def main():
 data={s:load(s) for s in BASE15}; rows=[]; ts=[]
 for v in (False,True):
  r,t=run(data,v); rows.append(r); ts.append(t); print(r)
 pd.DataFrame(rows).to_csv(OUT/"pc_base15_v3_gate20_oos_prev6m_summary.csv",index=False)
 pd.concat(ts,ignore_index=True).to_csv(OUT/"pc_base15_v3_gate20_oos_prev6m_trades.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
