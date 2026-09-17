from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; START_BALANCE=120.; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
SYMBOLS=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT"]
ACTIVATIONS=[.06,.08,.10]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def prep(df):
 d=purple_cloud(df.reset_index(drop=True));
 # confirmed close-only structural reversal: break previous N-bar swing extreme.
 for n in (3,5,8):
  d[f"prev_low_{n}"]=d.low.shift(1).rolling(n).min(); d[f"prev_high_{n}"]=d.high.shift(1).rolling(n).max()
  d[f"bear_break_{n}"]=d.close < d[f"prev_low_{n}"]; d[f"bull_break_{n}"]=d.close > d[f"prev_high_{n}"]
 return d

def load30(s):
 d=download_klines(s,"30m",START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True); return prep(d)
def load15(s):
 d=download_klines(s,"15m",START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True); return prep(d)
def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(d30,d15,name,activation=None,exit_tf="30m",swing=5):
 # Entry/reverse signals always exact 30m Purple Cloud. Faster structure is exit-only after activation.
 ev=[]
 for s,d in d30.items():
  for _,r in d.iterrows(): ev.append((r.time,0,s,"30",r))
 if activation is not None and exit_tf=="15m":
  for s,d in d15.items():
   for _,r in d.iterrows(): ev.append((r.time,1,s,"15",r))
 ev.sort(key=lambda x:(x[0],x[1],x[2])); bal=START_BALANCE; pos={}; last={}; tr=[]; skipped=0; mine=START_BALANCE; peak_eq=START_BALANCE; maxdd=0.; maxpos=0; maxgross=0.; structural_exits=0; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return NOTIONAL*len(pos)
 def close(s,px,tm,reason):
  nonlocal bal,structural_exits
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"exit_reason":reason,"armed":p["armed"],"peak_return_pct":p["best"]*100,"exit_return_pct":ret(p["side"],p["entry"],px)*100,"pnl_usdt":pnl})
  if reason.startswith("STRUCT"): structural_exits+=1
  del pos[s]
 for tm,_,s,tf,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s)
  if p:
   favorable=ret(p["side"],p["entry"],float(r.high) if p["side"]=="LONG" else float(r.low)); p["best"]=max(p["best"],favorable)
   if activation is not None and p["best"]>=activation: p["armed"]=True
  # Structural exit only on chosen timeframe and only after the trade has earned activation threshold.
  p=pos.get(s)
  if p and p["armed"] and tf==("15" if exit_tf=="15m" else "30"):
   broken=bool(r[f"bear_break_{swing}"]) if p["side"]=="LONG" else bool(r[f"bull_break_{swing}"])
   if broken: close(s,px,tm,f"STRUCT_{exit_tf}_{swing}")
  # Native PC entry/reversal remains 30m only.
  if tf=="30":
   sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None); p=pos.get(s)
   if p and sig and sig!=p["side"]: close(s,px,tm,"PC_OPPOSITE")
   if sig and s not in pos:
    if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
     pos[s]={"side":sig,"entry":px,"time":tm,"armed":False,"best":0.}; maxpos=max(maxpos,len(pos))
    else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak_eq=max(peak_eq,e); maxdd=min(maxdd,(e/peak_eq-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=d30[s].iloc[-1]; close(s,float(r.close),r.time,"END")
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 return {"variant":name,"activation_pct":None if activation is None else activation*100,"exit_tf":exit_tf,"swing_bars":swing,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"winrate":wins/len(t)*100 if len(t) else 0,"structural_exits":structural_exits,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq},t

def main():
 d30={s:load30(s) for s in SYMBOLS}; d15={s:load15(s) for s in SYMBOLS}; rows=[]; trades=[]
 r,t=run(d30,d15,"NO_EXIT"); rows.append(r); trades.append(t); print(r)
 for a in ACTIVATIONS:
  for tf in ("30m","15m"):
   for n in (3,5,8):
    name=f"A{int(a*100)}_{tf}_SWING{n}"; r,t=run(d30,d15,name,a,tf,n); rows.append(r); trades.append(t); print(r)
 summary=pd.DataFrame(rows).sort_values("final_balance",ascending=False); summary.to_csv(OUT/"pc_base14_armed_reversal_exit_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base14_armed_reversal_exit_trades.csv",index=False)
 print("\nFINAL RANKING\n",summary.to_string(index=False))
if __name__=="__main__": main()
