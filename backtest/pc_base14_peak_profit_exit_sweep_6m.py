from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; START_BALANCE=120.; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
SYMBOLS=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT"]
ACTIVATIONS=[.05,.06,.08,.10]; GIVEBACKS=[.01,.02,.03,.04]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def load(s):
 d=download_klines(s,"30m",START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True)
 return purple_cloud(d)[["time","high","low","close","pc_buy","pc_sell"]]
def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(data,name,activation=None,giveback=None):
 ev=[]
 for s,d in data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1])); bal=START_BALANCE; pos={}; last={}; tr=[]; skipped=0; maxpos=0; maxgross=0.; mine=START_BALANCE; peak_eq=START_BALANCE; maxdd=0.; liq=False; peak_exits=0
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return NOTIONAL*len(pos)
 def close(s,px,tm,reason):
  nonlocal bal,peak_exits
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  peak_r=ret(p["side"],p["entry"],p["best_px"]); exit_r=ret(p["side"],p["entry"],px)
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"exit_reason":reason,"pnl_usdt":pnl,"peak_return_pct":peak_r*100,"exit_return_pct":exit_r*100,"giveback_from_peak_pct_points":(peak_r-exit_r)*100})
  if reason=="PEAK_EXIT": peak_exits+=1
  del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); hi=float(r.high); lo=float(r.low); last[s]=px; p=pos.get(s)
  # Update favorable excursion using candle extremes. Exit at the configured trailing threshold when crossed.
  if p and activation is not None:
   if p["side"]=="LONG":
    p["best_px"]=max(p["best_px"],hi); best=ret("LONG",p["entry"],p["best_px"])
    if best>=activation:
     p["armed"]=True
     trail=p["best_px"]*(1-giveback)
     if lo<=trail:
      close(s,trail,tm,"PEAK_EXIT"); p=None; last[s]=trail
   else:
    p["best_px"]=min(p["best_px"],lo); best=ret("SHORT",p["entry"],p["best_px"])
    if best>=activation:
     p["armed"]=True
     trail=p["best_px"]*(1+giveback)
     if hi>=trail:
      close(s,trail,tm,"PEAK_EXIT"); p=None; last[s]=trail
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None); p=pos.get(s)
  if p and sig and sig!=p["side"]: close(s,px,tm,"PC_OPPOSITE")
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm,"best_px":px,"armed":False}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak_eq=max(peak_eq,e); maxdd=min(maxdd,(e/peak_eq-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time,"END")
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 row={"variant":name,"activation_pct":None if activation is None else activation*100,"giveback_pct":None if giveback is None else giveback*100,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"peak_exits":peak_exits,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq}
 return row,t

def main():
 data={s:load(s) for s in SYMBOLS}; rows=[]; trades=[]
 r,t=run(data,"NO_EXIT"); rows.append(r); trades.append(t); print(r)
 for a in ACTIVATIONS:
  for g in GIVEBACKS:
   name=f"A{int(a*100)}_G{int(g*100)}"; r,t=run(data,name,a,g); rows.append(r); trades.append(t); print(r)
 summary=pd.DataFrame(rows).sort_values("final_balance",ascending=False)
 summary.to_csv(OUT/"pc_base14_peak_profit_exit_sweep_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base14_peak_profit_exit_sweep_trades.csv",index=False)
 print("\nFINAL RANKING\n",summary.to_string(index=False))

if __name__=="__main__": main()
