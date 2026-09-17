from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"
START_BALANCE=120.; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
BASE14=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT"]
CANDIDATES=["FLOWUSDT","IOTAUSDT","DYMUSDT","IDUSDT","STXUSDT","NOTUSDT","GMXUSDT"]
VARIANTS=[("BASE14",BASE14)]+[(f"BASE14+{s.replace('USDT','')}",BASE14+[s]) for s in CANDIDATES]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def load(s):
 d=download_klines(s,"30m",START,END).reset_index(drop=True)
 d["time"]=pd.to_datetime(d["time"],utc=True)
 return purple_cloud(d)[["time","high","low","close","pc_buy","pc_sell"]]

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(all_data,name,symbols):
 data={s:all_data[s] for s in symbols}; ev=[]
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
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"pnl_usdt":pnl}); del pos[s]
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
 row={"variant":name,"symbols":len(symbols),"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq}
 by=t.groupby("symbol",as_index=False).agg(trades=("pnl_usdt","size"),pnl_usdt=("pnl_usdt","sum")) if len(t) else pd.DataFrame()
 if len(by): by.insert(0,"variant",name)
 return row,t,by

def main():
 needed=sorted(set(BASE14+CANDIDATES)); print("Loading",len(needed),"symbols...")
 data={s:load(s) for s in needed}; rows=[]; trades=[]; by=[]
 for name,symbols in VARIANTS:
  r,t,b=run(data,name,symbols); rows.append(r); trades.append(t)
  if len(b): by.append(b)
  print(r)
 summary=pd.DataFrame(rows).sort_values("final_balance",ascending=False)
 summary.to_csv(OUT/"pc_base14_candidate_additions_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base14_candidate_additions_trades.csv",index=False)
 if by: pd.concat(by,ignore_index=True).to_csv(OUT/"pc_base14_candidate_additions_by_symbol.csv",index=False)
 print("\nFINAL COMPARISON\n",summary.to_string(index=False))

if __name__=="__main__": main()
