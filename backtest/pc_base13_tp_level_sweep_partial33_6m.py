from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; START_BALANCE=120.0; NOTIONAL=60.0; LEVERAGE=20.0; MMR=0.005
PARTIAL=0.33
TP_LEVELS=[0.02,0.03,0.04,0.05,0.06]
BASE13=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT"]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def load(s):
 d=download_klines(s,"30m",START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True)
 return purple_cloud(d)[["time","high","low","close","pc_buy","pc_sell"]].copy()
def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(data,tp_pct):
 events=[]
 for s,d in data.items():
  for _,r in d.iterrows(): events.append((r.time,s,r))
 events.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; trades=[]; maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; skipped=0; liq=False
 def n(p): return NOTIONAL*p["rem"]
 def floating(): return sum(n(p)*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def equity(): return bal+floating()
 def gross(): return sum(n(p) for p in pos.values())
 def close(s,px,tm):
  nonlocal bal
  p=pos[s]; rp=n(p)*ret(p["side"],p["entry"],px); bal+=rp
  trades.append({"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"tp_hit":p["hit"],"pnl_usdt":p["partial_pnl"]+rp}); del pos[s]
 for tm,s,r in events:
  if liq: break
  px=float(r.close); hi=float(r.high); lo=float(r.low); last[s]=px; p=pos.get(s)
  if p and not p["hit"]:
   tp=p["entry"]*(1+tp_pct) if p["side"]=="LONG" else p["entry"]*(1-tp_pct)
   hit=hi>=tp if p["side"]=="LONG" else lo<=tp
   if hit:
    pp=NOTIONAL*PARTIAL*ret(p["side"],p["entry"],tp); bal+=pp; p["partial_pnl"]+=pp; p["rem"]=1-PARTIAL; p["hit"]=True
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None); p=pos.get(s)
  if p and sig and sig!=p["side"]: close(s,px,tm)
  if sig and s not in pos:
   if equity()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm,"rem":1.,"hit":False,"partial_pnl":0.}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=equity(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time)
 t=pd.DataFrame(trades); final=bal if not liq else equity(); wins=int((t.pnl_usdt>0).sum()); hits=int(t.tp_hit.sum())
 return {"tp_level_pct":tp_pct*100,"partial_pct":PARTIAL*100,"runner_pct":(1-PARTIAL)*100,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100,"tp_hits":hits,"tp_hit_rate_pct":hits/len(t)*100,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq},t

def main():
 data={s:load(s) for s in BASE13}; rows=[]; alltr=[]
 for tp in TP_LEVELS:
  row,t=run(data,tp); rows.append(row); t.insert(0,"tp_level_pct",tp*100); alltr.append(t); print(row)
 summary=pd.DataFrame(rows); summary.to_csv(OUT/"pc_base13_tp_level_sweep_partial33_summary.csv",index=False)
 pd.concat(alltr,ignore_index=True).to_csv(OUT/"pc_base13_tp_level_sweep_partial33_trades.csv",index=False)
 print("\nCOMPARISON\n",summary.to_string(index=False))
if __name__=="__main__": main()
