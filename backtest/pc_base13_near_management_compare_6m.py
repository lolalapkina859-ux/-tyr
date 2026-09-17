from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud
START="2026-03-01"; END="2026-09-01"; START_BALANCE=120.; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
SYMBOLS=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT"]
VARIANTS=[("NO_TP",None,0.),("TP6_PARTIAL33",.06,.33),("TP6_PARTIAL50",.06,.50)]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)
def load(s):
 d=download_klines(s,"30m",START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True); return purple_cloud(d)[["time","high","low","close","pc_buy","pc_sell"]]
def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1
def run(data,name,tp,part):
 ev=[]
 for s,d in data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1])); bal=START_BALANCE; pos={}; last={}; tr=[]; skipped=0; maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def n(p): return NOTIONAL*p["rem"]
 def floating(): return sum(n(p)*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return sum(n(p) for p in pos.values())
 def close(s,px,tm):
  nonlocal bal
  p=pos[s]; rp=n(p)*ret(p["side"],p["entry"],px); bal+=rp; tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"tp_hit":p["hit"],"pnl_usdt":p["partial"]+rp}); del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); hi=float(r.high); lo=float(r.low); last[s]=px; p=pos.get(s)
  if tp is not None and p and not p["hit"]:
   tpp=p["entry"]*(1+tp) if p["side"]=="LONG" else p["entry"]*(1-tp); hit=hi>=tpp if p["side"]=="LONG" else lo<=tpp
   if hit:
    pp=NOTIONAL*part*ret(p["side"],p["entry"],tpp); bal+=pp; p["partial"]+=pp; p["rem"]=1-part; p["hit"]=True
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None); p=pos.get(s)
  if p and sig and sig!=p["side"]: close(s,px,tm)
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm,"rem":1.,"hit":False,"partial":0.}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; close(s,float(r.close),r.time)
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()); hits=int(t.tp_hit.sum())
 row={"variant":name,"symbols":len(SYMBOLS),"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100,"tp_hits":hits,"tp_hit_rate_pct":hits/len(t)*100,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq}
 by=t.groupby("symbol",as_index=False).agg(trades=("pnl_usdt","size"),pnl_usdt=("pnl_usdt","sum"),tp_hits=("tp_hit","sum")); by.insert(0,"variant",name)
 return row,t,by
def main():
 data={s:load(s) for s in SYMBOLS}; rows=[]; trades=[]; by=[]
 for v,tp,p in VARIANTS:
  r,t,b=run(data,v,tp,p); rows.append(r); trades.append(t); by.append(b); print(r)
 pd.DataFrame(rows).to_csv(OUT/"pc_base13_near_management_compare_summary.csv",index=False); pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base13_near_management_compare_trades.csv",index=False); pd.concat(by,ignore_index=True).to_csv(OUT/"pc_base13_near_management_compare_by_symbol.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
