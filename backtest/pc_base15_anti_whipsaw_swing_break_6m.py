from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

# Anti-whipsaw experiment:
# Native exit remains unchanged. Only a reverse entry after a short-lived (<12h)
# position is delayed until price CLOSES through the latest confirmed swing.
# Swing logic is intentionally the simple confirmed "Pivot bars" idea from the
# uploaded Liquidity Tracker, tested at several pivot sizes.
VARIANTS=[("NATIVE",None),("SWING3",3),("SWING5",5),("SWING8",8),("SWING12",12)]
MAX_SHORT_BARS=24  # 12h on 30m

def add_swings(d,n):
 d=d.copy().reset_index(drop=True)
 # confirmed at i+n, but the level price belongs to pivot i; no lookahead use.
 ph=d.high.shift(n).where(d.high.shift(n)==d.high.rolling(2*n+1).max())
 pl=d.low.shift(n).where(d.low.shift(n)==d.low.rolling(2*n+1).min())
 d["swing_high"]=ph.ffill()
 d["swing_low"]=pl.ffill()
 return d

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(raw,name,n):
 data={s:(raw[s] if n is None else add_swings(raw[s],n)) for s in BASE15}
 ev=[]
 for s,d in data.items():
  for i,r in d.iterrows(): ev.append((r.time,s,i,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; pending={}; last={}; tr=[]
 maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 delayed=confirmed=cancelled=skips_margin=0
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return len(pos)*NOTIONAL
 def openpos(s,side,px,tm):
  nonlocal maxpos,skips_margin
  if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
   pos[s]={"side":side,"entry":px,"time":tm}; maxpos=max(maxpos,len(pos)); return True
  skips_margin+=1; return False
 def closepos(s,px,tm,bar):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,
   "entry_price":p["entry"],"exit_price":px,"pnl_usdt":pnl,"bars_held":bar-p["bar"] if "bar" in p else None})
  age=bar-p.get("bar",bar); del pos[s]; return age
 for tm,s,i,r in ev:
  if liq: break
  px=float(r.close); last[s]=px
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)

  # A fresh PC signal cancels an older pending confirmation on that symbol.
  if sig and s in pending and sig!=pending[s]["side"]:
   del pending[s]; cancelled+=1

  p=pos.get(s)
  if p and sig and sig!=p["side"]:
   age=i-p["bar"]; closepos(s,px,tm,i)
   if n is not None and age<MAX_SHORT_BARS:
    level=float(r.swing_high) if sig=="LONG" and pd.notna(r.swing_high) else (float(r.swing_low) if sig=="SHORT" and pd.notna(r.swing_low) else None)
    if level is not None:
     pending[s]={"side":sig,"level":level,"signal_time":tm}; delayed+=1
    else:
     openpos(s,sig,px,tm)
     if s in pos: pos[s]["bar"]=i
   else:
    openpos(s,sig,px,tm)
    if s in pos: pos[s]["bar"]=i
  elif sig and s not in pos and s not in pending:
   openpos(s,sig,px,tm)
   if s in pos: pos[s]["bar"]=i

  # Confirm only on a later CLOSED candle through the locked swing level.
  q=pending.get(s)
  if q and tm>q["signal_time"]:
   ok=(q["side"]=="LONG" and px>q["level"]) or (q["side"]=="SHORT" and px<q["level"])
   if ok:
    side=q["side"]; del pending[s]
    if openpos(s,side,px,tm):
     pos[s]["bar"]=i; confirmed+=1

  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e)
  maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; closepos(s,float(r.close),r.time,len(data[s])-1)
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 return {"variant":name,"final_balance":final,"net_profit_usdt":final-START_BALANCE,
  "closed_trades":len(t),"winrate":wins/len(t)*100 if len(t) else 0,"max_floating_dd_pct":maxdd,
  "min_equity":mine,"delayed_reversals":delayed,"confirmed_delayed_entries":confirmed,
  "cancelled_pending":cancelled,"skipped_entries_margin":skips_margin,
  "max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"liquidated":liq},t

def main():
 raw={s:eng.load(s).reset_index(drop=True) for s in BASE15}
 rows=[]; trades=[]
 for name,n in VARIANTS:
  r,t=run(raw,name,n); rows.append(r); trades.append(t); print(r)
 summary=pd.DataFrame(rows).sort_values("net_profit_usdt",ascending=False)
 summary.to_csv(OUT/"pc_base15_anti_whipsaw_swing_break_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_anti_whipsaw_swing_break_trades.csv",index=False)
 print("\nANTI-WHIPSAW SWING BREAK\n",summary.to_string(index=False))

if __name__=="__main__": main()
