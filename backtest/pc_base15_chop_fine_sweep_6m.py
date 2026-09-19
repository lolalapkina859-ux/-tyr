from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=float(os.getenv("PC_TEST_START_BALANCE","145.58352658"))
NOTIONAL=60.; LEVERAGE=20.; MMR=.005
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

# Entry-quality filters only. Exit remains native opposite Purple Cloud signal.
# All filters use information available on the confirmed entry candle.
VARIANTS=[("NATIVE",{})]+[(f"CHOP{x}",{"chop":float(x)}) for x in range(55,63)]

def indicators(d):
 d=d.copy().reset_index(drop=True)
 prev=d.close.shift(1)
 tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
 up=d.high.diff(); dn=-d.low.diff()
 plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=d.index)
 minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=d.index)
 atr=tr.ewm(alpha=1/14,adjust=False).mean()
 pdi=100*plus.ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
 mdi=100*minus.ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
 dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
 d["adx"]=dx.ewm(alpha=1/14,adjust=False).mean()
 n=20
 change=(d.close-d.close.shift(n)).abs()
 volatility=d.close.diff().abs().rolling(n).sum()
 d["er"]=change/volatility.replace(0,np.nan)
 atrsum=tr.rolling(14).sum()
 hh=d.high.rolling(14).max(); ll=d.low.rolling(14).min()
 d["chop"]=100*np.log10(atrsum/(hh-ll).replace(0,np.nan))/np.log10(14)
 d["ema50"]=d.close.ewm(span=50,adjust=False).mean()
 d["ema50_slope"]=d.ema50-d.ema50.shift(5)
 return d

def allowed(r,side,cfg):
 if cfg.get("adx") is not None and (pd.isna(r.adx) or r.adx<cfg["adx"]): return False
 if cfg.get("er") is not None and (pd.isna(r.er) or r.er<cfg["er"]): return False
 if cfg.get("chop") is not None and (pd.isna(r.chop) or r.chop>cfg["chop"]): return False
 if cfg.get("ema_slope"):
  if pd.isna(r.ema50_slope): return False
  if side=="LONG" and r.ema50_slope<=0: return False
  if side=="SHORT" and r.ema50_slope>=0: return False
 return True

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run(data,name,cfg):
 ev=[]
 for s,d in data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; tr=[]; skips_filter=skips_margin=0
 maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return len(pos)*NOTIONAL
 def closepos(s,px,tm):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  tr.append({"variant":name,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,
   "entry_price":p["entry"],"exit_price":px,"pnl_usdt":pnl})
  del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); last[s]=px
  sig="LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
  p=pos.get(s)
  # Critical: opposite PC signal ALWAYS closes existing position.
  if p and sig and sig!=p["side"]: closepos(s,px,tm)
  # Filter controls only whether the new/reverse entry is opened.
  if sig and s not in pos:
   if not allowed(r,sig,cfg): skips_filter+=1
   elif eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm}; maxpos=max(maxpos,len(pos))
   else: skips_margin+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e)
  maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; closepos(s,float(r.close),r.time)
 t=pd.DataFrame(tr); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 row={"variant":name,"final_balance":final,"net_profit_usdt":final-START_BALANCE,
  "closed_trades":len(t),"winrate":wins/len(t)*100 if len(t) else 0,
  "max_floating_dd_pct":maxdd,"min_equity":mine,"filtered_entries":skips_filter,
  "skipped_entries_margin":skips_margin,"max_simultaneous_positions":maxpos,
  "max_gross_notional":maxgross,"liquidated":liq}
 return row,t

def main():
 data={s:indicators(eng.load(s)) for s in BASE15}
 rows=[]; trades=[]
 for name,cfg in VARIANTS:
  r,t=run(data,name,cfg); rows.append(r); trades.append(t); print(r)
 summary=pd.DataFrame(rows).sort_values("net_profit_usdt",ascending=False)
 summary.to_csv(OUT/"pc_base15_chop_fine_sweep_summary.csv",index=False)
 pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base15_chop_fine_sweep_trades.csv",index=False)
 print("\nCHOP FINE SWEEP\n",summary.to_string(index=False))

if __name__=="__main__": main()
