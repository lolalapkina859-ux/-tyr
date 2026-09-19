from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"
START_BALANCE=145.58352658; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
BASE15=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT","FLOWUSDT"]
VARIANTS=["A_NATIVE_PC","B_NATIVE_LC","C_PC_LC_CONFIRM","D_PC_ENTRY_LC_EXIT"]
OUT=Path("backtest/data"); TMP=OUT/"lc_tmp"; OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
LC_ROOT=os.getenv("LC_ROOT","vendor/lorentzian-classification")
LC_PY=str(Path(LC_ROOT)/"ports/python")

def norm(s): return "".join(c.lower() for c in str(s) if c.isalnum())

def pick(df,*names):
 m={norm(c):c for c in df.columns}
 for n in names:
  if norm(n) in m: return m[norm(n)]
 raise KeyError(f"Missing any of {names}; columns={list(df.columns)}")

def truth(v):
 if pd.isna(v): return False
 if isinstance(v,bool): return v
 if isinstance(v,(int,float)): return v != 0
 return str(v).strip().lower() in ("1","true","yes","buy","sell")

def lc_signals(raw,symbol):
 inp=TMP/f"{symbol}_in.csv"; out=TMP/f"{symbol}_out.csv"
 raw[["time","open","high","low","close"]].to_csv(inp,index=False)
 env=os.environ.copy(); env["PYTHONPATH"]=LC_PY
 cmd=[sys.executable,"-m","lorentzian_classification","run",str(inp),"--include-full-history","--output",str(out)]
 subprocess.run(cmd,check=True,env=env)
 x=pd.read_csv(out)
 tcol=pick(x,"time"); bcol=pick(x,"buy","StartLongTrade"); scol=pick(x,"sell","StartShortTrade")
 try: elcol=pick(x,"StopBuy","endLongTrade","closeLong","longExit")
 except KeyError: elcol=None
 try: escol=pick(x,"StopSell","endShortTrade","closeShort","shortExit")
 except KeyError: escol=None
 try: posture=pick(x,"BacktestSignal","backTestSignal","direction","signal")
 except KeyError: posture=None
 z=pd.DataFrame({"time":pd.to_datetime(x[tcol],utc=True),"lc_buy":x[bcol].map(truth),"lc_sell":x[scol].map(truth)})
 z["lc_exit_long"]=x[elcol].map(truth) if elcol else False
 z["lc_exit_short"]=x[escol].map(truth) if escol else False
 if posture:
  p=pd.to_numeric(x[posture],errors="coerce").fillna(0)
  z["lc_posture"]=p.clip(-1,1)
 else:
  state=0; vals=[]
  for b,s,el,es in zip(z.lc_buy,z.lc_sell,z.lc_exit_long,z.lc_exit_short):
   if b: state=1
   elif s: state=-1
   elif (state==1 and el) or (state==-1 and es): state=0
   vals.append(state)
  z["lc_posture"]=vals
 return z

def prepare(symbol):
 d=download_klines(symbol,"30m",START,END).reset_index(drop=True)
 d["time"]=pd.to_datetime(d["time"],utc=True)
 pc=purple_cloud(d.copy())
 lc=lc_signals(d,symbol)
 cols=["time","open","high","low","close"]
 q=pc[cols+["pc_buy","pc_sell"]].merge(lc,on="time",how="left")
 for c in ["lc_buy","lc_sell","lc_exit_long","lc_exit_short"]: q[c]=q[c].fillna(False)
 q["lc_posture"]=q["lc_posture"].fillna(0)
 return q

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def events_for(row,variant,current_side):
 if variant=="A_NATIVE_PC":
  entry="LONG" if row.pc_buy else ("SHORT" if row.pc_sell else None); return entry,False
 if variant=="B_NATIVE_LC":
  entry="LONG" if row.lc_buy else ("SHORT" if row.lc_sell else None)
  exit_now=(current_side=="LONG" and row.lc_exit_long) or (current_side=="SHORT" and row.lc_exit_short)
  return entry,exit_now
 if variant=="C_PC_LC_CONFIRM":
  entry=None
  if row.pc_buy and row.lc_posture>0: entry="LONG"
  elif row.pc_sell and row.lc_posture<0: entry="SHORT"
  return entry,False
 # D: PC entries; LC's own fixed/dynamic exit event closes; stay flat until a fresh PC signal
 entry="LONG" if row.pc_buy else ("SHORT" if row.pc_sell else None)
 exit_now=(current_side=="LONG" and (row.lc_exit_long or row.lc_sell)) or (current_side=="SHORT" and (row.lc_exit_short or row.lc_buy))
 return entry,exit_now

def run(all_data,variant):
 ev=[]
 for s,d in all_data.items():
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; trades=[]; skipped=0; maxpos=0; maxgross=0.; mine=START_BALANCE; peak=START_BALANCE; maxdd=0.; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return NOTIONAL*len(pos)
 def closep(s,px,tm,reason):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  trades.append({"variant":variant,"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"entry_price":p["entry"],"exit_price":px,"pnl_usdt":pnl,"exit_reason":reason}); del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); last[s]=px; p=pos.get(s); side=p["side"] if p else None
  entry,exit_now=events_for(r,variant,side)
  if p and exit_now: closep(s,px,tm,"LC_EXIT")
  p=pos.get(s)
  if p and entry and entry!=p["side"]: closep(s,px,tm,"OPPOSITE_ENTRY")
  if entry and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":entry,"entry":px,"time":tm}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  for s in list(pos): closep(s,last[s],END,"END")
 t=pd.DataFrame(trades); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 long=t[t.side=="LONG"] if len(t) else t; short=t[t.side=="SHORT"] if len(t) else t
 row={"variant":variant,"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"avg_pnl_per_trade":t.pnl_usdt.mean() if len(t) else 0,"long_trades":len(long),"long_pnl":long.pnl_usdt.sum() if len(long) else 0,"short_trades":len(short),"short_pnl":short.pnl_usdt.sum() if len(short) else 0,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq}
 return row,t

def main():
 print("Lorentzian: official AI Edge Python port, defaults, include_full_history=True")
 data={s:prepare(s) for s in BASE15}
 rows=[]; ts=[]
 for v in VARIANTS:
  r,t=run(data,v); rows.append(r); ts.append(t); print(r)
 pd.DataFrame(rows).to_csv(OUT/"pc_lorentzian_abcd_base15_30m_6m_summary.csv",index=False)
 pd.concat(ts,ignore_index=True).to_csv(OUT/"pc_lorentzian_abcd_base15_30m_6m_trades.csv",index=False)

if __name__=="__main__": main()
