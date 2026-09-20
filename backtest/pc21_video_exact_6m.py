from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import atr, vwma, rma

START="2026-03-01"; END="2026-09-01"
START_BALANCE=145.58352658; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
SYMBOLS=["ZECUSDT","USELESSUSDT","FETUSDT","HYPEUSDT","JTOUSDT","VETUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
PERIOD=40; ALPHA=.9; BPT=.5; SPT=.5
ATR_PERIOD=14; ATR_MULT=2.
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def pc21(df):
 d=df.copy().reset_index(drop=True); n1=int(np.ceil(PERIOD/4)); n2=int(np.ceil(PERIOD/2))
 x2=atr(d,PERIOD)*ALPHA; xh=d.close+x2; xl=d.close-x2; hl2=(d.high+d.low)/2
 a1=vwma(hl2*d.volume,d.volume,n1)/vwma(d.volume,d.volume,n1)
 a2=vwma(hl2*d.volume,d.volume,n2)/vwma(d.volume,d.volume,n2)
 a3=2*a1-a2; a4=vwma(a3,d.volume,PERIOD); b1=rma(d.close,PERIOD); a5=2*a4*b1/(a4+b1)
 buy=(a5<=xl)&(d.close>b1*(1+BPT*.01)); sell=(a5>=xh)&(d.close<b1*(1-SPT*.01))
 xs=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)): xs[i]=1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
 changed=pd.Series(xs).ne(pd.Series(xs).shift(1))
 d["pc_buy"]=buy&changed; d["pc_sell"]=sell&changed
 d["ema200"]=d.close.ewm(span=200,adjust=False).mean()
 d["atr14"]=atr(d,ATR_PERIOD)
 return d

def load(s):
 d=download_klines(s,"30m",START,END).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True)
 return pc21(d)

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def main():
 data={s:load(s) for s in SYMBOLS}; events=[]
 for s,d in data.items():
  for i,r in d.iterrows(): events.append((r.time,s,i,r))
 events.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; trades=[]; skipped=0; peak=START_BALANCE; maxdd=0.; mine=START_BALANCE; maxpos=0; maxgross=0.; liq=False

 def floating():
  return sum(p["remaining"]*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return sum(p["remaining"] for p in pos.values())
 def realize(s,px,tm,qty,reason):
  nonlocal bal
  p=pos[s]; pnl=qty*ret(p["side"],p["entry"],px); bal+=pnl; p["remaining"]-=qty
  trades.append({"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"qty":qty,"exit_reason":reason,"pnl_usdt":pnl})
  if p["remaining"]<1e-9: del pos[s]

 for tm,s,i,r in events:
  if liq: break
  px=float(r.close); hi=float(r.high); lo=float(r.low); last[s]=px
  p=pos.get(s)
  if p:
   if p["side"]=="LONG":
    if not p["partial"]:
     if lo<=p["stop"]: realize(s,p["stop"],tm,p["remaining"],"SL")
     elif hi>=p["target"]:
      realize(s,p["target"],tm,NOTIONAL*.5,"TP50"); p=pos.get(s)
      if p: p["partial"]=True; p["stop"]=p["entry"]
    else:
     # Remaining 50% trails the lower ATR band, never loosening.
     band=px-ATR_MULT*float(r.atr14); p["stop"]=max(p["stop"],band)
     if lo<=p["stop"]: realize(s,p["stop"],tm,p["remaining"],"ATR_TRAIL")
   else:
    if not p["partial"]:
     if hi>=p["stop"]: realize(s,p["stop"],tm,p["remaining"],"SL")
     elif lo<=p["target"]:
      realize(s,p["target"],tm,NOTIONAL*.5,"TP50"); p=pos.get(s)
      if p: p["partial"]=True; p["stop"]=p["entry"]
    else:
     band=px+ATR_MULT*float(r.atr14); p["stop"]=min(p["stop"],band)
     if hi>=p["stop"]: realize(s,p["stop"],tm,p["remaining"],"ATR_TRAIL")

  # Video baseline filter: above EMA200 long only, below EMA200 short only.
  sig=None
  if bool(r.pc_buy) and px>float(r.ema200): sig="LONG"
  elif bool(r.pc_sell) and px<float(r.ema200): sig="SHORT"

  # Entries only while flat on that symbol. Trade management owns exits.
  if sig and s not in pos and np.isfinite(r.atr14):
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    a=float(r.atr14)
    if sig=="LONG":
     stop=px-ATR_MULT*a; target=px+ATR_MULT*a
    else:
     stop=px+ATR_MULT*a; target=px-ATR_MULT*a
    pos[s]={"side":sig,"entry":px,"time":tm,"remaining":NOTIONAL,"stop":stop,"target":target,"partial":False}
    maxpos=max(maxpos,len(pos))
   else: skipped+=1

  e=eq(); g=gross(); peak=max(peak,e); mine=min(mine,e); maxdd=min(maxdd,(e/peak-1)*100); maxgross=max(maxgross,g)
  if pos and e<=g*MMR: liq=True

 if not liq:
  for s in list(pos):
   r=data[s].iloc[-1]; realize(s,float(r.close),r.time,pos[s]["remaining"],"END")

 t=pd.DataFrame(trades); final=bal if not liq else eq()
 # Group partial fills back into logical trades for WR.
 if len(t):
  logical=t.groupby(["symbol","side","entry_time"],as_index=False).agg(pnl_usdt=("pnl_usdt","sum"))
  wins=int((logical.pnl_usdt>0).sum()); n=len(logical)
 else: logical=pd.DataFrame(); wins=n=0
 result={"start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"logical_trades":n,"wins":wins,"winrate":wins/n*100 if n else 0,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq}
 print("\nPURPLE CLOUD 2.1 VIDEO TEST")
 print("P40 A0.9 Pressure0.5 | EMA200 | ATR14 x2 | 50% then BE + ATR trail")
 print(result)
 if len(logical):
  by=logical.groupby("symbol",as_index=False).agg(trades=("pnl_usdt","size"),pnl_usdt=("pnl_usdt","sum"),wins=("pnl_usdt",lambda x:int((x>0).sum())))
  by["winrate"]=by.wins/by.trades*100; print("\nBY SYMBOL\n",by.to_string(index=False))
  by.to_csv(OUT/"pc21_video_exact_by_symbol.csv",index=False)
 t.to_csv(OUT/"pc21_video_exact_fills.csv",index=False)
 logical.to_csv(OUT/"pc21_video_exact_trades.csv",index=False)

if __name__=="__main__": main()
