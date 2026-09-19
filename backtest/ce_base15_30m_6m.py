from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines

START="2026-03-01"; END="2026-09-01"
START_BALANCE=145.58352658; NOTIONAL=60.; LEVERAGE=20.; MMR=.005
ATR_LEN=14; ATR_MULT=3.0
BASE15=["ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT","FLOWUSDT"]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()

def chandelier(d):
 d=d.copy().reset_index(drop=True)
 prev=d.close.shift(1)
 tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
 atr=rma(tr,ATR_LEN)*ATR_MULT
 # everget CE semantics: Close Extremums ON, one-bar shifted stop references, confirmed close
 long_stop=d.close.rolling(ATR_LEN).max()-atr
 short_stop=d.close.rolling(ATR_LEN).min()+atr
 long_prev=long_stop.shift(1); short_prev=short_stop.shift(1)
 long_stop=pd.Series([float("nan")]*len(d)); short_stop=pd.Series([float("nan")]*len(d))
 raw_long=d.close.rolling(ATR_LEN).max()-atr; raw_short=d.close.rolling(ATR_LEN).min()+atr
 for i in range(len(d)):
  if i==0 or pd.isna(raw_long.iloc[i]): continue
  lp=long_stop.iloc[i-1] if not pd.isna(long_stop.iloc[i-1]) else raw_long.iloc[i-1]
  sp=short_stop.iloc[i-1] if not pd.isna(short_stop.iloc[i-1]) else raw_short.iloc[i-1]
  long_stop.iloc[i]=max(raw_long.iloc[i],lp) if d.close.iloc[i-1]>lp else raw_long.iloc[i]
  short_stop.iloc[i]=min(raw_short.iloc[i],sp) if d.close.iloc[i-1]<sp else raw_short.iloc[i]
 direction=[1]*len(d)
 buy=[False]*len(d); sell=[False]*len(d)
 for i in range(1,len(d)):
  if pd.isna(long_stop.iloc[i-1]) or pd.isna(short_stop.iloc[i-1]):
   direction[i]=direction[i-1]; continue
  if d.close.iloc[i]>short_stop.iloc[i-1]: direction[i]=1
  elif d.close.iloc[i]<long_stop.iloc[i-1]: direction[i]=-1
  else: direction[i]=direction[i-1]
  buy[i]=direction[i]==1 and direction[i-1]==-1
  sell[i]=direction[i]==-1 and direction[i-1]==1
 d["ce_buy"]=buy; d["ce_sell"]=sell
 return d[["time","close","ce_buy","ce_sell"]]

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def main():
 ev=[]
 for s in BASE15:
  d=download_klines(s,"30m",START,END).reset_index(drop=True)
  d["time"]=pd.to_datetime(d["time"],utc=True)
  d=chandelier(d)
  for _,r in d.iterrows(): ev.append((r.time,s,r))
 ev.sort(key=lambda x:(x[0],x[1]))
 bal=START_BALANCE; pos={}; last={}; trades=[]; peak=START_BALANCE; mine=START_BALANCE; maxdd=0.; maxpos=0; maxgross=0.; skipped=0; liq=False
 def floating(): return sum(NOTIONAL*ret(p["side"],p["entry"],last.get(s,p["entry"])) for s,p in pos.items())
 def eq(): return bal+floating()
 def gross(): return NOTIONAL*len(pos)
 def closep(s,px,tm):
  nonlocal bal
  p=pos[s]; pnl=NOTIONAL*ret(p["side"],p["entry"],px); bal+=pnl
  trades.append({"symbol":s,"side":p["side"],"entry_time":p["time"],"exit_time":tm,"entry_price":p["entry"],"exit_price":px,"pnl_usdt":pnl}); del pos[s]
 for tm,s,r in ev:
  if liq: break
  px=float(r.close); last[s]=px
  sig="LONG" if bool(r.ce_buy) else ("SHORT" if bool(r.ce_sell) else None)
  p=pos.get(s)
  if p and sig and sig!=p["side"]: closep(s,px,tm)
  if sig and s not in pos:
   if eq()-gross()/LEVERAGE>=NOTIONAL/LEVERAGE:
    pos[s]={"side":sig,"entry":px,"time":tm}; maxpos=max(maxpos,len(pos))
   else: skipped+=1
  e=eq(); g=gross(); maxgross=max(maxgross,g); mine=min(mine,e); peak=max(peak,e); maxdd=min(maxdd,(e/peak-1)*100)
  if pos and e<=g*MMR: liq=True
 if not liq:
  # close remaining at final available close
  for s in list(pos):
   closep(s,last[s],END)
 t=pd.DataFrame(trades); final=bal if not liq else eq(); wins=int((t.pnl_usdt>0).sum()) if len(t) else 0
 row={"strategy":"CE14x3_CLOSE","symbols":15,"timeframe":"30m","start_balance":START_BALANCE,"final_balance":final,"net_profit_usdt":final-START_BALANCE,"return_pct":(final/START_BALANCE-1)*100,"closed_trades":len(t),"wins":wins,"winrate":wins/len(t)*100 if len(t) else 0,"avg_pnl_per_trade":t.pnl_usdt.mean() if len(t) else 0,"max_floating_dd_pct":maxdd,"min_equity":mine,"max_simultaneous_positions":maxpos,"max_gross_notional":maxgross,"skipped_entries_margin":skipped,"liquidated":liq}
 pd.DataFrame([row]).to_csv(OUT/"ce_base15_30m_6m_summary.csv",index=False)
 t.to_csv(OUT/"ce_base15_30m_6m_trades.csv",index=False)
 print(row)

if __name__=="__main__": main()
