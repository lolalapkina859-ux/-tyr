import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

ORIGINAL=["ZECUSDT","USELESSUSDT","FETUSDT","HYPEUSDT","JTOUSDT","VETUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
V2=["ZECUSDT","USELESSUSDT","WLDUSDT","ENAUSDT","XLMUSDT","PENDLEUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
PORTFOLIOS={"ORIGINAL15":ORIGINAL,"LTM_V2":V2}
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]
ATR_LEN=16; FLIP_ATR=4.5

def atr(d,n):
 p=d.close.shift(1);tr=pd.concat([(d.high-d.low),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False).mean()

def sig(d):
 d=d.reset_index(drop=True).copy();a=atr(d,ATR_LEN);t=np.full(len(d),np.nan);trend=1
 buy=np.zeros(len(d),bool);sell=np.zeros(len(d),bool);warm=max(ATR_LEN*3,60)
 for i in range(len(d)):
  if not np.isfinite(a.iloc[i]):continue
  s=d.close.iloc[i];u=s-a.iloc[i]*FLIP_ATR;l=s+a.iloc[i]*FLIP_ATR
  if i==0 or not np.isfinite(t[i-1]):t[i]=u if trend==1 else l;continue
  p=t[i-1]
  if trend==1 and s<p:
   trend=-1;t[i]=l
   if i>=warm:sell[i]=True
  elif trend==-1 and s>p:
   trend=1;t[i]=u
   if i>=warm:buy[i]=True
  elif trend==1:t[i]=max(u,p)
  else:t[i]=min(l,p)
 d["pc_buy"]=buy;d["pc_sell"]=sell;return d

def main():
 universe=sorted(set(ORIGINAL+V2)); rows=[];bys=[]
 ob,on=eng.START_BALANCE,eng.NOTIONAL;eng.START_BALANCE=145.58352658;eng.NOTIONAL=60.
 try:
  for pn,st,en in PERIODS:
   raw={}
   for s in universe:
    try:
     d=eng.download_klines(s,"30m",st,en).reset_index(drop=True)
     if len(d)<500:continue
     d["time"]=pd.to_datetime(d["time"],utc=True);x=sig(d)
     raw[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
    except Exception as e:print("SKIP",s,e)
   for name,symbols in PORTFOLIOS.items():
    avail=[s for s in symbols if s in raw];data={s:raw[s] for s in avail}
    r,tr,by=eng.run(data,pn+"_"+name,avail)
    r["period"]=pn;r["portfolio"]=name;r["symbols"]=len(avail);rows.append(r)
    if len(by):bys.append(by.assign(period=pn,portfolio=name))
 finally:eng.START_BALANCE=ob;eng.NOTIONAL=on
 pd.DataFrame(rows).to_csv("backtest/data/ltm_base15_v2_compare_summary.csv",index=False)
 if bys:pd.concat(bys,ignore_index=True).to_csv("backtest/data/ltm_base15_v2_compare_by_symbol.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
