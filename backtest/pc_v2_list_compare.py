from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

ORIGINAL15=["ZECUSDT","USELESSUSDT","FETUSDT","HYPEUSDT","JTOUSDT","VETUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
V2_LIST=["ZECUSDT","USELESSUSDT","WLDUSDT","ENAUSDT","XLMUSDT","PENDLEUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
PORTFOLIOS={"PC_ORIGINAL15":ORIGINAL15,"PC_V2_LIST":V2_LIST}
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]
PERIOD=20;ALPHA=1.5;BPT=.2;SPT=.2

def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def atr(d,n):
 p=d.close.shift(1)
 tr=pd.concat([(d.high-d.low),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return rma(tr,n)
def vwma(x,v,n): return (x*v).rolling(n).sum()/v.rolling(n).sum().replace(0,np.nan)
def purple_cloud(d):
 d=d.copy().reset_index(drop=True);n1=int(np.ceil(PERIOD/4));n2=int(np.ceil(PERIOD/2))
 x2=atr(d,PERIOD)*ALPHA;xh=d.close+x2;xl=d.close-x2;hl2=(d.high+d.low)/2
 a1=vwma(hl2*d.volume,d.volume,n1)/vwma(d.volume,d.volume,n1)
 a2=vwma(hl2*d.volume,d.volume,n2)/vwma(d.volume,d.volume,n2)
 a3=2*a1-a2;a4=vwma(a3,d.volume,PERIOD);b1=rma(d.close,PERIOD);a5=2*a4*b1/(a4+b1)
 buy=(a5<=xl)&(d.close>b1*(1+BPT*.01));sell=(a5>=xh)&(d.close<b1*(1-SPT*.01))
 xs=np.zeros(len(d),dtype=int)
 for i in range(1,len(d)): xs[i]=1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
 changed=pd.Series(xs).ne(pd.Series(xs).shift(1));d["pc_buy"]=buy&changed;d["pc_sell"]=sell&changed
 return d

def main():
 universe=sorted(set(ORIGINAL15+V2_LIST));rows=[];bys=[]
 ob,on=eng.START_BALANCE,eng.NOTIONAL;eng.START_BALANCE=145.58352658;eng.NOTIONAL=60.
 try:
  for pn,st,en in PERIODS:
   raw={}
   for s in universe:
    try:
     d=eng.download_klines(s,"30m",st,en).reset_index(drop=True)
     if len(d)<500:continue
     d["time"]=pd.to_datetime(d["time"],utc=True);x=purple_cloud(d)
     raw[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
    except Exception as e:print("SKIP",s,e)
   for name,symbols in PORTFOLIOS.items():
    avail=[s for s in symbols if s in raw];data={s:raw[s] for s in avail}
    r,tr,by=eng.run(data,pn+"_"+name,avail)
    r["period"]=pn;r["portfolio"]=name;r["symbols"]=len(avail);rows.append(r)
    if len(by):bys.append(by.assign(period=pn,portfolio=name))
 finally:eng.START_BALANCE=ob;eng.NOTIONAL=on
 pd.DataFrame(rows).to_csv("backtest/data/pc_v2_list_compare_summary.csv",index=False)
 if bys:pd.concat(bys,ignore_index=True).to_csv("backtest/data/pc_v2_list_compare_by_symbol.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
