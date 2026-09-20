import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

SYMBOLS=["ZECUSDT","USELESSUSDT","FETUSDT","HYPEUSDT","JTOUSDT","VETUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]
ATR_LEN=16
FLIP_ATR=4.5

def atr(d,n):
    p=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def signals(d):
    d=d.reset_index(drop=True).copy(); a=atr(d,ATR_LEN)
    trail=np.full(len(d),np.nan); trend=1
    buy=np.zeros(len(d),bool); sell=np.zeros(len(d),bool); warm=max(ATR_LEN*3,60)
    for i in range(len(d)):
        if not np.isfinite(a.iloc[i]): continue
        src=d.close.iloc[i]; up=src-a.iloc[i]*FLIP_ATR; dn=src+a.iloc[i]*FLIP_ATR
        if i==0 or not np.isfinite(trail[i-1]):
            trail[i]=up if trend==1 else dn; continue
        prev=trail[i-1]
        if trend==1 and src<prev:
            trend=-1; trail[i]=dn
            if i>=warm: sell[i]=True
        elif trend==-1 and src>prev:
            trend=1; trail[i]=up
            if i>=warm: buy[i]=True
        elif trend==1: trail[i]=max(up,prev)
        else: trail[i]=min(dn,prev)
    d["pc_buy"]=buy; d["pc_sell"]=sell
    return d

def main():
    ob,on=eng.START_BALANCE,eng.NOTIONAL
    eng.START_BALANCE=145.58352658; eng.NOTIONAL=60.
    summaries=[]; all_by=[]
    try:
        for pn,st,en in PERIODS:
            data={}; available=[]
            for s in SYMBOLS:
                try:
                    d=eng.download_klines(s,"30m",st,en).reset_index(drop=True)
                    if len(d)<500:
                        print("SKIP_SHORT_HISTORY",s,len(d)); continue
                    d["time"]=pd.to_datetime(d["time"],utc=True)
                    x=signals(d)
                    data[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
                    available.append(s)
                except Exception as e:
                    print("SKIP",s,e)
            r,tr,by=eng.run(data,pn+"_LTM_BASE15",available)
            r["period"]=pn; r["symbols"]=len(available); summaries.append(r)
            if len(by): all_by.append(by.assign(period=pn))
    finally:
        eng.START_BALANCE=ob; eng.NOTIONAL=on
    pd.DataFrame(summaries).to_csv("backtest/data/ltm_base15_portfolio_summary.csv",index=False)
    if all_by:
        pd.concat(all_by,ignore_index=True).to_csv("backtest/data/ltm_base15_by_symbol.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
if __name__=="__main__": main()
