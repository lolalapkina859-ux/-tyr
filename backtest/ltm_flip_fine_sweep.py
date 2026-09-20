import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]
DISTANCES=[4.2,4.4,4.6,4.8]
ATR_LEN=13

def atr(d,n=13):
    p=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def signals(d,m):
    d=d.reset_index(drop=True).copy(); a=atr(d,ATR_LEN)
    trail=np.full(len(d),np.nan); trend=1
    buy=np.zeros(len(d),bool); sell=np.zeros(len(d),bool)
    for i in range(len(d)):
        if not np.isfinite(a.iloc[i]): continue
        src=d.close.iloc[i]; up=src-a.iloc[i]*m; dn=src+a.iloc[i]*m
        if i==0 or not np.isfinite(trail[i-1]):
            trail[i]=up if trend==1 else dn; continue
        prev=trail[i-1]
        if trend==1 and src<prev:
            trend=-1; trail[i]=dn
            if i>=60:sell[i]=True
        elif trend==-1 and src>prev:
            trend=1; trail[i]=up
            if i>=60:buy[i]=True
        elif trend==1: trail[i]=max(up,prev)
        else: trail[i]=min(dn,prev)
    d["pc_buy"]=buy; d["pc_sell"]=sell
    return d

def main():
    ob,on=eng.START_BALANCE,eng.NOTIONAL
    eng.START_BALANCE=145.58352658; eng.NOTIONAL=60.
    rows=[]; bys=[]
    try:
        for pn,st,en in PERIODS:
            raw={}
            for s in SYMBOLS:
                d=eng.download_klines(s,"30m",st,en).reset_index(drop=True)
                d["time"]=pd.to_datetime(d["time"],utc=True); raw[s]=d
            for m in DISTANCES:
                data={}
                for s,d in raw.items():
                    x=signals(d,m)
                    data[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
                mode=f"LTM_FLIP_{m:.1f}ATR"
                r,tr,by=eng.run(data,pn+"_"+mode,SYMBOLS)
                r["period"]=pn; r["mode"]=mode; r["flip_atr"]=m; rows.append(r)
                if len(by): bys.append(by.assign(period=pn,mode=mode,flip_atr=m))
    finally:
        eng.START_BALANCE=ob; eng.NOTIONAL=on
    pd.DataFrame(rows).to_csv("backtest/data/ltm_flip_fine_sweep_summary.csv",index=False)
    if bys: pd.concat(bys,ignore_index=True).to_csv("backtest/data/ltm_flip_fine_sweep_by_symbol.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
