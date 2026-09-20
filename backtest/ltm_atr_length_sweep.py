import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]
ATR_LENGTHS=[10,13,16,20]
FLIP_ATR=4.5

def atr(d,n):
    p=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def signals(d,n):
    d=d.reset_index(drop=True).copy(); a=atr(d,n)
    trail=np.full(len(d),np.nan); trend=1
    buy=np.zeros(len(d),bool); sell=np.zeros(len(d),bool)
    warm=max(n*3,60)
    for i in range(len(d)):
        if not np.isfinite(a.iloc[i]): continue
        src=d.close.iloc[i]; up=src-a.iloc[i]*FLIP_ATR; dn=src+a.iloc[i]*FLIP_ATR
        if i==0 or not np.isfinite(trail[i-1]):
            trail[i]=up if trend==1 else dn; continue
        prev=trail[i-1]
        if trend==1 and src<prev:
            trend=-1; trail[i]=dn
            if i>=warm:sell[i]=True
        elif trend==-1 and src>prev:
            trend=1; trail[i]=up
            if i>=warm:buy[i]=True
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
            for n in ATR_LENGTHS:
                data={}
                for s,d in raw.items():
                    x=signals(d,n); data[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
                mode=f"LTM_FLIP_4.5_ATR_LEN{n}"
                r,tr,by=eng.run(data,pn+"_"+mode,SYMBOLS)
                r["period"]=pn; r["mode"]=mode; r["atr_len"]=n; r["flip_atr"]=FLIP_ATR; rows.append(r)
                if len(by): bys.append(by.assign(period=pn,mode=mode,atr_len=n,flip_atr=FLIP_ATR))
    finally:
        eng.START_BALANCE=ob; eng.NOTIONAL=on
    pd.DataFrame(rows).to_csv("backtest/data/ltm_atr_length_sweep_summary.csv",index=False)
    if bys: pd.concat(bys,ignore_index=True).to_csv("backtest/data/ltm_atr_length_sweep_by_symbol.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
