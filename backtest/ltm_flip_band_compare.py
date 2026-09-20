from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=145.58352658
NOTIONAL=60.
ATR_LEN=13
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]

def rma_tv(s,n):
    # TradingView-like Wilder RMA: SMA seed then recursive alpha=1/n
    a=s.to_numpy(float); out=np.full(len(a),np.nan)
    valid=np.where(np.isfinite(a))[0]
    if len(valid)<n:return pd.Series(out,index=s.index)
    start=valid[0]+n-1
    if start>=len(a):return pd.Series(out,index=s.index)
    out[start]=np.nanmean(a[valid[0]:start+1])
    for i in range(start+1,len(a)):
        out[i]=out[i-1]+(a[i]-out[i-1])/n if np.isfinite(a[i]) else out[i-1]
    return pd.Series(out,index=s.index)

def atr_tv(d,n):
    prev=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    return rma_tv(tr,n)

def ltm(d,flip_choice):
    src=d.close.reset_index(drop=True); at=atr_tv(d.reset_index(drop=True),ATR_LEN)
    # exact Balanced geometry from source: base 4.0, proportional step .25
    ms=[4.0,5.0,6.0,7.0]
    ts=[np.full(len(d),np.nan) for _ in range(4)]
    trend=np.ones(len(d),dtype=int)
    buy=np.zeros(len(d),bool); sell=np.zeros(len(d),bool)
    warm=max(ATR_LEN*3,60)
    for i in range(len(d)):
        if not np.isfinite(at.iloc[i]): continue
        upp=[src.iloc[i]-at.iloc[i]*m for m in ms]
        low=[src.iloc[i]+at.iloc[i]*m for m in ms]
        if i==0 or any(not np.isfinite(x[i-1]) for x in ts):
            for k in range(4): ts[k][i]=upp[k]
            continue
        trend[i]=trend[i-1]
        fp=ts[flip_choice-1][i-1]
        down=trend[i-1]==1 and src.iloc[i]<fp
        up=trend[i-1]==-1 and src.iloc[i]>fp
        if trend[i-1]==1:
            if down:
                trend[i]=-1
                for k in range(4): ts[k][i]=low[k]
                if i>=warm:sell[i]=True
            else:
                for k in range(4): ts[k][i]=max(upp[k],ts[k][i-1])
        else:
            if up:
                trend[i]=1
                for k in range(4): ts[k][i]=upp[k]
                if i>=warm:buy[i]=True
            else:
                for k in range(4): ts[k][i]=min(low[k],ts[k][i-1])
    x=d.copy(); x["pc_buy"]=buy; x["pc_sell"]=sell
    return x

def main():
    rows=[]; parts=[]
    oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
    eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
    try:
        for pname,start,end in PERIODS:
            raw={}
            for sym in SYMBOLS:
                d=eng.download_klines(sym,"30m",start,end).reset_index(drop=True)
                d["time"]=pd.to_datetime(d["time"],utc=True); raw[sym]=d
            for band in [2,3,4]:
                data={}
                for sym,d in raw.items():
                    x=ltm(d,band)
                    data[sym]=x[["time","high","low","close","pc_buy","pc_sell"]]
                mode=f"LTM_BALANCED_BAND{band}_FLIP"
                r,tr,by=eng.run(data,f"{pname}_{mode}",SYMBOLS)
                r["period"]=pname;r["mode"]=mode;rows.append(r)
                if len(by):parts.append(by.assign(period=pname,mode=mode))
                print("\n",r)
    finally:
        eng.START_BALANCE=oldb;eng.NOTIONAL=oldn
    pd.DataFrame(rows).to_csv("backtest/data/ltm_flip_band_compare_summary.csv",index=False)
    if parts:pd.concat(parts,ignore_index=True).to_csv("backtest/data/ltm_flip_band_compare_by_symbol.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
