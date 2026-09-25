import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

SYMBOLS=["ZECUSDT","USELESSUSDT","FETUSDT","HYPEUSDT","JTOUSDT","VETUSDT","XRPUSDT","ETHUSDT","UNIUSDT","INJUSDT","SEIUSDT","1000SHIBUSDT","DYDXUSDT","NEARUSDT","FLOWUSDT"]
PERIODS=[("MAR_SEP_2026","2026-03-01","2026-09-01"),("OOS_SEP_MAR","2025-09-01","2026-03-01")]
RF_PERIOD=100
RF_MULT=3.0

def ema(s,n):
    return s.ewm(span=n,adjust=False).mean()

def range_signals(d):
    d=d.reset_index(drop=True).copy()
    src=d.close.astype(float)
    wper=RF_PERIOD*2-1
    avrng=ema((src-src.shift(1)).abs(),RF_PERIOD)
    smrng=ema(avrng,wper)*RF_MULT

    filt=np.full(len(d),np.nan)
    upward=np.zeros(len(d),dtype=float)
    downward=np.zeros(len(d),dtype=float)
    long_cond=np.zeros(len(d),dtype=bool)
    short_cond=np.zeros(len(d),dtype=bool)
    cond_ini=np.zeros(len(d),dtype=int)
    buy=np.zeros(len(d),dtype=bool)
    sell=np.zeros(len(d),dtype=bool)

    for i in range(len(d)):
        x=float(src.iloc[i])
        r=float(smrng.iloc[i]) if np.isfinite(smrng.iloc[i]) else np.nan
        prev=filt[i-1] if i>0 and np.isfinite(filt[i-1]) else 0.0
        if not np.isfinite(r):
            filt[i]=x
        elif x>prev:
            filt[i]=prev if x-r<prev else x-r
        else:
            filt[i]=prev if x+r>prev else x+r

        if i>0:
            upward[i]=upward[i-1]+1 if filt[i]>filt[i-1] else (0 if filt[i]<filt[i-1] else upward[i-1])
            downward[i]=downward[i-1]+1 if filt[i]<filt[i-1] else (0 if filt[i]>filt[i-1] else downward[i-1])

        long_cond[i]=(x>filt[i] and upward[i]>0)
        short_cond[i]=(x<filt[i] and downward[i]>0)
        prev_state=cond_ini[i-1] if i>0 else 0
        cond_ini[i]=1 if long_cond[i] else (-1 if short_cond[i] else prev_state)
        buy[i]=long_cond[i] and prev_state==-1
        sell[i]=short_cond[i] and prev_state==1

    d["pc_buy"]=buy
    d["pc_sell"]=sell
    return d

def main():
    rows=[]; bys=[]
    oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
    eng.START_BALANCE=145.58352658; eng.NOTIONAL=60.
    try:
        for pn,st,en in PERIODS:
            raw={}
            for s in SYMBOLS:
                try:
                    d=eng.download_klines(s,"30m",st,en).reset_index(drop=True)
                    if len(d)<500: continue
                    d["time"]=pd.to_datetime(d["time"],utc=True)
                    x=range_signals(d)
                    raw[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
                except Exception as e:
                    print("SKIP",s,e)
            avail=[s for s in SYMBOLS if s in raw]
            r,tr,by=eng.run(raw,pn+"_RANGE100_3",avail)
            r["period"]=pn;r["variant"]="RANGE100_3";r["symbols"]=len(avail);rows.append(r)
            if len(by): bys.append(by.assign(period=pn,variant="RANGE100_3"))
    finally:
        eng.START_BALANCE=oldb;eng.NOTIONAL=oldn
    pd.DataFrame(rows).to_csv("backtest/data/range_filter_base15_summary.csv",index=False)
    if bys: pd.concat(bys,ignore_index=True).to_csv("backtest/data/range_filter_base15_by_symbol.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    main()
