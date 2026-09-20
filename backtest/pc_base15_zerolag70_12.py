from __future__ import annotations
import numpy as np, pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import purple_cloud, atr

SYMBOLS=list(eng.BASE14)+["FLOWUSDT"]
START_BALANCE=145.58352658
NOTIONAL=60.0
ZLEN=70
MULT=1.2

def zero_lag_trend(d):
    d=d.copy().reset_index(drop=True)
    lag=int(np.floor((ZLEN-1)/2))
    src=d.close
    zsrc=src+(src-src.shift(lag))
    zlema=zsrc.ewm(span=ZLEN,adjust=False).mean()
    volatility=atr(d,ZLEN).rolling(ZLEN*3).max()*MULT
    trend=np.zeros(len(d),dtype=int)
    for i in range(1,len(d)):
        trend[i]=trend[i-1]
        if np.isfinite(zlema.iloc[i]) and np.isfinite(volatility.iloc[i]):
            upper=zlema.iloc[i]+volatility.iloc[i]
            lower=zlema.iloc[i]-volatility.iloc[i]
            prev_upper=zlema.iloc[i-1]+volatility.iloc[i-1] if np.isfinite(volatility.iloc[i-1]) else np.nan
            prev_lower=zlema.iloc[i-1]-volatility.iloc[i-1] if np.isfinite(volatility.iloc[i-1]) else np.nan
            if np.isfinite(prev_upper) and src.iloc[i]>upper and src.iloc[i-1]<=prev_upper:
                trend[i]=1
            if np.isfinite(prev_lower) and src.iloc[i]<lower and src.iloc[i-1]>=prev_lower:
                trend[i]=-1
    return pd.Series(trend,index=d.index)

def load(s):
    d=eng.download_klines(s,"30m",eng.START,eng.END).reset_index(drop=True)
    d["time"]=pd.to_datetime(d["time"],utc=True)
    d=purple_cloud(d)
    z=zero_lag_trend(d)
    # Keep original PC events only when current 30m Zero Lag trend agrees.
    d["pc_buy"]=d["pc_buy"] & (z==1)
    d["pc_sell"]=d["pc_sell"] & (z==-1)
    return d[["time","high","low","close","pc_buy","pc_sell"]]

def main():
    print("BASE15 + AlgoAlpha Zero Lag trend filter | 30m | Length=70 Mult=1.2")
    data={s:load(s) for s in SYMBOLS}
    oldb,oldn=eng.START_BALANCE,eng.NOTIONAL
    try:
        eng.START_BALANCE=START_BALANCE; eng.NOTIONAL=NOTIONAL
        row,trades,by=eng.run(data,"BASE15_ZL70_1.2",SYMBOLS)
    finally:
        eng.START_BALANCE=oldb; eng.NOTIONAL=oldn
    print("\nRESULT\n",row)
    print("\nBY SYMBOL\n",by.to_string(index=False))
    trades.to_csv("backtest/data/pc_base15_zerolag70_12_trades.csv",index=False)
    by.to_csv("backtest/data/pc_base15_zerolag70_12_by_symbol.csv",index=False)

if __name__=="__main__":
    main()
