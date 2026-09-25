import time, requests, numpy as np, pandas as pd

SYMBOL="GOLD-XAUUSDT"
INTERVAL="5m"
START="2026-03-01"
END="2026-09-01"
ACCEL=0.12
START_MULT=2.0
SMOOTH=3
VWAP_BOOST=1.5
NOTIONAL=60.0
START_BALANCE=145.58352658

def fetch_bingx():
    url="https://open-api.bingx.com/openApi/swap/v3/quote/klines"
    start=int(pd.Timestamp(START,tz="UTC").timestamp()*1000)
    end=int(pd.Timestamp(END,tz="UTC").timestamp()*1000)-1
    out=[]
    cursor=end
    for _ in range(500):
        r=requests.get(url,params={"symbol":SYMBOL,"interval":INTERVAL,"limit":1440,"endTime":cursor},timeout=30)
        j=r.json()
        if j.get("code",0)!=0:
            raise RuntimeError(j)
        data=j.get("data",[])
        if not data: break
        rows=[]
        for x in data:
            if isinstance(x,dict):
                ts=int(x.get("time",x.get("timestamp",0)))
                rows.append([ts,float(x["open"]),float(x["high"]),float(x["low"]),float(x["close"]),float(x.get("volume",0))])
            else:
                # BingX public kline array fallback: time,open,close,high,low,volume...
                ts=int(x[0]); rows.append([ts,float(x[1]),float(x[3]),float(x[4]),float(x[2]),float(x[5])])
        out.extend(rows)
        mn=min(x[0] for x in rows)
        if mn<=start: break
        cursor=mn-1
        time.sleep(0.06)
    d=pd.DataFrame(out,columns=["time","open","high","low","close","volume"]).drop_duplicates("time").sort_values("time")
    d=d[(d.time>=start)&(d.time<=end)].reset_index(drop=True)
    if len(d)<1000:
        raise RuntimeError(f"Only {len(d)} candles downloaded; first={d.time.min() if len(d) else None} last={d.time.max() if len(d) else None}")
    d["time"]=pd.to_datetime(d.time,unit="ms",utc=True)
    return d

def rma(s,n):
    # TradingView-style RMA: SMA seed then recursive alpha=1/n
    a=s.to_numpy(float); out=np.full(len(a),np.nan)
    valid=np.where(np.isfinite(a))[0]
    if len(valid)<n:return pd.Series(out,index=s.index)
    k=valid[0]+n-1
    if k>=len(a):return pd.Series(out,index=s.index)
    out[k]=np.nanmean(a[k-n+1:k+1])
    for i in range(k+1,len(a)):
        out[i]=out[i-1]+(a[i]-out[i-1])/n
    return pd.Series(out,index=s.index)

def signals(d):
    prev=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    atr=rma(tr,14)
    atrslow=tr.rolling(100).mean()

    day=d.time.dt.floor("D")
    pv=(d.high+d.low)/2*d.volume
    cpv=pv.groupby(day).cumsum(); cv=d.volume.groupby(day).cumsum()
    vs=cpv/cv.replace(0,np.nan)
    vs=vs.fillna((d.high+d.low)/2)

    n=len(d); trend=np.ones(n,dtype=bool); arc=np.full(n,np.nan); vel=np.zeros(n)
    raw_b=np.zeros(n,bool);raw_s=np.zeros(n,bool);conf_b=np.zeros(n,bool);conf_s=np.zeros(n,bool)
    init=False
    for i in range(n):
        if not init and i>100 and np.isfinite(atrslow.iloc[i]):
            arc[i]=d.low.iloc[i]-atrslow.iloc[i]*START_MULT
            trend[i]=True;init=True
        elif init:
            arc[i]=arc[i-1]; trend[i]=trend[i-1]; vel[i]=vel[i-1]
            if d.close.iloc[i] < arc[i]: trend[i]=False
            if d.close.iloc[i] > arc[i]: trend[i]=True
            flipped=trend[i]!=trend[i-1]
            if flipped:
                if trend[i]:
                    arc[i]=d.low.iloc[i]-atrslow.iloc[i]*START_MULT; raw_b[i]=True
                else:
                    arc[i]=d.high.iloc[i]+atrslow.iloc[i]*START_MULT; raw_s[i]=True
                vel[i]=0.0
                agree=(d.close.iloc[i]>=vs.iloc[i]) if trend[i] else (d.close.iloc[i]<=vs.iloc[i])
                if agree:
                    conf_b[i]=trend[i]; conf_s[i]=not trend[i]
            # Pine bar_index % smooth == 0
            if i % SMOOTH==0:
                vd=0.0
                if np.isfinite(atr.iloc[i]) and atr.iloc[i]>0:
                    vd=min(abs(d.close.iloc[i]-vs.iloc[i])/(atr.iloc[i]*4),1.0)
                eff=ACCEL*(1+(VWAP_BOOST-1)*vd)
                vel[i]+=eff
                step=atrslow.iloc[i]*0.15
                arc[i]+=step*vel[i] if trend[i] else -step*vel[i]
    z=d.copy()
    z["raw_buy"]=raw_b;z["raw_sell"]=raw_s;z["conf_buy"]=conf_b;z["conf_sell"]=conf_s
    return z

def backtest(d,buycol,sellcol):
    side=0;entry=0.0;bal=START_BALANCE;peak=bal;maxdd=0.0;trades=[]
    for r in d.itertuples():
        sig=1 if getattr(r,buycol) else (-1 if getattr(r,sellcol) else 0)
        if not sig or sig==side: continue
        if side:
            pnl=NOTIONAL*side*(r.close/entry-1)
            bal+=pnl;trades.append(pnl)
        side=sig;entry=r.close
        peak=max(peak,bal);maxdd=min(maxdd,(bal-peak)/peak*100)
    if side and len(d):
        r=d.iloc[-1];pnl=NOTIONAL*side*(r.close/entry-1);bal+=pnl;trades.append(pnl)
    a=np.array(trades,float)
    return {"trades":len(a),"wins":int((a>0).sum()),"win_rate":float((a>0).mean()*100 if len(a) else 0),
            "net_pnl":float(a.sum()),"final_balance":float(bal),"return_pct":float((bal/START_BALANCE-1)*100),
            "max_closed_dd_pct":float(maxdd),"avg_trade":float(a.mean() if len(a) else 0)}

def main():
    d=fetch_bingx(); print("CANDLES",len(d),d.time.iloc[0],d.time.iloc[-1])
    s=signals(d)
    rows=[]
    for name,b,se in [("RAW_FLIPS","raw_buy","raw_sell"),("SESSION_VWAP_CONFIRMED_ONLY","conf_buy","conf_sell")]:
        r=backtest(s,b,se);r["variant"]=name;rows.append(r)
    pd.DataFrame(rows).to_csv("backtest/data/arc_vwap_xau_5m_summary.csv",index=False)
    s[["time","open","high","low","close","volume","raw_buy","raw_sell","conf_buy","conf_sell"]].to_csv("backtest/data/arc_vwap_xau_5m_signals.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":main()
