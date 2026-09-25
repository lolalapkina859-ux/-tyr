import time, requests, numpy as np, pandas as pd

SYMBOL="GOLD(XAU)-USDT"
INTERVAL="5m"
START="2026-03-01"
END="2026-09-01"
ACCEL=0.12
START_MULT=2.0
SMOOTH=3
VWAP_BOOST=1.5
NOTIONAL=60.0
LOT_SIZE=0.02
OZ_PER_LOT=100.0
START_BALANCE=145.58352658

def fetch_bingx():
    """Download XAUUSD 5-minute spot-gold bars from Dukascopy via dukascopy-node CLI."""
    import subprocess, tempfile, os, glob
    with tempfile.TemporaryDirectory() as td:
        cmd=["npx","-y","dukascopy-node","-i","xauusd","-from",START,"-to",END,"-t","m5","-f","csv","-dir",td]
        print("DOWNLOAD", " ".join(cmd))
        subprocess.run(cmd,check=True)
        files=glob.glob(os.path.join(td,"*.csv"))
        if not files:
            raise RuntimeError("Dukascopy downloader produced no CSV")
        d=pd.read_csv(files[0])
    d.columns=[str(x).strip().lower() for x in d.columns]
    # dukascopy-node CSV uses timestamp/open/high/low/close/volume
    tcol=next((x for x in ["timestamp","time","date","datetime"] if x in d.columns),None)
    if tcol is None:
        raise RuntimeError(f"Unknown Dukascopy columns: {list(d.columns)}")
    raw_ts=d[tcol]
    if pd.api.types.is_numeric_dtype(raw_ts) or pd.to_numeric(raw_ts,errors="coerce").notna().mean() > 0.95:
        # dukascopy-node emits Unix timestamps in milliseconds (e.g. 1612137600000).
        nums=pd.to_numeric(raw_ts,errors="coerce")
        d["time"]=pd.to_datetime(nums,unit="ms",utc=True,errors="coerce")
    else:
        d["time"]=pd.to_datetime(raw_ts,utc=True,errors="coerce")
    need=["open","high","low","close"]
    if any(x not in d.columns for x in need):
        raise RuntimeError(f"Missing OHLC columns: {list(d.columns)}")
    if "volume" not in d.columns:
        d["volume"]=1.0
    for x in ["open","high","low","close","volume"]:
        d[x]=pd.to_numeric(d[x],errors="coerce")
    d=d.dropna(subset=["time","open","high","low","close"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(d)<1000:
        raise RuntimeError(f"Only {len(d)} XAUUSD candles downloaded")
    return d[["time","open","high","low","close","volume"]]

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

def backtest_lots(d,buycol,sellcol):
    """XAUUSD PnL for fixed lot size. Standard contract assumption: 1.00 lot = 100 troy oz."""
    side=0; entry=0.0; equity=START_BALANCE; peak=equity; maxdd=0.0
    trades=[]; max_loss=0.0; max_win=0.0
    qty_oz=LOT_SIZE*OZ_PER_LOT
    for r in d.itertuples():
        sig=1 if getattr(r,buycol) else (-1 if getattr(r,sellcol) else 0)
        if not sig or sig==side:
            continue
        if side:
            pnl=qty_oz*side*(r.close-entry)
            equity+=pnl; trades.append(pnl)
            peak=max(peak,equity)
            maxdd=min(maxdd,equity-peak)
            max_loss=min(max_loss,pnl); max_win=max(max_win,pnl)
        side=sig; entry=r.close
    if side and len(d):
        px=float(d.close.iloc[-1])
        pnl=qty_oz*side*(px-entry)
        equity+=pnl; trades.append(pnl)
        peak=max(peak,equity); maxdd=min(maxdd,equity-peak)
        max_loss=min(max_loss,pnl); max_win=max(max_win,pnl)
    a=np.asarray(trades,float)
    return {"lot_size":LOT_SIZE,"oz_exposure":qty_oz,"trades":len(a),
            "wins":int((a>0).sum()),"win_rate":float((a>0).mean()*100 if len(a) else 0),
            "net_pnl_usd":float(a.sum()),"final_equity":float(equity),
            "max_closed_dd_usd":float(maxdd),"avg_trade_usd":float(a.mean() if len(a) else 0),
            "best_trade_usd":float(max_win),"worst_trade_usd":float(max_loss)}

def floating_risk_test(d,buycol,sellcol,lots=(0.001,0.002,0.003,0.005)):
    rows=[]
    for lot in lots:
        qty=lot*OZ_PER_LOT
        side=0; entry=0.0; balance=START_BALANCE; peak_equity=START_BALANCE
        min_equity=START_BALANCE; max_float_dd=0.0; liquidated=False; trades=0
        for r in d.itertuples():
            # mark-to-market before acting on the bar-close signal
            equity=balance if side==0 else balance + qty*side*(r.close-entry)
            peak_equity=max(peak_equity,equity)
            min_equity=min(min_equity,equity)
            if peak_equity>0:
                max_float_dd=min(max_float_dd,(equity-peak_equity)/peak_equity*100)
            if equity<=0:
                liquidated=True
            sig=1 if getattr(r,buycol) else (-1 if getattr(r,sellcol) else 0)
            if sig and sig!=side:
                if side:
                    balance += qty*side*(r.close-entry); trades+=1
                side=sig; entry=r.close
        if side and len(d):
            balance += qty*side*(float(d.close.iloc[-1])-entry); trades+=1
        rows.append({"lot":lot,"trades":trades,"net_pnl_usd":balance-START_BALANCE,
                     "final_balance":balance,"min_floating_equity":min_equity,
                     "max_floating_dd_pct":max_float_dd,"equity_below_zero":liquidated})
    return pd.DataFrame(rows)

def intrabar_risk_test(d,buycol,sellcol,lots=(0.001,0.002),spread_usd=0.30):
    """Conservative 5m risk: mark open trades at adverse candle extreme and charge spread per completed round-trip."""
    rows=[]
    for lot in lots:
        qty=lot*OZ_PER_LOT
        side=0; entry=0.0; balance=START_BALANCE
        peak=START_BALANCE; min_eq=START_BALANCE; maxdd=0.0; trades=0
        gross=0.0; costs=0.0
        for r in d.itertuples():
            if side:
                adverse=float(r.low) if side==1 else float(r.high)
                eq=balance + qty*side*(adverse-entry)
                peak=max(peak,balance)
                min_eq=min(min_eq,eq)
                if peak>0: maxdd=min(maxdd,(eq-peak)/peak*100)
            sig=1 if getattr(r,buycol) else (-1 if getattr(r,sellcol) else 0)
            if sig and sig!=side:
                if side:
                    pnl=qty*side*(float(r.close)-entry)
                    cost=qty*spread_usd
                    gross+=pnl; costs+=cost; balance+=pnl-cost; trades+=1
                side=sig; entry=float(r.close)
                peak=max(peak,balance); min_eq=min(min_eq,balance)
        if side and len(d):
            px=float(d.close.iloc[-1]); pnl=qty*side*(px-entry); cost=qty*spread_usd
            gross+=pnl; costs+=cost; balance+=pnl-cost; trades+=1
        rows.append({"lot":lot,"spread_usd_per_oz":spread_usd,"trades":trades,
                     "gross_pnl_usd":gross,"spread_cost_usd":costs,"net_pnl_usd":balance-START_BALANCE,
                     "final_balance":balance,"min_intrabar_equity":min_eq,
                     "max_intrabar_dd_pct":maxdd,"equity_below_zero":min_eq<=0})
    return pd.DataFrame(rows)

def bingx_cost_test(d,buycol,sellcol,lots=(0.001,0.002),spread_usd=0.30,taker_rate=0.0005):
    """Stress test: adverse HIGH/LOW equity + spread + taker fee on both entry and exit."""
    rows=[]
    for lot in lots:
        qty=lot*OZ_PER_LOT
        side=0; entry=0.0; balance=START_BALANCE; peak=START_BALANCE; min_eq=START_BALANCE
        maxdd=0.0; trades=0; gross=0.0; spread_cost=0.0; fees=0.0
        for r in d.itertuples():
            if side:
                adverse=float(r.low) if side==1 else float(r.high)
                eq=balance + qty*side*(adverse-entry)
                peak=max(peak,balance); min_eq=min(min_eq,eq)
                if peak>0: maxdd=min(maxdd,(eq-peak)/peak*100)
            sig=1 if getattr(r,buycol) else (-1 if getattr(r,sellcol) else 0)
            if sig and sig!=side:
                if side:
                    px=float(r.close); pnl=qty*side*(px-entry)
                    exit_fee=qty*px*taker_rate
                    sc=qty*spread_usd
                    gross+=pnl; fees+=exit_fee; spread_cost+=sc
                    balance+=pnl-exit_fee-sc; trades+=1
                px=float(r.close)
                entry_fee=qty*px*taker_rate
                fees+=entry_fee; balance-=entry_fee
                side=sig; entry=px
                peak=max(peak,balance); min_eq=min(min_eq,balance)
        if side and len(d):
            px=float(d.close.iloc[-1]); pnl=qty*side*(px-entry)
            exit_fee=qty*px*taker_rate; sc=qty*spread_usd
            gross+=pnl; fees+=exit_fee; spread_cost+=sc
            balance+=pnl-exit_fee-sc; trades+=1
        rows.append({"lot":lot,"taker_rate_pct":taker_rate*100,"spread_usd_per_oz":spread_usd,
                     "trades":trades,"gross_pnl_usd":gross,"trading_fees_usd":fees,
                     "spread_cost_usd":spread_cost,"net_pnl_usd":balance-START_BALANCE,
                     "final_balance":balance,"min_intrabar_equity":min_eq,
                     "max_intrabar_dd_pct":maxdd,"equity_below_zero":min_eq<=0})
    return pd.DataFrame(rows)

def main():
    d=fetch_bingx(); print("CANDLES",len(d),d.time.iloc[0],d.time.iloc[-1])
    s=signals(d)
    rows=[]
    for name,b,se in [("RAW_FLIPS","raw_buy","raw_sell"),("SESSION_VWAP_CONFIRMED_ONLY","conf_buy","conf_sell")]:
        r=backtest(s,b,se);r["variant"]=name;rows.append(r)
    pd.DataFrame(rows).to_csv("backtest/data/arc_vwap_xau_5m_summary.csv",index=False)
    lot_rows=[]
    for name,b,se in [("RAW_FLIPS","raw_buy","raw_sell"),("SESSION_VWAP_CONFIRMED_ONLY","conf_buy","conf_sell")]:
        r=backtest_lots(s,b,se); r["variant"]=name; lot_rows.append(r)
    pd.DataFrame(lot_rows).to_csv("backtest/data/arc_vwap_xau_5m_lot002_summary.csv",index=False)
    print("\n0.02 LOT RESULTS (1 lot = 100 oz):")
    print(pd.DataFrame(lot_rows).to_string(index=False))
    risk=floating_risk_test(s,"conf_buy","conf_sell")
    risk.to_csv("backtest/data/arc_vwap_xau_5m_floating_risk.csv",index=False)
    print("\nVWAP CONFIRMED FLOATING RISK — START BALANCE $150-ish:")
    print(risk.to_string(index=False))
    intrabar=intrabar_risk_test(s,"conf_buy","conf_sell",spread_usd=0.30)
    intrabar.to_csv("backtest/data/arc_vwap_xau_5m_intrabar_risk.csv",index=False)
    print("\nVWAP CONFIRMED INTRABAR HIGH/LOW RISK + $0.30 SPREAD/OZ:")
    print(intrabar.to_string(index=False))
    bingx=bingx_cost_test(s,"conf_buy","conf_sell",spread_usd=0.30,taker_rate=0.0005)
    bingx.to_csv("backtest/data/arc_vwap_xau_5m_bingx_costs.csv",index=False)
    print("\nVWAP CONFIRMED BINGX COST STRESS — 0.05% TAKER EACH SIDE + $0.30/OZ SPREAD:")
    print(bingx.to_string(index=False))
    s[["time","open","high","low","close","volume","raw_buy","raw_sell","conf_buy","conf_sell"]].to_csv("backtest/data/arc_vwap_xau_5m_signals.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":main()
