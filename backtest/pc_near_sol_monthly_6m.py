from __future__ import annotations

from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START="2026-03-01"; END="2026-09-01"; TF="30m"; NOTIONAL=60.0
SYMBOLS=["NEARUSDT","SOLUSDT"]
MONTHS=[("2026-03","2026-03-01","2026-04-01"),("2026-04","2026-04-01","2026-05-01"),("2026-05","2026-05-01","2026-06-01"),("2026-06","2026-06-01","2026-07-01"),("2026-07","2026-07-01","2026-08-01"),("2026-08","2026-08-01","2026-09-01")]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def ret(side,e,x): return x/e-1 if side=="LONG" else e/x-1

def run_month(d,sym,label,start,end):
    x=d[(d.time>=pd.Timestamp(start,tz="UTC"))&(d.time<pd.Timestamp(end,tz="UTC"))].reset_index(drop=True)
    pos=None; rows=[]
    for _,r in x.iterrows():
        buy,sell,px=bool(r.pc_buy),bool(r.pc_sell),float(r.close)
        if pos and ((pos[0]=="LONG" and sell) or (pos[0]=="SHORT" and buy)):
            rr=ret(pos[0],pos[1],px); rows.append([sym,label,pos[0],pos[2],r.time,pos[1],px,rr*100,NOTIONAL*rr,"OPPOSITE"]); pos=None
        if pos is None and (buy or sell): pos=("LONG" if buy else "SHORT",px,r.time)
    if pos and len(x):
        r=x.iloc[-1]; px=float(r.close); rr=ret(pos[0],pos[1],px); rows.append([sym,label,pos[0],pos[2],r.time,pos[1],px,rr*100,NOTIONAL*rr,"MONTH_END"])
    return rows

def main():
    trades=[]; failures=[]
    for sym in SYMBOLS:
        try:
            raw=download_klines(sym,TF,START,END).reset_index(drop=True); raw["time"]=pd.to_datetime(raw.time,utc=True)
            d=purple_cloud(raw); d["time"]=raw.time
            for label,a,b in MONTHS: trades += run_month(d,sym,label,a,b)
        except Exception as e: failures.append({"symbol":sym,"error":str(e)})
    cols=["symbol","month","side","entry_time","exit_time","entry","exit","return_pct","pnl_60_usdt","exit_reason"]
    t=pd.DataFrame(trades,columns=cols)
    if len(t):
        s=t.groupby(["symbol","month"]).agg(trades=("return_pct","size"),return_pct=("return_pct","sum"),pnl_60_usdt=("pnl_60_usdt","sum"),wins=("return_pct",lambda z:(z>0).sum())).reset_index()
        s["winrate"]=100*s.wins/s.trades
        longs=t[t.side=="LONG"].groupby(["symbol","month"]).return_pct.sum().rename("long_return_pct")
        shorts=t[t.side=="SHORT"].groupby(["symbol","month"]).return_pct.sum().rename("short_return_pct")
        s=s.merge(longs,on=["symbol","month"],how="left").merge(shorts,on=["symbol","month"],how="left").fillna(0)
    else: s=pd.DataFrame()
    s.to_csv(OUT/"pc_near_sol_monthly_6m_summary.csv",index=False); t.to_csv(OUT/"pc_near_sol_monthly_6m_trades.csv",index=False); pd.DataFrame(failures).to_csv(OUT/"pc_near_sol_monthly_6m_failures.csv",index=False)
    print(s.to_string(index=False))
if __name__=="__main__": main()
