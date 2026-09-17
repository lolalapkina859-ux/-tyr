from __future__ import annotations

from pathlib import Path
import time
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

TF = "30m"
PERIODS = [("MAR_JUN", "2026-03-01", "2026-06-01"), ("JUN_SEP", "2026-06-01", "2026-09-01")]
CURRENT14 = {"ZECUSDT","USELESSUSDT","HYPEUSDT","FETUSDT","JTOUSDT","XRPUSDT","VETUSDT","INJUSDT","ETHUSDT","DYDXUSDT","1000SHIBUSDT","UNIUSDT","SEIUSDT","NEARUSDT"}
# Broad crypto USD-M universe. Recent 2026 listings without a full six-month history
# will be skipped automatically rather than being compared unfairly.
CANDIDATES = [
"BTCUSDT","ADAUSDT","DOGEUSDT","SOLUSDT","BNBUSDT","LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","DOTUSDT","ATOMUSDT","APTUSDT","SUIUSDT","ARBUSDT","OPUSDT","TIAUSDT","FILUSDT","ETCUSDT","AAVEUSDT","ICPUSDT","HBARUSDT","TRXUSDT","EOSUSDT","IOTAUSDT","THETAUSDT","KAVAUSDT","ZILUSDT","ENJUSDT","CHZUSDT","1INCHUSDT","CELOUSDT","FLOWUSDT","MINAUSDT","ROSEUSDT","QTUMUSDT","NEOUSDT","DASHUSDT","ZRXUSDT","ANKRUSDT","BATUSDT","IOSTUSDT","ONTUSDT","WAVESUSDT","KSMUSDT","MASKUSDT","ENSUSDT","API3USDT","GMXUSDT","SSVUSDT","CFXUSDT","IDUSDT","RDNTUSDT","HOOKUSDT","MAGICUSDT","AGIXUSDT","HIGHUSDT","TRUUSDT","LQTYUSDT","JOEUSDT","CYBERUSDT","ARKMUSDT","BIGTIMEUSDT","NTRNUSDT","BEAMXUSDT","ACEUSDT","XAIUSDT","MANTAUSDT","ALTUSDT","DYMUSDT","PIXELUSDT","PORTALUSDT","ENAUSDT","WUSDT","TAOUSDT","REZUSDT","BBUSDT","IOUSDT","ZKUSDT","LISTAUSDT","ZROUSDT","TONUSDT","BANANAUSDT","RENDERUSDT","EIGENUSDT","NEIROUSDT","HMSTRUSDT","TURBOUSDT","PNUTUSDT","ACTUSDT","KAIAUSDT","MOVEUSDT","MEUSDT","VIRTUALUSDT","PENGUUSDT","BIOUSDT",
"XLMUSDT","ALGOUSDT","SANDUSDT","MANAUSDT","GALAUSDT","AXSUSDT","RUNEUSDT","CRVUSDT","LDOUSDT","STXUSDT","IMXUSDT","GRTUSDT","JUPUSDT","BONKUSDT","FLOKIUSDT","NOTUSDT","PYTHUSDT","WLDUSDT","ARUSDT","MKRUSDT","COMPUSDT","SNXUSDT","OMUSDT","ONDOUSDT","SAGAUSDT","STRKUSDT","AEVOUSDT","BLURUSDT","MEMEUSDT","1000SATSUSDT","ORDIUSDT","WIFUSDT","1000PEPEUSDT"
]
OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)

def run_symbol(sym,start,end):
    d=download_klines(sym,TF,start,end).reset_index(drop=True)
    # Require near-complete 3-month coverage, not merely 100 candles.
    if len(d)<3800: raise ValueError(f"insufficient 30m coverage: {len(d)} candles")
    d["time"]=pd.to_datetime(d["time"],utc=True); pc=purple_cloud(d); pos=None; rows=[]
    for i,b in d.iterrows():
        buy=bool(pc.iloc[i].pc_buy); sell=bool(pc.iloc[i].pc_sell)
        if pos and ((pos["side"]=="LONG" and sell) or (pos["side"]=="SHORT" and buy)):
            ex=float(b.close); r=ex/pos["entry"]-1 if pos["side"]=="LONG" else pos["entry"]/ex-1
            rows.append(r*100); pos=None
        if pos is None and (buy or sell): pos={"side":"LONG" if buy else "SHORT","entry":float(b.close)}
    if pos:
        ex=float(d.iloc[-1].close); r=ex/pos["entry"]-1 if pos["side"]=="LONG" else pos["entry"]/ex-1; rows.append(r*100)
    return rows

def main():
    symbols=sorted(set(CANDIDATES)-CURRENT14); stats=[]; skipped=[]
    print(f"FORGOTTEN CANDIDATE RESCREEN | {len(symbols)} symbols | current14 excluded")
    for k,s in enumerate(symbols,1):
        print(f"[{k}/{len(symbols)}] {s}")
        vals={}
        ok=True
        for name,start,end in PERIODS:
            try:
                x=run_symbol(s,start,end); vals[name]=sum(x); stats.append({"symbol":s,"period":name,"trades":len(x),"sum_return_pct":sum(x)})
            except Exception as e:
                skipped.append({"symbol":s,"period":name,"error":str(e)}); ok=False; print(" SKIP",name,e)
        if ok: print(" ",vals)
        time.sleep(.1)
    summary=pd.DataFrame(stats); summary.to_csv(OUT/"pc_forgotten_candidates_summary.csv",index=False)
    if len(summary):
        rank=summary.pivot(index="symbol",columns="period",values="sum_return_pct").reset_index()
        for c in ["MAR_JUN","JUN_SEP"]:
            if c not in rank: rank[c]=float("nan")
        rank=rank.dropna(subset=["MAR_JUN","JUN_SEP"])
        rank["both_positive"]=(rank.MAR_JUN>0)&(rank.JUN_SEP>0)
        rank["six_month_sum_pct"]=rank.MAR_JUN+rank.JUN_SEP
        rank["worst_period_pct"]=rank[["MAR_JUN","JUN_SEP"]].min(axis=1)
        rank=rank.sort_values(["both_positive","worst_period_pct","six_month_sum_pct"],ascending=False)
    else: rank=pd.DataFrame()
    rank.to_csv(OUT/"pc_forgotten_candidates_ranking.csv",index=False)
    winners=rank[rank.both_positive].copy() if len(rank) else pd.DataFrame(); winners.to_csv(OUT/"pc_forgotten_candidates_winners.csv",index=False)
    pd.DataFrame(skipped).to_csv(OUT/"pc_forgotten_candidates_skipped.csv",index=False)
    print("\nBOTH PERIODS POSITIVE\n",winners.to_string(index=False) if len(winners) else "NONE")

if __name__=="__main__": main()
