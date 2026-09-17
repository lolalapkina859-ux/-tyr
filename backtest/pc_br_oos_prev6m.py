from __future__ import annotations
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

SYMBOL='BRUSDT'
TF='30m'
PERIODS=[('SEP_DEC_2025','2025-09-01','2025-12-01'),('DEC_MAR_2026','2025-12-01','2026-03-01')]
OUT='backtest/data'


def run_period(name,start,end):
    d=download_klines(SYMBOL,TF,start,end)
    if d is None or len(d)<60:
        return {'period':name,'symbol':SYMBOL,'start':start,'end':end,'candles':0 if d is None else len(d),'trades':0,'sum_return_pct':None,'winrate_pct':None,'status':'insufficient_data'}, pd.DataFrame()
    pc=purple_cloud(d)
    sig=[]
    for i,r in pc.iterrows():
        side='LONG' if bool(r.pc_buy) else ('SHORT' if bool(r.pc_sell) else None)
        if side: sig.append((i,side,float(r.close)))
    trades=[]
    for j in range(len(sig)-1):
        i,side,entry=sig[j]; k,_,exitp=sig[j+1]
        ret=(exitp/entry-1)*100 if side=='LONG' else (entry/exitp-1)*100
        trades.append({'period':name,'symbol':SYMBOL,'entry_index':i,'exit_index':k,'side':side,'entry':entry,'exit':exitp,'return_pct':ret})
    t=pd.DataFrame(trades)
    return {'period':name,'symbol':SYMBOL,'start':start,'end':end,'candles':len(d),'trades':len(t),'sum_return_pct':float(t.return_pct.sum()) if len(t) else 0.0,'winrate_pct':float((t.return_pct>0).mean()*100) if len(t) else 0.0,'status':'ok'},t


def main():
    rows=[]; ts=[]
    for p,s,e in PERIODS:
        r,t=run_period(p,s,e); rows.append(r)
        if len(t): ts.append(t)
        print(r)
    summary=pd.DataFrame(rows)
    summary.to_csv(f'{OUT}/pc_br_oos_prev6m_summary.csv',index=False)
    (pd.concat(ts,ignore_index=True) if ts else pd.DataFrame()).to_csv(f'{OUT}/pc_br_oos_prev6m_trades.csv',index=False)
    print('\nPREVIOUS 6M OOS\n',summary.to_string(index=False))

if __name__=='__main__': main()
