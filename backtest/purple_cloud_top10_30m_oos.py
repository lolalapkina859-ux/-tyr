from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

# Out-of-sample period BEFORE the selection window (Jun-Sep 2026).
START='2026-03-01'; END='2026-06-01'; TF='30m'
SYMBOLS=['ZECUSDT','ADAUSDT','XRPUSDT','SOLUSDT','ETHUSDT','DOTUSDT','BTCUSDT','BNBUSDT','UNIUSDT','DOGEUSDT']

def run_symbol(sym):
 d=download_klines(sym,TF,START,END).reset_index(drop=True); d['time']=pd.to_datetime(d['time'],utc=True); pc=purple_cloud(d); rows=[]; pos=None
 for i,b in d.iterrows():
  buy=bool(pc.iloc[i]['pc_buy']); sell=bool(pc.iloc[i]['pc_sell'])
  if pos and ((pos['side']=='LONG' and sell) or (pos['side']=='SHORT' and buy)):
   ex=float(b.close); ret=(ex/pos['entry']-1) if pos['side']=='LONG' else (pos['entry']/ex-1)
   rows.append({'symbol':sym,'side':pos['side'],'entry_time':pos['time'],'exit_time':b.time,'entry':pos['entry'],'exit':ex,'return_pct':ret*100,'bars':i-pos['i']}); pos=None
  if pos is None and (buy or sell): pos={'side':'LONG' if buy else 'SHORT','entry':float(b.close),'time':b.time,'i':i}
 if pos:
  b=d.iloc[-1]; ex=float(b.close); ret=(ex/pos['entry']-1) if pos['side']=='LONG' else (pos['entry']/ex-1)
  rows.append({'symbol':sym,'side':pos['side'],'entry_time':pos['time'],'exit_time':b.time,'entry':pos['entry'],'exit':ex,'return_pct':ret*100,'bars':len(d)-1-pos['i']})
 return pd.DataFrame(rows)

def stats(x,s):
 w=x[x.return_pct>0]; l=x[x.return_pct<0]
 return {'symbol':s,'trades':len(x),'wins':len(w),'losses':len(l),'winrate':round(len(w)/len(x)*100,2),'sum_trade_returns_pct':round(x.return_pct.sum(),4),'avg_return_per_signal_pct':round(x.return_pct.mean(),4),'avg_winner_pct':round(w.return_pct.mean(),4) if len(w) else 0,'avg_loser_pct':round(l.return_pct.mean(),4) if len(l) else 0,'best_pct':round(x.return_pct.max(),4),'worst_pct':round(x.return_pct.min(),4),'avg_hold_hours':round(x.bars.mean()/2,2)}

def main():
 xs=[]; ss=[]
 for s in SYMBOLS:
  x=run_symbol(s); xs.append(x); ss.append(stats(x,s)); print(ss[-1])
 allx=pd.concat(xs,ignore_index=True); ss.append(stats(allx,'ALL')); out=Path('backtest/data'); out.mkdir(parents=True,exist_ok=True)
 allx.to_csv(out/'purple_cloud_top10_30m_oos_detail.csv',index=False); pd.DataFrame(ss).to_csv(out/'purple_cloud_top10_30m_oos_summary.csv',index=False); print(pd.DataFrame(ss).to_string(index=False))
if __name__=='__main__': main()
