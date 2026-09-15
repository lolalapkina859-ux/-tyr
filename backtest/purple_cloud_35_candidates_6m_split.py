from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

TF='30m'
PERIODS=[('MAR_JUN','2026-03-01','2026-06-01'),('JUN_SEP','2026-06-01','2026-09-01')]
# 35 fresh Binance USD-M candidates, excluding previously screened baskets.
SYMBOLS=['XLMUSDT','VETUSDT','ALGOUSDT','SANDUSDT','MANAUSDT','GALAUSDT','AXSUSDT','RUNEUSDT','CRVUSDT','LDOUSDT','STXUSDT','IMXUSDT','GRTUSDT','JUPUSDT','BONKUSDT','FLOKIUSDT','NOTUSDT','JTOUSDT','PYTHUSDT','WLDUSDT','ARUSDT','MKRUSDT','COMPUSDT','SNXUSDT','DYDXUSDT','OMUSDT','ONDOUSDT','SAGAUSDT','STRKUSDT','AEVOUSDT','BLURUSDT','MEMEUSDT','1000SHIBUSDT','1000SATSUSDT','ORDIUSDT']

def run_symbol(sym,start,end):
 d=download_klines(sym,TF,start,end).reset_index(drop=True)
 if len(d)<100: raise ValueError(f'not enough candles: {len(d)}')
 d['time']=pd.to_datetime(d['time'],utc=True); pc=purple_cloud(d); rows=[]; pos=None
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

def stat(x,s,p):
 w=x[x.return_pct>0]; l=x[x.return_pct<0]
 return {'symbol':s,'period':p,'trades':len(x),'wins':len(w),'losses':len(l),'winrate':round(100*len(w)/len(x),2) if len(x) else 0,'sum_return_pct':round(x.return_pct.sum(),4),'avg_signal_pct':round(x.return_pct.mean(),4) if len(x) else 0,'avg_winner_pct':round(w.return_pct.mean(),4) if len(w) else 0,'avg_loser_pct':round(l.return_pct.mean(),4) if len(l) else 0,'best_pct':round(x.return_pct.max(),4) if len(x) else 0,'worst_pct':round(x.return_pct.min(),4) if len(x) else 0,'avg_hold_hours':round(x.bars.mean()/2,2) if len(x) else 0}

def main():
 details=[]; stats=[]; skipped=[]
 for s in SYMBOLS:
  for name,start,end in PERIODS:
   try:
    x=run_symbol(s,start,end); x['period']=name; details.append(x); stats.append(stat(x,s,name)); print(stats[-1])
   except Exception as e:
    skipped.append({'symbol':s,'period':name,'error':str(e)}); print('SKIP',s,name,e)
 sm=pd.DataFrame(stats)
 if len(sm):
  piv=sm.pivot(index='symbol',columns='period',values='sum_return_pct').reset_index()
  for c in ['MAR_JUN','JUN_SEP']:
   if c not in piv: piv[c]=float('nan')
  piv['both_positive']=(piv.MAR_JUN>0)&(piv.JUN_SEP>0)
  piv['six_month_sum_pct']=piv[['MAR_JUN','JUN_SEP']].sum(axis=1,min_count=2)
  piv['worst_period_pct']=piv[['MAR_JUN','JUN_SEP']].min(axis=1)
  piv=piv.sort_values(['both_positive','worst_period_pct','six_month_sum_pct'],ascending=False)
 else: piv=pd.DataFrame()
 out=Path('backtest/data'); out.mkdir(parents=True,exist_ok=True)
 sm.to_csv(out/'pc35_6m_split_summary.csv',index=False); piv.to_csv(out/'pc35_6m_split_ranking.csv',index=False)
 pd.DataFrame(skipped).to_csv(out/'pc35_6m_split_skipped.csv',index=False)
 if details: pd.concat(details,ignore_index=True).to_csv(out/'pc35_6m_split_detail.csv',index=False)
 print('\nRANKING\n',piv.to_string(index=False)); print('\nSKIPPED\n',pd.DataFrame(skipped).to_string(index=False))
if __name__=='__main__': main()
