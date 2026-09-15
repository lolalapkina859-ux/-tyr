from pathlib import Path
import pandas as pd

START_BALANCE=100.0
POSITION_USD=3.0
INPUT=Path('backtest/data/purple_cloud_10coins_30m_no_sl_detail.csv')
OUT=Path('backtest/data')

def main():
 x=pd.read_csv(INPUT)
 x['entry_time']=pd.to_datetime(x['entry_time'],utc=True)
 x['exit_time']=pd.to_datetime(x['exit_time'],utc=True)
 # Fixed $3 notional per signal. PnL realized at exit. No fees/leverage.
 x['pnl_usd']=POSITION_USD*x['return_pct']/100.0
 events=x.sort_values(['exit_time','symbol']).copy()
 bal=START_BALANCE; peak=bal; maxdd=0.0; balances=[]
 for _,r in events.iterrows():
  bal+=r.pnl_usd; peak=max(peak,bal); dd=(bal-peak)/peak*100; maxdd=min(maxdd,dd); balances.append(bal)
 events['balance_after_exit']=balances
 final=bal; profit=final-START_BALANCE
 s=pd.DataFrame([{
  'start_balance_usd':START_BALANCE,'position_usd':POSITION_USD,'trades':len(events),
  'final_balance_usd':round(final,4),'net_profit_usd':round(profit,4),
  'return_on_deposit_pct':round(profit/START_BALANCE*100,4),
  'avg_pnl_per_signal_usd':round(events.pnl_usd.mean(),6),
  'avg_pnl_per_signal_pct_of_deposit':round(events.pnl_usd.mean()/START_BALANCE*100,6),
  'avg_winner_usd':round(events.loc[events.pnl_usd>0,'pnl_usd'].mean(),6),
  'avg_loser_usd':round(events.loc[events.pnl_usd<0,'pnl_usd'].mean(),6),
  'best_trade_usd':round(events.pnl_usd.max(),6),'worst_trade_usd':round(events.pnl_usd.min(),6),
  'max_closed_equity_drawdown_pct':round(maxdd,4)
 }])
 by=x.assign(pnl_usd=x.pnl_usd).groupby('symbol').agg(trades=('pnl_usd','size'),pnl_usd=('pnl_usd','sum'),avg_pnl_usd=('pnl_usd','mean')).reset_index()
 OUT.mkdir(parents=True,exist_ok=True); events.to_csv(OUT/'pc10_deposit100_pos3_equity.csv',index=False); by.to_csv(OUT/'pc10_deposit100_pos3_by_symbol.csv',index=False); s.to_csv(OUT/'pc10_deposit100_pos3_summary.csv',index=False)
 print(s.to_string(index=False)); print(by.to_string(index=False))
if __name__=='__main__': main()
