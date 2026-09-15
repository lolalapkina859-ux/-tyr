from pathlib import Path
import pandas as pd

START_BALANCE=100.0
MARGIN_USD=3.0
LEVERAGE=20.0
NOTIONAL_USD=MARGIN_USD*LEVERAGE
INPUT=Path('backtest/data/purple_cloud_10coins_30m_no_sl_detail.csv')
OUT=Path('backtest/data')

def main():
 x=pd.read_csv(INPUT)
 x['entry_time']=pd.to_datetime(x['entry_time'],utc=True)
 x['exit_time']=pd.to_datetime(x['exit_time'],utc=True)
 x['notional_usd']=NOTIONAL_USD
 x['initial_margin_usd']=MARGIN_USD
 x['pnl_usd']=NOTIONAL_USD*x['return_pct']/100.0
 # Cross account: all realized PnL goes to one common wallet. No isolated $3 loss cap.
 events=x.sort_values(['exit_time','symbol']).copy()
 bal=START_BALANCE; peak=bal; maxdd=0.0; balances=[]
 for _,r in events.iterrows():
  bal+=r.pnl_usd
  peak=max(peak,bal)
  dd=(bal-peak)/peak*100 if peak else 0
  maxdd=min(maxdd,dd)
  balances.append(bal)
 events['cross_balance_after_exit']=balances
 profit=bal-START_BALANCE
 winners=events[events.pnl_usd>0]; losers=events[events.pnl_usd<0]
 summary=pd.DataFrame([{
  'start_balance_usd':START_BALANCE,'cross_margin':True,'leverage':LEVERAGE,
  'initial_margin_per_position_usd':MARGIN_USD,'notional_per_position_usd':NOTIONAL_USD,
  'trades':len(events),'final_balance_usd':round(bal,4),'net_profit_usd':round(profit,4),
  'return_on_start_deposit_pct':round(profit/START_BALANCE*100,4),
  'avg_pnl_per_signal_usd':round(events.pnl_usd.mean(),5),
  'avg_winner_usd':round(winners.pnl_usd.mean(),5),'avg_loser_usd':round(losers.pnl_usd.mean(),5),
  'best_trade_usd':round(events.pnl_usd.max(),5),'worst_trade_usd':round(events.pnl_usd.min(),5),
  'max_closed_equity_drawdown_pct':round(maxdd,4)
 }])
 by=events.groupby('symbol').agg(trades=('pnl_usd','size'),pnl_usd=('pnl_usd','sum'),avg_pnl_usd=('pnl_usd','mean'),best_trade_usd=('pnl_usd','max'),worst_trade_usd=('pnl_usd','min')).reset_index()
 OUT.mkdir(parents=True,exist_ok=True)
 events.to_csv(OUT/'pc10_cross100_margin3_20x_equity.csv',index=False)
 by.to_csv(OUT/'pc10_cross100_margin3_20x_by_symbol.csv',index=False)
 summary.to_csv(OUT/'pc10_cross100_margin3_20x_summary.csv',index=False)
 print(summary.to_string(index=False)); print(by.to_string(index=False))
 print('NOTE: closed-equity model; exact intratrade cross liquidation/floating DD requires candle-by-candle portfolio simulation.')
if __name__=='__main__': main()
