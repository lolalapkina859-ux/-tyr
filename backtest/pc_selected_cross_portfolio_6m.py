from pathlib import Path
import pandas as pd, numpy as np
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START='2026-03-01'; END='2026-09-01'; TF='30m'
SYMBOLS=['ZECUSDT','USELESSUSDT','FETUSDT','HYPEUSDT','JTOUSDT','VETUSDT','XRPUSDT','ETHUSDT','UNIUSDT','INJUSDT','SEIUSDT','1000SHIBUSDT','DYDXUSDT']
START_BAL=100.0; LEVERAGE=20.0; NOTIONAL=60.0; IM=NOTIONAL/LEVERAGE; MMR=0.005

def load(sym):
 d=download_klines(sym,TF,START,END).reset_index(drop=True); d['time']=pd.to_datetime(d.time,utc=True); pc=purple_cloud(d)
 d['buy']=pc.pc_buy.astype(bool); d['sell']=pc.pc_sell.astype(bool); return d.set_index('time')

def main():
 data={}; failures=[]
 for s in SYMBOLS:
  try:
   d=load(s)
   if len(d): data[s]=d
  except Exception as e: failures.append((s,str(e))); print('SKIP',s,e)
 times=sorted(set().union(*[set(d.index) for d in data.values()]))
 wallet=START_BAL; pos={}; closed=[]; eqrows=[]; skipped=0; peak=START_BAL; maxdd=0; min_eq=START_BAL; maxpos=0; maxgross=0; liquidated=False; liq_time=None
 for t in times:
  bars={s:d.loc[t] for s,d in data.items() if t in d.index}
  # Mark-to-market before decisions.
  def upnl(s,p,price): return NOTIONAL*((price/p['entry']-1) if p['side']=='LONG' else (p['entry']/price-1))
  unreal=sum(upnl(s,p,float(bars[s].close)) for s,p in pos.items() if s in bars)
  equity=wallet+unreal; maint=len(pos)*NOTIONAL*MMR; used=len(pos)*IM; available=equity-used
  if equity <= maint:
   liquidated=True; liq_time=t
   for s,p in list(pos.items()):
    if s in bars:
     pnl=upnl(s,p,float(bars[s].close)); wallet+=pnl; closed.append({'symbol':s,'side':p['side'],'entry_time':p['time'],'exit_time':t,'entry':p['entry'],'exit':float(bars[s].close),'pnl_usdt':pnl,'return_pct':pnl/NOTIONAL*100,'exit_reason':'LIQUIDATION'})
   pos.clear(); eqrows.append({'time':t,'wallet':wallet,'unrealized':0,'equity':wallet,'positions':0,'gross_notional':0,'used_initial_margin':0,'maintenance_margin':0,'available_margin':wallet}); break
  # Exit/reverse on opposite NORMAL signal.
  reverse=[]
  for s,p in list(pos.items()):
   if s not in bars: continue
   b=bars[s]; opp=(p['side']=='LONG' and bool(b.sell)) or (p['side']=='SHORT' and bool(b.buy))
   if opp:
    ex=float(b.close); pnl=upnl(s,p,ex); wallet+=pnl; closed.append({'symbol':s,'side':p['side'],'entry_time':p['time'],'exit_time':t,'entry':p['entry'],'exit':ex,'pnl_usdt':pnl,'return_pct':pnl/NOTIONAL*100,'exit_reason':'OPPOSITE'}); del pos[s]; reverse.append(s)
  # Open signals, including reversal, only if cross available margin can support new $3 IM.
  for s,b in bars.items():
   if s in pos: continue
   side='LONG' if bool(b.buy) else ('SHORT' if bool(b.sell) else None)
   if not side: continue
   unreal=sum(upnl(k,p,float(bars[k].close)) for k,p in pos.items() if k in bars); equity=wallet+unreal; available=equity-len(pos)*IM
   if available >= IM:
    pos[s]={'side':side,'entry':float(b.close),'time':t}
   else: skipped+=1
  unreal=sum(upnl(s,p,float(bars[s].close)) for s,p in pos.items() if s in bars); equity=wallet+unreal; used=len(pos)*IM; maint=len(pos)*NOTIONAL*MMR; available=equity-used
  peak=max(peak,equity); dd=(equity/peak-1)*100 if peak else 0; maxdd=min(maxdd,dd); min_eq=min(min_eq,equity); maxpos=max(maxpos,len(pos)); maxgross=max(maxgross,len(pos)*NOTIONAL)
  eqrows.append({'time':t,'wallet':wallet,'unrealized':unreal,'equity':equity,'positions':len(pos),'gross_notional':len(pos)*NOTIONAL,'used_initial_margin':used,'maintenance_margin':maint,'available_margin':available,'drawdown_pct':dd})
 # Close remaining at end for final realized result.
 if not liquidated and times:
  t=times[-1]
  for s,p in list(pos.items()):
   if s in data and t in data[s].index:
    ex=float(data[s].loc[t].close); pnl=upnl(s,p,ex); wallet+=pnl; closed.append({'symbol':s,'side':p['side'],'entry_time':p['time'],'exit_time':t,'entry':p['entry'],'exit':ex,'pnl_usdt':pnl,'return_pct':pnl/NOTIONAL*100,'exit_reason':'END'})
 c=pd.DataFrame(closed); e=pd.DataFrame(eqrows)
 summary=pd.DataFrame([{'start_balance':START_BAL,'final_balance':wallet,'net_profit_usdt':wallet-START_BAL,'return_pct':(wallet/START_BAL-1)*100,'closed_trades':len(c),'wins':int((c.pnl_usdt>0).sum()) if len(c) else 0,'losses':int((c.pnl_usdt<0).sum()) if len(c) else 0,'winrate':float((c.pnl_usdt>0).mean()*100) if len(c) else 0,'max_floating_dd_pct':maxdd,'min_equity':min_eq,'max_simultaneous_positions':maxpos,'max_gross_notional':maxgross,'skipped_entries_margin':skipped,'liquidated':liquidated,'liquidation_time':liq_time,'leverage':LEVERAGE,'position_notional':NOTIONAL,'initial_margin_per_position':IM,'mmr_assumption':MMR,'symbols_loaded':len(data),'symbols_requested':len(SYMBOLS)}])
 bysym=c.groupby('symbol').agg(trades=('pnl_usdt','size'),pnl_usdt=('pnl_usdt','sum'),avg_pnl=('pnl_usdt','mean'),winrate=('pnl_usdt',lambda x:(x>0).mean()*100)).reset_index().sort_values('pnl_usdt',ascending=False) if len(c) else pd.DataFrame()
 out=Path('backtest/data'); out.mkdir(parents=True,exist_ok=True); summary.to_csv(out/'pc_selected_cross_6m_summary.csv',index=False); c.to_csv(out/'pc_selected_cross_6m_trades.csv',index=False); e.to_csv(out/'pc_selected_cross_6m_equity.csv',index=False); bysym.to_csv(out/'pc_selected_cross_6m_by_symbol.csv',index=False); pd.DataFrame(failures,columns=['symbol','error']).to_csv(out/'pc_selected_cross_6m_failures.csv',index=False)
 print(summary.to_string(index=False)); print('\nBY SYMBOL\n',bysym.to_string(index=False)); print('\nFAILURES',failures)
if __name__=='__main__': main()
