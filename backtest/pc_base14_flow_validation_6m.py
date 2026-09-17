from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud
from backtest.pc_base14_candidate_additions_6m import BASE14, load, run

OUT=Path("backtest/data"); OUT.mkdir(parents=True,exist_ok=True)
MONTHS=[("MAR","2026-03-01","2026-04-01"),("APR","2026-04-01","2026-05-01"),("MAY","2026-05-01","2026-06-01"),("JUN","2026-06-01","2026-07-01"),("JUL","2026-07-01","2026-08-01"),("AUG","2026-08-01","2026-09-01")]
FLOW_SET=BASE14+["FLOWUSDT"]
COMBOS=[("BASE14",BASE14),("BASE14+FLOW",FLOW_SET),("BASE14+FLOW+ID",FLOW_SET+["IDUSDT"]),("BASE14+FLOW+IOTA",FLOW_SET+["IOTAUSDT"]),("BASE14+FLOW+ID+IOTA",FLOW_SET+["IDUSDT","IOTAUSDT"])]

def load_period(s,start,end):
 d=download_klines(s,"30m",start,end).reset_index(drop=True); d["time"]=pd.to_datetime(d["time"],utc=True)
 return purple_cloud(d)[["time","high","low","close","pc_buy","pc_sell"]]

def run_period(data,name,symbols,start_balance=120.):
 # Reuse exact portfolio engine, temporarily replacing its module start balance.
 import backtest.pc_base14_candidate_additions_6m as eng
 old=eng.START_BALANCE; eng.START_BALANCE=start_balance
 try: return eng.run(data,name,symbols)
 finally: eng.START_BALANCE=old

def main():
 # Full 6m combination comparison.
 needed=sorted(set(BASE14+["FLOWUSDT","IDUSDT","IOTAUSDT"])); data={s:load(s) for s in needed}
 combo_rows=[]; combo_trades=[]
 for name,symbols in COMBOS:
  r,t,_=run(data,name,symbols); combo_rows.append(r); combo_trades.append(t); print("COMBO",r)
 pd.DataFrame(combo_rows).sort_values("final_balance",ascending=False).to_csv(OUT/"pc_base14_flow_combo_summary.csv",index=False)
 pd.concat(combo_trades,ignore_index=True).to_csv(OUT/"pc_base14_flow_combo_trades.csv",index=False)

 # Independent fresh-$120 monthly BASE14 vs BASE14+FLOW to test stability.
 monthly=[]
 for mon,start,end in MONTHS:
  md={s:load_period(s,start,end) for s in FLOW_SET}
  for name,symbols in [("BASE14",BASE14),("BASE14+FLOW",FLOW_SET)]:
   r,_,_=run_period(md,f"{mon}_{name}",symbols); r["month"]=mon; r["portfolio"]=name; monthly.append(r); print("MONTH",r)
 m=pd.DataFrame(monthly)
 m.to_csv(OUT/"pc_base14_flow_monthly_summary.csv",index=False)
 piv=m.pivot(index="month",columns="portfolio",values="net_profit_usdt").reset_index()
 if "BASE14" in piv and "BASE14+FLOW" in piv: piv["flow_increment_usdt"]=piv["BASE14+FLOW"]-piv["BASE14"]
 piv.to_csv(OUT/"pc_base14_flow_monthly_increment.csv",index=False)
 print("\nMONTHLY\n",m[["month","portfolio","final_balance","net_profit_usdt","max_floating_dd_pct","min_equity","skipped_entries_margin","liquidated"]].to_string(index=False))
 print("\nCOMBOS\n",pd.DataFrame(combo_rows).to_string(index=False))

if __name__=="__main__": main()
