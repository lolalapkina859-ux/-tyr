from __future__ import annotations
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

START_BALANCE=200.0
SIZES=[60.0,80.0,100.0]
OUT=eng.OUT

def main():
    data={s:eng.load(s) for s in eng.BASE14}
    rows=[]; trades=[]
    old_balance,old_notional=eng.START_BALANCE,eng.NOTIONAL
    try:
        for size in SIZES:
            eng.START_BALANCE=START_BALANCE
            eng.NOTIONAL=size
            name=f"BASE14_BAL200_NOTIONAL{int(size)}"
            r,t,_=eng.run(data,name,eng.BASE14)
            r["position_notional_usdt"]=size
            r["notional_to_start_balance_pct"]=size/START_BALANCE*100
            rows.append(r); trades.append(t)
            print(r)
    finally:
        eng.START_BALANCE=old_balance
        eng.NOTIONAL=old_notional
    summary=pd.DataFrame(rows).sort_values("final_balance",ascending=False)
    summary.to_csv(OUT/"pc_base14_balance200_size_sweep_summary.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(OUT/"pc_base14_balance200_size_sweep_trades.csv",index=False)
    print("\nFINAL COMPARISON\n",summary.to_string(index=False))

if __name__=="__main__": main()
