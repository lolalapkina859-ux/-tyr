from __future__ import annotations
import os
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
import backtest.purple_cloud_entry_test as pcmod

BASE15 = list(eng.BASE14) + ['FLOWUSDT']
START_BALANCE = float(os.getenv('PC_TEST_START_BALANCE', '145.58352658'))
NOTIONAL = 60.0
ALPHAS = [1.1, 1.5]
OUT = eng.OUT

def main():
    data = {s: eng.load(s) for s in BASE15}
    rows, trades = [], []
    old_balance, old_notional, old_alpha = eng.START_BALANCE, eng.NOTIONAL, pcmod.ALPHA
    # eng.purple_cloud was imported into the engine module; its function reads globals
    # from pcmod, so changing pcmod.ALPHA changes the exact PC calculation.
    try:
        eng.START_BALANCE = START_BALANCE
        eng.NOTIONAL = NOTIONAL
        for alpha in ALPHAS:
            pcmod.ALPHA = alpha
            name = f'BASE15_ALPHA_{alpha:.1f}'
            r, t, _ = eng.run(data, name, BASE15)
            r['alpha'] = alpha
            r['start_balance'] = START_BALANCE
            r['position_notional_usdt'] = NOTIONAL
            rows.append(r)
            t['alpha'] = alpha
            trades.append(t)
            print(r)
    finally:
        eng.START_BALANCE = old_balance
        eng.NOTIONAL = old_notional
        pcmod.ALPHA = old_alpha

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'pc_base15_alpha_11_vs_15_summary.csv', index=False)
    pd.concat(trades, ignore_index=True).to_csv(OUT / 'pc_base15_alpha_11_vs_15_trades.csv', index=False)
    print('\nALPHA COMPARISON\n', summary.to_string(index=False))

if __name__ == '__main__':
    main()
