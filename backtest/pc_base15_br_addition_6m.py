from __future__ import annotations
import os
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

BASE15 = list(eng.BASE14) + ['FLOWUSDT']
BASE16 = BASE15 + ['BRUSDT']
START_BALANCE = float(os.getenv('PC_TEST_START_BALANCE', '145.58352658'))
NOTIONAL = 60.0
OUT = eng.OUT


def main():
    symbols = list(dict.fromkeys(BASE16))
    data = {s: eng.load(s) for s in symbols}
    rows, trades = [], []
    old_balance, old_notional = eng.START_BALANCE, eng.NOTIONAL
    try:
        eng.START_BALANCE = START_BALANCE
        eng.NOTIONAL = NOTIONAL
        for name, universe in [('BASE15', BASE15), ('BASE15_PLUS_BR', BASE16)]:
            r, t, _ = eng.run(data, name, universe)
            r['start_balance'] = START_BALANCE
            r['position_notional_usdt'] = NOTIONAL
            rows.append(r); trades.append(t)
            print(r)
    finally:
        eng.START_BALANCE = old_balance
        eng.NOTIONAL = old_notional

    summary = pd.DataFrame(rows)
    base = summary.loc[summary.variant == 'BASE15'].iloc[0]
    br = summary.loc[summary.variant == 'BASE15_PLUS_BR'].iloc[0]
    summary['delta_final_vs_base15'] = summary.final_balance - float(base.final_balance)
    summary['delta_net_vs_base15'] = summary.net_profit - float(base.net_profit)
    summary['delta_dd_pp_vs_base15'] = summary.max_floating_dd_pct - float(base.max_floating_dd_pct)
    summary.to_csv(OUT / 'pc_base15_br_addition_summary.csv', index=False)
    pd.concat(trades, ignore_index=True).to_csv(OUT / 'pc_base15_br_addition_trades.csv', index=False)
    print('\nFINAL COMPARISON\n', summary.to_string(index=False))
    print(f"\nBR impact: final {float(base.final_balance):.2f} -> {float(br.final_balance):.2f}; net delta {float(br.net_profit-base.net_profit):+.2f}; DD delta {float(br.max_floating_dd_pct-base.max_floating_dd_pct):+.2f} pp")


if __name__ == '__main__':
    main()
