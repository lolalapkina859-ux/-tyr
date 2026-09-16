from __future__ import annotations

# Exact BASE13 unified engine, but with the user's current $120 starting balance.
# Strategy is unchanged: 30m Purple Cloud, fixed $60 notional, 20x,
# no TP/SL; close/reverse only on the opposite Purple Cloud signal.

import backtest.pc13_to_17_compare_6m as engine

engine.START_BALANCE = 120.0
engine.PORTFOLIOS = {"BASE13_BALANCE120": engine.BASE13}
engine.ALL_SYMBOLS = sorted(set(engine.BASE13))

if __name__ == "__main__":
    engine.main()
