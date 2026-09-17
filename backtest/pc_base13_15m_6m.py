from __future__ import annotations

# BASE13 verification on 15m using the exact validated unified portfolio engine.
# Only timeframe changes from the live/reference 30m test to 15m.
# $120 start balance, fixed $60 notional, 20x, no TP/SL,
# close/reverse only on opposite Purple Cloud signal.

import backtest.pc13_to_17_compare_6m as engine

engine.TF = "15m"
engine.START_BALANCE = 120.0
engine.PORTFOLIOS = {"BASE13_15M_BALANCE120": engine.BASE13}
engine.ALL_SYMBOLS = sorted(set(engine.BASE13))

if __name__ == "__main__":
    engine.main()
