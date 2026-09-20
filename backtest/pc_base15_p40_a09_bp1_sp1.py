from __future__ import annotations
import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng
from backtest.purple_cloud_entry_test import atr, vwma, rma

BASE15 = list(eng.BASE14) + ["FLOWUSDT"]
START_BALANCE = 145.58352658
NOTIONAL = 60.0

PERIOD = 40
ALPHA = 0.9
BPT = 1.0
SPT = 1.0

def purple_cloud_test(df):
    d = df.copy().reset_index(drop=True)
    n1 = int(np.ceil(PERIOD / 4))
    n2 = int(np.ceil(PERIOD / 2))

    # No separate ATR Bands / ATR multiplier layer.
    # Keep only Purple Cloud's native alpha envelope.
    x2 = atr(d, PERIOD) * ALPHA
    xh = d.close + x2
    xl = d.close - x2

    hl2 = (d.high + d.low) / 2
    a1 = vwma(hl2 * d.volume, d.volume, n1) / vwma(d.volume, d.volume, n1)
    a2 = vwma(hl2 * d.volume, d.volume, n2) / vwma(d.volume, d.volume, n2)
    a3 = 2 * a1 - a2
    a4 = vwma(a3, d.volume, PERIOD)
    b1 = rma(d.close, PERIOD)
    a5 = 2 * a4 * b1 / (a4 + b1)

    buy = (a5 <= xl) & (d.close > b1 * (1 + BPT * 0.01))
    sell = (a5 >= xh) & (d.close < b1 * (1 - SPT * 0.01))

    xs = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        xs[i] = 1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])

    changed = pd.Series(xs).ne(pd.Series(xs).shift(1))
    d["pc_buy"] = buy & changed
    d["pc_sell"] = sell & changed
    return d

def load_raw(symbol):
    d = eng.download_klines(symbol, "30m", eng.START, eng.END).reset_index(drop=True)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d

def main():
    print("TEST: BASE15 30m | Period=40 Alpha=0.9 BPT=1 SPT=1 | no extra ATR bands")
    print("Range:", eng.START, "to", eng.END)
    print("Loading", len(BASE15), "symbols...")
    data = {}
    for s in BASE15:
        p = purple_cloud_test(load_raw(s))
        data[s] = p[["time","high","low","close","pc_buy","pc_sell"]]

    old_balance, old_notional = eng.START_BALANCE, eng.NOTIONAL
    try:
        eng.START_BALANCE = START_BALANCE
        eng.NOTIONAL = NOTIONAL
        result, trades, by_symbol = eng.run(data, "P40_A09_BP1_SP1", BASE15)
    finally:
        eng.START_BALANCE = old_balance
        eng.NOTIONAL = old_notional

    print("\nRESULT")
    print(result)
    print("\nBY SYMBOL")
    print(by_symbol.to_string(index=False))
    trades.to_csv("backtest/data/pc_base15_p40_a09_bp1_sp1_trades.csv", index=False)
    by_symbol.to_csv("backtest/data/pc_base15_p40_a09_bp1_sp1_by_symbol.csv", index=False)

if __name__ == "__main__":
    main()
