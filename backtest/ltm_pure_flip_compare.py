import numpy as np
import pandas as pd
import backtest.pc_base14_candidate_additions_6m as eng

SYMBOLS = list(eng.BASE14) + ["FLOWUSDT"]
PERIODS = [
    ("MAR_SEP_2026", "2026-03-01", "2026-09-01"),
    ("OOS_SEP_MAR", "2025-09-01", "2026-03-01"),
]
START_BALANCE = 145.58352658
NOTIONAL = 60.0
ATR_LEN = 13

def rma(s, n):
    return s.ewm(alpha=1/n, adjust=False).mean()

def atr(d, n=13):
    prev = d.close.shift(1)
    tr = pd.concat([(d.high-d.low), (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1)
    return rma(tr, n)

def ltm_flip(d, flip_band):
    d = d.reset_index(drop=True).copy()
    a = atr(d, ATR_LEN)
    mults = [4.0, 5.0, 6.0, 7.0]  # Balanced: base 4.0, step .25
    trails = [np.full(len(d), np.nan) for _ in range(4)]
    trend = 1
    buy = np.zeros(len(d), dtype=bool)
    sell = np.zeros(len(d), dtype=bool)

    for i in range(len(d)):
        if not np.isfinite(a.iloc[i]):
            continue
        src = d.close.iloc[i]
        upper = [src - a.iloc[i]*m for m in mults]
        lower = [src + a.iloc[i]*m for m in mults]

        if i == 0 or not np.isfinite(trails[flip_band-1][i-1]):
            for k in range(4):
                trails[k][i] = upper[k] if trend == 1 else lower[k]
            continue

        flip_prev = trails[flip_band-1][i-1]

        if trend == 1 and src < flip_prev:
            trend = -1
            for k in range(4):
                trails[k][i] = lower[k]
            if i >= 60:
                sell[i] = True
        elif trend == -1 and src > flip_prev:
            trend = 1
            for k in range(4):
                trails[k][i] = upper[k]
            if i >= 60:
                buy[i] = True
        elif trend == 1:
            for k in range(4):
                trails[k][i] = max(upper[k], trails[k][i-1])
        else:
            for k in range(4):
                trails[k][i] = min(lower[k], trails[k][i-1])

    d["pc_buy"] = buy
    d["pc_sell"] = sell
    return d

def main():
    old_balance, old_notional = eng.START_BALANCE, eng.NOTIONAL
    eng.START_BALANCE, eng.NOTIONAL = START_BALANCE, NOTIONAL
    rows, symbol_rows = [], []
    try:
        for period, start, end in PERIODS:
            raw = {}
            for symbol in SYMBOLS:
                d = eng.download_klines(symbol, "30m", start, end).reset_index(drop=True)
                d["time"] = pd.to_datetime(d["time"], utc=True)
                raw[symbol] = d

            for band in (2, 3, 4):
                data = {}
                for symbol, d in raw.items():
                    x = ltm_flip(d, band)
                    data[symbol] = x[["time","high","low","close","pc_buy","pc_sell"]]

                mode = f"LTM_BALANCED_BAND{band}_FLIP"
                summary, trades, by_symbol = eng.run(data, f"{period}_{mode}", SYMBOLS)
                summary["period"] = period
                summary["mode"] = mode
                rows.append(summary)
                if len(by_symbol):
                    symbol_rows.append(by_symbol.assign(period=period, mode=mode))

        pd.DataFrame(rows).to_csv("backtest/data/ltm_pure_flip_compare_summary.csv", index=False)
        if symbol_rows:
            pd.concat(symbol_rows, ignore_index=True).to_csv("backtest/data/ltm_pure_flip_compare_by_symbol.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    finally:
        eng.START_BALANCE, eng.NOTIONAL = old_balance, old_notional

if __name__ == "__main__":
    main()
