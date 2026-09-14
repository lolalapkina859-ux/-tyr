from __future__ import annotations

import argparse
import pandas as pd
from backtest.download_binance import download_klines


def summarize(df: pd.DataFrame) -> dict:
    closed = df[df['status'] == 'CLOSED'].copy()
    wins = closed[closed['highest_tp'] >= 1]
    losses = closed[closed['highest_tp'] == 0]
    return {
        'setups': int(len(df)), 'closed': int(len(closed)),
        'wins': int(len(wins)), 'losses': int(len(losses)),
        'win_rate_pct': round(100 * len(wins) / len(closed), 2) if len(closed) else 0.0,
        'total_r': round(float(closed['realized_r'].sum()), 4) if len(closed) else 0.0,
        'avg_r': round(float(closed['realized_r'].mean()), 4) if len(closed) else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    base = pd.read_csv(a.baseline)
    base['signal_time'] = pd.to_datetime(base['signal_time'], utc=True, errors='coerce')
    h1 = download_klines(a.symbol.upper(), '1h', a.start, a.end)
    h1['time'] = pd.to_datetime(h1['time'], utc=True)
    h1 = h1.sort_values('time').reset_index(drop=True)
    h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
    h1['prev_close'] = h1['close'].shift(1)
    h1['dir'] = 'NEUTRAL'
    h1.loc[(h1['close'] > h1['prev_close']) & (h1['close'] > h1['ema20']), 'dir'] = 'LONG'
    h1.loc[(h1['close'] < h1['prev_close']) & (h1['close'] < h1['ema20']), 'dir'] = 'SHORT'

    rows = []
    for _, trade in base.iterrows():
        out = trade.to_dict()
        if str(trade['side']) == 'LONG':
            out['h1_direction'] = 'NOT_FILTERED'
            rows.append(out)
            continue
        t = trade['signal_time']
        if pd.isna(t):
            continue
        closed = h1[(h1['time'] + pd.Timedelta(hours=1)) <= t]
        if not closed.empty and str(closed.iloc[-1]['dir']) == 'SHORT':
            out['h1_direction'] = 'SHORT'
            rows.append(out)

    filtered = pd.DataFrame(rows)
    filtered.to_csv(a.out, index=False)
    print('\n=== OB 1.10 BASELINE ===')
    print(summarize(base))
    print('\n=== ALL LONG + 1H-FILTERED SHORT ===')
    print(summarize(filtered))
    for side in ('LONG', 'SHORT'):
        print(side, summarize(filtered[filtered['side'] == side]))

if __name__ == '__main__':
    main()
