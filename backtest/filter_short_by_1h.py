from __future__ import annotations

import argparse
import pandas as pd

from backtest.download_binance import download_klines


def summarize(df: pd.DataFrame) -> dict:
    closed = df[df['status'] == 'CLOSED'].copy()
    wins = closed[closed['highest_tp'] >= 1]
    losses = closed[closed['highest_tp'] == 0]
    return {
        'setups': int(len(df)),
        'closed': int(len(closed)),
        'wins': int(len(wins)),
        'losses': int(len(losses)),
        'win_rate_pct': round(100 * len(wins) / len(closed), 2) if len(closed) else 0.0,
        'total_r': round(float(closed['realized_r'].sum()), 4) if len(closed) else 0.0,
        'avg_r': round(float(closed['realized_r'].mean()), 4) if len(closed) else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description='Fast post-filter: keep all LONG, require bearish closed 1H for SHORT')
    p.add_argument('--baseline', default='backtest/data/backtest_btc_ob_filtered_1_10.csv')
    p.add_argument('--start', default='2026-03-01T00:00:00+00:00')
    p.add_argument('--end', default='2026-09-01T00:00:00+00:00')
    p.add_argument('--out', default='backtest_btc_short_1h_filter.csv')
    args = p.parse_args()

    base = pd.read_csv(args.baseline)
    base['signal_time'] = pd.to_datetime(base['signal_time'], utc=True, errors='coerce')

    h1 = download_klines('BTCUSDT', '1h', args.start, args.end)
    h1['time'] = pd.to_datetime(h1['time'], utc=True)
    h1 = h1.sort_values('time').reset_index(drop=True)
    h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
    h1['prev_close'] = h1['close'].shift(1)
    h1['dir'] = 'NEUTRAL'
    h1.loc[(h1['close'] > h1['prev_close']) & (h1['close'] > h1['ema20']), 'dir'] = 'LONG'
    h1.loc[(h1['close'] < h1['prev_close']) & (h1['close'] < h1['ema20']), 'dir'] = 'SHORT'

    rows = []
    for _, trade in base.iterrows():
        side = str(trade['side'])
        out = trade.to_dict()

        # LONG is deliberately untouched.
        if side == 'LONG':
            out['h1_direction'] = 'NOT_FILTERED'
            rows.append(out)
            continue

        # Only SHORT requires a bearish fully closed 1H candle.
        t = trade['signal_time']
        if pd.isna(t):
            continue
        closed = h1[(h1['time'] + pd.Timedelta(hours=1)) <= t]
        if closed.empty:
            continue
        d = str(closed.iloc[-1]['dir'])
        if d == 'SHORT':
            out['h1_direction'] = d
            rows.append(out)

    filtered = pd.DataFrame(rows)
    filtered.to_csv(args.out, index=False)

    print('\n=== BASELINE: 15M LIMIT + OB 1.10 ===')
    print(summarize(base))
    print('\n=== HYBRID: ALL LONG + 1H-FILTERED SHORT ===')
    print(summarize(filtered))
    for side in ('LONG', 'SHORT'):
        print(side, summarize(filtered[filtered['side'] == side]))
    print('results:', args.out)


if __name__ == '__main__':
    main()
