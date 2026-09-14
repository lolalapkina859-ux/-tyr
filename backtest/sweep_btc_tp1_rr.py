from __future__ import annotations

import argparse
import pandas as pd


def stats(df: pd.DataFrame) -> dict:
    closed = df[df['status'] == 'CLOSED'].copy()
    wins = closed[closed['highest_tp'] >= 1]
    losses = closed[closed['highest_tp'] == 0]
    return {
        'setups': len(df),
        'closed': len(closed),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_pct': 100 * len(wins) / len(closed) if len(closed) else 0.0,
        'total_r': float(closed['realized_r'].sum()) if len(closed) else 0.0,
        'avg_r': float(closed['realized_r'].mean()) if len(closed) else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', default='backtest/data/backtest_btc_ob_filtered_1_10.csv')
    p.add_argument('--out', default='backtest_btc_tp1_rr_sweep.csv')
    args = p.parse_args()

    df = pd.read_csv(args.baseline)
    risk = (df['entry'] - df['sl']).abs()
    df['tp1_r'] = (df['tp1'] - df['entry']).abs() / risk.replace(0, pd.NA)

    rows = []
    for threshold in (0.70, 0.80, 0.90, 1.00, 1.10, 1.20):
        f = df[df['tp1_r'] >= threshold].copy()
        s = stats(f)
        s['min_tp1_r'] = threshold
        for side in ('LONG', 'SHORT'):
            ss = stats(f[f['side'] == side])
            s[f'{side.lower()}_closed'] = ss['closed']
            s[f'{side.lower()}_wr'] = ss['win_rate_pct']
            s[f'{side.lower()}_total_r'] = ss['total_r']
            s[f'{side.lower()}_avg_r'] = ss['avg_r']
        rows.append(s)

    result = pd.DataFrame(rows)[[
        'min_tp1_r','setups','closed','wins','losses','win_rate_pct','total_r','avg_r',
        'long_closed','long_wr','long_total_r','long_avg_r',
        'short_closed','short_wr','short_total_r','short_avg_r'
    ]]
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print('results:', args.out)


if __name__ == '__main__':
    main()
