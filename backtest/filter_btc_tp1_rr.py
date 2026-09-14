from __future__ import annotations

import argparse
import pandas as pd


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
    p = argparse.ArgumentParser(description='Fast BTC post-filter: require structural TP1 minimum R:R')
    p.add_argument('--baseline', default='backtest/data/backtest_btc_ob_filtered_1_10.csv')
    p.add_argument('--min-tp1-r', type=float, default=1.0)
    p.add_argument('--out', default='backtest_btc_tp1_min_1r.csv')
    args = p.parse_args()

    df = pd.read_csv(args.baseline)
    risk = (df['entry'] - df['sl']).abs()
    reward = (df['tp1'] - df['entry']).abs()
    df['tp1_r'] = reward / risk.replace(0, pd.NA)

    filtered = df[df['tp1_r'] >= args.min_tp1_r].copy()
    filtered.to_csv(args.out, index=False)

    print('\n=== BASELINE ===')
    print(summarize(df))
    print(f'\n=== TP1 >= {args.min_tp1_r:.2f}R ===')
    print(summarize(filtered))
    print('removed_setups:', len(df) - len(filtered))
    for side in ('LONG', 'SHORT'):
        print(side, summarize(filtered[filtered['side'] == side]))
    print('results:', args.out)


if __name__ == '__main__':
    main()
