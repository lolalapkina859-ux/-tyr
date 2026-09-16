from __future__ import annotations

import pandas as pd
from backtest.pc13_to_17_compare_6m import BASE13, OUT, load_symbol, run_portfolio

MONTHS = [
    ("2026-03", "2026-03-01", "2026-04-01"),
    ("2026-04", "2026-04-01", "2026-05-01"),
    ("2026-05", "2026-05-01", "2026-06-01"),
    ("2026-06", "2026-06-01", "2026-07-01"),
    ("2026-07", "2026-07-01", "2026-08-01"),
    ("2026-08", "2026-08-01", "2026-09-01"),
]


def main():
    data, failures = {}, []
    for s in BASE13:
        try:
            data[s] = load_symbol(s)
            print(f"LOADED {s}: {len(data[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
            print("SKIP", s, e)

    missing = sorted(set(BASE13) - set(data))
    pd.DataFrame(failures).to_csv(OUT / "pc_base13_monthly_6m_failures.csv", index=False)
    if missing:
        raise RuntimeError(f"Missing required BASE13 symbols: {missing}")

    # First reproduce the exact full 6-month BASE13 result with the validated engine.
    full_summary, full_trades, full_equity = run_portfolio("BASE13_FULL_6M", BASE13, data)
    pd.DataFrame([full_summary]).to_csv(OUT / "pc_base13_monthly_6m_full_summary.csv", index=False)

    # Then run every calendar month independently: fresh $100 at each month boundary.
    monthly_summaries, monthly_trades = [], []
    for month, start, end in MONTHS:
        st = pd.Timestamp(start, tz="UTC")
        en = pd.Timestamp(end, tz="UTC")
        month_data = {}
        for s, d in data.items():
            md = d[(d.time >= st) & (d.time < en)].copy().reset_index(drop=True)
            if md.empty:
                raise RuntimeError(f"No data for {s} in {month}")
            month_data[s] = md

        summary, trades, _ = run_portfolio(month, BASE13, month_data)
        summary["month"] = month
        monthly_summaries.append(summary)
        if len(trades):
            trades["month"] = month
            monthly_trades.append(trades)
        print(pd.DataFrame([summary]).to_string(index=False))

    ms = pd.DataFrame(monthly_summaries)
    ms.to_csv(OUT / "pc_base13_monthly_6m_summary.csv", index=False)
    if monthly_trades:
        pd.concat(monthly_trades, ignore_index=True).to_csv(OUT / "pc_base13_monthly_6m_trades.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "pc_base13_monthly_6m_trades.csv", index=False)

    # Per-symbol attribution from the continuous 6-month BASE13 run.
    if len(full_trades):
        ps = full_trades.groupby("symbol").agg(
            trades=("pnl_usdt", "size"),
            pnl_usdt=("pnl_usdt", "sum"),
            avg_pnl=("pnl_usdt", "mean"),
            wins=("pnl_usdt", lambda x: int((x > 0).sum())),
        ).reset_index()
        ps["winrate"] = ps.wins / ps.trades * 100.0
        ps = ps.sort_values("pnl_usdt", ascending=False)
        ps.to_csv(OUT / "pc_base13_monthly_6m_per_symbol.csv", index=False)

    print("\nFULL 6M BASE13 VERIFICATION")
    print(pd.DataFrame([full_summary]).to_string(index=False))
    print("\nMONTHLY INDEPENDENT CHECK")
    print(ms[["month", "final_balance", "net_profit_usdt", "max_floating_dd_pct", "min_equity", "closed_trades", "winrate", "skipped_entries_margin", "liquidated"]].to_string(index=False))


if __name__ == "__main__":
    main()
