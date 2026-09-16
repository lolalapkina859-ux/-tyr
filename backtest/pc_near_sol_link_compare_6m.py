from __future__ import annotations

# Reuse the exact unified cross-margin engine from the validated BASE13 comparison.
from backtest.pc13_to_17_compare_6m import (
    BASE13, START_BALANCE, OUT, load_symbol, run_portfolio
)
import pandas as pd

PORTFOLIOS = {
    "BASE13": BASE13,
    "BASE13_NEAR": BASE13 + ["NEARUSDT"],
    "BASE13_SOL": BASE13 + ["SOLUSDT"],
    "BASE13_LINK": BASE13 + ["LINKUSDT"],
    "BASE13_NEAR_SOL": BASE13 + ["NEARUSDT", "SOLUSDT"],
    "BASE13_NEAR_LINK": BASE13 + ["NEARUSDT", "LINKUSDT"],
    "BASE13_NEAR_SOL_LINK": BASE13 + ["NEARUSDT", "SOLUSDT", "LINKUSDT"],
}
ALL_SYMBOLS = sorted(set(sum(PORTFOLIOS.values(), [])))


def main():
    data, failures = {}, []
    for s in ALL_SYMBOLS:
        try:
            data[s] = load_symbol(s)
            print(f"LOADED {s}: {len(data[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
            print("SKIP", s, e)

    missing = sorted(set(ALL_SYMBOLS) - set(data))
    if missing:
        pd.DataFrame(failures).to_csv(OUT / "pc_near_sol_link_compare_6m_failures.csv", index=False)
        raise RuntimeError(f"Missing required symbols: {missing}")

    summaries, all_trades, all_equity = [], [], []
    for name, symbols in PORTFOLIOS.items():
        print("\n" + "=" * 80)
        print("RUN", name, "symbols=", len(symbols))
        summary, trades, equity = run_portfolio(name, symbols, data)
        summaries.append(summary)
        all_trades.append(trades)
        all_equity.append(equity)
        print(pd.DataFrame([summary]).to_string(index=False))

    sm = pd.DataFrame(summaries)
    base_final = float(sm.loc[sm.portfolio == "BASE13", "final_balance"].iloc[0])
    base_dd = float(sm.loc[sm.portfolio == "BASE13", "max_floating_dd_pct"].iloc[0])
    sm["delta_final_vs_base"] = sm.final_balance - base_final
    sm["delta_dd_vs_base_pp"] = sm.max_floating_dd_pct - base_dd
    sm = sm.sort_values("final_balance", ascending=False).reset_index(drop=True)

    sm.to_csv(OUT / "pc_near_sol_link_compare_6m_summary.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "pc_near_sol_link_compare_6m_trades.csv", index=False)
    pd.concat(all_equity, ignore_index=True).to_csv(OUT / "pc_near_sol_link_compare_6m_equity.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "pc_near_sol_link_compare_6m_failures.csv", index=False)

    print("\nFINAL COMPARISON\n")
    cols = ["portfolio", "symbols", "final_balance", "net_profit_usdt", "max_floating_dd_pct", "min_equity", "closed_trades", "winrate", "skipped_entries_margin", "liquidated", "delta_final_vs_base", "delta_dd_vs_base_pp"]
    print(sm[cols].to_string(index=False))


if __name__ == "__main__":
    main()
