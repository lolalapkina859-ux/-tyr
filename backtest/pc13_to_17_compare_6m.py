from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START = "2026-03-01"
END = "2026-09-01"
TF = "30m"
START_BALANCE = 100.0
NOTIONAL = 60.0
LEVERAGE = 20.0
INITIAL_MARGIN = NOTIONAL / LEVERAGE
MMR = 0.005

BASE13 = [
    "ZECUSDT", "USELESSUSDT", "HYPEUSDT", "FETUSDT", "JTOUSDT",
    "XRPUSDT", "VETUSDT", "INJUSDT", "ETHUSDT", "DYDXUSDT",
    "1000SHIBUSDT", "UNIUSDT", "SEIUSDT",
]

# Add candidates one by one in the requested order.
PORTFOLIOS = {
    "13_BASE": BASE13,
    "14_IOTA": BASE13 + ["IOTAUSDT"],
    "15_IOTA_NEAR": BASE13 + ["IOTAUSDT", "NEARUSDT"],
    "16_IOTA_NEAR_DYM": BASE13 + ["IOTAUSDT", "NEARUSDT", "DYMUSDT"],
    "17_IOTA_NEAR_DYM_FLOW": BASE13 + ["IOTAUSDT", "NEARUSDT", "DYMUSDT", "FLOWUSDT"],
}

ALL_SYMBOLS = sorted(set(sum(PORTFOLIOS.values(), [])))
OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)


def load_symbol(symbol: str) -> pd.DataFrame:
    d = download_klines(symbol, TF, START, END).reset_index(drop=True)
    if len(d) < 100:
        raise ValueError(f"not enough candles: {len(d)}")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    pc = purple_cloud(d)
    return pc[["time", "close", "pc_buy", "pc_sell"]].copy()


def trade_return(side: str, entry: float, exit_: float) -> float:
    return (exit_ / entry - 1.0) if side == "LONG" else (entry / exit_ - 1.0)


def run_portfolio(name: str, symbols: list[str], data: dict[str, pd.DataFrame]):
    events = []
    for s in symbols:
        d = data[s]
        for i, r in d.iterrows():
            events.append((r.time, s, i, float(r.close), bool(r.pc_buy), bool(r.pc_sell)))
    events.sort(key=lambda x: (x[0], x[1]))

    balance = START_BALANCE
    positions = {}
    last_price = {}
    trades = []
    equity_rows = []
    skipped_margin = 0
    liquidated = False
    liquidation_time = None
    max_positions = 0
    min_equity = START_BALANCE
    peak_equity = START_BALANCE
    max_dd_pct = 0.0

    def floating_pnl():
        total = 0.0
        for sym, p in positions.items():
            px = last_price.get(sym, p["entry"])
            total += NOTIONAL * trade_return(p["side"], p["entry"], px)
        return total

    def equity():
        return balance + floating_pnl()

    for tm, sym, i, close, buy, sell in events:
        if liquidated:
            break
        last_price[sym] = close
        pos = positions.get(sym)
        opposite = pos and ((pos["side"] == "LONG" and sell) or (pos["side"] == "SHORT" and buy))

        if opposite:
            ret = trade_return(pos["side"], pos["entry"], close)
            pnl = NOTIONAL * ret
            balance += pnl
            trades.append({
                "portfolio": name, "symbol": sym, "side": pos["side"],
                "entry_time": pos["time"], "exit_time": tm,
                "entry": pos["entry"], "exit": close,
                "pnl_usdt": pnl, "return_pct": ret * 100.0,
                "exit_reason": "OPPOSITE",
            })
            del positions[sym]

        if sym not in positions and (buy or sell):
            used_margin = len(positions) * INITIAL_MARGIN
            available = equity() - used_margin
            if available >= INITIAL_MARGIN:
                positions[sym] = {
                    "side": "LONG" if buy else "SHORT",
                    "entry": close,
                    "time": tm,
                }
                max_positions = max(max_positions, len(positions))
            else:
                skipped_margin += 1

        eq = equity()
        gross = len(positions) * NOTIONAL
        maintenance = gross * MMR
        min_equity = min(min_equity, eq)
        peak_equity = max(peak_equity, eq)
        dd = (eq / peak_equity - 1.0) * 100.0 if peak_equity else 0.0
        max_dd_pct = min(max_dd_pct, dd)
        equity_rows.append({
            "portfolio": name, "time": tm, "balance": balance,
            "equity": eq, "open_positions": len(positions),
            "gross_notional": gross,
        })

        if positions and eq <= maintenance:
            liquidated = True
            liquidation_time = tm

    if not liquidated:
        for sym, pos in list(positions.items()):
            b = data[sym].iloc[-1]
            ex = float(b.close)
            ret = trade_return(pos["side"], pos["entry"], ex)
            pnl = NOTIONAL * ret
            balance += pnl
            trades.append({
                "portfolio": name, "symbol": sym, "side": pos["side"],
                "entry_time": pos["time"], "exit_time": b.time,
                "entry": pos["entry"], "exit": ex,
                "pnl_usdt": pnl, "return_pct": ret * 100.0,
                "exit_reason": "END",
            })
        positions.clear()

    t = pd.DataFrame(trades)
    final_value = balance if not liquidated else equity()
    wins = int((t.pnl_usdt > 0).sum()) if len(t) else 0
    summary = {
        "portfolio": name,
        "symbols": len(symbols),
        "symbol_list": ",".join(symbols),
        "start_balance": START_BALANCE,
        "final_balance": final_value,
        "net_profit_usdt": final_value - START_BALANCE,
        "return_pct": (final_value / START_BALANCE - 1.0) * 100.0,
        "closed_trades": len(t),
        "wins": wins,
        "losses": len(t) - wins,
        "winrate": 100.0 * wins / len(t) if len(t) else 0.0,
        "max_floating_dd_pct": max_dd_pct,
        "min_equity": min_equity,
        "max_simultaneous_positions": max_positions,
        "max_gross_notional": max_positions * NOTIONAL,
        "skipped_entries_margin": skipped_margin,
        "liquidated": liquidated,
        "liquidation_time": liquidation_time,
    }
    return summary, t, pd.DataFrame(equity_rows)


def main():
    data = {}
    failures = []
    for s in ALL_SYMBOLS:
        try:
            data[s] = load_symbol(s)
            print(f"LOADED {s}: {len(data[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
            print("SKIP", s, e)

    missing = sorted(set(ALL_SYMBOLS) - set(data))
    if missing:
        raise RuntimeError(f"Missing required symbols: {missing}")

    summaries = []
    all_trades = []
    all_equity = []

    for name, symbols in PORTFOLIOS.items():
        print("\n" + "=" * 80)
        print("RUN", name, "symbols=", len(symbols))
        summary, trades, equity = run_portfolio(name, symbols, data)
        summaries.append(summary)
        all_trades.append(trades)
        all_equity.append(equity)
        print(pd.DataFrame([summary]).to_string(index=False))

    sm = pd.DataFrame(summaries)
    sm["delta_final_vs_13"] = sm["final_balance"] - float(sm.iloc[0]["final_balance"])
    sm["delta_dd_vs_13_pp"] = sm["max_floating_dd_pct"] - float(sm.iloc[0]["max_floating_dd_pct"])
    sm["added_vs_previous_final"] = sm["final_balance"].diff()
    sm["added_vs_previous_dd_pp"] = sm["max_floating_dd_pct"].diff()

    sm.to_csv(OUT / "pc13_to_17_compare_6m_summary.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "pc13_to_17_compare_6m_trades.csv", index=False)
    pd.concat(all_equity, ignore_index=True).to_csv(OUT / "pc13_to_17_compare_6m_equity.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "pc13_to_17_compare_6m_failures.csv", index=False)

    print("\nFINAL COMPARISON\n")
    cols = [
        "portfolio", "symbols", "final_balance", "net_profit_usdt", "return_pct",
        "max_floating_dd_pct", "min_equity", "closed_trades", "winrate",
        "max_simultaneous_positions", "skipped_entries_margin", "liquidated",
        "delta_final_vs_13", "added_vs_previous_final",
        "delta_dd_vs_13_pp", "added_vs_previous_dd_pp",
    ]
    print(sm[cols].to_string(index=False))


if __name__ == "__main__":
    main()
