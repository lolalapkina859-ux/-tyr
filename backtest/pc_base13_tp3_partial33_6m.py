from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

START = "2026-03-01"
END = "2026-09-01"
START_BALANCE = 120.0
NOTIONAL = 60.0
LEVERAGE = 20.0
INITIAL_MARGIN = NOTIONAL / LEVERAGE
MMR = 0.005
TP_PCT = 0.03
TP_FRACTION = 0.33
RUNNER_FRACTION = 1.0 - TP_FRACTION

BASE13 = [
    "ZECUSDT", "USELESSUSDT", "HYPEUSDT", "FETUSDT", "JTOUSDT",
    "XRPUSDT", "VETUSDT", "INJUSDT", "ETHUSDT", "DYDXUSDT",
    "1000SHIBUSDT", "UNIUSDT", "SEIUSDT",
]

OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)


def load(symbol):
    d = download_klines(symbol, "30m", START, END).reset_index(drop=True)
    if len(d) < 100:
        raise ValueError(f"{symbol}: not enough candles: {len(d)}")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return purple_cloud(d)[["time", "open", "high", "low", "close", "pc_buy", "pc_sell"]].copy()


def price_ret(side, entry, exit_):
    return exit_ / entry - 1.0 if side == "LONG" else entry / exit_ - 1.0


def main():
    data, failures = {}, []
    for s in BASE13:
        try:
            data[s] = load(s)
            print(f"LOADED {s}: {len(data[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
    missing = [s for s in BASE13 if s not in data]
    if missing:
        raise RuntimeError(f"Missing required symbols: {missing}")

    # Process every 30m candle across all symbols chronologically.
    # TP is detected from candle HIGH/LOW and filled exactly at +3% price.
    # 33% of original $60 notional is realized once. Remaining 67% stays
    # until the normal opposite Purple Cloud 30m signal.
    events = []
    for s, df in data.items():
        for _, r in df.iterrows():
            events.append((r.time, s, r))
    events.sort(key=lambda x: (x[0], x[1]))

    balance = START_BALANCE
    positions = {}
    last_price = {}
    trades = []
    partials = []
    equity_rows = []
    skipped = 0
    max_positions = 0
    max_gross = 0.0
    min_equity = START_BALANCE
    peak_equity = START_BALANCE
    max_dd = 0.0
    liquidated = False
    liquidation_time = None

    def open_notional(p):
        return NOTIONAL * p["remaining_fraction"]

    def floating_pnl():
        total = 0.0
        for sym, p in positions.items():
            px = last_price.get(sym, p["entry"])
            total += open_notional(p) * price_ret(p["side"], p["entry"], px)
        return total

    def equity():
        return balance + floating_pnl()

    def gross_notional():
        return sum(open_notional(p) for p in positions.values())

    def close_runner(sym, price, tm, reason):
        nonlocal balance
        p = positions[sym]
        n = open_notional(p)
        r = price_ret(p["side"], p["entry"], price)
        pnl = n * r
        balance += pnl
        total_pnl = p["realized_partial_pnl"] + pnl
        trades.append({
            "symbol": sym, "side": p["side"], "entry_time": p["time"], "exit_time": tm,
            "entry": p["entry"], "exit": price, "tp3_hit": p["tp_hit"],
            "partial_pnl_usdt": p["realized_partial_pnl"], "runner_pnl_usdt": pnl,
            "total_trade_pnl_usdt": total_pnl,
            "runner_fraction_at_exit": p["remaining_fraction"], "exit_reason": reason,
        })
        del positions[sym]

    for tm, sym, r in events:
        if liquidated:
            break
        close = float(r.close)
        high = float(r.high)
        low = float(r.low)
        last_price[sym] = close

        p = positions.get(sym)

        # Existing position: first check intrabar TP3 partial.
        if p and not p["tp_hit"]:
            tp_price = p["entry"] * (1.0 + TP_PCT) if p["side"] == "LONG" else p["entry"] * (1.0 - TP_PCT)
            hit = high >= tp_price if p["side"] == "LONG" else low <= tp_price
            if hit:
                partial_notional = NOTIONAL * TP_FRACTION
                pnl = partial_notional * price_ret(p["side"], p["entry"], tp_price)
                balance += pnl
                p["tp_hit"] = True
                p["remaining_fraction"] = RUNNER_FRACTION
                p["realized_partial_pnl"] += pnl
                partials.append({
                    "symbol": sym, "side": p["side"], "entry_time": p["time"], "tp_time": tm,
                    "entry": p["entry"], "tp_price": tp_price, "closed_fraction": TP_FRACTION,
                    "closed_notional": partial_notional, "pnl_usdt": pnl,
                })

        # Normal Purple Cloud opposite signal closes whatever remains.
        p = positions.get(sym)
        sig = "LONG" if bool(r.pc_buy) else ("SHORT" if bool(r.pc_sell) else None)
        if p and sig and sig != p["side"]:
            close_runner(sym, close, tm, "OPPOSITE_30M")
            p = None

        # Same original strategy behavior: fresh PC signal opens/reverses.
        if sig and sym not in positions:
            used_margin = gross_notional() / LEVERAGE
            available = equity() - used_margin
            if available >= INITIAL_MARGIN:
                positions[sym] = {
                    "side": sig, "entry": close, "time": tm,
                    "remaining_fraction": 1.0, "tp_hit": False,
                    "realized_partial_pnl": 0.0,
                }
                max_positions = max(max_positions, len(positions))
            else:
                skipped += 1

        eq = equity()
        gross = gross_notional()
        max_gross = max(max_gross, gross)
        maint = gross * MMR
        min_equity = min(min_equity, eq)
        peak_equity = max(peak_equity, eq)
        dd = (eq / peak_equity - 1.0) * 100.0 if peak_equity else 0.0
        max_dd = min(max_dd, dd)
        equity_rows.append({"time": tm, "balance": balance, "equity": eq, "open_positions": len(positions), "gross_notional": gross})
        if positions and eq <= maint:
            liquidated = True
            liquidation_time = tm

    if not liquidated:
        for sym in list(positions):
            row = data[sym].iloc[-1]
            close_runner(sym, float(row.close), row.time, "END")

    t = pd.DataFrame(trades)
    pt = pd.DataFrame(partials)
    final = balance if not liquidated else equity()
    wins = int((t.total_trade_pnl_usdt > 0).sum()) if len(t) else 0
    tp_hits = int(t.tp3_hit.sum()) if len(t) else 0
    summary = pd.DataFrame([{
        "portfolio": "BASE13_30M_TP3_PARTIAL33",
        "symbols": len(BASE13), "start_balance": START_BALANCE, "final_balance": final,
        "net_profit_usdt": final - START_BALANCE, "return_pct": (final / START_BALANCE - 1.0) * 100.0,
        "closed_trades": len(t), "wins": wins, "losses": len(t)-wins,
        "winrate": wins/len(t)*100 if len(t) else 0.0,
        "tp3_hits": tp_hits, "tp3_hit_rate_pct": tp_hits/len(t)*100 if len(t) else 0.0,
        "tp_pct": TP_PCT*100, "tp_closed_fraction_pct": TP_FRACTION*100,
        "runner_fraction_pct": RUNNER_FRACTION*100,
        "max_floating_dd_pct": max_dd, "min_equity": min_equity,
        "max_simultaneous_positions": max_positions, "max_gross_notional": max_gross,
        "skipped_entries_margin": skipped, "liquidated": liquidated, "liquidation_time": liquidation_time,
    }])

    if len(t):
        by = t.groupby("symbol", as_index=False).agg(
            trades=("total_trade_pnl_usdt", "size"), pnl_usdt=("total_trade_pnl_usdt", "sum"),
            avg_pnl=("total_trade_pnl_usdt", "mean"), tp3_hits=("tp3_hit", "sum"))
        wr = t.assign(win=t.total_trade_pnl_usdt > 0).groupby("symbol")["win"].mean()*100
        by["winrate"] = by.symbol.map(wr)
        by["tp3_hit_rate_pct"] = by.tp3_hits / by.trades * 100
        by = by.sort_values("pnl_usdt", ascending=False)
    else:
        by = pd.DataFrame()

    prefix = OUT / "pc_base13_tp3_partial33"
    summary.to_csv(str(prefix)+"_summary.csv", index=False)
    t.to_csv(str(prefix)+"_trades.csv", index=False)
    pt.to_csv(str(prefix)+"_partials.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(str(prefix)+"_equity.csv", index=False)
    by.to_csv(str(prefix)+"_by_symbol.csv", index=False)
    pd.DataFrame(failures).to_csv(str(prefix)+"_failures.csv", index=False)
    print("\nSUMMARY\n", summary.to_string(index=False))
    print("\nBY SYMBOL\n", by.to_string(index=False))

if __name__ == "__main__":
    main()
