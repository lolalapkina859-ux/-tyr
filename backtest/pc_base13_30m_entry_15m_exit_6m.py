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

BASE13 = [
    "ZECUSDT", "USELESSUSDT", "HYPEUSDT", "FETUSDT", "JTOUSDT",
    "XRPUSDT", "VETUSDT", "INJUSDT", "ETHUSDT", "DYDXUSDT",
    "1000SHIBUSDT", "UNIUSDT", "SEIUSDT",
]

OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)


def load_pc(symbol, tf):
    d = download_klines(symbol, tf, START, END).reset_index(drop=True)
    if len(d) < 100:
        raise ValueError(f"{symbol} {tf}: not enough candles: {len(d)}")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    p = purple_cloud(d)
    return p[["time", "close", "pc_buy", "pc_sell"]].copy()


def ret(side, entry, exit_):
    return exit_ / entry - 1 if side == "LONG" else entry / exit_ - 1


def main():
    d15, d30, failures = {}, {}, []
    for s in BASE13:
        try:
            d15[s] = load_pc(s, "15m")
            d30[s] = load_pc(s, "30m")
            print(f"LOADED {s}: 30m={len(d30[s])} 15m={len(d15[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
    missing = [s for s in BASE13 if s not in d15 or s not in d30]
    if missing:
        raise RuntimeError(f"Missing symbols: {missing}")

    # 30m signals are ENTRY ONLY. A 30m signal opens a position if flat.
    # An opposite 15m signal CLOSES the position early, but NEVER reverses it.
    # After a 15m exit the symbol stays flat until the NEXT fresh 30m signal.
    events = []
    for s in BASE13:
        for _, r in d15[s].iterrows():
            if bool(r.pc_buy) or bool(r.pc_sell):
                events.append((r.time, 0, s, "15m", float(r.close), bool(r.pc_buy), bool(r.pc_sell)))
        for _, r in d30[s].iterrows():
            if bool(r.pc_buy) or bool(r.pc_sell):
                events.append((r.time, 1, s, "30m", float(r.close), bool(r.pc_buy), bool(r.pc_sell)))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    balance = START_BALANCE
    pos = {}
    last_price = {}
    trades, eqrows = [], []
    skipped = 0
    maxpos = 0
    min_eq = START_BALANCE
    peak_eq = START_BALANCE
    maxdd = 0.0
    liquidated = False
    liq_time = None

    def floating():
        return sum(NOTIONAL * ret(p["side"], p["entry"], last_price.get(s, p["entry"])) for s, p in pos.items())

    def equity():
        return balance + floating()

    def close_position(sym, price, tm, reason):
        nonlocal balance
        p = pos[sym]
        r = ret(p["side"], p["entry"], price)
        pnl = NOTIONAL * r
        balance += pnl
        trades.append({"symbol": sym, "side": p["side"], "entry_time": p["time"], "exit_time": tm,
                       "entry": p["entry"], "exit": price, "pnl_usdt": pnl, "return_pct": r*100,
                       "entry_tf": "30m", "exit_tf": "15m", "exit_reason": reason})
        del pos[sym]

    for tm, _, sym, tf, price, buy, sell in events:
        if liquidated:
            break
        last_price[sym] = price
        signal = "LONG" if buy else "SHORT"

        if tf == "15m":
            if sym in pos and pos[sym]["side"] != signal:
                close_position(sym, price, tm, "OPPOSITE_15M")
        else:
            # 30m is entry only. If already in a position, do nothing here;
            # exits are controlled exclusively by opposite 15m signals.
            if sym not in pos:
                used = len(pos) * INITIAL_MARGIN
                available = equity() - used
                if available >= INITIAL_MARGIN:
                    pos[sym] = {"side": signal, "entry": price, "time": tm}
                    maxpos = max(maxpos, len(pos))
                else:
                    skipped += 1

        eq = equity()
        gross = len(pos) * NOTIONAL
        maint = gross * MMR
        min_eq = min(min_eq, eq)
        peak_eq = max(peak_eq, eq)
        dd = (eq / peak_eq - 1) * 100 if peak_eq else 0
        maxdd = min(maxdd, dd)
        eqrows.append({"time": tm, "balance": balance, "equity": eq, "open_positions": len(pos), "gross_notional": gross})
        if pos and eq <= maint:
            liquidated = True
            liq_time = tm

    if not liquidated:
        for sym in list(pos):
            row = d15[sym].iloc[-1]
            close_position(sym, float(row.close), row.time, "END")

    t = pd.DataFrame(trades)
    final = balance if not liquidated else equity()
    wins = int((t.pnl_usdt > 0).sum()) if len(t) else 0
    summary = pd.DataFrame([{
        "portfolio": "BASE13_30M_ENTRY_15M_EXIT", "symbols": 13,
        "start_balance": START_BALANCE, "final_balance": final,
        "net_profit_usdt": final-START_BALANCE, "return_pct": (final/START_BALANCE-1)*100,
        "closed_trades": len(t), "wins": wins, "losses": len(t)-wins,
        "winrate": wins/len(t)*100 if len(t) else 0,
        "max_floating_dd_pct": maxdd, "min_equity": min_eq,
        "max_simultaneous_positions": maxpos, "max_gross_notional": maxpos*NOTIONAL,
        "skipped_entries_margin": skipped, "liquidated": liquidated, "liquidation_time": liq_time,
    }])

    if len(t):
        by = t.groupby("symbol", as_index=False).agg(trades=("pnl_usdt","size"), pnl_usdt=("pnl_usdt","sum"), avg_pnl=("pnl_usdt","mean"))
        wr = t.assign(win=t.pnl_usdt>0).groupby("symbol")["win"].mean()*100
        by["winrate"] = by.symbol.map(wr)
        by = by.sort_values("pnl_usdt", ascending=False)
    else:
        by = pd.DataFrame()

    prefix = OUT / "pc_base13_30m_entry_15m_exit"
    summary.to_csv(str(prefix)+"_summary.csv", index=False)
    t.to_csv(str(prefix)+"_trades.csv", index=False)
    pd.DataFrame(eqrows).to_csv(str(prefix)+"_equity.csv", index=False)
    by.to_csv(str(prefix)+"_by_symbol.csv", index=False)
    pd.DataFrame(failures).to_csv(str(prefix)+"_failures.csv", index=False)
    print("\nSUMMARY\n", summary.to_string(index=False))
    print("\nBY SYMBOL\n", by.to_string(index=False))

if __name__ == "__main__":
    main()
