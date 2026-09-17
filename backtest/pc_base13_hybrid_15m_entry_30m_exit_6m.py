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


def load_pc(symbol: str, tf: str) -> pd.DataFrame:
    d = download_klines(symbol, tf, START, END).reset_index(drop=True)
    if len(d) < 100:
        raise ValueError(f"{symbol} {tf}: not enough candles: {len(d)}")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    pc = purple_cloud(d)
    return pc[["time", "close", "pc_buy", "pc_sell"]].copy()


def trade_return(side: str, entry: float, exit_: float) -> float:
    return (exit_ / entry - 1.0) if side == "LONG" else (entry / exit_ - 1.0)


def main():
    data15 = {}
    data30 = {}
    failures = []
    for s in BASE13:
        try:
            data15[s] = load_pc(s, "15m")
            data30[s] = load_pc(s, "30m")
            print(f"LOADED {s}: 15m={len(data15[s])}, 30m={len(data30[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
            print("SKIP", s, e)

    missing = [s for s in BASE13 if s not in data15 or s not in data30]
    if missing:
        raise RuntimeError(f"Missing required symbols: {missing}")

    # Event semantics:
    # 30m Purple Cloud signals define the allowed regime and exits.
    # On a new 30m signal, an opposite open position is closed immediately at
    # the 30m signal close. Then we WAIT for the next matching 15m Purple Cloud
    # signal to enter. 15m opposite signals never close an existing position.
    events = []
    for s in BASE13:
        for _, r in data30[s].iterrows():
            if bool(r.pc_buy) or bool(r.pc_sell):
                events.append((r.time, 0, s, "30m", float(r.close), bool(r.pc_buy), bool(r.pc_sell)))
        for _, r in data15[s].iterrows():
            if bool(r.pc_buy) or bool(r.pc_sell):
                events.append((r.time, 1, s, "15m", float(r.close), bool(r.pc_buy), bool(r.pc_sell)))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    balance = START_BALANCE
    positions = {}
    regime = {s: None for s in BASE13}
    last_price = {}
    trades = []
    equity_rows = []
    skipped_margin = 0
    max_positions = 0
    min_equity = START_BALANCE
    peak_equity = START_BALANCE
    max_dd_pct = 0.0
    liquidated = False
    liquidation_time = None

    def floating_pnl():
        total = 0.0
        for sym, p in positions.items():
            px = last_price.get(sym, p["entry"])
            total += NOTIONAL * trade_return(p["side"], p["entry"], px)
        return total

    def equity():
        return balance + floating_pnl()

    for tm, _, sym, tf, close, buy, sell in events:
        if liquidated:
            break
        last_price[sym] = close

        if tf == "30m":
            new_regime = "LONG" if buy else "SHORT"
            regime[sym] = new_regime
            pos = positions.get(sym)
            if pos and pos["side"] != new_regime:
                ret = trade_return(pos["side"], pos["entry"], close)
                pnl = NOTIONAL * ret
                balance += pnl
                trades.append({
                    "symbol": sym, "side": pos["side"],
                    "entry_time": pos["time"], "exit_time": tm,
                    "entry": pos["entry"], "exit": close,
                    "pnl_usdt": pnl, "return_pct": ret * 100.0,
                    "exit_reason": "OPPOSITE_30M",
                    "entry_tf": "15m", "exit_tf": "30m",
                })
                del positions[sym]

        else:  # 15m entry trigger only
            sig = "LONG" if buy else "SHORT"
            if sym not in positions and regime[sym] == sig:
                used_margin = len(positions) * INITIAL_MARGIN
                available = equity() - used_margin
                if available >= INITIAL_MARGIN:
                    positions[sym] = {"side": sig, "entry": close, "time": tm}
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
            "time": tm, "balance": balance, "equity": eq,
            "open_positions": len(positions), "gross_notional": gross,
        })
        if positions and eq <= maintenance:
            liquidated = True
            liquidation_time = tm

    if not liquidated:
        for sym, pos in list(positions.items()):
            b = data15[sym].iloc[-1]
            ex = float(b.close)
            ret = trade_return(pos["side"], pos["entry"], ex)
            pnl = NOTIONAL * ret
            balance += pnl
            trades.append({
                "symbol": sym, "side": pos["side"],
                "entry_time": pos["time"], "exit_time": b.time,
                "entry": pos["entry"], "exit": ex,
                "pnl_usdt": pnl, "return_pct": ret * 100.0,
                "exit_reason": "END", "entry_tf": "15m", "exit_tf": "30m",
            })
        positions.clear()

    t = pd.DataFrame(trades)
    final_value = balance if not liquidated else equity()
    wins = int((t.pnl_usdt > 0).sum()) if len(t) else 0
    summary = pd.DataFrame([{
        "portfolio": "BASE13_HYBRID_15M_ENTRY_30M_EXIT",
        "symbols": len(BASE13),
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
    }])

    by_symbol = (t.groupby("symbol", as_index=False)
        .agg(trades=("pnl_usdt", "size"), pnl_usdt=("pnl_usdt", "sum"), avg_pnl=("pnl_usdt", "mean"))) if len(t) else pd.DataFrame()
    if len(t):
        wr = t.assign(win=t.pnl_usdt > 0).groupby("symbol")["win"].mean().mul(100)
        by_symbol["winrate"] = by_symbol.symbol.map(wr)
        by_symbol = by_symbol.sort_values("pnl_usdt", ascending=False)

    summary.to_csv(OUT / "pc_base13_hybrid_15m_entry_30m_exit_summary.csv", index=False)
    t.to_csv(OUT / "pc_base13_hybrid_15m_entry_30m_exit_trades.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(OUT / "pc_base13_hybrid_15m_entry_30m_exit_equity.csv", index=False)
    by_symbol.to_csv(OUT / "pc_base13_hybrid_15m_entry_30m_exit_by_symbol.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "pc_base13_hybrid_15m_entry_30m_exit_failures.csv", index=False)

    print("\nSUMMARY\n", summary.to_string(index=False))
    print("\nBY SYMBOL\n", by_symbol.to_string(index=False))


if __name__ == "__main__":
    main()
