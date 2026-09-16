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

# Existing 13 + four new candidates from the broad screen.
SYMBOLS = [
    "ZECUSDT", "USELESSUSDT", "HYPEUSDT", "FETUSDT", "JTOUSDT",
    "XRPUSDT", "VETUSDT", "INJUSDT", "ETHUSDT", "DYDXUSDT",
    "1000SHIBUSDT", "UNIUSDT", "SEIUSDT",
    "FLOWUSDT", "IOTAUSDT", "DYMUSDT", "NEARUSDT",
]

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


def main():
    data = {}
    failures = []
    for s in SYMBOLS:
        try:
            data[s] = load_symbol(s)
            print(f"LOADED {s}: {len(data[s])}")
        except Exception as e:
            failures.append({"symbol": s, "error": str(e)})
            print("SKIP", s, e)

    if not data:
        raise RuntimeError("No symbols loaded")

    # Build a common 30m event stream. Each symbol is processed at its own candle close.
    events = []
    for s, d in data.items():
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
                "symbol": sym, "side": pos["side"], "entry_time": pos["time"],
                "exit_time": tm, "entry": pos["entry"], "exit": close,
                "pnl_usdt": pnl, "return_pct": ret * 100.0, "exit_reason": "OPPOSITE",
            })
            del positions[sym]
            pos = None

        # Same opposite candle immediately reverses, matching the Purple Cloud state change.
        if sym not in positions and (buy or sell):
            used_margin = len(positions) * INITIAL_MARGIN
            available = equity() - used_margin
            if available >= INITIAL_MARGIN:
                positions[sym] = {
                    "side": "LONG" if buy else "SHORT",
                    "entry": close, "time": tm,
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
        equity_rows.append({"time": tm, "balance": balance, "equity": eq, "open_positions": len(positions), "gross_notional": gross})

        if positions and eq <= maintenance:
            liquidated = True
            liquidation_time = tm
            print("LIQUIDATED", tm, "equity", eq, "maintenance", maintenance)

    # Force-close remaining positions at each symbol's last available close.
    if not liquidated:
        for sym, pos in list(positions.items()):
            d = data[sym]
            b = d.iloc[-1]
            ex = float(b.close)
            ret = trade_return(pos["side"], pos["entry"], ex)
            pnl = NOTIONAL * ret
            balance += pnl
            trades.append({
                "symbol": sym, "side": pos["side"], "entry_time": pos["time"],
                "exit_time": b.time, "entry": pos["entry"], "exit": ex,
                "pnl_usdt": pnl, "return_pct": ret * 100.0, "exit_reason": "END",
            })
        positions.clear()

    t = pd.DataFrame(trades)
    e = pd.DataFrame(equity_rows)
    if len(t):
        by = t.groupby("symbol").agg(
            trades=("pnl_usdt", "size"),
            pnl_usdt=("pnl_usdt", "sum"),
            avg_pnl=("pnl_usdt", "mean"),
            winrate=("pnl_usdt", lambda x: 100.0 * (x > 0).mean()),
        ).reset_index().sort_values("pnl_usdt", ascending=False)
    else:
        by = pd.DataFrame()

    wins = int((t.pnl_usdt > 0).sum()) if len(t) else 0
    losses = int((t.pnl_usdt <= 0).sum()) if len(t) else 0
    summary = pd.DataFrame([{
        "start_balance": START_BALANCE,
        "final_balance": balance if not liquidated else equity(),
        "net_profit_usdt": (balance if not liquidated else equity()) - START_BALANCE,
        "return_pct": ((balance if not liquidated else equity()) / START_BALANCE - 1.0) * 100.0,
        "closed_trades": len(t), "wins": wins, "losses": losses,
        "winrate": 100.0 * wins / len(t) if len(t) else 0.0,
        "max_floating_dd_pct": max_dd_pct, "min_equity": min_equity,
        "max_simultaneous_positions": max_positions,
        "max_gross_notional": max_positions * NOTIONAL,
        "skipped_entries_margin": skipped_margin,
        "liquidated": liquidated, "liquidation_time": liquidation_time,
        "leverage": LEVERAGE, "position_notional": NOTIONAL,
        "initial_margin_per_position": INITIAL_MARGIN, "mmr_assumption": MMR,
        "symbols_loaded": len(data), "symbols_requested": len(SYMBOLS),
    }])

    t.to_csv(OUT / "pc17_unified_cross_6m_trades.csv", index=False)
    e.to_csv(OUT / "pc17_unified_cross_6m_equity.csv", index=False)
    by.to_csv(OUT / "pc17_unified_cross_6m_by_symbol.csv", index=False)
    summary.to_csv(OUT / "pc17_unified_cross_6m_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "pc17_unified_cross_6m_failures.csv", index=False)

    print("\nSUMMARY\n", summary.to_string(index=False))
    print("\nBY SYMBOL\n", by.to_string(index=False))


if __name__ == "__main__":
    main()
