from __future__ import annotations

from itertools import combinations
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
CANDIDATES = ["IOTAUSDT", "NEARUSDT", "DYMUSDT", "FLOWUSDT"]
ALL_SYMBOLS = sorted(set(BASE13 + CANDIDATES))
OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)


def load_symbol(symbol):
    d = download_klines(symbol, TF, START, END).reset_index(drop=True)
    if len(d) < 100:
        raise ValueError(f"not enough candles: {len(d)}")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    pc = purple_cloud(d)
    return pc[["time", "close", "pc_buy", "pc_sell"]].copy()


def trade_return(side, entry, exit_):
    return (exit_ / entry - 1.0) if side == "LONG" else (entry / exit_ - 1.0)


def run_portfolio(name, added, data):
    symbols = BASE13 + list(added)
    events = []
    for s in symbols:
        for _, r in data[s].iterrows():
            events.append((r.time, s, float(r.close), bool(r.pc_buy), bool(r.pc_sell)))
    events.sort(key=lambda x: (x[0], x[1]))

    balance = START_BALANCE
    positions, last_price = {}, {}
    trades = []
    skipped_margin = 0
    liquidated = False
    liquidation_time = None
    max_positions = 0
    min_equity = START_BALANCE
    peak_equity = START_BALANCE
    max_dd_pct = 0.0

    def floating_pnl():
        return sum(NOTIONAL * trade_return(p["side"], p["entry"], last_price.get(s, p["entry"])) for s, p in positions.items())

    def equity():
        return balance + floating_pnl()

    for tm, sym, close, buy, sell in events:
        if liquidated:
            break
        last_price[sym] = close
        pos = positions.get(sym)
        opposite = pos and ((pos["side"] == "LONG" and sell) or (pos["side"] == "SHORT" and buy))

        if opposite:
            ret = trade_return(pos["side"], pos["entry"], close)
            pnl = NOTIONAL * ret
            balance += pnl
            trades.append({"combo": name, "symbol": sym, "side": pos["side"], "entry_time": pos["time"], "exit_time": tm, "entry": pos["entry"], "exit": close, "pnl_usdt": pnl, "return_pct": ret * 100, "exit_reason": "OPPOSITE"})
            del positions[sym]

        if sym not in positions and (buy or sell):
            used_margin = len(positions) * INITIAL_MARGIN
            if equity() - used_margin >= INITIAL_MARGIN:
                positions[sym] = {"side": "LONG" if buy else "SHORT", "entry": close, "time": tm}
                max_positions = max(max_positions, len(positions))
            else:
                skipped_margin += 1

        eq = equity()
        gross = len(positions) * NOTIONAL
        maintenance = gross * MMR
        min_equity = min(min_equity, eq)
        peak_equity = max(peak_equity, eq)
        dd = (eq / peak_equity - 1) * 100 if peak_equity else 0
        max_dd_pct = min(max_dd_pct, dd)
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
            trades.append({"combo": name, "symbol": sym, "side": pos["side"], "entry_time": pos["time"], "exit_time": b.time, "entry": pos["entry"], "exit": ex, "pnl_usdt": pnl, "return_pct": ret * 100, "exit_reason": "END"})

    t = pd.DataFrame(trades)
    final_value = balance if not liquidated else equity()
    wins = int((t.pnl_usdt > 0).sum()) if len(t) else 0
    return {
        "combo": name,
        "added_count": len(added),
        "added_symbols": ",".join(added) if added else "NONE",
        "total_symbols": len(symbols),
        "final_balance": final_value,
        "net_profit_usdt": final_value - START_BALANCE,
        "return_pct": (final_value / START_BALANCE - 1) * 100,
        "closed_trades": len(t),
        "winrate": 100 * wins / len(t) if len(t) else 0,
        "max_floating_dd_pct": max_dd_pct,
        "min_equity": min_equity,
        "max_simultaneous_positions": max_positions,
        "max_gross_notional": max_positions * NOTIONAL,
        "skipped_entries_margin": skipped_margin,
        "liquidated": liquidated,
        "liquidation_time": liquidation_time,
    }, t


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
        raise RuntimeError(f"Missing required symbols: {missing}")

    combos = []
    for n in range(0, len(CANDIDATES) + 1):
        combos.extend(combinations(CANDIDATES, n))
    assert len(combos) == 16

    summaries, trade_frames = [], []
    for idx, added in enumerate(combos, 1):
        short = [x.replace("USDT", "") for x in added]
        name = "BASE13" if not added else "BASE13+" + "+".join(short)
        print(f"\n[{idx}/16] {name}")
        sm, tr = run_portfolio(name, added, data)
        summaries.append(sm)
        trade_frames.append(tr)
        print(pd.DataFrame([sm]).to_string(index=False))

    df = pd.DataFrame(summaries)
    base = df[df.combo == "BASE13"].iloc[0]
    df["delta_final_vs_base"] = df.final_balance - base.final_balance
    df["delta_dd_vs_base_pp"] = df.max_floating_dd_pct - base.max_floating_dd_pct
    df["dd_depth_pct"] = -df.max_floating_dd_pct
    # Objective helper only: higher final balance and shallower DD are both visible; no hidden optimization.
    df = df.sort_values(["liquidated", "final_balance", "max_floating_dd_pct"], ascending=[True, False, False]).reset_index(drop=True)
    df["rank_by_final_balance"] = range(1, len(df) + 1)

    df.to_csv(OUT / "pc_candidate_combos_6m_summary.csv", index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "pc_candidate_combos_6m_trades.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "pc_candidate_combos_6m_failures.csv", index=False)

    print("\nALL 16 COMBINATIONS\n")
    cols = ["rank_by_final_balance", "combo", "total_symbols", "final_balance", "net_profit_usdt", "max_floating_dd_pct", "min_equity", "closed_trades", "winrate", "max_simultaneous_positions", "skipped_entries_margin", "liquidated", "delta_final_vs_base", "delta_dd_vs_base_pp"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
