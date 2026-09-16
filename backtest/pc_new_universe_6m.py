from __future__ import annotations

from pathlib import Path
import time
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud

TF = "30m"
PERIODS = [
    ("MAR_JUN", "2026-03-01", "2026-06-01"),
    ("JUN_SEP", "2026-06-01", "2026-09-01"),
]
OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDED = {
    "BTCUSDT", "ETHUSDT", "ZECUSDT",
    "XLMUSDT", "VETUSDT", "ALGOUSDT", "SANDUSDT", "MANAUSDT",
    "GALAUSDT", "AXSUSDT", "RUNEUSDT", "CRVUSDT", "LDOUSDT",
    "STXUSDT", "IMXUSDT", "GRTUSDT", "JUPUSDT", "BONKUSDT",
    "FLOKIUSDT", "NOTUSDT", "JTOUSDT", "PYTHUSDT", "WLDUSDT",
    "ARUSDT", "MKRUSDT", "COMPUSDT", "SNXUSDT", "DYDXUSDT",
    "OMUSDT", "ONDOUSDT", "SAGAUSDT", "STRKUSDT", "AEVOUSDT",
    "BLURUSDT", "MEMEUSDT", "1000SHIBUSDT", "1000SATSUSDT", "ORDIUSDT",
}


def get_universe():
    # Static USD-M candidate universe: avoids fapi.binance.com, which can return
    # HTTP 451 from GitHub-hosted runners. Historical candles still come from
    # data.binance.vision through download_binance.py.
    candidates = [
        "ADAUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT", "LTCUSDT", "BCHUSDT",
        "LINKUSDT", "AVAXUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
        "SUIUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT",
        "FILUSDT", "ETCUSDT", "AAVEUSDT", "UNIUSDT", "ICPUSDT", "HBARUSDT",
        "TRXUSDT", "EOSUSDT", "IOTAUSDT", "THETAUSDT", "KAVAUSDT", "ZILUSDT",
        "ENJUSDT", "CHZUSDT", "1INCHUSDT", "CELOUSDT", "FLOWUSDT", "MINAUSDT",
        "ROSEUSDT", "QTUMUSDT", "NEOUSDT", "DASHUSDT", "ZRXUSDT", "ANKRUSDT",
        "BATUSDT", "IOSTUSDT", "ONTUSDT", "WAVESUSDT", "KSMUSDT", "MASKUSDT",
        "ENSUSDT", "API3USDT", "GMXUSDT", "SSVUSDT", "CFXUSDT", "IDUSDT",
        "RDNTUSDT", "HOOKUSDT", "MAGICUSDT", "AGIXUSDT", "HIGHUSDT", "TRUUSDT",
        "LQTYUSDT", "JOEUSDT", "CYBERUSDT", "ARKMUSDT", "BIGTIMEUSDT", "NTRNUSDT",
        "BEAMXUSDT", "ACEUSDT", "XAIUSDT", "MANTAUSDT", "ALTUSDT", "DYMUSDT",
        "PIXELUSDT", "PORTALUSDT", "ENAUSDT", "WUSDT", "TAOUSDT", "REZUSDT",
        "BBUSDT", "IOUSDT", "ZKUSDT", "LISTAUSDT", "ZROUSDT", "TONUSDT",
        "BANANAUSDT", "RENDERUSDT", "EIGENUSDT", "NEIROUSDT", "HMSTRUSDT",
        "TURBOUSDT", "PNUTUSDT", "ACTUSDT", "KAIAUSDT", "MOVEUSDT", "MEUSDT",
        "VIRTUALUSDT", "PENGUUSDT", "BIOUSDT",
    ]
    return sorted({s for s in candidates if s not in EXCLUDED})


def run_symbol(sym, start, end):
    d = download_klines(sym, TF, start, end).reset_index(drop=True)
    if len(d) < 100:
        raise ValueError(f"not enough candles: {len(d)}")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    pc = purple_cloud(d)
    rows = []
    pos = None
    for i, b in d.iterrows():
        buy = bool(pc.iloc[i]["pc_buy"])
        sell = bool(pc.iloc[i]["pc_sell"])
        if pos and ((pos["side"] == "LONG" and sell) or (pos["side"] == "SHORT" and buy)):
            ex = float(b.close)
            ret = ex / pos["entry"] - 1 if pos["side"] == "LONG" else pos["entry"] / ex - 1
            rows.append({
                "symbol": sym, "side": pos["side"], "entry_time": pos["time"],
                "exit_time": b.time, "entry": pos["entry"], "exit": ex,
                "return_pct": ret * 100, "bars": i - pos["i"],
            })
            pos = None
        if pos is None and (buy or sell):
            pos = {"side": "LONG" if buy else "SHORT", "entry": float(b.close), "time": b.time, "i": i}
    if pos:
        b = d.iloc[-1]
        ex = float(b.close)
        ret = ex / pos["entry"] - 1 if pos["side"] == "LONG" else pos["entry"] / ex - 1
        rows.append({
            "symbol": sym, "side": pos["side"], "entry_time": pos["time"],
            "exit_time": b.time, "entry": pos["entry"], "exit": ex,
            "return_pct": ret * 100, "bars": len(d) - 1 - pos["i"],
        })
    return pd.DataFrame(rows)


def stat(x, symbol, period):
    if len(x) == 0:
        return {"symbol": symbol, "period": period, "trades": 0, "wins": 0, "losses": 0,
                "winrate": 0, "sum_return_pct": 0, "avg_signal_pct": 0,
                "avg_winner_pct": 0, "avg_loser_pct": 0, "best_pct": 0,
                "worst_pct": 0, "avg_hold_hours": 0}
    wins = x[x.return_pct > 0]
    losses = x[x.return_pct < 0]
    return {
        "symbol": symbol, "period": period, "trades": len(x), "wins": len(wins),
        "losses": len(losses), "winrate": round(100 * len(wins) / len(x), 2),
        "sum_return_pct": round(x.return_pct.sum(), 4),
        "avg_signal_pct": round(x.return_pct.mean(), 4),
        "avg_winner_pct": round(wins.return_pct.mean(), 4) if len(wins) else 0,
        "avg_loser_pct": round(losses.return_pct.mean(), 4) if len(losses) else 0,
        "best_pct": round(x.return_pct.max(), 4), "worst_pct": round(x.return_pct.min(), 4),
        "avg_hold_hours": round(x.bars.mean() / 2, 2),
    }


def main():
    symbols = get_universe()
    print("=" * 70)
    print("PURPLE CLOUD NEW UNIVERSE")
    print("=" * 70)
    print("Symbols to test:", len(symbols))
    details, stats, skipped = [], [], []
    for num, symbol in enumerate(symbols, 1):
        print(f"\n{'=' * 70}\n[{num}/{len(symbols)}] TESTING {symbol}\n{'=' * 70}")
        for period, start, end in PERIODS:
            try:
                x = run_symbol(symbol, start, end)
                x["period"] = period
                details.append(x)
                result = stat(x, symbol, period)
                stats.append(result)
                print(result)
            except Exception as e:
                skipped.append({"symbol": symbol, "period": period, "error": str(e)})
                print("SKIP", symbol, period, e)
        time.sleep(0.15)

    summary = pd.DataFrame(stats)
    summary.to_csv(OUT / "pc_new_universe_6m_summary.csv", index=False)
    if len(summary):
        ranking = summary.pivot(index="symbol", columns="period", values="sum_return_pct").reset_index()
        for c in ["MAR_JUN", "JUN_SEP"]:
            if c not in ranking:
                ranking[c] = float("nan")
        ranking["both_positive"] = (ranking.MAR_JUN > 0) & (ranking.JUN_SEP > 0)
        ranking["six_month_sum_pct"] = ranking[["MAR_JUN", "JUN_SEP"]].sum(axis=1, min_count=2)
        ranking["worst_period_pct"] = ranking[["MAR_JUN", "JUN_SEP"]].min(axis=1)
        ranking = ranking.sort_values(["both_positive", "worst_period_pct", "six_month_sum_pct"], ascending=False)
    else:
        ranking = pd.DataFrame()
    ranking.to_csv(OUT / "pc_new_universe_6m_ranking.csv", index=False)
    winners = ranking[ranking["both_positive"]].copy() if len(ranking) else pd.DataFrame()
    winners.to_csv(OUT / "pc_new_universe_6m_winners.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "pc_new_universe_6m_skipped.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_csv(OUT / "pc_new_universe_6m_detail.csv", index=False)
    print("\nFINAL RANKING")
    print(ranking.head(50).to_string(index=False) if len(ranking) else "No ranking")
    print("\nBOTH PERIODS POSITIVE")
    print(winners.to_string(index=False) if len(winners) else "No new symbols profitable in BOTH periods.")
    print("\nDONE\nResults saved to:", OUT.resolve())


if __name__ == "__main__":
    main()
