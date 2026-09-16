from __future__ import annotations

from pathlib import Path
import time
import requests
import pandas as pd

from backtest.download_binance import download_klines
from backtest.purple_cloud_entry_test import purple_cloud


# ============================================================
# PURPLE CLOUD — NEW UNIVERSE 6 MONTH SCREEN
# ============================================================

TF = "30m"

PERIODS = [
    ("MAR_JUN", "2026-03-01", "2026-06-01"),
    ("JUN_SEP", "2026-06-01", "2026-09-01"),
]

OUT = Path("backtest/data")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# УЖЕ ПРОТЕСТИРОВАННЫЕ / НЕ НУЖНО ТЕСТИРОВАТЬ ПОВТОРНО
# ============================================================

EXCLUDED = {
    # Основные / ранее протестированные
    "BTCUSDT",
    "ETHUSDT",
    "ZECUSDT",

    # предыдущая корзина 35
    "XLMUSDT",
    "VETUSDT",
    "ALGOUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "GALAUSDT",
    "AXSUSDT",
    "RUNEUSDT",
    "CRVUSDT",
    "LDOUSDT",
    "STXUSDT",
    "IMXUSDT",
    "GRTUSDT",
    "JUPUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "NOTUSDT",
    "JTOUSDT",
    "PYTHUSDT",
    "WLDUSDT",
    "ARUSDT",
    "MKRUSDT",
    "COMPUSDT",
    "SNXUSDT",
    "DYDXUSDT",
    "OMUSDT",
    "ONDOUSDT",
    "SAGAUSDT",
    "STRKUSDT",
    "AEVOUSDT",
    "BLURUSDT",
    "MEMEUSDT",
    "1000SHIBUSDT",
    "1000SATSUSDT",
    "ORDIUSDT",
}


# ============================================================
# GET CURRENT BINANCE USD-M PERPETUAL UNIVERSE
# ============================================================

def get_universe():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "PurpleCloud-Backtest"}
    )
    r.raise_for_status()

    data = r.json()

    symbols = []

    for s in data.get("symbols", []):
        symbol = s.get("symbol")

        if not symbol:
            continue

        if s.get("quoteAsset") != "USDT":
            continue

        if s.get("contractType") != "PERPETUAL":
            continue

        if s.get("status") != "TRADING":
            continue

        if symbol in EXCLUDED:
            continue

        symbols.append(symbol)

    return sorted(set(symbols))


# ============================================================
# RUN ONE SYMBOL / ONE PERIOD
# ============================================================

def run_symbol(sym, start, end):

    d = download_klines(
        sym,
        TF,
        start,
        end
    ).reset_index(drop=True)

    if len(d) < 100:
        raise ValueError(
            f"not enough candles: {len(d)}"
        )

    d["time"] = pd.to_datetime(
        d["time"],
        utc=True
    )

    pc = purple_cloud(d)

    rows = []

    pos = None

    for i, b in d.iterrows():

        buy = bool(pc.iloc[i]["pc_buy"])
        sell = bool(pc.iloc[i]["pc_sell"])

        # ----------------------------------------
        # EXIT ON OPPOSITE PURPLE CLOUD SIGNAL
        # ----------------------------------------

        if pos and (
            (pos["side"] == "LONG" and sell)
            or
            (pos["side"] == "SHORT" and buy)
        ):

            ex = float(b.close)

            if pos["side"] == "LONG":
                ret = ex / pos["entry"] - 1
            else:
                ret = pos["entry"] / ex - 1

            rows.append({
                "symbol": sym,
                "side": pos["side"],

                "entry_time": pos["time"],
                "exit_time": b.time,

                "entry": pos["entry"],
                "exit": ex,

                "return_pct": ret * 100,

                "bars": i - pos["i"],
            })

            pos = None

        # ----------------------------------------
        # OPEN POSITION
        # ----------------------------------------

        if pos is None and (buy or sell):

            pos = {
                "side": "LONG" if buy else "SHORT",
                "entry": float(b.close),
                "time": b.time,
                "i": i,
            }

    # --------------------------------------------
    # FORCE CLOSE AT END OF TEST PERIOD
    # --------------------------------------------

    if pos:

        b = d.iloc[-1]

        ex = float(b.close)

        if pos["side"] == "LONG":
            ret = ex / pos["entry"] - 1
        else:
            ret = pos["entry"] / ex - 1

        rows.append({
            "symbol": sym,
            "side": pos["side"],

            "entry_time": pos["time"],
            "exit_time": b.time,

            "entry": pos["entry"],
            "exit": ex,

            "return_pct": ret * 100,

            "bars": len(d) - 1 - pos["i"],
        })

    return pd.DataFrame(rows)


# ============================================================
# STATISTICS
# ============================================================

def stat(x, symbol, period):

    if len(x) == 0:
        return {
            "symbol": symbol,
            "period": period,

            "trades": 0,
            "wins": 0,
            "losses": 0,

            "winrate": 0,

            "sum_return_pct": 0,
            "avg_signal_pct": 0,

            "avg_winner_pct": 0,
            "avg_loser_pct": 0,

            "best_pct": 0,
            "worst_pct": 0,

            "avg_hold_hours": 0,
        }

    wins = x[x.return_pct > 0]
    losses = x[x.return_pct < 0]

    return {
        "symbol": symbol,
        "period": period,

        "trades": len(x),

        "wins": len(wins),
        "losses": len(losses),

        "winrate": round(
            100 * len(wins) / len(x),
            2
        ),

        "sum_return_pct": round(
            x.return_pct.sum(),
            4
        ),

        "avg_signal_pct": round(
            x.return_pct.mean(),
            4
        ),

        "avg_winner_pct": round(
            wins.return_pct.mean(),
            4
        ) if len(wins) else 0,

        "avg_loser_pct": round(
            losses.return_pct.mean(),
            4
        ) if len(losses) else 0,

        "best_pct": round(
            x.return_pct.max(),
            4
        ),

        "worst_pct": round(
            x.return_pct.min(),
            4
        ),

        # 30m = 0.5 hour
        "avg_hold_hours": round(
            x.bars.mean() / 2,
            2
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    symbols = get_universe()

    print("=" * 70)
    print("PURPLE CLOUD NEW UNIVERSE")
    print("=" * 70)

    print("Symbols to test:", len(symbols))
    print()

    details = []
    stats = []
    skipped = []

    total = len(symbols)

    for num, symbol in enumerate(symbols, 1):

        print()
        print("=" * 70)
        print(
            f"[{num}/{total}] TESTING {symbol}"
        )
        print("=" * 70)

        for period, start, end in PERIODS:

            try:

                x = run_symbol(
                    symbol,
                    start,
                    end
                )

                x["period"] = period

                details.append(x)

                result = stat(
                    x,
                    symbol,
                    period
                )

                stats.append(result)

                print(result)

            except Exception as e:

                err = {
                    "symbol": symbol,
                    "period": period,
                    "error": str(e),
                }

                skipped.append(err)

                print(
                    "SKIP",
                    symbol,
                    period,
                    e
                )

        # маленькая пауза между монетами
        time.sleep(0.15)

    # ========================================================
    # RAW SUMMARY
    # ========================================================

    summary = pd.DataFrame(stats)

    summary.to_csv(
        OUT / "pc_new_universe_6m_summary.csv",
        index=False
    )

    # ========================================================
    # RANKING
    # ========================================================

    if len(summary):

        ranking = summary.pivot(
            index="symbol",
            columns="period",
            values="sum_return_pct"
        ).reset_index()

        for c in ["MAR_JUN", "JUN_SEP"]:

            if c not in ranking:
                ranking[c] = float("nan")

        # прибыль в ОБОИХ периодах
        ranking["both_positive"] = (
            (ranking["MAR_JUN"] > 0)
            &
            (ranking["JUN_SEP"] > 0)
        )

        ranking["six_month_sum_pct"] = (
            ranking[
                ["MAR_JUN", "JUN_SEP"]
            ]
            .sum(
                axis=1,
                min_count=2
            )
        )

        # насколько хорош худший из двух периодов
        ranking["worst_period_pct"] = (
            ranking[
                ["MAR_JUN", "JUN_SEP"]
            ].min(axis=1)
        )

        ranking = ranking.sort_values(
            [
                "both_positive",
                "worst_period_pct",
                "six_month_sum_pct",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )

    else:

        ranking = pd.DataFrame()

    ranking.to_csv(
        OUT / "pc_new_universe_6m_ranking.csv",
        index=False
    )

    # ========================================================
    # ONLY PROFITABLE BOTH PERIODS
    # ========================================================

    if len(ranking):

        winners = ranking[
            ranking["both_positive"]
        ].copy()

    else:

        winners = pd.DataFrame()

    winners.to_csv(
        OUT / "pc_new_universe_6m_winners.csv",
        index=False
    )

    # ========================================================
    # SKIPPED
    # ========================================================

    pd.DataFrame(
        skipped
    ).to_csv(
        OUT / "pc_new_universe_6m_skipped.csv",
        index=False
    )

    # ========================================================
    # ALL TRADES
    # ========================================================

    if details:

        pd.concat(
            details,
            ignore_index=True
        ).to_csv(
            OUT / "pc_new_universe_6m_detail.csv",
            index=False
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    if len(ranking):

        print(
            ranking.head(50).to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("BOTH PERIODS POSITIVE")
    print("=" * 70)

    if len(winners):

        print(
            winners.to_string(
                index=False
            )
        )

    else:

        print(
            "No new symbols profitable "
            "in BOTH periods."
        )

    print()
    print("DONE")
    print(
        "Results saved to:",
        OUT.resolve()
    )


if __name__ == "__main__":
    main()
