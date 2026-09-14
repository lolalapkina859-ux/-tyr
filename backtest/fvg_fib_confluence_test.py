from __future__ import annotations

from pathlib import Path
import pandas as pd

DETAIL = Path("backtest/data/cross_asset_fvg_depth_detail.csv")
OUT = Path("backtest/data/fvg75_fib_confluence_summary.csv")

# Test whether the FVG75 entry itself lies in the 0.50-0.618 retracement
# of the most recent causal impulse leg available at signal time.
FIB_LO = 0.50
FIB_HI = 0.618
LOOKBACK = 96  # 24h on 15m


def fib_depth(side: str, entry: float, hist: pd.DataFrame) -> float | None:
    if len(hist) < 20:
        return None
    h = hist.tail(LOOKBACK)
    if side == "LONG":
        hi_i = h["high"].astype(float).idxmax()
        before = h.loc[:hi_i]
        if before.empty:
            return None
        hi = float(h.loc[hi_i, "high"])
        lo = float(before["low"].min())
        rng = hi - lo
        return (hi - entry) / rng if rng > 0 else None
    lo_i = h["low"].astype(float).idxmin()
    before = h.loc[:lo_i]
    if before.empty:
        return None
    lo = float(h.loc[lo_i, "low"])
    hi = float(before["high"].max())
    rng = hi - lo
    return (entry - lo) / rng if rng > 0 else None


def stats(df: pd.DataFrame, label: str) -> dict:
    closed = df[df.status == "CLOSED"]
    wins = closed[closed.highest_tp >= 1]
    r = float(closed.realized_r.sum()) if len(closed) else 0.0
    return {"group": label, "setups": len(df), "closed": len(closed),
            "wins": len(wins), "wr": round(100*len(wins)/len(closed),2) if len(closed) else 0,
            "total_r": round(r,4), "avg_r": round(r/len(closed),4) if len(closed) else 0}


def main():
    if not DETAIL.exists():
        raise FileNotFoundError(f"Missing {DETAIL}; run FVG depth cross-asset test first")
    d = pd.read_csv(DETAIL)
    d = d[d.depth_pct == 75].copy()
    d.signal_time = pd.to_datetime(d.signal_time, utc=True)

    # Download candles only through existing project helper, then calculate Fib causally.
    from backtest.download_binance import download_klines
    symbols = {"BTC":"BTCUSDT", "ETH":"ETHUSDT", "ZEC":"ZECUSDT"}
    d["fib_depth"] = pd.NA
    for asset, symbol in symbols.items():
        c = download_klines(symbol, "15m", "2026-03-01", "2026-09-01").copy()
        c["time"] = pd.to_datetime(c.time, utc=True)
        mask = d.asset == asset
        for i, row in d[mask].iterrows():
            hist = c[c.time <= row.signal_time]
            d.at[i, "fib_depth"] = fib_depth(str(row.side), float(row.entry), hist)

    d["fib_depth"] = pd.to_numeric(d.fib_depth, errors="coerce")
    d["fib_50_618"] = d.fib_depth.between(FIB_LO, FIB_HI, inclusive="both")

    rows=[]
    for asset in ["ALL","BTC","ETH","ZEC"]:
        x = d if asset == "ALL" else d[d.asset == asset]
        rows.append({"asset":asset, **stats(x, "FVG75_ALL")})
        rows.append({"asset":asset, **stats(x[x.fib_50_618], "FVG75_FIB_50_618")})
        rows.append({"asset":asset, **stats(x[~x.fib_50_618], "FVG75_OUTSIDE_FIB")})
    out=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT,index=False)
    d.to_csv(Path("backtest/data/fvg75_fib_confluence_detail.csv"),index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
