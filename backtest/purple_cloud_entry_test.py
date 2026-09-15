from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from backtest.download_binance import download_klines

START = "2026-03-01"
END = "2026-09-01"
ATR_LEN = 20
ST_FACTOR = 2.0
PERIOD = 20
ALPHA = 1.5
BPT = 0.2
SPT = 0.2
LOOKAHEAD_BARS = 12  # 3h after Trade Vision signal

ASSETS = {
    "BTC": ("BTCUSDT", Path("backtest/data/backtest_btc_ob_filtered_1_10.csv")),
    "ETH": ("ETHUSDT", Path("backtest/data/backtest_eth_ob_filtered_1_10.csv")),
    "ZEC": ("ZECUSDT", Path("backtest/data/backtest_zec_ob_filtered_1_10.csv")),
}


def production_gate(row):
    score = int(row["score"])
    return score >= (88 if str(row["side"]) == "SHORT" else 70)


def rma(s, n):
    return s.ewm(alpha=1/n, adjust=False).mean()


def atr(df, n):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def vwma(x, volume, n):
    return (x * volume).rolling(n).sum() / volume.rolling(n).sum().replace(0, np.nan)


def supertrend_direction(df, n=20, factor=2.0):
    a = atr(df, n)
    hl2 = (df.high + df.low) / 2
    upper = hl2 + factor*a
    lower = hl2 - factor*a
    fup = upper.copy(); flo = lower.copy()
    direction = pd.Series(np.nan, index=df.index)
    st = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        fup.iloc[i] = upper.iloc[i] if (upper.iloc[i] < fup.iloc[i-1] or df.close.iloc[i-1] > fup.iloc[i-1]) else fup.iloc[i-1]
        flo.iloc[i] = lower.iloc[i] if (lower.iloc[i] > flo.iloc[i-1] or df.close.iloc[i-1] < flo.iloc[i-1]) else flo.iloc[i-1]
        if pd.isna(st.iloc[i-1]):
            direction.iloc[i] = 1
        elif st.iloc[i-1] == fup.iloc[i-1]:
            direction.iloc[i] = -1 if df.close.iloc[i] > fup.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df.close.iloc[i] < flo.iloc[i] else -1
        st.iloc[i] = flo.iloc[i] if direction.iloc[i] < 0 else fup.iloc[i]
    return direction


def purple_cloud(df):
    d = df.copy().reset_index(drop=True)
    n1 = int(np.ceil(PERIOD/4)); n2 = int(np.ceil(PERIOD/2))
    x2 = atr(d, PERIOD) * ALPHA
    xh = d.close + x2; xl = d.close - x2
    hl2 = (d.high+d.low)/2
    a1 = vwma(hl2*d.volume, d.volume, n1) / vwma(d.volume, d.volume, n1)
    a2 = vwma(hl2*d.volume, d.volume, n2) / vwma(d.volume, d.volume, n2)
    a3 = 2*a1-a2
    a4 = vwma(a3, d.volume, PERIOD)
    b1 = rma(d.close, PERIOD)
    a5 = 2*a4*b1/(a4+b1)
    buy = (a5 <= xl) & (d.close > b1*(1+BPT*0.01))
    sell = (a5 >= xh) & (d.close < b1*(1-SPT*0.01))
    xs = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        xs[i] = 1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
    changed = pd.Series(xs).ne(pd.Series(xs).shift(1))
    direction = supertrend_direction(d, ATR_LEN, ST_FACTOR)
    d["pc_buy"] = buy & changed
    d["pc_sell"] = sell & changed
    d["pc_strong_buy"] = d.pc_buy & (direction < 0)
    d["pc_strong_sell"] = d.pc_sell & (direction > 0)
    return d


def summarize(df, label):
    c = df[(df.status == "CLOSED") & df[label]].copy()
    wins = int((c.realized_r > 0).sum())
    total = float(c.realized_r.sum())
    return {"filter": label, "closed":len(c), "wins":wins, "losses":len(c)-wins,
            "winrate":round(100*wins/len(c),2) if len(c) else 0,
            "total_r":round(total,4), "avg_r":round(total/len(c),4) if len(c) else 0}


def run_asset(asset, symbol, path):
    candles = download_klines(symbol, "15m", START, END).copy()
    candles.time = pd.to_datetime(candles.time, utc=True)
    pc = purple_cloud(candles)
    sig = pd.read_csv(path)
    sig.signal_time = pd.to_datetime(sig.signal_time, utc=True, errors="coerce")
    sig.realized_r = pd.to_numeric(sig.realized_r, errors="coerce")
    sig.score = pd.to_numeric(sig.score, errors="coerce")
    sig = sig[sig.apply(production_gate, axis=1)].copy()
    normal=[]; strong=[]
    for _, r in sig.iterrows():
        idx = candles.index[candles.time <= r.signal_time]
        if len(idx)==0:
            normal.append(False); strong.append(False); continue
        i=int(idx[-1]); w=pc.iloc[i:min(i+LOOKAHEAD_BARS+1,len(pc))]
        if r.side == "LONG":
            normal.append(bool(w.pc_buy.any())); strong.append(bool(w.pc_strong_buy.any()))
        else:
            normal.append(bool(w.pc_sell.any())); strong.append(bool(w.pc_strong_sell.any()))
    sig["PC_NORMAL"] = normal; sig["PC_STRONG"] = strong; sig["asset"] = asset
    return sig


def main():
    all_df=[]
    for asset,(symbol,path) in ASSETS.items(): all_df.append(run_asset(asset,symbol,path))
    d=pd.concat(all_df,ignore_index=True)
    rows=[]
    for asset in ["BTC","ETH","ZEC","ALL"]:
        x=d if asset=="ALL" else d[d.asset==asset]
        base=x[x.status=="CLOSED"]
        wins=int((base.realized_r>0).sum()); total=float(base.realized_r.sum())
        rows.append({"asset":asset,"filter":"BASELINE","closed":len(base),"wins":wins,"losses":len(base)-wins,"winrate":round(100*wins/len(base),2) if len(base) else 0,"total_r":round(total,4),"avg_r":round(total/len(base),4) if len(base) else 0})
        for label in ["PC_NORMAL","PC_STRONG"]:
            z=summarize(x,label); z["asset"]=asset; rows.append(z)
    out=Path("backtest/data"); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"purple_cloud_entry_summary.csv",index=False)
    d.to_csv(out/"purple_cloud_entry_detail.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__": main()
