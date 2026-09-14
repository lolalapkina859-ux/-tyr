from __future__ import annotations

import argparse
import pandas as pd
from backtest.download_binance import download_klines

TP_WEIGHTS=(0.25,0.25,0.25,0.25)

def rr(side, entry, sl, target):
    risk=abs(entry-sl)
    if risk<=0:return 0.0
    return max(0.0, ((target-entry) if side=='LONG' else (entry-target))/risk)

def replay(row, bars, mult):
    side=str(row.side); entry=float(row.entry); old_sl=float(row.sl)
    risk=abs(entry-old_sl)
    sl=entry-risk*mult if side=='LONG' else entry+risk*mult
    targets=[float(row[c]) for c in ('tp1','tp2','tp3','tp4') if c in row.index and pd.notna(row[c])]
    if not targets:return ('NO_TARGET',0,0.0)
    t=pd.to_datetime(row.entry_time,utc=True,errors='coerce')
    if pd.isna(t):return ('NO_FILL',0,0.0)
    x=bars[bars.time>=t].head(672) # max 7 days after fill
    highest=0
    first=True
    for b in x.itertuples():
        stop=entry if highest>=1 else sl
        hit_sl=(b.low<=stop) if side=='LONG' else (b.high>=stop)
        hit_tp=sum(1 for p in targets if (b.high>=p if side=='LONG' else b.low<=p))
        if first:
            first=False
            if hit_sl or hit_tp>0:return ('AMBIGUOUS',0,0.0)
            continue
        new=max(highest,hit_tp)
        if hit_sl and new>highest:return ('AMBIGUOUS',highest,0.0)
        highest=new
        if highest>=len(targets):break
        if hit_sl:break
    if highest<=0:
        return ('CLOSED',0,-1.0)
    realized=sum(TP_WEIGHTS[i]*rr(side,entry,sl,targets[i]) for i in range(min(highest,len(targets),4)))
    return ('CLOSED',highest,realized)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--symbol',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--start',default='2026-03-01T00:00:00+00:00')
    p.add_argument('--end',default='2026-09-01T00:00:00+00:00')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    df=pd.read_csv(a.baseline)
    bars=download_klines(a.symbol,'15m',a.start,a.end)
    bars['time']=pd.to_datetime(bars['time'],utc=True)
    rows=[]
    for mult in (1.00,1.15,1.25,1.35):
        vals=[]
        for _,r in df.iterrows():
            if str(r.side)!='SHORT' or str(r.status)!='CLOSED':continue
            st,tp,rv=replay(r,bars,mult); vals.append((st,tp,rv))
        closed=[v for v in vals if v[0]=='CLOSED']; wins=[v for v in closed if v[1]>=1]
        rows.append({'symbol':a.symbol,'sl_mult':mult,'closed':len(closed),'wins':len(wins),'losses':len(closed)-len(wins),'wr':100*len(wins)/len(closed) if closed else 0,'total_r':sum(v[2] for v in closed),'avg_r':sum(v[2] for v in closed)/len(closed) if closed else 0,'ambiguous':sum(v[0]=='AMBIGUOUS' for v in vals)})
    out=pd.DataFrame(rows); out.to_csv(a.out,index=False); print(out.to_string(index=False))
if __name__=='__main__':main()
