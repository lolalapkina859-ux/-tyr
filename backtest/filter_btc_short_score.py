from __future__ import annotations
import argparse
import pandas as pd

def stats(df):
    c=df[df.status=='CLOSED']; w=c[c.highest_tp>=1]; l=c[c.highest_tp==0]
    return dict(setups=len(df),closed=len(c),wins=len(w),losses=len(l),wr=round(100*len(w)/len(c),2) if len(c) else 0,total_r=round(float(c.realized_r.sum()),4),avg_r=round(float(c.realized_r.mean()),4) if len(c) else 0)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',default='backtest/data/backtest_btc_ob_filtered_1_10.csv'); p.add_argument('--out',default='backtest_btc_short_score_sweep.csv'); a=p.parse_args()
    df=pd.read_csv(a.baseline); rows=[]
    for threshold in (70,75,80,85,90,95):
        f=df[(df.side=='LONG')|((df.side=='SHORT')&(df.score>=threshold))].copy()
        s=stats(f); s['short_min_score']=threshold
        for side in ('LONG','SHORT'):
            ss=stats(f[f.side==side]); s[f'{side.lower()}_closed']=ss['closed']; s[f'{side.lower()}_wr']=ss['wr']; s[f'{side.lower()}_total_r']=ss['total_r']; s[f'{side.lower()}_avg_r']=ss['avg_r']
        rows.append(s)
    out=pd.DataFrame(rows); out.to_csv(a.out,index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
