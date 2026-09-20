from backtest.ltm_base15_v2_compare import *

V2_FULL=V2
V2_NO_PENDLE=[s for s in V2 if s!="PENDLEUSDT"]
V2_NO_SHIB=[s for s in V2 if s!="1000SHIBUSDT"]
V2_NO_BOTH=[s for s in V2 if s not in ("PENDLEUSDT","1000SHIBUSDT")]
PORTFOLIOS2={"V2_FULL":V2_FULL,"V2_NO_PENDLE":V2_NO_PENDLE,"V2_NO_SHIB":V2_NO_SHIB,"V2_NO_BOTH":V2_NO_BOTH}

def main():
 universe=sorted(set(V2_FULL));rows=[];bys=[]
 ob,on=eng.START_BALANCE,eng.NOTIONAL;eng.START_BALANCE=145.58352658;eng.NOTIONAL=60.
 try:
  for pn,st,en in PERIODS:
   raw={}
   for s in universe:
    try:
     d=eng.download_klines(s,"30m",st,en).reset_index(drop=True)
     if len(d)<500:continue
     d["time"]=pd.to_datetime(d["time"],utc=True);x=sig(d)
     raw[s]=x[["time","high","low","close","pc_buy","pc_sell"]]
    except Exception as e:print("SKIP",s,e)
   for name,symbols in PORTFOLIOS2.items():
    avail=[s for s in symbols if s in raw];data={s:raw[s] for s in avail}
    r,tr,by=eng.run(data,pn+"_"+name,avail)
    r["period"]=pn;r["portfolio"]=name;r["symbols"]=len(avail);rows.append(r)
    if len(by):bys.append(by.assign(period=pn,portfolio=name))
 finally:eng.START_BALANCE=ob;eng.NOTIONAL=on
 pd.DataFrame(rows).to_csv("backtest/data/ltm_v2_prune_summary.csv",index=False)
 if bys:pd.concat(bys,ignore_index=True).to_csv("backtest/data/ltm_v2_prune_by_symbol.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
