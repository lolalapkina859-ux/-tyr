from __future__ import annotations
import json, os, time
from pathlib import Path
from datetime import datetime, timezone

from exchange.bybit import get_klines
from backtest.purple_cloud_entry_test import purple_cloud

SYMBOLS = ['ZECUSDT','USELESSUSDT','FETUSDT','HYPEUSDT','JTOUSDT','VETUSDT','XRPUSDT','ETHUSDT','UNIUSDT','INJUSDT','SEIUSDT','1000SHIBUSDT','DYDXUSDT']
INTERVAL = '30'
NOTIONAL_USDT = float(os.getenv('PC_NOTIONAL_USDT', '60'))
LEVERAGE = int(os.getenv('PC_LEVERAGE', '20'))
PAPER = os.getenv('PC_PAPER', 'true').lower() == 'true'
STATE_PATH = Path(os.getenv('PC_STATE_PATH', '/data/purple_cloud_paper_state.json'))
SCAN_SECONDS = int(os.getenv('PC_SCAN_SECONDS', '60'))


def load_state():
    try:
        if STATE_PATH.exists(): return json.loads(STATE_PATH.read_text())
    except Exception as e: print('[PC] state read error:', e)
    return {'positions': {}, 'last_signal_candle': {}, 'events': []}


def save_state(st):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix('.tmp'); tmp.write_text(json.dumps(st, indent=2)); tmp.replace(STATE_PATH)


def emit(st, symbol, action, side, price, candle, note=''):
    ev={'ts':datetime.now(timezone.utc).isoformat(),'symbol':symbol,'action':action,'side':side,'price':price,'signal_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE,'mode':'PAPER' if PAPER else 'LIVE_DISABLED','note':note}
    st['events'].append(ev); st['events']=st['events'][-1000:]
    print('[PC EVENT]', json.dumps(ev))


def process_symbol(st, symbol):
    df=get_klines(symbol, INTERVAL, 250)
    if df is None or len(df)<60: return
    # BingX can include the currently-forming candle. Trade only on a fully closed 30m candle.
    now=pd.Timestamp.now(tz='UTC')
    closed=df[df['time'] + pd.Timedelta(minutes=30) <= now].copy()
    if len(closed)<60: return
    pc=purple_cloud(closed)
    row=pc.iloc[-1]; candle=closed.iloc[-1]['time'].isoformat(); price=float(closed.iloc[-1]['close'])
    signal='LONG' if bool(row.pc_buy) else ('SHORT' if bool(row.pc_sell) else None)
    if not signal or st['last_signal_candle'].get(symbol)==candle: return
    st['last_signal_candle'][symbol]=candle
    old=st['positions'].get(symbol)
    if old and old['side']==signal:
        emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle); return
    if old:
        pnl=NOTIONAL_USDT*((price/old['entry']-1) if old['side']=='LONG' else (old['entry']/price-1))
        emit(st,symbol,'CLOSE',old['side'],price,candle,f'paper_pnl={pnl:.4f} USDT'); st['positions'].pop(symbol,None)
    # PAPER only. Real order placement intentionally not enabled yet.
    if PAPER:
        st['positions'][symbol]={'side':signal,'entry':price,'opened_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE}
        emit(st,symbol,'OPEN',signal,price,candle)
    else:
        emit(st,symbol,'LIVE_BLOCKED',signal,price,candle,'Real BingX orders are not enabled in this safety stage')


def main():
    if not PAPER: print('[PC] WARNING: PC_PAPER=false, but LIVE orders remain intentionally blocked.')
    print(f'[PC] Purple Cloud NORMAL 30M | {len(SYMBOLS)} symbols | ${NOTIONAL_USDT} notional | {LEVERAGE}x | PAPER={PAPER}')
    while True:
        st=load_state()
        for s in SYMBOLS:
            try: process_symbol(st,s)
            except Exception as e: print('[PC]',s,'error:',e)
            save_state(st)
        time.sleep(SCAN_SECONDS)


# local import kept here so startup failures remain obvious
import pandas as pd
if __name__=='__main__': main()
