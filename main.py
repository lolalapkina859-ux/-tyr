import time
from datetime import datetime, timezone

from config import WATCHED_SYMBOLS, SCAN_INTERVAL_SECONDS, MIN_SIGNAL_SCORE
from exchange.bybit import get_klines
from analysis.engine import analyze
from telegram.notifier import send_signal
from state.store import load_state, already_sent, mark_sent

def signal_key(sig, candle_time):
    return f"{sig.symbol}:{sig.side}:{candle_time}:{sig.score}"

def run():
    state = load_state()
    print("=== TRADE VISION 24/7 — LIQUIDITY SIGNAL BOT ===")
    print(f"Symbols: {WATCHED_SYMBOLS}")
    print(f"Min score: {MIN_SIGNAL_SCORE}")

    while True:
        started = datetime.now(timezone.utc)
        print(f"\n[{started.isoformat()}] Scan started")

        for symbol in WATCHED_SYMBOLS:
            try:
                df4h = get_klines(symbol, "240", 400)
                time.sleep(0.15)
                df15 = get_klines(symbol, "15", 500)
                time.sleep(0.15)

                if df4h is None or df15 is None or len(df4h) < 100 or len(df15) < 150:
                    print(f"{symbol}: not enough data")
                    continue

                sig = analyze(symbol, df4h, df15)
                if sig is None:
                    print(f"{symbol}: no setup")
                    continue

                print(f"{symbol}: {sig.side} score={sig.score}")

                if sig.score < MIN_SIGNAL_SCORE:
                    continue

                closed_15m_time = df15.iloc[-2]["time"].isoformat()
                key = signal_key(sig, closed_15m_time)

                if already_sent(state, key):
                    continue

                if send_signal(sig):
                    state = mark_sent(state, key)

            except Exception as exc:
                print(f"{symbol}: ERROR {exc}")

        time.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    run()
