import time
from datetime import datetime, timezone

from config import (
    WATCHED_SYMBOLS,
    SCAN_INTERVAL_SECONDS,
    MIN_SIGNAL_SCORE,
)

from exchange.bybit import get_klines, get_top_symbols
from analysis.engine_hybrid import analyze
from telegram.notifier import send_signal
from telegram.entry_notifier import send_entry_filled
from state.store import load_state, already_sent, mark_sent
from state.tracker import (
    register_signal,
    update_symbol,
    get_summary,
    build_scenario_key,
    has_active_scenario,
)

TOP_SYMBOLS_LIMIT = 50
TOP_SYMBOLS_REFRESH_SECONDS = 60 * 60


def signal_key(sig):
    zone_low = round(float(sig.zone_low), 8)
    zone_high = round(float(sig.zone_high), 8)
    entry = round(float(sig.entry), 8)
    return f"{sig.symbol}:{sig.side}:{sig.entry_type}:{zone_low}:{zone_high}:{entry}"


def load_symbols():
    symbols = get_top_symbols(TOP_SYMBOLS_LIMIT)
    if symbols:
        print(f"[BingX] Using TOP {len(symbols)} symbols")
        return symbols
    print("[BingX] Failed to load TOP symbols. Using WATCHED_SYMBOLS fallback.")
    return WATCHED_SYMBOLS


def print_tracker_events(symbol, events):
    for event in events:
        event_type = event.get("type")
        if event_type == "ENTRY_FILLED":
            print(f"[TRACKER] {symbol} ENTRY FILLED ✅ at {event['price']}")
        elif event_type == "TP_HIT":
            print(f"[TRACKER] {symbol} TP{event['tp']} HIT ✅ R={event['rr']:.2f}")
        elif event_type == "SL_HIT":
            print(f"[TRACKER] {symbol} SL HIT ❌ Result=-1R")
        elif event_type == "BE_HIT":
            print(f"[TRACKER] {symbol} BREAKEVEN 🟡 after TP{event.get('highest_tp', 1)} Result=0R")
        elif event_type == "EXPIRED":
            print(f"[TRACKER] {symbol} setup EXPIRED ⏳")
        elif event_type == "AMBIGUOUS":
            print(f"[TRACKER] {symbol} AMBIGUOUS ⚠️")
        elif event_type == "CLOSED_TP":
            print(f"[TRACKER] {symbol} CLOSED ON FINAL TP ✅ R={event['result_r']:.2f}")


def run():
    state = load_state()
    print("=== TRADE VISION 24/7 — LIQUIDITY SIGNAL BOT ===")
    print(f"Min score: {MIN_SIGNAL_SCORE}")
    print(f"Scan interval: {SCAN_INTERVAL_SECONDS} sec")
    symbols = load_symbols()
    print(f"Symbols ({len(symbols)}): {symbols}")
    last_symbols_update = time.time()
    scan_number = 0

    while True:
        scan_number += 1
        started = datetime.now(timezone.utc)
        print(f"\n[{started.isoformat()}] Scan #{scan_number} started")

        if time.time() - last_symbols_update >= TOP_SYMBOLS_REFRESH_SECONDS:
            print("[BingX] Refreshing TOP symbols...")
            new_symbols = get_top_symbols(TOP_SYMBOLS_LIMIT)
            if new_symbols:
                symbols = new_symbols
                print(f"[BingX] TOP symbols updated: {symbols}")
            else:
                print("[BingX] TOP symbols refresh failed. Keeping previous list.")
            last_symbols_update = time.time()

        for symbol in symbols:
            try:
                print(f"\n--- {symbol} ---")
                df4h = get_klines(symbol, "240", 400)
                time.sleep(0.20)
                df15 = get_klines(symbol, "15", 500)
                time.sleep(0.20)

                if df4h is None:
                    print(f"{symbol}: 4H data unavailable")
                    continue
                if df15 is None:
                    print(f"{symbol}: 15M data unavailable")
                    continue
                if len(df4h) < 100:
                    print(f"{symbol}: not enough 4H data ({len(df4h)})")
                    continue
                if len(df15) < 150:
                    print(f"{symbol}: not enough 15M data ({len(df15)})")
                    continue

                tracker_events = update_symbol(symbol, df15)
                if tracker_events:
                    print_tracker_events(symbol, tracker_events)
                    current_price = float(df15.iloc[-1]["close"])
                    for event in tracker_events:
                        if event.get("type") == "ENTRY_FILLED":
                            sent_entry = send_entry_filled(event, current_price)
                            print(
                                f"[Telegram ENTRY] {symbol} "
                                + ("sent ✅" if sent_entry else "send failed")
                            )

                sig = analyze(symbol, df4h, df15)
                if sig is None:
                    print(f"{symbol}: no setup")
                    continue

                print(
                    f"{symbol}: {sig.side} score={sig.score} "
                    f"status={sig.entry_status} entry_type={sig.entry_type}"
                )

                if sig.score < MIN_SIGNAL_SCORE:
                    print(f"{symbol}: signal ignored ({sig.score} < {MIN_SIGNAL_SCORE})")
                    continue

                scenario = build_scenario_key(sig.symbol, sig.side, sig.reasons)
                print(f"{symbol}: scenario_id={scenario}")
                if has_active_scenario(scenario):
                    print(f"{symbol}: active scenario skipped ♻️ ({scenario})")
                    continue

                key = signal_key(sig)
                print(f"{symbol}: setup_id={key}")
                if already_sent(state, key):
                    print(f"{symbol}: duplicate setup skipped ♻️")
                    continue

                sent = send_signal(sig)
                if sent:
                    closed_15m_time = df15.iloc[-2]["time"].isoformat()
                    live_row = df15.iloc[-1]
                    live_snapshot = {
                        "time": live_row["time"].isoformat(),
                        "high": float(live_row["high"]),
                        "low": float(live_row["low"]),
                        "close": float(live_row["close"]),
                    }
                    register_signal(sig, key, closed_15m_time, live_snapshot)
                    state = mark_sent(state, key)
                    print(f"{symbol}: Telegram signal sent ✅")
                    print(f"[TRACKER] {symbol} signal registered 📊")
                else:
                    print(f"{symbol}: Telegram send failed")

            except Exception as exc:
                print(f"{symbol}: ERROR {exc}")

        try:
            summary = get_summary()
            print("\n=== SIGNAL TRACKER V2 ===")
            print(f"Total: {summary['total']}")
            print(f"Pending: {summary['pending']}")
            print(f"Filled: {summary['filled']}")
            print(f"Open: {summary['open']}")
            print(f"Closed: {summary['closed']}")
            print(f"Expired: {summary['expired']}")
            print(f"Ambiguous: {summary['ambiguous']}")
            print(f"Duplicates: {summary.get('duplicates', 0)}")
            print(f"Wins (TP1+): {summary['wins']}")
            print(f"Losses: {summary['losses']}")
            print(f"BE after TP1: {summary.get('breakeven', 0)}")
            print(f"WinRate: {summary['win_rate']:.1f}%")
            print(f"TP1: {summary['tp1']}")
            print(f"TP2: {summary['tp2']}")
            print(f"TP3: {summary['tp3']}")
            print(f"TP4: {summary['tp4']}")

            pending_list = summary.get("pending_signals", [])
            if pending_list:
                print("\n⏳ PENDING")
                for item in pending_list:
                    print(f"{item['symbol']} {item['side']} | LIMIT {item['entry']} | ⭐{item['score']}")

            open_list = summary.get("open_signals", [])
            if open_list:
                print("\n🟢 OPEN")
                for item in open_list:
                    tp_text = f"TP{item['highest_tp']} ✅" if item["highest_tp"] > 0 else "TP1 ⏳"
                    print(
                        f"{item['symbol']} {item['side']} | Entry {item['entry']} | "
                        f"{tp_text} | Best {item['max_r']:+.2f}R | ⭐{item['score']}"
                    )

            closed_list = summary.get("closed_signals", [])
            if closed_list:
                print("\n🏁 CLOSED")
                for item in closed_list:
                    result_r = item.get("result_r")
                    if result_r is not None and float(result_r) < 0:
                        result_text = f"SL ❌ {float(result_r):+.2f}R"
                    elif result_r is not None and float(result_r) == 0.0 and item["highest_tp"] >= 1:
                        result_text = "WIN ✅ TP1 + BE"
                    elif result_r is not None:
                        result_text = f"WIN ✅ {float(result_r):+.2f}R"
                    else:
                        result_text = "CLOSED"
                    print(
                        f"{item['symbol']} {item['side']} | {result_text} | "
                        f"TP{item['highest_tp']} | ⭐{item['score']}"
                    )

            expired_list = summary.get("expired_signals", [])
            if expired_list:
                print("\n⌛ EXPIRED")
                for item in expired_list:
                    print(
                        f"{item['symbol']} {item['side']} | LIMIT NOT FILLED | "
                        f"Entry {item['entry']} | ⭐{item['score']}"
                    )

            duplicate_list = summary.get("duplicate_signals", [])
            if duplicate_list:
                print("\n♻️ DUPLICATES")
                for item in duplicate_list:
                    print(
                        f"{item['symbol']} {item['side']} | IGNORED OLD COPY | "
                        f"Entry {item['entry']} | ⭐{item['score']}"
                    )

        except Exception as exc:
            print(f"[TRACKER] Summary error: {exc}")

        print(f"\nScan finished. Sleeping {SCAN_INTERVAL_SECONDS} sec...")
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
