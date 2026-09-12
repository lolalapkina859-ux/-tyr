import time
from datetime import datetime, timezone

from config import (
    WATCHED_SYMBOLS,
    SCAN_INTERVAL_SECONDS,
    MIN_SIGNAL_SCORE,
)

from exchange.bybit import (
    get_klines,
    get_top_symbols,
)

from analysis.engine import analyze
from telegram.notifier import send_signal

from state.store import (
    load_state,
    already_sent,
    mark_sent,
)

from state.tracker import (
    register_signal,
    update_symbol,
    get_summary,
)


TOP_SYMBOLS_LIMIT = 30
TOP_SYMBOLS_REFRESH_SECONDS = 60 * 60


def signal_key(sig):
    zone_low = round(float(sig.zone_low), 8)
    zone_high = round(float(sig.zone_high), 8)
    entry = round(float(sig.entry), 8)

    return (
        f"{sig.symbol}:"
        f"{sig.side}:"
        f"{sig.entry_type}:"
        f"{zone_low}:"
        f"{zone_high}:"
        f"{entry}"
    )


def load_symbols():
    symbols = get_top_symbols(
        TOP_SYMBOLS_LIMIT
    )

    if symbols:
        print(
            f"[BingX] Using TOP "
            f"{len(symbols)} symbols"
        )
        return symbols

    print(
        "[BingX] Failed to load TOP symbols. "
        "Using WATCHED_SYMBOLS fallback."
    )

    return WATCHED_SYMBOLS


def print_tracker_events(
    symbol,
    events,
):
    for event in events:
        event_type = event.get(
            "type"
        )

        if event_type == "ENTRY_FILLED":
            print(
                f"[TRACKER] "
                f"{symbol} ENTRY FILLED ✅ "
                f"at {event['price']}"
            )

        elif event_type == "TP_HIT":
            print(
                f"[TRACKER] "
                f"{symbol} "
                f"TP{event['tp']} HIT ✅ "
                f"R={event['rr']:.2f}"
            )

        elif event_type == "SL_HIT":
            print(
                f"[TRACKER] "
                f"{symbol} "
                f"SL HIT ❌ "
                f"Result=-1R"
            )

        elif event_type == "BE_HIT":
            print(
                f"[TRACKER] "
                f"{symbol} "
                f"BREAKEVEN 🟡 "
                f"after TP{event.get('highest_tp', 1)} "
                f"Result=0R"
            )

        elif event_type == "EXPIRED":
            print(
                f"[TRACKER] "
                f"{symbol} setup EXPIRED ⏳"
            )

        elif event_type == "AMBIGUOUS":
            print(
                f"[TRACKER] "
                f"{symbol} AMBIGUOUS ⚠️"
            )

        elif event_type == "CLOSED_TP":
            print(
                f"[TRACKER] "
                f"{symbol} CLOSED ON FINAL TP ✅ "
                f"R={event['result_r']:.2f}"
            )


def run():
    state = load_state()

    print(
        "=== TRADE VISION 24/7 — "
        "LIQUIDITY SIGNAL BOT ==="
    )

    print(
        f"Min score: "
        f"{MIN_SIGNAL_SCORE}"
    )

    print(
        f"Scan interval: "
        f"{SCAN_INTERVAL_SECONDS} sec"
    )

    symbols = load_symbols()

    print(
        f"Symbols ({len(symbols)}): "
        f"{symbols}"
    )

    last_symbols_update = (
        time.time()
    )

    scan_number = 0

    while True:
        scan_number += 1

        started = datetime.now(
            timezone.utc
        )

        print(
            f"\n[{started.isoformat()}] "
            f"Scan #{scan_number} started"
        )

        if (
            time.time()
            - last_symbols_update
            >= TOP_SYMBOLS_REFRESH_SECONDS
        ):
            print(
                "[BingX] Refreshing TOP symbols..."
            )

            new_symbols = get_top_symbols(
                TOP_SYMBOLS_LIMIT
            )

            if new_symbols:
                symbols = new_symbols

                print(
                    "[BingX] "
                    f"TOP symbols updated: "
                    f"{symbols}"
                )

            else:
                print(
                    "[BingX] TOP symbols refresh failed. "
                    "Keeping previous list."
                )

            last_symbols_update = (
                time.time()
            )

        for symbol in symbols:
            try:
                print(
                    f"\n--- {symbol} ---"
                )

                df4h = get_klines(
                    symbol,
                    "240",
                    400,
                )

                time.sleep(
                    0.20
                )

                df15 = get_klines(
                    symbol,
                    "15",
                    500,
                )

                time.sleep(
                    0.20
                )

                if df4h is None:
                    print(
                        f"{symbol}: 4H data unavailable"
                    )
                    continue

                if df15 is None:
                    print(
                        f"{symbol}: 15M data unavailable"
                    )
                    continue

                if len(df4h) < 100:
                    print(
                        f"{symbol}: not enough 4H data "
                        f"({len(df4h)})"
                    )
                    continue

                if len(df15) < 150:
                    print(
                        f"{symbol}: not enough 15M data "
                        f"({len(df15)})"
                    )
                    continue

                tracker_events = update_symbol(
                    symbol,
                    df15,
                )

                if tracker_events:
                    print_tracker_events(
                        symbol,
                        tracker_events,
                    )

                sig = analyze(
                    symbol,
                    df4h,
                    df15,
                )

                if sig is None:
                    print(
                        f"{symbol}: no setup"
                    )
                    continue

                print(
                    f"{symbol}: "
                    f"{sig.side} "
                    f"score={sig.score} "
                    f"status={sig.entry_status} "
                    f"entry_type={sig.entry_type}"
                )

                if (
                    sig.score
                    < MIN_SIGNAL_SCORE
                ):
                    print(
                        f"{symbol}: signal ignored "
                        f"({sig.score} < "
                        f"{MIN_SIGNAL_SCORE})"
                    )
                    continue

                key = signal_key(
                    sig
                )

                print(
                    f"{symbol}: setup_id={key}"
                )

                if already_sent(
                    state,
                    key,
                ):
                    print(
                        f"{symbol}: duplicate setup skipped ♻️"
                    )
                    continue

                sent = send_signal(
                    sig
                )

                if sent:
                    closed_15m_time = (
                        df15
                        .iloc[-2]["time"]
                        .isoformat()
                    )

                    live_row = (
                        df15
                        .iloc[-1]
                    )

                    live_snapshot = {
                        "time": live_row["time"].isoformat(),
                        "high": float(live_row["high"]),
                        "low": float(live_row["low"]),
                        "close": float(live_row["close"]),
                    }

                    register_signal(
                        sig,
                        key,
                        closed_15m_time,
                        live_snapshot,
                    )

                    state = mark_sent(
                        state,
                        key,
                    )

                    print(
                        f"{symbol}: Telegram signal sent ✅"
                    )

                    print(
                        f"[TRACKER] "
                        f"{symbol} signal registered 📊"
                    )

                else:
                    print(
                        f"{symbol}: Telegram send failed"
                    )

            except Exception as exc:
                print(
                    f"{symbol}: ERROR {exc}"
                )

        try:
            summary = get_summary()

            print(
                "\n=== SIGNAL TRACKER V2 ==="
            )

            print(
                f"Total: {summary['total']}"
            )

            print(
                f"Pending: {summary['pending']}"
            )

            print(
                f"Filled: {summary['filled']}"
            )

            print(
                f"Open: {summary['open']}"
            )

            print(
                f"Closed: {summary['closed']}"
            )

            print(
                f"Expired: {summary['expired']}"
            )

            print(
                f"Ambiguous: {summary['ambiguous']}"
            )

            print(
                f"Wins: {summary['wins']}"
            )

            print(
                f"Losses: {summary['losses']}"
            )

            print(
                f"BreakEven: {summary.get('breakeven', 0)}"
            )

            print(
                f"WinRate: {summary['win_rate']:.1f}%"
            )

            print(
                f"TP1: {summary['tp1']}"
            )

            print(
                f"TP2: {summary['tp2']}"
            )

            print(
                f"TP3: {summary['tp3']}"
            )

            print(
                f"TP4: {summary['tp4']}"
            )

            # ==================================================
            # DETAILED TRACKER LISTS
            # ==================================================

            pending_list = summary.get(
                "pending_signals",
                [],
            )

            if pending_list:
                print("\n⏳ PENDING")

                for item in pending_list:
                    print(
                        f"{item['symbol']} "
                        f"{item['side']} | "
                        f"LIMIT {item['entry']} | "
                        f"⭐{item['score']}"
                    )

            open_list = summary.get(
                "open_signals",
                [],
            )

            if open_list:
                print("\n🟢 OPEN")

                for item in open_list:
                    if item["highest_tp"] > 0:
                        tp_text = (
                            f"TP{item['highest_tp']} ✅"
                        )
                    else:
                        tp_text = "TP1 ⏳"

                    print(
                        f"{item['symbol']} "
                        f"{item['side']} | "
                        f"Entry {item['entry']} | "
                        f"{tp_text} | "
                        f"Best {item['max_r']:+.2f}R | "
                        f"⭐{item['score']}"
                    )

            closed_list = summary.get(
                "closed_signals",
                [],
            )

            if closed_list:
                print("\n🏁 CLOSED")

                for item in closed_list:
                    result_r = item.get(
                        "result_r"
                    )

                    if (
                        result_r is not None
                        and float(result_r) < 0
                    ):
                        result_text = (
                            f"SL ❌ "
                            f"{float(result_r):+.2f}R"
                        )

                    elif (
                        result_r is not None
                        and float(result_r) == 0.0
                        and item["highest_tp"] >= 1
                    ):
                        result_text = (
                            "BE 🟡 +0.00R"
                        )

                    elif result_r is not None:
                        result_text = (
                            f"WIN ✅ "
                            f"{float(result_r):+.2f}R"
                        )

                    else:
                        result_text = "CLOSED"

                    print(
                        f"{item['symbol']} "
                        f"{item['side']} | "
                        f"{result_text} | "
                        f"TP{item['highest_tp']} | "
                        f"⭐{item['score']}"
                    )

            expired_list = summary.get(
                "expired_signals",
                [],
            )

            if expired_list:
                print("\n⌛ EXPIRED")

                for item in expired_list:
                    print(
                        f"{item['symbol']} "
                        f"{item['side']} | "
                        f"LIMIT NOT FILLED | "
                        f"Entry {item['entry']} | "
                        f"⭐{item['score']}"
                    )

        except Exception as exc:
            print(
                "[TRACKER] "
                f"Summary error: {exc}"
            )

        print(
            f"\nScan finished. Sleeping "
            f"{SCAN_INTERVAL_SECONDS} sec..."
        )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    run()
