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


def signal_key(sig, candle_time):
    return (
        f"{sig.symbol}:"
        f"{sig.side}:"
        f"{candle_time}:"
        f"{sig.score}"
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

        if event_type == "TP_HIT":

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

        # ==================================================
        # UPDATE TOP-30
        # ==================================================

        if (
            time.time()
            - last_symbols_update
            >= TOP_SYMBOLS_REFRESH_SECONDS
        ):

            print(
                "[BingX] Refreshing "
                "TOP symbols..."
            )

            new_symbols = (
                get_top_symbols(
                    TOP_SYMBOLS_LIMIT
                )
            )

            if new_symbols:

                symbols = (
                    new_symbols
                )

                print(
                    "[BingX] "
                    f"TOP symbols updated: "
                    f"{symbols}"
                )

            else:

                print(
                    "[BingX] TOP symbols "
                    "refresh failed. "
                    "Keeping previous list."
                )

            last_symbols_update = (
                time.time()
            )

        # ==================================================
        # SCAN SYMBOLS
        # ==================================================

        for symbol in symbols:

            try:

                print(
                    f"\n--- {symbol} ---"
                )

                # ==========================================
                # 4H
                # ==========================================

                df4h = get_klines(
                    symbol,
                    "240",
                    400,
                )

                time.sleep(
                    0.20
                )

                # ==========================================
                # 15M
                # ==========================================

                df15 = get_klines(
                    symbol,
                    "15",
                    500,
                )

                time.sleep(
                    0.20
                )

                # ==========================================
                # DATA CHECK
                # ==========================================

                if df4h is None:

                    print(
                        f"{symbol}: "
                        "4H data unavailable"
                    )

                    continue

                if df15 is None:

                    print(
                        f"{symbol}: "
                        "15M data unavailable"
                    )

                    continue

                if len(df4h) < 100:

                    print(
                        f"{symbol}: "
                        f"not enough 4H data "
                        f"({len(df4h)})"
                    )

                    continue

                if len(df15) < 150:

                    print(
                        f"{symbol}: "
                        f"not enough 15M data "
                        f"({len(df15)})"
                    )

                    continue

                # ==========================================
                # TRACK EXISTING SIGNALS
                # ==========================================

                tracker_events = (
                    update_symbol(
                        symbol,
                        df15,
                    )
                )

                if tracker_events:

                    print_tracker_events(
                        symbol,
                        tracker_events,
                    )

                # ==========================================
                # ANALYSIS
                # ==========================================

                sig = analyze(
                    symbol,
                    df4h,
                    df15,
                )

                if sig is None:

                    print(
                        f"{symbol}: "
                        "no setup"
                    )

                    continue

                print(
                    f"{symbol}: "
                    f"{sig.side} "
                    f"score={sig.score}"
                )

                # ==========================================
                # SCORE FILTER
                # ==========================================

                if (
                    sig.score
                    < MIN_SIGNAL_SCORE
                ):

                    print(
                        f"{symbol}: "
                        "signal ignored "
                        f"({sig.score} < "
                        f"{MIN_SIGNAL_SCORE})"
                    )

                    continue

                # ==========================================
                # SIGNAL ID
                # ==========================================

                closed_15m_time = (
                    df15
                    .iloc[-2]["time"]
                    .isoformat()
                )

                key = signal_key(
                    sig,
                    closed_15m_time,
                )

                # ==========================================
                # DUPLICATE CHECK
                # ==========================================

                if already_sent(
                    state,
                    key,
                ):

                    print(
                        f"{symbol}: "
                        "signal already sent"
                    )

                    continue

                # ==========================================
                # TELEGRAM
                # ==========================================

                sent = send_signal(
                    sig
                )

                if sent:

                    # --------------------------------------
                    # SAVE TO STATISTICS
                    # --------------------------------------

                    register_signal(
                        sig,
                        key,
                        closed_15m_time,
                    )

                    # --------------------------------------
                    # MARK TELEGRAM SENT
                    # --------------------------------------

                    state = mark_sent(
                        state,
                        key,
                    )

                    print(
                        f"{symbol}: "
                        "Telegram signal sent ✅"
                    )

                    print(
                        f"[TRACKER] "
                        f"{symbol} signal "
                        "registered 📊"
                    )

                else:

                    print(
                        f"{symbol}: "
                        "Telegram send failed"
                    )

            except Exception as exc:

                print(
                    f"{symbol}: "
                    f"ERROR {exc}"
                )

        # ==================================================
        # STATISTICS
        # ==================================================

        try:

            summary = (
                get_summary()
            )

            print(
                "\n=== SIGNAL TRACKER ==="
            )

            print(
                f"Total: "
                f"{summary['total']}"
            )

            print(
                f"Open: "
                f"{summary['open']}"
            )

            print(
                f"Closed: "
                f"{summary['closed']}"
            )

            print(
                f"TP1: "
                f"{summary['tp1']}"
            )

            print(
                f"TP2: "
                f"{summary['tp2']}"
            )

            print(
                f"TP3: "
                f"{summary['tp3']}"
            )

            print(
                f"TP4: "
                f"{summary['tp4']}"
            )

            print(
                f"Losses: "
                f"{summary['losses']}"
            )

        except Exception as exc:

            print(
                "[TRACKER] "
                f"Summary error: {exc}"
            )

        # ==================================================
        # WAIT
        # ==================================================

        print(
            f"\nScan finished. "
            f"Sleeping "
            f"{SCAN_INTERVAL_SECONDS} sec..."
        )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    run()
