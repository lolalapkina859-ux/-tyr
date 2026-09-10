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


TOP_SYMBOLS_LIMIT = 30
TOP_SYMBOLS_REFRESH_SECONDS = 60 * 60  # обновлять TOP-30 раз в час


def signal_key(sig, candle_time):
    return f"{sig.symbol}:{sig.side}:{candle_time}:{sig.score}"


def load_symbols():
    """
    Получаем TOP-30 BingX USDT perpetual по объёму.
    Если BingX временно не отвечает — используем WATCHED_SYMBOLS из config.py.
    """

    symbols = get_top_symbols(TOP_SYMBOLS_LIMIT)

    if symbols:
        print(
            f"[BingX] Using TOP {len(symbols)} symbols"
        )
        return symbols

    print(
        "[BingX] Failed to load TOP symbols. "
        "Using WATCHED_SYMBOLS fallback."
    )

    return WATCHED_SYMBOLS


def run():

    state = load_state()

    print(
        "=== TRADE VISION 24/7 — LIQUIDITY SIGNAL BOT ==="
    )

    print(
        f"Min score: {MIN_SIGNAL_SCORE}"
    )

    print(
        f"Scan interval: {SCAN_INTERVAL_SECONDS} sec"
    )

    # Получаем TOP-30 при запуске
    symbols = load_symbols()

    print(
        f"Symbols ({len(symbols)}): {symbols}"
    )

    last_symbols_update = time.time()

    while True:

        started = datetime.now(
            timezone.utc
        )

        print(
            f"\n[{started.isoformat()}] Scan started"
        )

        # ==================================================
        # UPDATE TOP-30 ONCE PER HOUR
        # ==================================================

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
                    f"[BingX] TOP symbols updated: "
                    f"{symbols}"
                )

            else:

                print(
                    "[BingX] TOP symbols refresh failed. "
                    "Keeping previous list."
                )

            last_symbols_update = time.time()

        # ==================================================
        # SCAN
        # ==================================================

        for symbol in symbols:

            try:

                print(
                    f"\n--- {symbol} ---"
                )

                # ==============================
                # 4H DATA
                # ==============================

                df4h = get_klines(
                    symbol,
                    "240",
                    400,
                )

                time.sleep(0.20)

                # ==============================
                # 15M DATA
                # ==============================

                df15 = get_klines(
                    symbol,
                    "15",
                    500,
                )

                time.sleep(0.20)

                # ==============================
                # DATA VALIDATION
                # ==============================

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

                # ==============================
                # ANALYSIS
                # ==============================

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
                    f"score={sig.score}"
                )

                # ==============================
                # SCORE FILTER
                # ==============================

                if sig.score < MIN_SIGNAL_SCORE:

                    print(
                        f"{symbol}: signal ignored "
                        f"({sig.score} < "
                        f"{MIN_SIGNAL_SCORE})"
                    )

                    continue

                # ==============================
                # DUPLICATE PROTECTION
                # ==============================

                closed_15m_time = (
                    df15
                    .iloc[-2]["time"]
                    .isoformat()
                )

                key = signal_key(
                    sig,
                    closed_15m_time,
                )

                if already_sent(
                    state,
                    key,
                ):

                    print(
                        f"{symbol}: signal already sent"
                    )

                    continue

                # ==============================
                # TELEGRAM
                # ==============================

                sent = send_signal(
                    sig
                )

                if sent:

                    state = mark_sent(
                        state,
                        key,
                    )

                    print(
                        f"{symbol}: Telegram signal sent ✅"
                    )

                else:

                    print(
                        f"{symbol}: Telegram send failed"
                    )

            except Exception as exc:

                print(
                    f"{symbol}: ERROR {exc}"
                )

        # ==================================================
        # WAIT FOR NEXT SCAN
        # ==================================================

        print(
            f"\nScan finished. "
            f"Sleeping {SCAN_INTERVAL_SECONDS} sec..."
        )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    run()
