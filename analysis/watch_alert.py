from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from analysis.liquidity import detect_event


# =========================================================
# EARLY HTF LIQUIDITY WATCH
# =========================================================
#
# This is NOT a trade signal and is never registered in tracker.
# It only warns that important HTF liquidity has been swept/reclaimed
# and that the bot is now waiting for 15M structure confirmation.
# =========================================================

WATCH_LEVELS = {
    "PMH",
    "PML",
    "PWH",
    "PWL",
    "PDH",
    "PDL",
    "PSH",
    "PSL",
}

LEVEL_NAMES = {
    "PMH": "максимум прошлого месяца",
    "PML": "минимум прошлого месяца",
    "PWH": "максимум прошлой недели",
    "PWL": "минимум прошлой недели",
    "PDH": "максимум прошлого дня",
    "PDL": "минимум прошлого дня",
    "PSH": "предыдущий swing high",
    "PSL": "предыдущий swing low",
}

DATA_DIR = Path(
    os.getenv(
        "RAILWAY_VOLUME_MOUNT_PATH",
        "/data",
    )
)

try:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
except Exception:
    pass

WATCH_STATE_PATH = (
    DATA_DIR
    / "watch_state.json"
)


def _fmt_price(value: float) -> str:
    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:.4f}"

    return (
        f"{value:.8f}"
        .rstrip("0")
        .rstrip(".")
    )


def _load_state() -> dict:
    if not WATCH_STATE_PATH.exists():
        return {}

    try:
        data = json.loads(
            WATCH_STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

    except Exception as exc:
        print(
            f"[WATCH] state load error: {exc}"
        )

    return {}


def _save_state(state: dict) -> None:
    if len(state) > 3000:
        keys = list(
            state.keys()
        )[-2000:]

        state = {
            key: state[key]
            for key in keys
        }

    try:
        WATCH_STATE_PATH.write_text(
            json.dumps(
                state,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            f"[WATCH] state save error: {exc}"
        )


def _find_htf_level(sig) -> str | None:
    for reason in sig.reasons:
        parts = str(reason).split()

        if (
            len(parts) >= 3
            and parts[0] == "4H"
            and parts[1] in WATCH_LEVELS
            and "liquidity sweep" in str(reason)
        ):
            return parts[1]

    return None


def _structure_missing(sig) -> bool:
    return any(
        "15M structure NOT confirmed"
        in str(reason)
        for reason in sig.reasons
    )


def _event_time_iso(
    h4,
    event: dict,
) -> str:
    bar = event.get("bar")

    try:
        row = h4.loc[bar]
    except Exception:
        try:
            row = h4.iloc[int(bar)]
        except Exception:
            row = h4.iloc[-2]

    value = row["time"]

    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _build_message(
    symbol: str,
    side: str,
    level: str,
    level_price: float,
    current_price: float,
) -> str:
    icon = (
        "🟢"
        if side == "LONG"
        else "🔴"
    )

    direction = (
        "ниже"
        if side == "LONG"
        else "выше"
    )

    reclaim_text = (
        "вернулась выше уровня"
        if side == "LONG"
        else "вернулась ниже уровня"
    )

    description = LEVEL_NAMES.get(
        level,
        level,
    )

    return "\n".join(
        [
            f"👀 <b>SETUP WATCH — #{symbol}</b>",
            "",
            f"{icon} Возможный <b>{side}</b>-сценарий формируется.",
            "",
            "💧 <b>СНЯТА ОСНОВНАЯ HTF ЛИКВИДНОСТЬ</b>",
            (
                f"• Цена сняла ликвидность {direction} "
                f"<b>{level}</b> — {description}."
            ),
            (
                f"• Уровень: <b>{_fmt_price(level_price)}</b>"
            ),
            (
                f"• Current: <b>{_fmt_price(current_price)}</b>"
            ),
            (
                f"• Цена {reclaim_text}."
            ),
            "",
            "⏳ <b>ЧТО ЖДЁМ</b>",
            "• bullish/bearish BOS на 15M по направлению сценария;",
            "• подтверждение momentum / Money Flow;",
            "• FVG / Order Block для точного LIMIT-входа.",
            "",
            "⚠️ <b>СЕЙЧАС НЕ ВХОДИТЬ</b>",
            "Это раннее предупреждение, а не торговый сигнал.",
            "",
            "👁 <b>TRADE VISION 24/7</b>",
            "<i>Liquidity taken → now waiting for confirmation</i>",
        ]
    )


def maybe_send_watch(
    symbol: str,
    sig,
    h4,
    levels4: dict,
) -> bool:
    """
    Send exactly one early alert per HTF sweep candle.

    Conditions:
        - important 4H liquidity sweep/reclaim exists;
        - 15M BOS/structure is NOT confirmed yet;
        - alert for this exact sweep candle was not sent before.

    This function never writes to signal tracker/statistics.
    """

    if not _structure_missing(sig):
        return False

    level = _find_htf_level(sig)

    if level is None:
        return False

    expected_type = (
        "sweep_low"
        if sig.side == "LONG"
        else "sweep_high"
    )

    matching = [
        event
        for event in detect_event(
            h4,
            levels4,
            lookback=3,
        )
        if (
            event.get("type")
            == expected_type
            and event.get("level")
            == level
        )
    ]

    if not matching:
        return False

    event = matching[-1]

    event_time = _event_time_iso(
        h4,
        event,
    )

    key = (
        f"WATCH:{symbol}:"
        f"{sig.side}:"
        f"{level}:"
        f"{event_time}"
    )

    state = _load_state()

    if state.get(key):
        return False

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        print(
            f"[WATCH] Telegram token/chat id missing: {key}"
        )
        return False

    text = _build_message(
        symbol=symbol,
        side=sig.side,
        level=level,
        level_price=float(
            event["price"]
        ),
        current_price=float(
            sig.current_price
        ),
    )

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=15,
        )

        if not response.ok:
            print(
                "[WATCH] Telegram error:",
                response.text,
            )
            return False

        state[key] = True
        _save_state(
            state
        )

        print(
            f"[WATCH] {symbol} {sig.side} {level} sent 👀"
        )

        return True

    except Exception as exc:
        print(
            f"[WATCH] Telegram exception: {exc}"
        )
        return False
