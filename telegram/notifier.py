import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from analysis.engine import Signal


# =========================================================
# PRICE FORMAT
# =========================================================

def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"

    if x >= 1:
        return f"{x:.4f}"

    return f"{x:.8f}".rstrip("0").rstrip(".")


# =========================================================
# LEVEL NAMES
# =========================================================

LEVEL_NAMES = {
    "PMH": "максимум прошлого месяца",
    "PML": "минимум прошлого месяца",

    "PWH": "максимум прошлой недели",
    "PWL": "минимум прошлой недели",

    "PDH": "максимум прошлого дня",
    "PDL": "минимум прошлого дня",

    "PSH": "предыдущий swing high",
    "PSL": "предыдущий swing low",

    "ASIAH": "максимум азиатской сессии",
    "ASIAL": "минимум азиатской сессии",

    "LONH": "максимум лондонской сессии",
    "LONL": "минимум лондонской сессии",

    "NYH": "максимум New York",
    "NYL": "минимум New York",
}


def level_description(level: str) -> str:
    return LEVEL_NAMES.get(level, level)


# =========================================================
# SIGNAL STATUS
# =========================================================

def signal_status(score: int) -> str:

    if score >= 85:
        return "🔥 СИЛЬНЫЙ СИГНАЛ"

    if score >= 70:
        return "🟢 ХОРОШИЙ СИГНАЛ"

    if score >= 50:
        return "🟡 WATCH"

    return "⚪ СЛАБЫЙ СЕТАП"


# =========================================================
# REASON TRANSLATION
# =========================================================

def translate_reason(reason: str) -> str:

    text = reason

    replacements = {
        "liquidity sweep":
            "снятие ликвидности",

        "WaveTrend oversold":
            "WaveTrend в перепроданности",

        "WaveTrend overbought":
            "WaveTrend в перекупленности",

        "WaveTrend bullish cross":
            "бычий разворот WaveTrend",

        "WaveTrend bearish cross":
            "медвежий разворот WaveTrend",

        "momentum turning bullish":
            "momentum разворачивается вверх",

        "momentum turning bearish":
            "momentum разворачивается вниз",

        "Money Flow rising":
            "Money Flow растёт",

        "Money Flow falling":
            "Money Flow снижается",

        "bullish BOS / structure shift":
            "бычий BOS / смена структуры",

        "bearish BOS / structure shift":
            "медвежий BOS / смена структуры",

        "confirmation still weak":
            "подтверждение пока слабое",

        "TP1 R:R":
            "R:R до TP1",

        "TP2 R:R":
            "R:R до TP2",
    }

    for eng, ru in replacements.items():
        text = text.replace(
            eng,
            ru
        )

    return text


# =========================================================
# FIND REASONS
# =========================================================

def find_reason(
    sig: Signal,
    timeframe: str,
    keyword: str
) -> str | None:

    for reason in sig.reasons:

        lower = reason.lower()

        if (
            timeframe.lower() in lower
            and keyword.lower() in lower
        ):
            return reason

    return None


def find_liquidity_level(
    sig: Signal,
    timeframe: str
) -> str | None:

    for reason in sig.reasons:

        if (
            timeframe.lower()
            not in reason.lower()
        ):
            continue

        if "liquidity sweep" not in reason.lower():
            continue

        parts = reason.split()

        if len(parts) >= 2:
            return parts[1]

    return None


# =========================================================
# 4H CONTEXT
# =========================================================

def build_4h_context(
    sig: Signal
) -> list[str]:

    lines = []

    level = find_liquidity_level(
        sig,
        "4H"
    )

    reasons_text = (
        " ".join(sig.reasons)
        .lower()
    )

    if level:

        desc = level_description(
            level
        )

        if sig.side == "LONG":

            lines.append(
                f"Цена сняла ликвидность ниже "
                f"<b>{level}</b> — {desc} — "
                f"и вернулась обратно выше уровня."
            )

            lines.append(
                "Такое поведение больше похоже "
                "на liquidity sweep, чем на "
                "устойчивое продолжение движения вниз."
            )

        else:

            lines.append(
                f"Цена сняла ликвидность выше "
                f"<b>{level}</b> — {desc} — "
                f"и вернулась обратно под уровень."
            )

            lines.append(
                "Такое поведение больше похоже "
                "на liquidity sweep, чем на "
                "устойчивое продолжение движения вверх."
            )

    if (
        "4h wavetrend oversold"
        in reasons_text
    ):
        lines.append(
            "WaveTrend находится в зоне "
            "перепроданности — продавцы могут "
            "начинать терять давление."
        )

    if (
        "4h wavetrend overbought"
        in reasons_text
    ):
        lines.append(
            "WaveTrend находится в зоне "
            "перекупленности — покупательский "
            "импульс может ослабевать."
        )

    if (
        "4h momentum turning bullish"
        in reasons_text
    ):
        lines.append(
            "Momentum на 4H начинает "
            "разворачиваться вверх."
        )

    if (
        "4h momentum turning bearish"
        in reasons_text
    ):
        lines.append(
            "Momentum на 4H начинает "
            "разворачиваться вниз."
        )

    if (
        "4h money flow rising"
        in reasons_text
    ):
        lines.append(
            "Money Flow на 4H растёт — "
            "покупательское давление усиливается."
        )

    if (
        "4h money flow falling"
        in reasons_text
    ):
        lines.append(
            "Money Flow на 4H снижается — "
            "покупательское давление ослабевает."
        )

    if not lines:

        lines.append(
            "На 4H есть реакция цены "
            "на ключевую ликвидность."
        )

    return lines


# =========================================================
# 15M CONFIRMATION
# =========================================================

def build_15m_confirmation(
    sig: Signal
) -> list[str]:

    lines = []

    level = find_liquidity_level(
        sig,
        "15M"
    )

    reasons_text = (
        " ".join(sig.reasons)
        .lower()
    )

    if level:

        desc = level_description(
            level
        )

        if sig.side == "LONG":

            lines.append(
                f"На 15M снята локальная "
                f"ликвидность ниже "
                f"<b>{level}</b> — {desc}."
            )

        else:

            lines.append(
                f"На 15M снята локальная "
                f"ликвидность выше "
                f"<b>{level}</b> — {desc}."
            )

    if (
        "15m bullish bos"
        in reasons_text
    ):
        lines.append(
            "Есть bullish BOS / смена структуры — "
            "рынок начинает подтверждать LONG."
        )

    if (
        "15m bearish bos"
        in reasons_text
    ):
        lines.append(
            "Есть bearish BOS / смена структуры — "
            "рынок начинает подтверждать SHORT."
        )

    if (
        "15m wavetrend bullish cross"
        in reasons_text
    ):
        lines.append(
            "WaveTrend дал бычий cross."
        )

    elif (
        "15m momentum turning bullish"
        in reasons_text
    ):
        lines.append(
            "Momentum разворачивается вверх."
        )

    if (
        "15m wavetrend bearish cross"
        in reasons_text
    ):
        lines.append(
            "WaveTrend дал медвежий cross."
        )

    elif (
        "15m momentum turning bearish"
        in reasons_text
    ):
        lines.append(
            "Momentum разворачивается вниз."
        )

    if (
        "15m money flow rising"
        in reasons_text
    ):
        lines.append(
            "Money Flow растёт — "
            "подтверждается давление покупателей."
        )

    if (
        "15m money flow falling"
        in reasons_text
    ):
        lines.append(
            "Money Flow снижается — "
            "покупатели теряют силу."
        )

    if not lines:

        lines.append(
            "На 15M есть подтверждение "
            "реакции после снятия ликвидности."
        )

    return lines


# =========================================================
# FINAL CONCLUSION
# =========================================================

def build_conclusion(
    sig: Signal
) -> list[str]:

    lines = []

    has_4h_liquidity = (
        find_liquidity_level(
            sig,
            "4H"
        )
        is not None
    )

    has_15m_liquidity = (
        find_liquidity_level(
            sig,
            "15M"
        )
        is not None
    )

    reasons_text = (
        " ".join(sig.reasons)
        .lower()
    )

    structure_confirmed = (
        "15m bullish bos"
        in reasons_text
        or
        "15m bearish bos"
        in reasons_text
    )

    if (
        has_4h_liquidity
        and has_15m_liquidity
        and structure_confirmed
    ):

        lines.append(
            "HTF liquidity и LTF confirmation "
            "совпали — это усиливает сценарий."
        )

    elif (
        has_4h_liquidity
        and structure_confirmed
    ):

        lines.append(
            "4H liquidity подтверждается "
            "сменой структуры на 15M."
        )

    elif has_15m_liquidity:

        lines.append(
            "Сигнал сформирован после "
            "локального sweep и подтверждения "
            "на 15M."
        )

    if sig.side == "LONG":

        lines.append(
            "Пока структура удерживается выше "
            "уровня отмены сценария, "
            "приоритет остаётся за LONG."
        )

    else:

        lines.append(
            "Пока структура удерживается ниже "
            "уровня отмены сценария, "
            "приоритет остаётся за SHORT."
        )

    return lines


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

def build_message(
    sig: Signal
) -> str:

    icon = (
        "🟢"
        if sig.side == "LONG"
        else "🔴"
    )

    status = signal_status(
        sig.score
    )

    risk = abs(
        sig.entry - sig.sl
    )

    lines = [
        f"{icon} <b>{sig.side} — #{sig.symbol}</b>",
        "",
        status,
        f"⭐ <b>Рейтинг: {sig.score}/100</b>",
        "",
    ]

    # =====================================================
    # 4H CONTEXT
    # =====================================================

    lines.append(
        "📊 <b>КОНТЕКСТ 4H</b>"
    )

    for text in build_4h_context(
        sig
    ):
        lines.append(
            f"• {text}"
        )

    # =====================================================
    # 15M CONFIRMATION
    # =====================================================

    lines += [
        "",
        "⚡ <b>ПОДТВЕРЖДЕНИЕ 15M</b>",
    ]

    for text in build_15m_confirmation(
        sig
    ):
        lines.append(
            f"• {text}"
        )

    # =====================================================
    # PLAN
    # =====================================================

    lines += [
        "",
        "📍 <b>ТОРГОВЫЙ ПЛАН</b>",
        f"Entry: <b>{fmt_price(sig.entry)}</b>",
        f"SL: <b>{fmt_price(sig.sl)}</b>",
        "",
    ]

    # =====================================================
    # TARGETS + R:R
    # =====================================================

    for i, (
        price,
        name
    ) in enumerate(
        sig.targets,
        1
    ):

        price = float(
            price
        )

        reward = abs(
            price - sig.entry
        )

        rr = (
            reward / risk
            if risk > 0
            else 0
        )

        description = (
            level_description(
                name
            )
        )

        lines.append(
            f"🎯 <b>TP{i}:</b> "
            f"{fmt_price(price)} "
            f"<i>({name})</i>"
        )

        lines.append(
            f"   ↳ R:R <b>1:{rr:.2f}</b> "
            f"• {description}"
        )

    # =====================================================
    # CONCLUSION
    # =====================================================

    lines += [
        "",
        "🧠 <b>ПОЧЕМУ ЭТОТ СИГНАЛ</b>",
    ]

    for text in build_conclusion(
        sig
    ):
        lines.append(
            f"• {text}"
        )

    # =====================================================
    # INVALIDATION
    # =====================================================

    invalid_price = fmt_price(
        sig.sl
    )

    if sig.side == "LONG":

        invalid_text = (
            f"Закрепление 15M ниже "
            f"<b>{invalid_price}</b> "
            f"ломает текущий LONG-сценарий."
        )

    else:

        invalid_text = (
            f"Закрепление 15M выше "
            f"<b>{invalid_price}</b> "
            f"ломает текущий SHORT-сценарий."
        )

    lines += [
        "",
        "❌ <b>ОТМЕНА СЦЕНАРИЯ</b>",
        invalid_text,
        "",
        "👁 <b>TRADE VISION 24/7</b>",
        "<i>Liquidity → Reaction → Confirmation</i>",
        "",
        "⚠️ Технический сценарий. Контролируй риск.",
    ]

    return "\n".join(
        lines
    )


# =========================================================
# SEND TELEGRAM
# =========================================================

def send_signal(
    sig: Signal
) -> bool:

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):

        print(
            "[Telegram] token/chat id missing"
        )

        print(
            build_message(sig)
        )

        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": build_message(sig),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        r = requests.post(
            url,
            json=payload,
            timeout=15,
        )

        if not r.ok:

            print(
                "[Telegram]",
                r.text,
            )

        return r.ok

    except Exception as exc:

        print(
            "[Telegram]",
            exc,
        )

        return False
