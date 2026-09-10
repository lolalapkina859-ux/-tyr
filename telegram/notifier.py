import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from analysis.engine import Signal


def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")


def translate_reason(reason: str) -> str:
    replacements = {
        "liquidity sweep": "снятие ликвидности",
        "WaveTrend oversold": "WaveTrend в перепроданности",
        "WaveTrend overbought": "WaveTrend в перекупленности",
        "WaveTrend bullish cross": "бычий разворот WaveTrend",
        "WaveTrend bearish cross": "медвежий разворот WaveTrend",
        "momentum turning bullish": "momentum разворачивается вверх",
        "momentum turning bearish": "momentum разворачивается вниз",
        "Money Flow rising": "Money Flow растёт",
        "Money Flow falling": "Money Flow снижается",
        "bullish BOS / structure shift": "бычий BOS / смена структуры",
        "bearish BOS / structure shift": "медвежий BOS / смена структуры",
        "confirmation still weak": "подтверждение пока слабое",
        "TP1 R:R": "R:R до TP1",
        "TP2 R:R": "R:R до TP2",
    }

    text = reason

    for eng, ru in replacements.items():
        text = text.replace(eng, ru)

    return text


def signal_status(score: int) -> str:
    if score >= 85:
        return "🔥 СИЛЬНЫЙ СИГНАЛ"
    if score >= 70:
        return "🟢 ХОРОШИЙ СИГНАЛ"
    if score >= 50:
        return "🟡 WATCH"

    return "⚪ СЛАБЫЙ СЕТАП"


def build_explanation(sig: Signal) -> list[str]:

    reasons_text = " ".join(sig.reasons).lower()

    lines = []

    if sig.side == "LONG":

        if "4h" in reasons_text and "sweep" in reasons_text:
            lines.append(
                "На 4H цена сняла ликвидность снизу и вернулась обратно в диапазон."
            )

        if "oversold" in reasons_text:
            lines.append(
                "WaveTrend показывает перепроданность, поэтому давление продавцов может ослабевать."
            )

        if "money flow rising" in reasons_text:
            lines.append(
                "Money Flow начинает расти — появляется приток покупательского давления."
            )

        if (
            "bullish bos" in reasons_text
            or "structure shift" in reasons_text
        ):
            lines.append(
                "На 15M появилась бычья смена структуры, что подтверждает локальный разворот."
            )

        if "15m" in reasons_text and "sweep" in reasons_text:
            lines.append(
                "На 15M также снята локальная ликвидность, после чего цена получила реакцию вверх."
            )

        if not lines:
            lines.append(
                "LONG сформирован после реакции цены на ликвидность и подтверждения momentum."
            )

        lines.append(
            "Приоритет сохраняется вверх, пока цена не нарушит уровень отмены сценария."
        )

    else:

        if "4h" in reasons_text and "sweep" in reasons_text:
            lines.append(
                "На 4H цена сняла ликвидность сверху и вернулась обратно под уровень."
            )

        if "overbought" in reasons_text:
            lines.append(
                "WaveTrend показывает перекупленность, поэтому давление покупателей может ослабевать."
            )

        if "money flow falling" in reasons_text:
            lines.append(
                "Money Flow снижается — покупательская сила ослабевает."
            )

        if (
            "bearish bos" in reasons_text
            or "structure shift" in reasons_text
        ):
            lines.append(
                "На 15M появилась медвежья смена структуры, что подтверждает движение вниз."
            )

        if "15m" in reasons_text and "sweep" in reasons_text:
            lines.append(
                "На 15M также снята локальная ликвидность сверху и появилась реакция вниз."
            )

        if not lines:
            lines.append(
                "SHORT сформирован после реакции цены на ликвидность и подтверждения momentum."
            )

        lines.append(
            "Приоритет сохраняется вниз, пока цена не нарушит уровень отмены сценария."
        )

    return lines


def build_message(sig: Signal) -> str:

    icon = "🟢" if sig.side == "LONG" else "🔴"

    status = signal_status(sig.score)

    lines = [
        f"{icon} <b>{sig.side} — #{sig.symbol}</b>",
        "",
        f"{status}",
        f"⭐ <b>Рейтинг: {sig.score}/100</b>",
        "",
        "💧 <b>ЛИКВИДНОСТЬ И ПОДТВЕРЖДЕНИЯ</b>",
    ]

    for reason in sig.reasons[:8]:
        lines.append(
            f"• {translate_reason(reason)}"
        )

    lines += [
        "",
        "📍 <b>ТОРГОВЫЙ ПЛАН</b>",
        f"Entry: <b>{fmt_price(sig.entry)}</b>",
        f"SL: <b>{fmt_price(sig.sl)}</b>",
        "",
    ]

    for i, (price, name) in enumerate(sig.targets, 1):

        lines.append(
            f"🎯 TP{i}: <b>{fmt_price(price)}</b> "
            f"<i>({name})</i>"
        )

    risk = abs(
        sig.entry - sig.sl
    )

    if risk > 0 and sig.targets:

        rr_last = (
            abs(
                sig.targets[-1][0]
                - sig.entry
            )
            / risk
        )

        lines += [
            "",
            f"⚖️ Потенциал до последней цели: "
            f"<b>1:{rr_last:.2f}</b>"
        ]

    lines += [
        "",
        "🧠 <b>ПОЧЕМУ ЭТОТ СИГНАЛ</b>",
    ]

    for explanation in build_explanation(sig):

        lines.append(
            f"• {explanation}"
        )

    lines += [
        "",
        "❌ <b>ОТМЕНА СЦЕНАРИЯ</b>",
        sig.invalidation,
        "",
        "👁 <b>TRADE VISION 24/7</b>",
        "<i>Liquidity → Reaction → Confirmation</i>",
        "",
        "⚠️ Технический сценарий. Контролируй риск.",
    ]

    return "\n".join(lines)


def send_signal(sig: Signal) -> bool:

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
