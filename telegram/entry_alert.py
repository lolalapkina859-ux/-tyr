import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


MAX_MARKET_CHASE_R = 0.20


def _fmt_price(x: float) -> str:
    x = float(x)
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")


def _send(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram ENTRY] token/chat id missing")
        print(text)
        return False

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
            print("[Telegram ENTRY]", response.text)
        return response.ok
    except Exception as exc:
        print("[Telegram ENTRY]", exc)
        return False


def send_entry_touch_alert(
    trade: dict,
    current_price: float,
) -> bool:
    """
    Telegram-only follow-up for a WAIT_FOR_RETRACE setup whose LIMIT
    has actually been touched by live price.

    This does not place orders and does not change strategy/tracker logic.
    If price is still inside the original entry zone, or has moved no more
    than MAX_MARKET_CHASE_R in the favorable direction from LIMIT, show
    ENTRY NOW. Otherwise mark the opportunity as missed / do not chase.
    """

    side = str(trade.get("side", ""))
    symbol = str(trade.get("symbol", ""))
    score = int(trade.get("score", 0))

    entry = float(trade["entry"])
    sl = float(trade["sl"])
    current = float(current_price)
    risk = float(trade.get("risk") or abs(entry - sl))

    zone_low = float(trade.get("zone_low", entry))
    zone_high = float(trade.get("zone_high", entry))

    if risk > 0:
        favorable_r = (
            (current - entry) / risk
            if side == "LONG"
            else (entry - current) / risk
        )
    else:
        favorable_r = 0.0

    in_zone = zone_low <= current <= zone_high
    near_entry = 0.0 <= favorable_r <= MAX_MARKET_CHASE_R
    market_ok = in_zone or near_entry

    targets = trade.get("targets", []) or []
    tp1 = None
    if targets:
        first = targets[0]
        if isinstance(first, dict):
            tp1 = first.get("price")
        else:
            try:
                tp1 = first[0]
            except Exception:
                tp1 = None

    icon = "🟢" if side == "LONG" else "🔴"

    if market_ok:
        lines = [
            f"🚀 <b>ENTRY NOW — #{symbol}</b>",
            "",
            f"{icon} <b>{side}</b> • ⭐ <b>{score}/100</b>",
            "✅ LIMIT только что был задет.",
            "Если лимитка не стояла — цена ещё рядом с рабочим входом.",
            "",
            f"💵 Current: <b>{_fmt_price(current)}</b>",
            f"🎯 LIMIT: <b>{_fmt_price(entry)}</b>",
            f"📍 Zone: <b>{_fmt_price(zone_low)} — {_fmt_price(zone_high)}</b>",
            f"🛑 SL: <b>{_fmt_price(sl)}</b>",
        ]

        if tp1 is not None:
            lines.append(
                f"🏁 TP1: <b>{_fmt_price(float(tp1))}</b>"
            )

        lines += [
            "",
            f"📐 От LIMIT: <b>{favorable_r:+.2f}R</b>",
            f"⚠️ Не догонять, если цена уйдёт дальше +{MAX_MARKET_CHASE_R:.2f}R.",
            "",
            "👁 <b>TRADE VISION 24/7</b>",
        ]
    else:
        lines = [
            f"⚠️ <b>ENTRY MISSED — #{symbol}</b>",
            "",
            f"{icon} <b>{side}</b> • ⭐ <b>{score}/100</b>",
            "LIMIT был задет, но текущая цена уже ушла от рабочего входа.",
            "<b>НЕ ДОГОНЯТЬ ПО РЫНКУ.</b>",
            "",
            f"💵 Current: <b>{_fmt_price(current)}</b>",
            f"🎯 LIMIT: <b>{_fmt_price(entry)}</b>",
            f"🛑 SL: <b>{_fmt_price(sl)}</b>",
            f"📐 От LIMIT: <b>{favorable_r:+.2f}R</b>",
            "",
            "👁 <b>TRADE VISION 24/7</b>",
        ]

    return _send("\n".join(lines))
