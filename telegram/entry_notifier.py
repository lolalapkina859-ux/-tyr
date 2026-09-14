import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")


def send_entry_filled(event: dict, current_price: float) -> bool:
    """Telegram-only reminder when a tracked LIMIT is actually touched."""
    symbol = event.get("symbol", "")
    side = event.get("side", "")
    entry = float(event.get("price", current_price))
    icon = "🟢" if side == "LONG" else "🔴"

    text = "\n".join([
        f"🚀 <b>ENTRY NOW — #{symbol}</b>",
        "",
        f"{icon} <b>{side}</b>",
        f"🎯 LIMIT touched: <b>{fmt_price(entry)}</b>",
        f"💵 Current: <b>{fmt_price(float(current_price))}</b>",
        "",
        "✅ Лимитная зона была активирована.",
        "Если ордер не был выставлен — проверь текущую цену и входи MARKET только если цена всё ещё рядом с LIMIT.",
        "⚠️ Если цена уже резко ушла от Entry — не догонять.",
        "",
        "👁 <b>TRADE VISION 24/7</b>",
    ])

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
        response = requests.post(url, json=payload, timeout=15)
        if not response.ok:
            print("[Telegram ENTRY]", response.text)
        return response.ok
    except Exception as exc:
        print("[Telegram ENTRY]", exc)
        return False
