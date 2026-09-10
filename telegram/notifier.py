import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from analysis.engine import Signal

def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")

def build_message(sig: Signal) -> str:
    icon = "🟢" if sig.side == "LONG" else "🔴"

    lines = [
        f"{icon} <b>{sig.symbol} {sig.side}</b>",
        "",
        f"⭐ <b>Signal Score: {sig.score}/100</b>",
        "",
        "🧠 <b>ANALYSIS</b>",
    ]
    lines += [f"• {r}" for r in sig.reasons[:7]]

    lines += [
        "",
        f"🎯 <b>ENTRY:</b> {fmt_price(sig.entry)}",
        f"🛑 <b>SL:</b> {fmt_price(sig.sl)}",
        "",
    ]

    for i, (price, name) in enumerate(sig.targets, 1):
        lines.append(f"🎯 <b>TP{i}:</b> {fmt_price(price)}  <i>({name})</i>")

    risk = abs(sig.entry - sig.sl)
    if risk > 0 and sig.targets:
        rr = abs(sig.targets[-1][0] - sig.entry) / risk
        lines += ["", f"⚖️ <b>Potential R:R:</b> 1:{rr:.2f}"]

    lines += [
        "",
        f"❌ <b>Invalidation:</b> {sig.invalidation}",
        "",
        "👁 <b>TRADE VISION 24/7</b>",
        "<i>Liquidity → Reaction → Confirmation</i>",
        "",
        "⚠️ Educational signal. Manage risk."
    ]
    return "\n".join(lines)

def send_signal(sig: Signal) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] token/chat id missing")
        print(build_message(sig))
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": build_message(sig),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            print("[Telegram]", r.text)
        return r.ok
    except Exception as exc:
        print("[Telegram]", exc)
        return False
