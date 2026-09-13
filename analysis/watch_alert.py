from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from analysis.liquidity import detect_event

# WATCH V2: only major monthly/weekly/daily liquidity.
# PSH/PSL are intentionally excluded to reduce noise.
MAJOR_LEVELS = {"PMH", "PML", "PWH", "PWL"}
DAILY_LEVELS = {"PDH", "PDL"}
WATCH_LEVELS = MAJOR_LEVELS | DAILY_LEVELS

LEVEL_NAMES = {
    "PMH": "максимум прошлого месяца",
    "PML": "минимум прошлого месяца",
    "PWH": "максимум прошлой недели",
    "PWL": "минимум прошлой недели",
    "PDH": "максимум прошлого дня",
    "PDL": "минимум прошлого дня",
}

DATA_DIR = Path(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

WATCH_STATE_PATH = DATA_DIR / "watch_state_v2.json"


def _fmt_price(value: float) -> str:
    value = float(value)
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _load_state() -> dict:
    if not WATCH_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(WATCH_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[WATCH V2] state load error: {exc}")
        return {}


def _save_state(state: dict) -> None:
    if len(state) > 3000:
        keys = list(state.keys())[-2000:]
        state = {key: state[key] for key in keys}
    try:
        WATCH_STATE_PATH.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[WATCH V2] state save error: {exc}")


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
        "15M structure NOT confirmed" in str(reason)
        for reason in sig.reasons
    )


def _momentum_ok(df, side: str) -> bool:
    """Major PM/PW WATCH needs 4H momentum already leaning with the scenario."""
    if df is None or len(df) < 5:
        return False
    if side == "LONG":
        mf_ok = df["mf"].iloc[-2] > df["mf"].iloc[-3] > df["mf"].iloc[-4]
        wt_ok = df["wt1"].iloc[-2] > df["wt1"].iloc[-3]
    else:
        mf_ok = df["mf"].iloc[-2] < df["mf"].iloc[-3] < df["mf"].iloc[-4]
        wt_ok = df["wt1"].iloc[-2] < df["wt1"].iloc[-3]
    return bool(mf_ok or wt_ok)


def _daily_reaction_ok(m15, side: str) -> bool:
    """PDH/PDL are common: demand both 15M WT and Money Flow reaction."""
    if m15 is None or len(m15) < 5:
        return False
    if side == "LONG":
        return bool(
            m15["wt1"].iloc[-2] > m15["wt1"].iloc[-3]
            and m15["mf"].iloc[-2] > m15["mf"].iloc[-3]
        )
    return bool(
        m15["wt1"].iloc[-2] < m15["wt1"].iloc[-3]
        and m15["mf"].iloc[-2] < m15["mf"].iloc[-3]
    )


def _not_too_late(m15, current_price: float, level_price: float) -> bool:
    """Skip WATCH if price already ran too far away from reclaimed liquidity."""
    try:
        atr15 = float(m15["atr"].iloc[-2])
    except Exception:
        return False
    max_distance = max(2.0 * atr15, abs(float(current_price)) * 0.015)
    return abs(float(current_price) - float(level_price)) <= max_distance


def _event_time_iso(h4, event: dict) -> str:
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


def _build_message(symbol, side, level, level_price, current_price) -> str:
    icon = "🟢" if side == "LONG" else "🔴"
    direction = "ниже" if side == "LONG" else "выше"
    reclaim_text = "вернулась выше уровня" if side == "LONG" else "вернулась ниже уровня"
    description = LEVEL_NAMES.get(level, level)

    return "\n".join([
        f"👀 <b>SETUP WATCH — #{symbol}</b>",
        "",
        f"{icon} Возможный <b>{side}</b>-сценарий формируется.",
        "",
        "💧 <b>СНЯТА ОСНОВНАЯ HTF ЛИКВИДНОСТЬ</b>",
        f"• Цена сняла ликвидность {direction} <b>{level}</b> — {description}.",
        f"• Уровень: <b>{_fmt_price(level_price)}</b>",
        f"• Current: <b>{_fmt_price(current_price)}</b>",
        f"• Цена {reclaim_text}.",
        "",
        "⏳ <b>ЧТО ЖДЁМ</b>",
        "• BOS / CHoCH на 15M по направлению сценария;",
        "• FVG / Order Block для точного LIMIT-входа.",
        "",
        "⚠️ <b>СЕЙЧАС НЕ ВХОДИТЬ</b>",
        "Reaction/momentum уже есть, но структура входа ещё не подтверждена.",
        "",
        "👁 <b>TRADE VISION 24/7</b>",
        "<i>Liquidity taken → reaction → waiting for BOS</i>",
    ])


def maybe_send_watch(symbol: str, sig, h4, m15, levels4: dict) -> bool:
    """
    WATCH V2 filters:
      PM/PW -> sweep + reclaim + aligned 4H momentum.
      PDH/PDL -> same + 15M WT/MF reaction.
      PSH/PSL -> no WATCH.
      All -> skip late alerts + persistent one-alert-per-sweep dedupe.
    """
    if not _structure_missing(sig):
        return False

    level = _find_htf_level(sig)
    if level is None:
        return False

    if not _momentum_ok(h4, sig.side):
        return False

    if level in DAILY_LEVELS and not _daily_reaction_ok(m15, sig.side):
        return False

    expected_type = "sweep_low" if sig.side == "LONG" else "sweep_high"
    matching = [
        event
        for event in detect_event(h4, levels4, lookback=3)
        if event.get("type") == expected_type and event.get("level") == level
    ]
    if not matching:
        return False

    event = matching[-1]
    level_price = float(event["price"])
    current_price = float(sig.current_price)

    if not _not_too_late(m15, current_price, level_price):
        print(f"[WATCH V2] {symbol} {sig.side} {level} skipped: late")
        return False

    event_time = _event_time_iso(h4, event)
    key = f"WATCH_V2:{symbol}:{sig.side}:{level}:{event_time}"
    state = _load_state()
    if state.get(key):
        return False

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[WATCH V2] Telegram token/chat id missing: {key}")
        return False

    text = _build_message(symbol, sig.side, level, level_price, current_price)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        if not response.ok:
            print("[WATCH V2] Telegram error:", response.text)
            return False

        state[key] = True
        _save_state(state)
        print(f"[WATCH V2] {symbol} {sig.side} {level} sent 👀")
        return True
    except Exception as exc:
        print(f"[WATCH V2] Telegram exception: {exc}")
        return False
