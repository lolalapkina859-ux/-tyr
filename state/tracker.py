import json
from datetime import datetime, timezone
from pathlib import Path


# Если позже подключим Railway Volume на /data,
# статистика будет сохраняться между redeploy.
DATA_DIR = (
    Path("/data")
    if Path("/data").exists()
    else Path(".")
)

PATH = DATA_DIR / "signal_stats.json"


def _load() -> dict:
    if not PATH.exists():
        return {
            "signals": []
        }

    try:
        data = json.loads(
            PATH.read_text()
        )

        if not isinstance(
            data,
            dict
        ):
            return {
                "signals": []
            }

        data.setdefault(
            "signals",
            []
        )

        return data

    except Exception:
        return {
            "signals": []
        }


def _save(
    data: dict
) -> None:

    PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    PATH.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )


# =========================================================
# REGISTER NEW SIGNAL
# =========================================================

def register_signal(
    sig,
    signal_key: str,
    candle_time: str,
) -> None:

    data = _load()

    # Не записываем один и тот же сигнал повторно.
    for item in data["signals"]:

        if (
            item.get("key")
            == signal_key
        ):
            return

    risk = abs(
        float(sig.entry)
        - float(sig.sl)
    )

    item = {
        "key": signal_key,

        "symbol": sig.symbol,
        "side": sig.side,
        "score": int(sig.score),

        "entry": float(sig.entry),
        "sl": float(sig.sl),

        "targets": [
            {
                "price": float(price),
                "name": name,
            }
            for price, name
            in sig.targets
        ],

        "reasons": list(
            sig.reasons
        ),

        "signal_candle_time":
            candle_time,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status": "OPEN",

        "highest_tp": 0,

        "risk": risk,

        "max_r": 0.0,
        "max_adverse_r": 0.0,

        "closed_at": None,
        "result_r": None,
    }

    data["signals"].append(
        item
    )

    _save(
        data
    )


# =========================================================
# R MULTIPLE
# =========================================================

def _r_multiple(
    side: str,
    entry: float,
    price: float,
    risk: float,
) -> float:

    if risk <= 0:
        return 0.0

    if side == "LONG":

        return (
            price - entry
        ) / risk

    return (
        entry - price
    ) / risk


# =========================================================
# UPDATE SIGNAL RESULTS
# =========================================================

def update_symbol(
    symbol: str,
    df15,
) -> list[dict]:

    """
    Проверяет все OPEN сигналы по символу.

    Возвращает события:
    TP_HIT
    SL_HIT
    """

    data = _load()

    events = []

    changed = False

    if (
        df15 is None
        or len(df15) < 3
    ):
        return events

    # Только закрытые свечи.
    closed = (
        df15.iloc[:-1]
    )

    for item in data["signals"]:

        if (
            item.get("symbol")
            != symbol
        ):
            continue

        if (
            item.get("status")
            != "OPEN"
        ):
            continue

        # -----------------------------------------
        # Берём свечи после появления сигнала
        # -----------------------------------------

        start = item.get(
            "signal_candle_time"
        )

        try:

            candles = closed[
                closed["time"]
                >= start
            ]

        except Exception:

            candles = closed

        if len(candles) == 0:
            continue

        entry = float(
            item["entry"]
        )

        sl = float(
            item["sl"]
        )

        risk = float(
            item.get("risk")
            or abs(
                entry - sl
            )
        )

        side = item["side"]

        # -----------------------------------------
        # MFE / MAE
        # -----------------------------------------

        max_high = float(
            candles["high"].max()
        )

        min_low = float(
            candles["low"].min()
        )

        favorable_price = (
            max_high
            if side == "LONG"
            else min_low
        )

        adverse_price = (
            min_low
            if side == "LONG"
            else max_high
        )

        max_r = _r_multiple(
            side,
            entry,
            favorable_price,
            risk,
        )

        adverse_r = _r_multiple(
            side,
            entry,
            adverse_price,
            risk,
        )

        item["max_r"] = max(
            float(
                item.get(
                    "max_r",
                    0.0
                )
            ),
            float(max_r),
        )

        item[
            "max_adverse_r"
        ] = min(
            float(
                item.get(
                    "max_adverse_r",
                    0.0
                )
            ),
            float(adverse_r),
        )

        highest_tp = int(
            item.get(
                "highest_tp",
                0
            )
        )

        # -----------------------------------------
        # Проверяем свечи по порядку
        # -----------------------------------------

        for _, row in (
            candles.iterrows()
        ):

            high = float(
                row["high"]
            )

            low = float(
                row["low"]
            )

            candle_time = (
                row["time"]
                .isoformat()
            )

            # =====================================
            # STOP LOSS
            # =====================================

            sl_hit = (
                low <= sl
                if side == "LONG"
                else high >= sl
            )

            # Если SL и TP коснулись внутри
            # одной 15M свечи, считаем
            # консервативно: сначала SL.
            if sl_hit:

                item[
                    "status"
                ] = "CLOSED"

                item[
                    "closed_at"
                ] = candle_time

                item[
                    "result_r"
                ] = -1.0

                events.append({
                    "type":
                        "SL_HIT",

                    "symbol":
                        symbol,

                    "side":
                        side,

                    "score":
                        item["score"],

                    "result_r":
                        -1.0,

                    "key":
                        item["key"],
                })

                changed = True

                break

            # =====================================
            # TAKE PROFITS
            # =====================================

            for idx, target in enumerate(
                item.get(
                    "targets",
                    []
                ),
                start=1,
            ):

                if idx <= highest_tp:
                    continue

                tp = float(
                    target["price"]
                )

                tp_hit = (
                    high >= tp
                    if side == "LONG"
                    else low <= tp
                )

                if not tp_hit:
                    continue

                highest_tp = idx

                item[
                    "highest_tp"
                ] = idx

                rr = (
                    abs(
                        tp - entry
                    )
                    / risk
                    if risk > 0
                    else 0.0
                )

                events.append({
                    "type":
                        "TP_HIT",

                    "tp":
                        idx,

                    "symbol":
                        symbol,

                    "side":
                        side,

                    "score":
                        item["score"],

                    "rr":
                        rr,

                    "price":
                        tp,

                    "key":
                        item["key"],
                })

                changed = True

        changed = True

    if changed:

        _save(
            data
        )

    return events


# =========================================================
# SUMMARY
# =========================================================

def get_summary() -> dict:

    data = _load()

    signals = data.get(
        "signals",
        []
    )

    closed = [
        x
        for x in signals
        if x.get("status")
        == "CLOSED"
    ]

    open_signals = [
        x
        for x in signals
        if x.get("status")
        == "OPEN"
    ]

    losses = sum(
        1
        for x in closed
        if float(
            x.get("result_r")
            or 0
        ) < 0
    )

    tp1 = sum(
        1
        for x in signals
        if int(
            x.get(
                "highest_tp",
                0
            )
        ) >= 1
    )

    tp2 = sum(
        1
        for x in signals
        if int(
            x.get(
                "highest_tp",
                0
            )
        ) >= 2
    )

    tp3 = sum(
        1
        for x in signals
        if int(
            x.get(
                "highest_tp",
                0
            )
        ) >= 3
    )

    tp4 = sum(
        1
        for x in signals
        if int(
            x.get(
                "highest_tp",
                0
            )
        ) >= 4
    )

    return {
        "total":
            len(signals),

        "open":
            len(open_signals),

        "closed":
            len(closed),

        "losses":
            losses,

        "tp1":
            tp1,

        "tp2":
            tp2,

        "tp3":
            tp3,

        "tp4":
            tp4,
    }
