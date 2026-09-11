import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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

PATH = DATA_DIR / "signal_stats_v2.json"

# 48 x 15m = 12 hours.
# If a WAIT_FOR_RETRACE setup is not filled within this window,
# it is treated as expired rather than as an open trade.
PENDING_EXPIRY_BARS = 48


def _empty_data() -> dict:
    return {
        "signals": []
    }


def _load() -> dict:
    if not PATH.exists():
        return _empty_data()

    try:
        data = json.loads(
            PATH.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            return _empty_data()

        data.setdefault(
            "signals",
            []
        )

        return data

    except Exception as exc:
        print(
            f"[TRACKER] load error: {exc}"
        )
        return _empty_data()


def _save(
    data: dict
) -> None:

    PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PATH.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _iso(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def _sorted_targets(
    sig,
) -> list[dict]:

    items = [
        {
            "price": float(price),
            "name": name,
        }
        for price, name
        in sig.targets
    ]

    reverse = (
        sig.side == "SHORT"
    )

    items.sort(
        key=lambda x: x["price"],
        reverse=reverse,
    )

    return items


# =========================================================
# REGISTER NEW SIGNAL
# =========================================================

def register_signal(
    sig,
    signal_key: str,
    candle_time: str,
    live_snapshot: dict | None = None,
) -> None:

    data = _load()

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

    now = datetime.now(
        timezone.utc
    ).isoformat()

    entry_status = getattr(
        sig,
        "entry_status",
        "ENTER_NOW",
    )

    # WAIT_FOR_RETRACE is not a trade yet.
    # ENTER_NOW is treated as filled at notification time.
    if entry_status == "WAIT_FOR_RETRACE":
        status = "PENDING"
        entry_filled_at = None
    else:
        status = "OPEN"
        entry_filled_at = now

    item = {
        "key": signal_key,

        "symbol": sig.symbol,
        "side": sig.side,
        "score": int(sig.score),

        "entry_status": entry_status,
        "entry_type": getattr(
            sig,
            "entry_type",
            "UNKNOWN",
        ),

        "zone_low": float(
            getattr(
                sig,
                "zone_low",
                sig.entry,
            )
        ),

        "zone_high": float(
            getattr(
                sig,
                "zone_high",
                sig.entry,
            )
        ),

        "current_price_at_signal": float(
            getattr(
                sig,
                "current_price",
                sig.entry,
            )
        ),

        "entry": float(sig.entry),
        "sl": float(sig.sl),

        "targets": _sorted_targets(
            sig
        ),

        "reasons": list(
            sig.reasons
        ),

        # Last CLOSED 15m candle that existed when signal was sent.
        # We never evaluate that candle for result tracking.
        "signal_candle_time": candle_time,

        # Real notification/registration time.
        "created_at": now,

        # Snapshot of the LIVE 15m candle at the moment Telegram signal
        # was registered. This prevents pre-signal wick contamination.
        "signal_live_time": (
            live_snapshot.get("time")
            if live_snapshot
            else None
        ),
        "signal_live_high": (
            float(live_snapshot.get("high"))
            if live_snapshot
            and live_snapshot.get("high") is not None
            else None
        ),
        "signal_live_low": (
            float(live_snapshot.get("low"))
            if live_snapshot
            and live_snapshot.get("low") is not None
            else None
        ),
        "signal_live_close": (
            float(live_snapshot.get("close"))
            if live_snapshot
            and live_snapshot.get("close") is not None
            else float(
                getattr(
                    sig,
                    "current_price",
                    sig.entry,
                )
            )
        ),

        "status": status,
        "entry_filled_at": entry_filled_at,

        "highest_tp": 0,

        "risk": risk,

        "max_r": 0.0,
        "max_adverse_r": 0.0,

        "closed_at": None,
        "result_r": None,

        "expired_at": None,
        "ambiguous_at": None,
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


def _entry_touched(
    entry: float,
    high: float,
    low: float,
) -> bool:
    return (
        low <= entry <= high
    )


def _sl_hit(
    side: str,
    sl: float,
    high: float,
    low: float,
) -> bool:
    if side == "LONG":
        return low <= sl

    return high >= sl


def _target_hit(
    side: str,
    tp: float,
    high: float,
    low: float,
) -> bool:
    if side == "LONG":
        return high >= tp

    return low <= tp


def _future_closed_candles(
    closed,
    signal_candle_time: str,
):
    """
    Include candles from the live signal candle onward.

    The first/live candle is handled separately with the stored
    signal-time high/low snapshot, so price action that happened
    before the Telegram signal cannot falsely fill an entry.
    """

    if len(closed) == 0:
        return closed

    try:
        signal_ts = pd.to_datetime(
            signal_candle_time,
            utc=True,
        )

        times = pd.to_datetime(
            closed["time"],
            utc=True,
        )

        return closed[
            times >= signal_ts
        ]

    except Exception:
        return closed.iloc[-1:0]


def _post_signal_range(
    item: dict,
    row,
) -> tuple[float, float]:
    """
    Return only the price range that is valid AFTER signal registration.

    If this is the same live 15m candle that existed when the signal was
    created, ignore its old pre-signal wick and only allow newly extended
    highs/lows from the saved snapshot.

    For all later candles, use the full candle high/low.
    """

    high = float(row["high"])
    low = float(row["low"])

    live_time = item.get("signal_live_time")

    if not live_time:
        return high, low

    try:
        row_time = pd.to_datetime(
            row["time"],
            utc=True,
        )

        saved_time = pd.to_datetime(
            live_time,
            utc=True,
        )

        if row_time != saved_time:
            return high, low

    except Exception:
        return high, low

    saved_high = item.get(
        "signal_live_high"
    )
    saved_low = item.get(
        "signal_live_low"
    )
    saved_close = float(
        item.get(
            "signal_live_close",
            item.get(
                "current_price_at_signal",
                item["entry"],
            ),
        )
    )

    # Same candle: only newly-created movement after signal counts.
    valid_high = saved_close
    valid_low = saved_close

    if (
        saved_high is not None
        and high > float(saved_high)
    ):
        valid_high = high

    if (
        saved_low is not None
        and low < float(saved_low)
    ):
        valid_low = low

    return (
        max(valid_high, valid_low),
        min(valid_high, valid_low),
    )


def _close_ambiguous(
    item: dict,
    candle_time: str,
    events: list[dict],
):
    item["status"] = "AMBIGUOUS"
    item["closed_at"] = candle_time
    item["ambiguous_at"] = candle_time
    item["result_r"] = None

    events.append({
        "type": "AMBIGUOUS",
        "symbol": item["symbol"],
        "side": item["side"],
        "score": item["score"],
        "key": item["key"],
    })


# =========================================================
# UPDATE SIGNAL RESULTS
# =========================================================

def update_symbol(
    symbol: str,
    df15,
) -> list[dict]:

    data = _load()

    events = []
    changed = False

    if (
        df15 is None
        or len(df15) < 3
    ):
        return events

    # Track touches intrabar too.
    # BingX returns the current live 15M candle with running high/low,
    # so LIMIT / TP / SL can be detected on the next 60s scan
    # instead of waiting for the 15M candle to close.
    closed = (
        df15
        .copy()
    )

    for item in data["signals"]:

        if (
            item.get("symbol")
            != symbol
        ):
            continue

        status = item.get(
            "status",
            "OPEN",
        )

        # Migration for old tracker rows.
        if status == "OPEN":
            item.setdefault(
                "entry_filled_at",
                item.get("created_at"),
            )

        if status not in (
            "PENDING",
            "OPEN",
        ):
            continue

        candles = _future_closed_candles(
            closed,
            item.get(
                "signal_candle_time",
                "",
            ),
        )

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

        # =================================================
        # PENDING: WAIT FOR ACTUAL ENTRY
        # =================================================

        if status == "PENDING":

            # Expire stale limit/retrace setup.
            if len(candles) >= PENDING_EXPIRY_BARS:

                last_time = _iso(
                    candles.iloc[
                        PENDING_EXPIRY_BARS - 1
                    ]["time"]
                )

                item["status"] = "EXPIRED"
                item["expired_at"] = last_time
                item["closed_at"] = last_time
                item["result_r"] = None

                events.append({
                    "type": "EXPIRED",
                    "symbol": symbol,
                    "side": side,
                    "key": item["key"],
                })

                changed = True
                continue

            fill_index = None

            for idx, row in candles.iterrows():

                high, low = _post_signal_range(
                    item,
                    row,
                )

                if _entry_touched(
                    entry,
                    high,
                    low,
                ):
                    fill_index = idx
                    fill_time = _iso(
                        row["time"]
                    )

                    item["status"] = "OPEN"
                    item["entry_filled_at"] = fill_time

                    events.append({
                        "type": "ENTRY_FILLED",
                        "symbol": symbol,
                        "side": side,
                        "price": entry,
                        "key": item["key"],
                    })

                    changed = True
                    break

            if fill_index is None:
                continue

            # From here we evaluate only from the fill candle onward.
            candles = candles.loc[
                fill_index:
            ]

        # =================================================
        # OPEN: evaluate MFE/MAE + TP/SL in chronological order
        # =================================================

        highest_tp = int(
            item.get(
                "highest_tp",
                0,
            )
        )

        targets = item.get(
            "targets",
            [],
        )

        for _, row in candles.iterrows():

            high, low = _post_signal_range(
                item,
                row,
            )

            candle_time = _iso(
                row["time"]
            )

            # MFE / MAE only AFTER entry is active.
            favorable_price = (
                high
                if side == "LONG"
                else low
            )

            adverse_price = (
                low
                if side == "LONG"
                else high
            )

            item["max_r"] = max(
                float(
                    item.get(
                        "max_r",
                        0.0,
                    )
                ),
                float(
                    _r_multiple(
                        side,
                        entry,
                        favorable_price,
                        risk,
                    )
                ),
            )

            item["max_adverse_r"] = min(
                float(
                    item.get(
                        "max_adverse_r",
                        0.0,
                    )
                ),
                float(
                    _r_multiple(
                        side,
                        entry,
                        adverse_price,
                        risk,
                    )
                ),
            )

            sl_touched = _sl_hit(
                side,
                sl,
                high,
                low,
            )

            newly_hit = []

            for idx, target in enumerate(
                targets,
                start=1,
            ):

                if idx <= highest_tp:
                    continue

                tp = float(
                    target["price"]
                )

                if _target_hit(
                    side,
                    tp,
                    high,
                    low,
                ):
                    newly_hit.append(
                        idx
                    )

            # If the same 15m candle hits SL and one or more TPs,
            # intrabar order is unknowable -> AMBIGUOUS.
            if (
                sl_touched
                and newly_hit
            ):
                _close_ambiguous(
                    item,
                    candle_time,
                    events,
                )

                changed = True
                break

            if sl_touched:

                item["status"] = "CLOSED"
                item["closed_at"] = candle_time
                item["result_r"] = -1.0

                events.append({
                    "type": "SL_HIT",
                    "symbol": symbol,
                    "side": side,
                    "score": item["score"],
                    "result_r": -1.0,
                    "key": item["key"],
                })

                changed = True
                break

            if newly_hit:

                for idx in newly_hit:
                    target = targets[
                        idx - 1
                    ]

                    tp = float(
                        target["price"]
                    )

                    highest_tp = max(
                        highest_tp,
                        idx,
                    )

                    item[
                        "highest_tp"
                    ] = highest_tp

                    rr = (
                        abs(
                            tp - entry
                        )
                        / risk
                        if risk > 0
                        else 0.0
                    )

                    events.append({
                        "type": "TP_HIT",
                        "tp": idx,
                        "symbol": symbol,
                        "side": side,
                        "score": item["score"],
                        "rr": rr,
                        "price": tp,
                        "key": item["key"],
                    })

                    changed = True

                # Final TP closes the trade.
                if (
                    targets
                    and highest_tp
                    >= len(targets)
                ):
                    final_tp = float(
                        targets[-1]["price"]
                    )

                    final_rr = (
                        abs(
                            final_tp - entry
                        )
                        / risk
                        if risk > 0
                        else 0.0
                    )

                    item["status"] = "CLOSED"
                    item["closed_at"] = candle_time
                    item["result_r"] = final_rr

                    events.append({
                        "type": "CLOSED_TP",
                        "symbol": symbol,
                        "side": side,
                        "result_r": final_rr,
                        "key": item["key"],
                    })

                    changed = True
                    break

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
        [],
    )

    pending = [
        x
        for x in signals
        if x.get("status")
        == "PENDING"
    ]

    open_signals = [
        x
        for x in signals
        if x.get("status")
        == "OPEN"
    ]

    closed = [
        x
        for x in signals
        if x.get("status")
        == "CLOSED"
    ]

    expired = [
        x
        for x in signals
        if x.get("status")
        == "EXPIRED"
    ]

    ambiguous = [
        x
        for x in signals
        if x.get("status")
        == "AMBIGUOUS"
    ]

    filled = [
        x
        for x in signals
        if x.get(
            "entry_filled_at"
        )
    ]

    losses = sum(
        1
        for x in closed
        if float(
            x.get("result_r")
            or 0
        ) < 0
    )

    wins = sum(
        1
        for x in closed
        if float(
            x.get("result_r")
            or 0
        ) > 0
    )

    def tp_count(level: int) -> int:
        return sum(
            1
            for x in signals
            if int(
                x.get(
                    "highest_tp",
                    0,
                )
            ) >= level
        )

    closed_decided = (
        wins
        + losses
    )

    win_rate = (
        wins
        / closed_decided
        * 100.0
        if closed_decided > 0
        else 0.0
    )

    return {
        "total": len(signals),

        "pending": len(pending),
        "open": len(open_signals),
        "closed": len(closed),
        "expired": len(expired),
        "ambiguous": len(ambiguous),

        "filled": len(filled),

        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,

        "tp1": tp_count(1),
        "tp2": tp_count(2),
        "tp3": tp_count(3),
        "tp4": tp_count(4),
    }
