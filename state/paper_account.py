import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
TRACKER_PATH = DATA_DIR / "signal_stats_v2.json"
PAPER_PATH = DATA_DIR / "paper_account_v1.json"

START_BALANCE = 1000.0
POSITION_NOTIONAL = 60.0


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path, default):
    try:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    except Exception as exc:
        print(f"[PAPER] load error {path.name}: {exc}")
    return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _state():
    state = _load_json(PAPER_PATH, {})
    if not state:
        state = {
            "version": 1,
            "started_at": _now_iso(),
            "start_balance": START_BALANCE,
            "position_notional": POSITION_NOTIONAL,
        }
        _save_json(PAPER_PATH, state)
        print(f"[PAPER] NEW account: balance={START_BALANCE:.2f}, fixed_notional={POSITION_NOTIONAL:.2f}")
    return state


def _after_start(item, started_at):
    created = str(item.get("created_at") or "")
    return bool(created and created >= started_at)


def _signed_return(side, entry, price):
    if entry <= 0:
        return 0.0
    if side == "LONG":
        return (price - entry) / entry
    return (entry - price) / entry


def _confirmed_realized_pnl(item):
    entry = float(item.get("entry") or 0.0)
    side = str(item.get("side") or "")
    highest_tp = int(item.get("highest_tp") or 0)
    targets = item.get("targets") or []

    pnl = 0.0
    partial_notional = POSITION_NOTIONAL / 4.0

    confirmed = min(highest_tp, len(targets), 4)
    for index in range(confirmed):
        price = float(targets[index]["price"])
        pnl += partial_notional * _signed_return(side, entry, price)

    status = item.get("status")
    result_r = item.get("result_r")

    if status == "CLOSED" and highest_tp == 0 and result_r is not None and float(result_r) < 0:
        sl = float(item.get("sl") or entry)
        pnl = POSITION_NOTIONAL * _signed_return(side, entry, sl)

    return pnl


def get_paper_summary():
    state = _state()
    tracker = _load_json(TRACKER_PATH, {"signals": []})

    started_at = state["started_at"]
    signals = [
        x for x in tracker.get("signals", [])
        if _after_start(x, started_at) and x.get("status") != "DUPLICATE"
    ]

    filled = [x for x in signals if x.get("entry_filled_at")]
    open_positions = [x for x in filled if x.get("status") == "OPEN"]
    closed = [x for x in filled if x.get("status") == "CLOSED"]
    ambiguous = [x for x in filled if x.get("status") == "AMBIGUOUS"]

    pnl_by_trade = [_confirmed_realized_pnl(x) for x in filled]
    realized_pnl = sum(pnl_by_trade)
    balance = START_BALANCE + realized_pnl

    positive = [x for x in pnl_by_trade if x > 0]
    negative = [x for x in pnl_by_trade if x < 0]
    gross_profit = sum(positive)
    gross_loss = abs(sum(negative))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    wins = len(positive)
    losses = len(negative)
    decided = wins + losses
    money_win_rate = wins / decided * 100.0 if decided else 0.0

    tp_counts = {}
    for level in range(1, 5):
        tp_counts[level] = sum(1 for x in filled if int(x.get("highest_tp") or 0) >= level)

    closed_for_dd = sorted(
        [x for x in filled if x.get("closed_at")],
        key=lambda x: str(x.get("closed_at") or ""),
    )
    equity = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    for item in closed_for_dd:
        equity += _confirmed_realized_pnl(item)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "started_at": started_at,
        "start_balance": START_BALANCE,
        "position_notional": POSITION_NOTIONAL,
        "balance": balance,
        "realized_pnl": realized_pnl,
        "return_pct": (balance / START_BALANCE - 1.0) * 100.0,
        "signals": len(signals),
        "filled": len(filled),
        "open": len(open_positions),
        "closed": len(closed),
        "ambiguous": len(ambiguous),
        "wins": wins,
        "losses": losses,
        "money_win_rate": money_win_rate,
        "tp1": tp_counts[1],
        "tp2": tp_counts[2],
        "tp3": tp_counts[3],
        "tp4": tp_counts[4],
        "profit_factor": profit_factor,
        "max_dd": max_dd,
    }
