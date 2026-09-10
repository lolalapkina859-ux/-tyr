import json
from pathlib import Path

PATH = Path("state.json")

def load_state():
    if not PATH.exists():
        return {}
    try:
        return json.loads(PATH.read_text())
    except Exception:
        return {}

def save_state(state):
    PATH.write_text(json.dumps(state, indent=2))

def already_sent(state, key: str) -> bool:
    return bool(state.get(key))

def mark_sent(state, key: str):
    state[key] = True
    # Keep state reasonably small on a long-running Railway worker.
    if len(state) > 5000:
        keys = list(state.keys())[-3000:]
        state = {k: state[k] for k in keys}
    save_state(state)
    return state
