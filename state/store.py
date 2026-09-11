import json
import os
from pathlib import Path


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

PATH = DATA_DIR / "signal_state.json"


def load_state():
    if not PATH.exists():
        return {}

    try:
        data = json.loads(
            PATH.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

        return {}

    except Exception as exc:
        print(
            f"[STATE] load error: {exc}"
        )
        return {}


def save_state(state):
    try:
        PATH.write_text(
            json.dumps(
                state,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except Exception as exc:
        print(
            f"[STATE] save error: {exc}"
        )


def already_sent(
    state,
    key: str,
) -> bool:
    return bool(
        state.get(key)
    )


def mark_sent(
    state,
    key: str,
):
    state[key] = True

    if len(state) > 5000:
        keys = list(
            state.keys()
        )[-3000:]

        state = {
            k: state[k]
            for k in keys
        }

    save_state(
        state
    )

    return state
