import time
import requests
import pandas as pd
from config import BYBIT_BASE_URL

def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame | None:
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            rows = data.get("result", {}).get("list", [])
            if data.get("retCode") != 0 or not rows:
                time.sleep(1 + attempt)
                continue

            df = pd.DataFrame(
                rows,
                columns=["time", "open", "high", "low", "close", "volume", "turnover"]
            )
            df = df.iloc[::-1].reset_index(drop=True)

            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
            return df.dropna().reset_index(drop=True)

        except Exception as exc:
            print(f"[Bybit] {symbol} {interval} attempt {attempt+1}: {exc}")
            time.sleep(1 + attempt)

    return None
