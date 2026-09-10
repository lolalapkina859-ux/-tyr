import time
import requests
import pandas as pd

BINGX_BASE_URL = "https://open-api.bingx.com"


def _format_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("-", "")

    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT"

    return symbol


INTERVAL_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "D": "1d",
    "W": "1w",
}


def get_klines(
    symbol: str,
    interval: str,
    limit: int = 500
) -> pd.DataFrame | None:

    bingx_symbol = _format_symbol(symbol)
    bingx_interval = INTERVAL_MAP.get(str(interval), str(interval))

    url = f"{BINGX_BASE_URL}/openApi/swap/v3/quote/klines"

    params = {
        "symbol": bingx_symbol,
        "interval": bingx_interval,
        "limit": limit,
    }

    headers = {
        "User-Agent": "TradeVision24-7"
    }

    for attempt in range(3):

        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=20
            )

            r.raise_for_status()

            data = r.json()

            if data.get("code") not in (0, "0", None):
                raise RuntimeError(
                    f"BingX error: {data.get('code')} "
                    f"{data.get('msg')}"
                )

            rows = data.get("data", [])

            if not rows:
                raise RuntimeError("No kline data received")

            normalized = []

            for row in rows:

                if isinstance(row, list):

                    normalized.append([
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                    ])

                elif isinstance(row, dict):

                    normalized.append([
                        row.get("time") or row.get("openTime"),
                        row.get("open"),
                        row.get("high"),
                        row.get("low"),
                        row.get("close"),
                        row.get("volume"),
                    ])

            df = pd.DataFrame(
                normalized,
                columns=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df["time"] = pd.to_datetime(
                pd.to_numeric(
                    df["time"],
                    errors="coerce"
                ),
                unit="ms",
                utc=True
            )

            df = (
                df
                .dropna()
                .sort_values("time")
                .drop_duplicates("time")
                .reset_index(drop=True)
            )

            print(
                f"[BingX] {symbol} "
                f"{bingx_interval}: "
                f"{len(df)} candles"
            )

            return df

        except Exception as exc:

            print(
                f"[BingX] {symbol} "
                f"{bingx_interval} "
                f"attempt {attempt + 1}: "
                f"{exc}"
            )

            time.sleep(1 + attempt)

    return None
