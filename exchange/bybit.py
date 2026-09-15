import time
import requests
import pandas as pd


BINGX_BASE_URL = "https://open-api.bingx.com"

# Non-crypto synthetic/index/commodity contracts that must never enter
# the Trade Vision crypto scanner, even if they rank in BingX TOP volume.
EXCLUDED_SYMBOLS = {
    "NCSKASML2USDUSDT",
    "NCCOXAG2USDUSDT",
    "NCCOGOLD2USDUSDT",
    "NCSINASDAQ1002USDUSDT",
}


# =========================================================
# SYMBOL FORMAT
# =========================================================

def _format_symbol(symbol: str) -> str:
    """
    BTCUSDT -> BTC-USDT
    ETHUSDT -> ETH-USDT
    """

    symbol = symbol.upper().replace("-", "")

    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT"

    return symbol


# =========================================================
# INTERVAL MAP
# =========================================================

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


# =========================================================
# GET KLINES
# =========================================================

def get_klines(
    symbol: str,
    interval: str,
    limit: int = 500
) -> pd.DataFrame | None:

    bingx_symbol = _format_symbol(symbol)

    bingx_interval = INTERVAL_MAP.get(
        str(interval),
        str(interval)
    )

    url = (
        f"{BINGX_BASE_URL}"
        f"/openApi/swap/v3/quote/klines"
    )

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

            if data.get("code") not in (
                0,
                "0",
                None,
            ):
                raise RuntimeError(
                    f"BingX error: "
                    f"{data.get('code')} "
                    f"{data.get('msg')}"
                )

            rows = data.get(
                "data",
                []
            )

            if not rows:
                raise RuntimeError(
                    "No kline data received"
                )

            normalized = []

            for row in rows:

                if isinstance(row, list):
                    if len(row) < 6:
                        continue
                    normalized.append([
                        row[0], row[1], row[2], row[3], row[4], row[5]
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

            if not normalized:
                raise RuntimeError("No valid kline rows")

            df = pd.DataFrame(
                normalized,
                columns=["time", "open", "high", "low", "close", "volume"]
            )

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["time"] = pd.to_datetime(
                pd.to_numeric(df["time"], errors="coerce"),
                unit="ms",
                utc=True
            )

            df = (
                df.dropna()
                .sort_values("time")
                .drop_duplicates("time")
                .reset_index(drop=True)
            )

            print(f"[BingX] {symbol} {bingx_interval}: {len(df)} candles")
            return df

        except Exception as exc:
            print(
                f"[BingX] {symbol} {bingx_interval} "
                f"attempt {attempt + 1}: {exc}"
            )
            time.sleep(1 + attempt)

    return None


# =========================================================
# GET TOP SYMBOLS BY 24H VOLUME
# =========================================================

def get_top_symbols(limit: int = 30) -> list[str]:
    """Gets the most liquid BingX USDT perpetual crypto pairs."""

    url = f"{BINGX_BASE_URL}/openApi/swap/v2/quote/ticker"
    headers = {"User-Agent": "TradeVision24-7"}

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("code") not in (0, "0", None):
            raise RuntimeError(
                f"BingX error: {data.get('code')} {data.get('msg')}"
            )

        rows = data.get("data", [])
        if not rows:
            raise RuntimeError("No ticker data received")

        ranked = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            symbol = row.get("symbol", "")
            if not symbol.endswith("-USDT"):
                continue

            clean_symbol = symbol.replace("-", "").upper()

            # Hard exclusion before ranking so excluded contracts do not
            # consume slots in TOP_SYMBOLS_LIMIT.
            if clean_symbol in EXCLUDED_SYMBOLS:
                continue

            quote_volume = (
                row.get("quoteVolume")
                or row.get("turnover24h")
                or row.get("quoteVol")
            )

            if quote_volume is None:
                try:
                    base_volume = float(row.get("volume", 0))
                    last_price = float(
                        row.get("lastPrice")
                        or row.get("last")
                        or row.get("price")
                        or 0
                    )
                    quote_volume = base_volume * last_price
                except (TypeError, ValueError):
                    quote_volume = 0

            try:
                quote_volume = float(quote_volume)
            except (TypeError, ValueError):
                quote_volume = 0

            if quote_volume <= 0:
                continue

            ranked.append((clean_symbol, quote_volume))

        ranked.sort(key=lambda x: x[1], reverse=True)

        symbols = [symbol for symbol, _ in ranked[:limit]]

        print(f"[BingX] TOP {len(symbols)} symbols: {symbols}")
        return symbols

    except Exception as exc:
        print(f"[BingX] TOP symbols error: {exc}")
        return []
