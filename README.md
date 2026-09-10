# Trade Vision 24/7 — Liquidity Signal Bot

Railway-ready Python scanner for Bybit perpetual markets.

## Logic v1

The bot uses:

- 4H context
- 15M confirmation
- PDH / PDL
- PWH / PWL
- PMH / PML
- previous swing high / low
- Asia / London / New York ranges
- liquidity sweep detection
- WaveTrend 9 / 12 / 3
- extreme level ±60
- Money Flow 21 / 9
- 15M structure shift
- dynamic Entry / SL / TP levels
- Telegram signal score

The WaveTrend and Money Flow formulas are ported from the supplied Liquidity Tracker Pine logic.

## Railway

1. Push this folder to GitHub.
2. Create a new Railway project from the GitHub repo.
3. Add Variables:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
WATCHED_SYMBOLS=BTCUSDT,ETHUSDT,HYPEUSDT,FARTCOINUSDT,ZAMAUSDT
MIN_SIGNAL_SCORE=70
SCAN_INTERVAL_SECONDS=60
```

4. Railway should detect the Procfile and run:

```bash
python main.py
```

## Important

Version 1 is **signal-only**. It does not place trades.

That is intentional: first compare Telegram calls against manual chart analysis and tune the score/entry logic before enabling execution.

## Recommended next upgrades

- exact equal-high / equal-low engine from the Pine script
- explicit sweep vs liquidity-run state machine
- bullish/bearish divergence engine
- half-timeframe Money Flow
- FVG detection and 50% FVG entries
- persistent database instead of JSON state
- chart snapshot attached to Telegram
- optional Bybit execution module after validation
