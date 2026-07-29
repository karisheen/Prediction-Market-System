# Prediction Market System

A crypto-first research and decision-support system for estimating calibrated
prediction-market probabilities, ranking conservative opportunities, preserving
an audit history, and delivering manual-review alerts to Discord.

This first vertical slice evaluates binary contracts of the form “Will a crypto
asset be above a specified price at expiry?” It does **not** place trades.

## What is implemented

- A lognormal structural probability model for crypto price thresholds.
- A market-anchored forecast blended in log-odds space.
- Explicit uncertainty, fees, slippage, resolution haircuts, liquidity checks,
  fractional Kelly sizing, and a no-trade `WATCH` state.
- An unauthenticated Kalshi REST adapter for market metadata and executable
  order-book quotes.
- Historical Kalshi ingestion for archived markets, candlesticks, rule snapshots,
  resolutions, and scheduled series/event fee changes.
- Public Coinbase spot candles plus Deribit DVOL, funding history, perpetual basis,
  and open-interest snapshots.
- Kalshi event live-data snapshots with provider-specific payloads preserved.
- A no-look-ahead context assembler that enforces source cutoffs, complete realized-
  volatility windows, and explicit staleness limits.
- Append-oriented SQLite storage for forecasts, recommendations, alert events, and
  point-in-time venue research data.
- Discord webhook delivery with idempotent retries and in-place market updates.
- A CLI for manual evaluations, historical ingestion, research sync, and paper alerts.

The current model is a testable baseline, not evidence of a durable trading edge.
It must be calibrated and validated on untouched historical and forward data
before real-money use.

## Architecture

```text
Market snapshot ─┐
                 ├─> probability engine ─> conservative edge/risk checks
Crypto snapshot ─┘                              │
                                               ├─> SQLite audit history
                                               └─> Discord manual-review alert

Kalshi archive ──> markets + candles + rules + resolutions + fee changes ─┐
Coinbase ───────> completed spot candles + realized volatility ────────────────┤
Deribit ────────> DVOL + funding + forward derivatives snapshots ──────────────┤
Kalshi events ──> timestamped venue context ────────────────────────────────────┤
                                                                                 └─> SQLite point-in-time store
```

The SQLite history is authoritative. Discord is only a notification surface and
never receives exchange credentials, wallet keys, or authority to trade.

## Setup

Requirements:

- `uv`
- Python 3.11 or newer (managed automatically by `uv`)

```bash
uv sync
cp .env.example .env
uv run pms init-db
```

No Discord credential is required to evaluate and store opportunities locally.

## Run a paper evaluation

```bash
uv run pms evaluate \
  --market-id btc-100k-example \
  --question "Will BTC be above 100000 USD at expiry?" \
  --venue example \
  --expires-at "2030-12-31T23:59:00Z" \
  --symbol BTC \
  --spot 110000 \
  --strike 100000 \
  --volatility 0.55 \
  --yes-bid 0.40 \
  --yes-ask 0.42 \
  --no-bid 0.57 \
  --no-ask 0.59 \
  --yes-ask-size 500 \
  --no-ask-size 500 \
  --resolution-rule "Resolves YES if the venue's stated BTC index is above 100000 USD at expiry."
```

Review saved evaluations:

```bash
uv run pms history
```

All inputs must represent the same point in time. The command accepts
`--observed-at` for historical evaluations; omitting it uses the current UTC time.

## Live Kalshi data

Kalshi is the first venue adapter. Public REST market and order-book reads do not
require credentials:

```bash
uv run pms kalshi-markets --series KALSHI_SERIES_TICKER
uv run pms kalshi-inspect --ticker KALSHI_MARKET_TICKER
```

For a supported terminal price-threshold contract, fetch current Kalshi quotes
and evaluate them against user-supplied crypto inputs:

```bash
uv run pms kalshi-evaluate \
  --ticker KALSHI_MARKET_TICKER \
  --symbol BTC \
  --spot 110000 \
  --volatility 0.55
```

The command intentionally rejects markets that can close early. Contracts that
resolve when a price touches a barrier at any point before expiry require a
path-dependent barrier model; applying the terminal-price model to them would be
mathematically wrong.

See [the venue decision](docs/venue-decision.md) for why Kalshi is first and
Polymarket is planned as a second read-only signal source.

## Historical Kalshi ingestion

Archive a bounded set of settled markets and hourly candlesticks from a Kalshi
series:

```bash
uv run pms kalshi-sync-history \
  --series KXBTC \
  --start "2025-01-01T00:00:00Z" \
  --end "2025-12-31T23:59:59Z" \
  --period 60 \
  --max-markets 100
```

`--period` accepts Kalshi's 1-minute, 60-minute, or 1440-minute intervals. The
start and end timestamps are inclusive and must include a timezone. The sync uses
Kalshi's public historical endpoints and does not require credentials.

Each run records a point-in-time market and rule snapshot. Settled outcomes are
materialized separately with settlement values and timestamps. Candlesticks and
scheduled fee-change records use source identifiers as stable keys, so rerunning
the same range does not duplicate those immutable rows. Event fee records preserve
explicit `null` overrides because they mean “clear the event override and inherit
the series fee.”

## Point-in-time research data

Synchronize completed Coinbase spot candles, Deribit DVOL and funding history, a
current Deribit perpetual snapshot, and optional Kalshi event context:

```bash
uv run pms sync-research-data \
  --symbol BTC \
  --start "2025-11-30T23:00:00Z" \
  --end "2025-12-31T23:00:00Z" \
  --interval 60 \
  --realized-window-days 30 \
  --event-ticker KALSHI_EVENT_TICKER
```

The end timestamp is exclusive. `--interval` accepts 1, 60, or 1440 minutes so
Coinbase candles and Deribit DVOL observations share a boundary. Sync runs and
provider row counts are recorded in SQLite. Immutable source records are
idempotent; current derivatives and event payloads are timestamped snapshots.

Inspect exactly what would have been available at a historical cutoff:

```bash
uv run pms research-context \
  --symbol BTC \
  --as-of "2025-12-31T23:00:00Z" \
  --interval 60 \
  --realized-window-days 30 \
  --event-ticker KALSHI_EVENT_TICKER
```

The assembler only selects source timestamps at or before `--as-of`. Required
spot data fails closed when missing or stale, and realized volatility requires a
complete trailing window. Optional DVOL, funding, derivatives, and event inputs
are omitted with warnings when missing or stale. A current derivatives or event
snapshot fetched during a historical sync is therefore stored for forward use
but cannot leak into the historical context.

Coinbase is a public continuous-price source, not necessarily the exact benchmark
named in a Kalshi resolution rule. The provider and raw payload remain attached
to every observation so that benchmark mismatch is auditable. Historical funding
and DVOL are backfilled; basis and open interest are captured forward because the
public Deribit ticker endpoint exposes their current state.

## Discord delivery

For the initial one-way notifier:

1. Create a private Discord channel.
2. In Discord, open **Channel Settings → Integrations → Webhooks**.
3. Create and copy a webhook URL.
4. Put it in your local `.env`:

```dotenv
PMS_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/REPLACE_ME
```

Never commit `.env` or paste the webhook into source code. Test delivery with:

```bash
uv run pms discord-test
```

Add `--send-discord` to `pms evaluate` to send an actionable `ENTER YES` or
`ENTER NO` result. `WATCH` evaluations are saved but do not create notifications.

## Configuration

All settings use the `PMS_` prefix. See `.env.example` for the available risk and
cost assumptions. Defaults are deliberately conservative but are not universally
correct. The initial Kalshi fee coefficient approximates the published general
taker formula; the historical store now preserves scheduled series and event fee
changes, but backtests must apply the effective fee at each evaluation timestamp
and reproduce Kalshi's fee rounding.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Next build stages

1. Backtest with executable quotes, latency, partial fills, and walk-forward splits.
2. Add a barrier-hitting model for supported early-close crypto contracts.
3. Calibrate uncertainty from held-out outcomes instead of the initial fixed margin.
4. Run Discord-delivered paper alerts through multiple market regimes.

Prediction markets involve financial, legal, venue, resolution, and counterparty
risk. This software is for research and manual decision support, not a guarantee
of profit or financial advice.
