# Prediction Market System

A crypto-first research and decision-support system for estimating calibrated
prediction-market probabilities, ranking conservative opportunities, preserving
an audit history, and delivering manual-review alerts to Discord.

This first vertical slice evaluates binary crypto terminal ranges plus upper and
lower terminal/touch thresholds. It does **not** place trades.

## What is implemented

- Lognormal terminal-price models for bounded crypto ranges and upper/lower
  thresholds, plus geometric-Brownian first-passage models for touch barriers.
- A market-anchored forecast blended in log-odds space.
- Held-out, time-ordered uncertainty calibration plus explicit fees, slippage,
  resolution haircuts, liquidity checks, fractional Kelly sizing, and a no-trade
  `WATCH` state.
- An unauthenticated Kalshi REST adapter for paginated market metadata and
  side-specific executable order-book quotes.
- Historical Kalshi ingestion for archived markets, candlesticks, rule snapshots,
  resolutions, and scheduled series/event fee changes.
- Public Coinbase spot candles plus Deribit DVOL, funding history, perpetual basis,
  and open-interest snapshots.
- Kalshi event live-data snapshots with provider-specific payloads preserved.
- A no-look-ahead context assembler that enforces source cutoffs, complete realized-
  volatility windows, and explicit staleness limits.
- A persisted walk-forward backtester with delayed executable quotes, adverse
  intraperiod pricing, partial fills, and point-in-time fee schedules.
- Append-oriented SQLite storage for forecasts, recommendations, alert events,
  per-market paper-cycle checks, point-in-time venue research data, and backtest runs.
- Discord webhook delivery with idempotent retries and in-place market updates.
- A CLI for manual/live evaluations, ingestion, research sync, walk-forward
  backtesting, one-shot multi-market paper-alert cycles, and regime-coverage reporting.

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
                                                                                 ├─> SQLite point-in-time store
                                                                                 └─> walk-forward backtest
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

For a supported fixed-time terminal range/threshold or explicit early-close touch
barrier, fetch current Kalshi quotes and evaluate them against user-supplied crypto inputs:

```bash
uv run pms kalshi-evaluate \
  --ticker KALSHI_MARKET_TICKER \
  --symbol BTC \
  --spot 110000 \
  --volatility 0.55
```

Live Kalshi evaluation requires a matching held-out calibration profile for the
symbol, structural model, and model version. Profiles are produced and persisted
by `pms backtest`. `--allow-uncalibrated` permits local research with the configured
fixed margin, but uncalibrated Discord alerts are rejected.

Terminal markets use the probability of finishing within a bounded range or beyond
a threshold. Early-close markets use the continuous-time first-passage probability
of touching a threshold before expiry. Upper and lower barriers, already-crossed
barriers, physical drift, and numerically extreme tails are supported. A market
with only one executable side remains evaluable on that side; missing liquidity is
never synthesized for the other side.

Classification fails closed. Range contracts require positive, increasing bounds
and an explicit fixed-time terminal observation. An early-close threshold is
accepted only when Kalshi's strike metadata identifies its direction and its
resolution rules explicitly describe terminal or touch semantics. Ambiguous rules
remain unsupported rather than being routed to a mathematically incorrect model.

The first-passage model assumes continuous geometric Brownian motion with constant
volatility and drift over the remaining contract life. It does not model jumps,
discrete benchmark sampling, exchange outages, or intraperiod volatility changes.
Those mismatches must remain part of the uncertainty and resolution-risk review.

See [the venue decision](docs/venue-decision.md) for why Kalshi is first and
Polymarket is planned as a second read-only signal source.

## Historical Kalshi ingestion

Archive every contract's metadata and resolved outcome for settled KXBTC events,
plus bounded candlestick history for each structural model:

```bash
uv run pms kalshi-sync-history \
  --series KXBTC \
  --start "2025-01-01T00:00:00Z" \
  --end "2025-12-31T23:59:59Z" \
  --period 60 \
  --max-events 500 \
  --range-contracts-per-event 5 \
  --history-hours 24
```

`--period` accepts Kalshi's 1-minute, 60-minute, or 1440-minute intervals. The
start and end timestamps are inclusive and must include a timezone. Events are
the pagination unit, so one dense hourly ladder cannot consume the event budget.
Within each event, candle history covers one median-strike contract per threshold
model plus an outcome-independent range sample selected by strike position—never
by resolved outcome. The sync uses Kalshi's public endpoints and does not require
credentials.

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

## Walk-forward backtesting

Replay resolved markets from a stored Kalshi series:

```bash
uv run pms backtest \
  --series KXBTC \
  --symbol BTC \
  --start "2025-01-01T00:00:00Z" \
  --end "2026-01-01T00:00:00Z" \
  --period 60 \
  --realized-window-days 30 \
  --train-days 90 \
  --test-days 30 \
  --step-days 30 \
  --latency-seconds 30 \
  --calibration-lead-minutes 5 \
  --max-volume-participation 0.10 \
  --minimum-calibration-samples 30 \
  --calibration-bins 5 \
  --calibration-confidence 0.95 \
  --minimum-validation-events 20 \
  --minimum-validation-folds 2
```

Run `kalshi-sync-history` and `sync-research-data` first. Research coverage must
begin early enough to provide the complete realized-volatility window at every
training and test timestamp. For each fold, only markets whose outcomes settled
by the training cutoff are eligible for calibration. The five-minute default
calibration lead matches KXBTC's completed hourly market candle before its
five-minute expected-expiration timestamp. Each event contributes one
outcome-weighted sample per structural model, so hundreds of mutually exclusive
contracts from one range ladder cannot masquerade as independent evidence. The
following non-overlapping test window remains untouched until evaluation.

Calibration samples are isolated by symbol, structural model, and model version,
then grouped by event before equal-frequency binning. Each bin compares its mean
forecast with a Wilson confidence interval for the event-level observed outcome
frequency; the larger distance to either interval bound becomes that bin's
uncertainty margin. The default requires 30 independent resolved events per
model at 95% confidence. Signals without a qualifying profile fail closed. Use
`--allow-uncalibrated` only to inspect fixed-margin behavior.

Signals use the executable bid/ask at a completed market candle. Execution uses
the first later candle satisfying the latency assumption and its adverse quote
extreme: the YES ask high or the complementary NO ask derived from the YES bid
low. Fills are whole contracts, capped by `--max-volume-participation`, and can
be partial. Each market can produce at most one filled entry.

When a scheduled fee record exists, the backtester selects the point-in-time
series fee and event override at signal and execution time. When the venue reports
no fee changes, it uses the configured current fee coefficient rather than
discarding every signal. Explicit null event overrides restore the series fee.
Taker fees are rounded upward to cents. Fold metrics, event-grouped calibration
profiles, selected structural model, uncertainty source and margin, individual
fills, cost, P&L, return on cost, and event-weighted Brier score are persisted.
Deployment approval is denied unless the configured independent-event,
held-out-fold, return-on-cost, and Brier thresholds all pass. Profiles and their
approval decisions are indexed for subsequent live evaluation.

Candlestick volume is a participation constraint, not historical order-book
depth. Adverse candle extremes provide a conservative latency/slippage bound but
cannot reconstruct the exact queue position or fill path.

## Paper-alert regime campaign

Refresh the slower research and regime state independently:

```bash
uv run pms paper-alert-research \
  --series KXBTC \
  --symbol BTC \
  --interval 60 \
  --realized-window-days 30
```

Run a delivery-free live shadow evaluation against that stored state:

```bash
uv run pms paper-alerts \
  --series KXBTC \
  --symbol BTC \
  --interval 60 \
  --realized-window-days 30
```

Each evaluation adds a completed one-minute Coinbase decision price, follows
Kalshi cursors across up to 1,000 open markets, and evaluates every supported
range or threshold contract. A stale decision price fails the cycle closed. All
evaluations, including `WATCH`, remain in SQLite; every skipped market is recorded
with its rejection reason. Shadow mode never sends Discord messages. Add
`--send-discord` only after backtesting has persisted an approval for the exact
calibration profile; unapproved models fail closed even when delivery is requested.

The research command classifies a 5% absolute trailing-return threshold for
`uptrend`, `range`, and `downtrend`, plus 40% and 80% annualized realized-volatility
boundaries for `low`, `typical`, and `high` volatility. The exact thresholds and
source window are stored with every regime observation.

Both commands deliberately run one cycle and exit. Schedule research hourly and
shadow evaluation every five minutes with `launchd`, cron, or another supervisor.
This keeps failures observable, prevents overlapping runs, and avoids repeatedly
downloading the full research window during fast market scans. Review accumulated
forward coverage with:

```bash
uv run pms paper-alert-status --series KXBTC --symbol BTC
```

Run `pms backtest` first to produce held-out calibration profiles in the same
database. Regime coverage is forward evidence gathered over time; adding the
runner does not itself establish that the model has survived multiple real market
regimes.


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
correct. Live evaluation uses the configured fee coefficient. Backtests instead
select scheduled series/event fees at each signal and execution timestamp and
reproduce Kalshi's upward cent rounding.
The configured uncertainty margin is retained only for manual evaluations and
explicit `--allow-uncalibrated` research. Calibrated backtests and live evaluations
derive probability-specific margins from settled training outcomes.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Current operating stage

Run the calibrated paper-alert command on a fixed schedule and review
`paper-alert-status` until untouched forward observations cover the intended
trend/volatility matrix. This is an empirical operating campaign, not a claim of a
validated edge.

