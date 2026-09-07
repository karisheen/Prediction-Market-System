# Prediction Market System

A crypto-first prediction-market research, calibration, backtesting, and
decision-support system. It combines public Kalshi market data with Coinbase and
Deribit research inputs, estimates contract probabilities, applies conservative
execution and risk checks, preserves the full decision trail in SQLite, and can
send approved opportunities to Discord for manual review.

The system is **read-only with respect to trading venues**. It does not authenticate
to an exchange, place orders, hold funds, or manage positions.

## Scope at a glance

| Area | Current scope |
| --- | --- |
| Venue | Kalshi public REST APIs; KXBTC is the operational focus |
| Contracts | Fixed-time crypto ranges, upper/lower terminal thresholds, and explicitly defined touch barriers |
| Market data | Paginated markets and events, executable order books, historical candles, rules, resolutions, and fee changes |
| Research data | Coinbase spot candles and realized volatility; Deribit DVOL, funding, basis, and open interest |
| Models | Lognormal terminal probabilities, geometric-Brownian first-passage probabilities, and market-anchored log-odds blending |
| Evidence | No-look-ahead walk-forward replay, event-grouped calibration, held-out model approval, and forward shadow observations |
| Decisions | `WATCH`, `ENTER YES`, or `ENTER NO`, with fees, slippage, uncertainty, liquidity, expiry, and exposure constraints |
| Operations | Hourly research refreshes and independent five-minute market scans, each as an observable one-shot process |
| Output | Append-oriented SQLite audit records and optional manual-review Discord alerts |
| Execution | No automated trading, exchange credentials, wallet access, or portfolio management |

## Core capabilities

- Parse Kalshi contract metadata and resolution rules into explicit terminal-range,
  terminal-threshold, or touch-barrier models; ambiguous contracts fail closed.
- Evaluate two-sided and one-sided executable order books without inventing missing
  liquidity.
- Blend structural and market-implied probabilities, then widen them with
  probability-specific held-out calibration uncertainty.
- Apply venue fees, slippage, resolution haircuts, minimum-liquidity checks,
  fractional Kelly sizing, per-market limits, and aggregate per-event exposure caps.
- Ingest settled history by event while sampling dense mutually exclusive ladders
  without using resolved outcomes to choose contracts.
- Replay point-in-time data with delayed adverse execution, volume-constrained
  partial fills, independent-event calibration, return metrics, and event-weighted
  Brier scores.
- Persist explicit model approval or rejection evidence. Discord delivery requires
  approval for the exact calibration profile used by the live forecast.
- Run high-frequency shadow scans without Discord delivery while retaining every
  forecast, recommendation, candidate, rejection, and failure reason.

This is a testable research and decision-support baseline—not evidence of a durable
trading edge. Real-money use requires independent review, substantially broader
historical and forward validation, and a separate execution and position-management
system.

## System workflow

```text
HISTORICAL EVIDENCE
Kalshi events + contracts + candles + outcomes + fees ─┐
Coinbase/Deribit point-in-time research ────────────────┴─> walk-forward replay
                                                               │
                                                               ├─> calibration profiles
                                                               └─> model approval/rejection

LIVE RESEARCH (hourly)
Coinbase + Deribit ─> point-in-time research context ─> regime snapshot ─> SQLite

LIVE EVALUATION (every five minutes)
completed Coinbase decision price + Kalshi markets/order books
                              │
                              v
contract parser ─> probability model ─> calibrated uncertainty ─> risk/cost checks
                                                                        │
                                      SQLite audit <────────────────────┤
                                                                        ├─> WATCH
                                                                        └─> entry candidate
                                                                              │
                                                  exact profile approved? ─────┤
                                                                              ├─ no: audit only
                                                                              └─ yes: Discord manual review
```

SQLite is authoritative. Discord is only a notification surface and never
receives venue credentials, wallet keys, or authority to trade.

## Setup

Requirements:

- `uv`
- Python 3.11 or newer (managed automatically by `uv`)

```bash
uv sync
cp .env.example .env
uv run pms init-db
```

The database defaults to `data/prediction_markets.db`. A Discord webhook is
optional; local evaluation, ingestion, backtesting, and shadow operation do not
require one.

## CLI map

| Command | Purpose |
| --- | --- |
| `init-db` | Initialize or migrate the SQLite audit store |
| `evaluate` | Evaluate a manually supplied binary market snapshot |
| `kalshi-markets` / `kalshi-inspect` | Browse public live Kalshi markets and metadata |
| `kalshi-evaluate` | Evaluate one live Kalshi contract |
| `kalshi-sync-history` | Ingest settled events, contract metadata, sampled candles, rules, outcomes, and fee changes |
| `sync-research-data` / `research-context` | Persist research inputs and reconstruct an as-of context |
| `backtest` | Run walk-forward calibration, execution replay, and model approval |
| `paper-alert-research` | Refresh slower research data and persist the current regime |
| `paper-alerts` | Scan all open contracts; shadow-only unless `--send-discord` is supplied |
| `paper-alert-status` | Report cycles since the previous request, all-time resolved alert profitability, and regime coverage |
| `paper-alert-maintain` | Preview or apply bounded detailed `WATCH` retention with daily rollups |
| `paper-alert-archive` / `paper-alert-validate` | Archive completed UTC days with a work budget; run the frozen validation campaign |
| `history` | Review persisted forecasts and recommendations |
| `discord-test` | Send a non-trading webhook health check |
| `doctor` | Read-only integrity, capacity, freshness, cycle-gap, and unresolved-delivery telemetry |
| `explain-forecast` | Replay one forecast's manifest, contract snapshot, and research-input context |
| `shadow-report` / `campaign-report` / `compare-runs` | Forward ledger scoring, frozen campaign and holdout audit, paired run comparison |
| `db-backup` / `db-restore` | Online SQLite backup with a restore drill; restore to a new path only |
| `alerts-unresolved` / `alerts-reconcile` | List uncertain Discord deliveries and record the operator-observed outcome |

## Recommended operating sequence

1. Initialize SQLite and configure conservative venue/risk assumptions.
2. Ingest Kalshi event history and matching Coinbase/Deribit research history.
3. Run walk-forward backtests to create calibration profiles and persisted approval
   decisions.
4. Schedule `paper-alert-research` hourly.
5. Schedule `paper-alerts` every five minutes in its default shadow mode.
6. Schedule `paper-alert-maintain --apply` daily to bound detailed `WATCH` storage.
7. Review forward candidates in SQLite and track regime coverage with
   `paper-alert-status`; enable `--send-discord` only when the exact live profile
   has passed the configured approval gates.

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

For a supported fixed-time terminal range or threshold, fetch current Kalshi
quotes and evaluate them against user-supplied crypto inputs:

```bash
uv run pms kalshi-evaluate \
  --ticker KALSHI_MARKET_TICKER \
  --symbol BTC \
  --spot 110000 \
  --volatility 0.55
```

Live Kalshi evaluation requires a matching held-out calibration profile for the
symbol, structural model, model version, and recipe identity. `pms backtest`
persists research profiles and gate results; those runs are not deployable
approval. Managed delivery additionally requires an active frozen validation
campaign and a matching deployment-policy fingerprint. `--allow-uncalibrated`
permits local research with the configured fixed margin, but uncalibrated
Discord alerts are rejected.

Terminal markets use the probability of finishing within a bounded range or beyond
a threshold at the contract's benchmark observation time, which is parsed from
the rules and kept separate from trading close, expected settlement, and the time
the outcome becomes available. Contracts that settle on an average over a stated
window use the discrete arithmetic-average moments of that window rather than the
terminal spot distribution. Physical drift and numerically extreme tails are
supported. A market with only one executable side remains evaluable on that side;
missing liquidity is never synthesized for the other side.

Classification fails closed. Range contracts require positive, increasing bounds
and an explicit fixed-time terminal observation. Averaging contracts require an
unambiguous observation window. Touch/path-dependent contracts are classified as
barriers but are not evaluable: deciding whether a barrier was crossed during the
contractual observation period needs benchmark path history that Kalshi does not
publish, so the current spot alone must never stand in for it. Ambiguous rules
remain unsupported rather than being routed to a mathematically incorrect model,
and unsupported contracts remain visible in archived market universes. Live
evaluation and historical replay share one eligibility function.

The structural model assumes geometric Brownian motion with constant volatility
and drift over the remaining observation horizon. It does not model jumps,
exchange outages, or intraperiod volatility changes. Those mismatches must remain
part of the uncertainty and resolution-risk review. Every forecast carries a
recipe identity: the structural model, model version, and a fingerprint of all
forecast-affecting configuration and research-input provenance. Execution, fee,
slippage, sizing, and freshness assumptions are a separate deployment-policy
identity. Corrected model behavior changes the model version; a changed
operational policy invalidates delivery approval without invalidating the
recipe. Legacy calibration profiles and approval decisions remain preserved but
cannot be reused when required identity or campaign metadata is missing.

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

The assembler only selects source timestamps at or before `--as-of`, using the
provider revision that had been retrieved by then. Required spot data fails closed
when missing, stale, or future-dated, and realized volatility requires a complete,
correctly spaced trailing window: missing, duplicate, irregular, or incomplete
intervals fail closed rather than being interpolated. Changed provider payloads are
stored as additional revisions instead of overwriting the earlier observation.
Optional DVOL, funding, derivatives, and event inputs are omitted with warnings
when missing or stale. A current derivatives or event
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
  --spot-interval 1 \
  --research-interval 60 \
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

`--period` is the Kalshi quote/execution grid. `--spot-interval` (default 1
minute) and `--research-interval` (default 60 minutes) are independent Coinbase
series used for the live decision price and realized-volatility features. Those
two research intervals must match live scanning for historical evidence to be
recipe-compatible; changing either one is an intentional incompatibility.

Run `kalshi-sync-history` and `sync-research-data` first. Research coverage must
begin early enough to provide the complete realized-volatility window at every
training and test timestamp. For each fold, only markets whose outcomes settled
by the training cutoff are eligible for calibration. The five-minute default
calibration lead matches KXBTC's completed hourly market candle before its
five-minute expected-expiration timestamp. Each event contributes one
outcome-weighted sample per structural model, so hundreds of mutually exclusive
contracts from one range ladder cannot masquerade as independent evidence. The
following non-overlapping test window remains untouched until evaluation.

Calibration samples are isolated by symbol, structural model, model version, and
recipe identity, then partitioned into fixed probability regions and horizon
bands; regions are never merged to reach a sample threshold, because opposite
conditional errors within a ladder would cancel. Each region clusters samples by
event and compares its mean forecast with a Bonferroni-adjusted Hoeffding
interval for the event-level outcome frequency; the larger distance to either
interval bound becomes the uncertainty margin, and sparse regions receive a
margin that makes them unusable. A forecast whose probability or horizon falls
outside every supported region receives the maximal margin instead of borrowing
the nearest bin. The default requires 30 independent resolved events per model
at 95% confidence. Profiles fitted for a different recipe or model version, or
with research-only inputs, cannot approve the live recipe. Signals without a
qualifying profile fail closed. Use `--allow-uncalibrated` only to inspect
fixed-margin behavior.

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
Passing those research gates records `accepted_for_paper_alerts` on the run.
That is not managed-delivery approval. Deployable approval also requires an
active frozen `paper-alert-validate` campaign whose stored deployment-policy
fingerprint matches the live engine, and a non-research-only profile.

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

`--interval` is the research/realized-volatility cadence (hourly by default), not
the decision price. Each evaluation adds a completed one-minute Coinbase decision
price, archives recently completed one-minute Kalshi candles at actual receipt
time, follows Kalshi cursors across up to 1,000 open markets, and evaluates every supported
range or threshold contract with the same eligibility rules as historical replay.
Freshness is enforced at each market's evaluation boundary, not just at scan
start: a decision price that becomes stale during a slow scan, or any future-dated
input, fails the remaining markets closed. Every evaluation and skipped-market
reason is initially recorded in SQLite, and every forecast is appended to a
compact forward-evidence ledger that exists independently of Discord delivery. The managed
deployment retains detailed `WATCH` evaluations for 14 days, then preserves daily
counts while keeping entry candidates, deliveries, failures, and model evidence
indefinitely. The managed schedule is shadow-only and uses neither Discord delivery
flag. Enable `--send-discord` through a deliberate deployment-policy change only
after backtesting has approved the exact calibration profile.
`--allow-unapproved-discord` remains an explicitly initiated local/manual-review
override and still cannot execute trades.

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

The first request reports all recorded cycles. Later requests count cycles from the
persisted series/symbol checkpoint. Alert totals are all-time so an alert that was
unresolved during an earlier request is included after its Kalshi resolution is
archived. Resolved, profitable, and unresolved counts are shown separately; an
alert is profitable when its recommended side matches the archived resolution.

Validation evidence maintenance is automated with two idempotent commands:

```bash
uv run pms paper-alert-archive \
  --series KXBTC --symbol BTC \
  --campaign-start 2026-08-07T00:00:00Z

uv run pms paper-alert-validate \
  --series KXBTC --symbol BTC \
  --campaign-start 2026-08-07T00:00:00Z \
  --send-discord
```

Preview bounded `WATCH` retention before applying it:

```bash
uv run pms paper-alert-maintain \
  --series KXBTC \
  --watch-retention-days 14
```

Archive completed UTC days daily. The archive treats its catch-up size as a work
budget: it repairs the oldest missing or incomplete windows first instead of
permanently abandoning them, and a window is complete only when its market and
spot series are contiguous. Validation can run weekly: before the configured
chronological window is complete it updates one Discord readiness message; once
ready, it runs the walk-forward backtest and replaces that message with the exact
per-model approval gates and decisions.

The first `paper-alert-validate` run preregisters the campaign: the start date,
fold geometry, latency and participation assumptions, every approval gate, the
engine configuration, model version, and deployment-policy fingerprint are
hashed into a campaign identity that is stored and linked to each validation
run. Ordinary `pms backtest` research runs cannot create deployable approval.
Later campaign runs with different settings are refused so gates cannot be
loosened after seeing results; `--replace-campaign` explicitly supersedes the
registration while preserving the old one as evidence, and prior approvals under
the superseded campaign fail closed. Held-out events consumed by a validation
run are recorded, so a rerun over the same events is reported as a descriptive
re-examination and cannot approve a profile.

Run `pms backtest` first to produce held-out calibration profiles in the same
database. Regime coverage is forward evidence gathered over time; adding the
runner does not itself establish that the model has survived multiple real market
regimes.

The managed macOS deployment uses a verified release directory and an atomic
`app` symlink rather than copying source over the running installation. See the
[operational hardening change record](docs/operations-hardening.md) for deployment,
retention, rollback, verification, and future infrastructure work.

## Discord delivery

Discord messages are manual-review instructions, not trade executions. An entry
alert includes the recommended side, maximum price, paper exposure cap, exact YES
condition, event ticker, contract ticker, calibrated probability interval, edge,
regime, costs, and resolution-risk context. Delivery is idempotent by market and
updates an existing Discord message when the recommendation changes.

Every delivery path passes one authorization check that requires a persisted
approval for the exact calibration profile and a research context that the
immutable store can reproduce. A delivery attempt is claimed in SQLite before any
network call. If Discord's response is ambiguous (timeout, 5xx, connection loss),
the attempt is recorded as `uncertain`, further alerts for that market are
blocked, and `doctor` reports the count. The system never guesses the remote
outcome: use `alerts-unresolved` to list attempts and `alerts-reconcile` to record
what the channel actually shows. Webhook tokens and other credentials are redacted
from logs, exceptions, and persisted failure records.

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

Add `--send-discord` to `pms evaluate` for a manual evaluation, or to
`pms paper-alerts` for scheduled delivery. `WATCH` evaluations never create
notifications. The paper-alert runner additionally requires a persisted approval
for the exact calibration profile; missing or rejected approval is audited and
fails delivery closed.

## Evidence and audit

Every backtest, validation, and forward evaluation writes an immutable run
manifest containing the git revision and source hashes, model and recipe
identities, engine and backtest configuration, dataset boundaries, and campaign
identity. Forecasts reference their exact contract snapshot and a content-
addressed research context, so a forecast can be replayed with the inputs it
actually used:

```bash
uv run pms doctor
uv run pms explain-forecast FORECAST_ID
uv run pms shadow-report --series KXBTC
uv run pms campaign-report --series KXBTC --symbol BTC
uv run pms compare-runs FIRST_RUN_ID SECOND_RUN_ID
uv run pms db-backup --destination backups/audit.sqlite3
uv run pms db-restore --source backups/audit.sqlite3 --destination restored.sqlite3
```

`shadow-report` scores only forward-shadow ledger observations, including `WATCH`
forecasts, against the same-population market price. It reports forward sample
sizes, event-weighted Brier and log-loss, and how many non-forward ledger rows
were excluded. Historical replay and manual research remain explainable but do
not enter those counts. Notification activity lives in paper-alert status and
Discord delivery records, not in `shadow-report`. Repeated scans of the
same market do not inflate independent evidence counts. `db-restore` writes to a new path and never switches configuration.
Initializing a legacy database takes a verified backup before migrating.

## Configuration

All settings use the `PMS_` prefix. See `.env.example` for database, bankroll,
edge, uncertainty, structural-weight, cost, liquidity, sizing, aggregate event
exposure, spot-freshness, and expiry assumptions. Defaults are deliberately
conservative but are not universally correct.

Live evaluation uses the configured fee coefficient. Backtests select scheduled
series and event fees at each signal and execution timestamp; when Kalshi reports
no fee changes, they fall back to the configured current coefficient. Kalshi fees
are rounded upward to cents.

The configured fixed uncertainty margin is retained only for manual evaluations
and explicit `--allow-uncalibrated` research. Calibrated backtests and live
evaluations derive probability-specific margins from settled training outcomes.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Current operating stage

The intended operating mode is shadow-first: refresh research hourly, evaluate all
supported open contracts every five minutes, and accumulate untouched forward
observations across trend and volatility regimes. Entry candidates are research
observations, not permission to trade.

Model approval is data-dependent and stored in SQLite; the repository never
assumes a model is approved merely because code or tests pass. Discord delivery
remains locked for a profile until its independent-event calibration, held-out
event/fold coverage, return-on-cost threshold, and event-weighted Brier threshold
all pass. This campaign is empirical evidence collection, not a claim of a
validated edge.

