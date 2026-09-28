# Phase 1 protocol v1 (frozen)

Machine-readable source of truth: [`experiments/phase1/protocol-v1.json`](../experiments/phase1/protocol-v1.json).
If this document and the JSON disagree, the JSON governs.

| Field | Value |
| --- | --- |
| `protocol_version` | `pms-phase1-v1` |
| `protocol_id` | `sha256:531a074fbaa7be281ca15df07c6ebf8c9385278903b8c9732371130f805991d1` |
| `frozen_at` | `2026-09-26T07:58:04+00:00` |
| Scope | research-only, `KXBTC` / `BTC`, terminal-range contracts |
| Model | `crypto-terminal-range-market-anchor-arithmetic-60s`, version `2.1.0`, `structural_weight` 0.5 |

The protocol was frozen before any local database read and before any outcome was inspected.

## Identity

`protocol_id = "sha256:" + content_id(manifest without protocol_id)`, where `content_id` is
`prediction_market_system.evidence.content_id`: SHA-256 of
`json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)` over the parsed JSON.
Every top-level field except `protocol_id` (including `frozen_at` and the narrative rule fields) is
hashed. `load_protocol` recomputes and rejects a mismatch.

## Question

Does the fixed engine's structural probability, or its 50/50 log-odds blend, forecast resolved
`KXBTC` 60-second averaged terminal-range outcomes better than the displayed executable YES ask?

| Arm | Source |
| --- | --- |
| `structural` | `opportunity.forecast.structural_probability_yes` |
| `blend` | `opportunity.forecast.probability_yes` |
| `market` (primary baseline, `yes_ask`) | `opportunity.market.yes_ask` |
| `recorded_midpoint_anchor` (secondary) | `opportunity.forecast.market_probability_yes` |

The primary baseline is the displayed YES ask, a price a buyer could actually pay at decision time.
The secondary baseline is the engine's recorded market anchor, the log-odds average of available
bid/ask midpoints. It is descriptive only and is **not** an executable price.

Paired differences: `structural_minus_market`, `blend_minus_market`, `blend_minus_structural`
(negative favours the first arm).

## Windows (UTC, half-open `[start, end)`, applied to decision time)

| Window | Start | End |
| --- | --- | --- |
| Training (calibration evidence constraint) | 2026-05-29 | 2026-08-27 |
| Validation | 2026-08-27 | 2026-09-26 |
| Validation fold 1 / 2 | 2026-08-27 / 2026-09-11 | 2026-09-11 / 2026-09-26 |
| Retrospective archive | 2026-07-01 | 2026-09-26 |
| Report as-of | 2026-09-26T00:00:00Z | |
| Future holdout | 2026-09-27 | 2026-11-26 |
| Holdout fold 1 / 2 | 2026-09-27 / 2026-10-27 | 2026-10-27 / 2026-11-26 |

The training and validation windows only constrain which existing calibration evidence may be
recorded against a forecast. The protocol does not claim that such data exist or were fitted.

## Source kinds

- `synthetic-fixture`: declarative synthetic observations run through the real engine. Never
  empirical evidence.
- `retrospective-local`: previously recorded `forward-evaluation` ledger rows in the archive window.
  This is already-seen local evidence and is never described as untouched.
- `forward-holdout`: forward-evaluation rows in the holdout window recorded before outcomes existed.
  Not implemented in Phase 1 and never inferred from retrospective provenance.

## Timestamp rules

- Every timestamp needs an explicit offset and is normalized to UTC. Naive or unparseable values
  exclude the row (`invalid_timestamp`).
- Decision time is `opportunity.market.observed_at`; `forecast.generated_at` must equal it
  (`forecast_time_mismatch`).
- Event date is the UTC date of the explicit `observation_end_at`. All rows of an event must share
  it; otherwise the whole event is excluded (`event_date_conflict`) instead of counting twice.
- Fold membership uses the event's earliest decision time.

## Availability rules

- Crypto input and every research-context source must be available at or before decision time
  (`future_input`); input age must be within 0–120 s (`stale_input`). Synthetic fixtures do not
  bypass freshness.
- `retrospective-local` rows need a complete content-addressed research context whose `content_id`
  equals `inputs.crypto.input_provenance.research_context_id`, whose `spot_end_at` equals the crypto
  observation time, and which passes `ResearchContext.validate_at(decision_time, 120)`
  (`research_context_unavailable` / `research_context_invalid`). Legacy rows without provenance are
  excluded, never reconstructed.
- Synthetic fixtures may use manual features only when explicitly labelled synthetic.
- `run_manifest_sha256 == content_id(run_manifest)` and `input_id == content_id(inputs)`
  (`identity_mismatch`).
- `retrospective-local` requires evidence class `forward-evaluation`; `historical-evaluation`,
  `manual-evaluation`, and `ambiguous-legacy` are excluded (`evidence_class`).
- A recorded calibration profile must match symbol, model name, version, and recipe id, have
  `training_start >= 2026-05-29`, `cutoff_at <= min(training_end, decision time)`, and at least 30
  independent events; otherwise the row is excluded (`calibration_profile_invalid`). Rows with the
  fixed margin stay eligible. Calibration only changes the uncertainty interval and recommendation,
  never the scored probabilities.
- An outcome counts only when `result` is `yes`/`no` and it was observed at or before the report
  as-of time; otherwise the row is unresolved (kept, unscored). A resolution observed at or before
  decision time excludes the row (`outcome_leak`).

## Eligibility (all required; failures are listed with a reason, never dropped)

1. `series_id == KXBTC` and crypto symbol `BTC` (`out_of_scope`).
2. Terminal range with `0 < lower_bound < upper_bound` and a 60 s settlement window
   (`unsupported_contract`).
3. Exact model name, version `2.1.0`, `structural_weight` 0.5 in both the recorded recipe and engine
   config, and `recipe_id` equal to the recorded recipe fingerprint (`model_mismatch`).
4. Explicit venue `event_id`; no fallback to market id or contract label (`missing_event_id`).
5. Explicit `observation_end_at`; `observation_start_at` null or exactly 60 s before it
   (`missing_observation_provenance`).
6. Decision before averaging begins and at least 300 s before `min(expires_at, observation_end_at)`
   (`timing_ineligible`).
7. YES ask present, finite, in `[0, 1]`, with displayed size ≥ 1 (`no_displayed_ask`).
8. Consistent quotes (`yes_bid <= yes_ask`, `no_bid <= no_ask`, all finite in `[0, 1]`) and valid
   structural, blend, and midpoint probabilities with `lower <= probability <= upper`
   (`invalid_quote` / `invalid_arm`). A bad arm excludes the whole row.
9. Decision time inside the source kind's window (`outside_window`).
10. Duplicate `observation_id`: keep the first by `(observed_at, observation_id)`
    (`duplicate_observation`).
11. Every recommendation state, including WATCH, is eligible. No selection on entry or edge.

Unsupported structures: threshold and touch-barrier contracts; ranges without positive increasing
bounds or an explicit fixed-time terminal observation; any averaging other than explicit simple
arithmetic averaging of the 60 s before observation end; non-binary or non-$1 markets; inactive
markets; metadata from the future; conflicting observation clocks; elapsed averaging prefixes.

## Metrics

- Brier `(p - y)^2`; log loss with `p` clamped to `[1e-12, 1 - 1e-12]`.
- Scored population: resolved eligible rows with all three arms valid. Unresolved eligible rows are
  reported in observations and coverage.
- Event weighting: average within each event, then equally across events. Repeated contracts or
  timestamps never increase the independent event count.
- Calibration: 10 equal-width bins per arm; each row weighted `1 / scored rows in its event`; each
  bin reports forecast count, event count, weight sum, weighted mean probability, and weighted mean
  outcome.
- Coverage: eligible/resolved/unresolved rows and events, exclusions by reason, recommendation-state
  mix, distinct event dates, qualifying folds.

## Uncertainty

- 2000 replicates, seed 1729, Python `random.Random(1729)`, clusters sorted by stable id first.
  95 % percentile intervals.
- Primary: event cluster bootstrap.
- Secondary: UTC event-date block bootstrap; statistic is the sum of sampled event scores divided by
  the sampled event count.
- The same resamples are used for every arm and paired difference.
- Fewer than two clusters gives a null interval, not a zero-width one.
- Regimes (`market_regime` trend and volatility; missing is `unknown`, never inferred from outcomes)
  are descriptive. Regime intervals only with at least two independent event clusters.

## Minimums and folds

| Requirement | Value |
| --- | --- |
| Scored resolved events | 20 |
| Distinct event dates | 20 |
| Calibration profile independent events | 30 |
| Qualifying folds | 2 |
| Scored events for a fold to qualify | 10 |

## Conclusions

- `synthetic-fixture`: always `inconclusive`; pipeline demonstration only.
- `retrospective-local`: always `inconclusive` for any unseen-event claim, whatever the point
  estimate, p-value, or interval sign.
- `forward-holdout`: `inconclusive` unless all minimums hold. A directional statement additionally
  needs both the event and date-block intervals to exclude zero in the same direction for both Brier
  and log loss.
- Low event or date counts, missing qualifying calibration or holdout folds, and unresolved or
  excluded coverage are always listed as limitations.
- No fitting, tuning, threshold search, or model selection happens under this protocol.

## Costs (Phase 1C only, after 1B)

`decision_costs` mirror the engine defaults with a 0.07 quadratic fee coefficient. Scenarios:

| Scenario | Slippage (bps) | Latency (s) | Fee multiplier | Resolution haircut |
| --- | --- | --- | --- | --- |
| base | 25 | 30 | 1.0 | 0.01 |
| adverse | 50 | 60 | 1.5 | 0.02 |
| severe | 100 | 120 | 2.0 | 0.03 |

Fills use the actual later YES ask at or after decision time plus latency, with slippage and
`fee_multiplier × 0.07 × p(1 − p)` fees under the engine's rounding, capped at 10 % of the later
displayed size. With no later snapshot or a price above the limit, the order does not fill; partial
displayed size is a partial fill. Every scenario is reported and conclusions use the most adverse
one. The top-level `minimum_ask_size` (1) gates baseline eligibility; `decision_costs.minimum_ask_size`
(10) gates paper decisions only. Costs never change probability metrics or eligibility.

## Replacement

This file and its JSON are immutable once any analysis has used them. Changes to eligibility,
metrics, windows, or constants need a new `protocol_version`, a new file path, a new `protocol_id`,
and an explicit trial record. v1 stays unchanged.
