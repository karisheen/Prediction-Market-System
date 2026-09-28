# Phase 1 protocol v2 (frozen)

Machine-readable source of truth: [`experiments/phase1/protocol-v2.json`](../experiments/phase1/protocol-v2.json).
If this document and the JSON disagree, the JSON governs. v2 is the default Phase 1 protocol.

| Field | Value |
| --- | --- |
| `protocol_version` | `pms-phase1-v2` |
| `protocol_id` | `sha256:ca38fcdf3ef1b0659270d1a31a8b1d4cb418b0dffdd06560f5898c3deb5421fe` |
| `frozen_at` | `2026-09-26T08:07:09+00:00` |
| Supersedes | `pms-phase1-v1`, `sha256:531a074fbaa7be281ca15df07c6ebf8c9385278903b8c9732371130f805991d1`, frozen `2026-09-26T07:58:04+00:00` ([`protocol-v1.json`](../experiments/phase1/protocol-v1.json), [doc](phase1-protocol-v1.md)) |
| Scope | research-only, `KXBTC` / `BTC`, terminal-range contracts |
| Model | `crypto-terminal-range-market-anchor-arithmetic-60s`, version `2.1.0`, `structural_weight` 0.5 |

The protocol was frozen before any local database read and before any outcome was inspected.

## Why v2 replaces v1

This replacement happened before any outcomes existed. v1 was never used for analysis, and no
outcome or local database was read under it. v1 is preserved byte-for-byte with its original id.
v2 fixes five problems in v1:

- v1 did not pin the production feature recipe, which was requested before freeze.
- v1 costed paper execution at the YES ask only, although the engine chooses YES or NO.
- v1 kept the first copy of conflicting duplicate observations, so the result depended on input
  order.
- v1 left the freshness of the follow-on execution snapshot unbounded.
- v1 allowed regime intervals, although regimes are a few categories, not independent clusters.

All other rules are carried over from v1 unchanged.

`supersedes` also records the SHA-256 of the v1 JSON file (`9dc37ce8…`) and of the v1 document
(`bc60f9b5…`), so the preserved bytes can be checked.

## Trial record

`trial_record` lists every protocol identity that existed before this freeze. None was used for a
comparison report, a local database read, or an outcome inspection.

| Identity | Frozen at (UTC) | Status |
| --- | --- | --- |
| v1 `sha256:531a074f…` | 07:58:04 | Superseded by v2; preserved byte-for-byte |
| v1 `sha256:5c41045c…` | 07:59:49 | Withdrawn in-place edit of v1, reverted to the original before use |
| v2 `sha256:10cdb564…` | 08:03:02 | Withdrawn provisional v2 draft, replaced in place before use |
| v2 `sha256:ca38fcdf…` | 08:07:09 | Active (this protocol) |

The final v2 differs from the `10cdb564` draft in three ways. It adds the trial record and the v1
byte hashes. The execution price limit uses the engine's cent-rounded whole-contract fee. Two
different execution snapshots with the same timestamp now produce a nonfill instead of an
order-dependent choice.

## Identity

`protocol_id = "sha256:" + content_id(manifest without protocol_id)`, where `content_id` is
`prediction_market_system.evidence.content_id`: SHA-256 of
`json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)` over the parsed JSON.
Every top-level field except `protocol_id` is hashed, including `frozen_at`, `supersedes`, and the
narrative rule fields. `load_protocol` recomputes the id and rejects a mismatch.

## Production feature recipe

Production rows (`retrospective-local`, `forward-holdout`) must use the recipe produced by
`ResearchContext.to_crypto_snapshot`. `expected_annual_return` is 0.0 (top level).
`production_features` uses the same key names as `inputs.crypto.feature_recipe`, so each value can
be compared directly:

| `production_features` key | Value |
| --- | --- |
| `semantics` | `point-in-time-complete-windows-v2` |
| `spot_interval_seconds` | 60 |
| `realized_interval_seconds` | 3600 |
| `realized_window_seconds` | 2592000 (30 days) |
| `volatility_selection` | `matching-interval-dvol-else-realized` |

A production row is excluded (`feature_recipe_mismatch`) unless all of the following hold:

- The expected annual return in the recipe and in the crypto input is 0.0.
- `inputs.crypto.feature_recipe` equals `inputs.recipe.features`.
- Every pinned key matches.
- The research context reproduces the pinned values and the same selected volatility kind,
  provider, and window.

`optional_max_age_seconds` varies by CLI path, so it is recorded but not pinned.

The source-selection branches are declared in advance. In the `implied` branch, an admissible
matching-interval DVOL implied volatility was selected. In the `realized` branch, the 30-day hourly
realized volatility was selected. Each distinct combination of selected volatility kind, provider,
window, and `recipe_id` is a separate branch identity, reported with its own row, event, and date
counts and descriptive point metrics. Branches are pooled only because they follow one fixed
selection rule. They are never collapsed into one recipe identity and never select the population
or the conclusion.

Synthetic fixture rows are exempt from this rule only when explicitly labelled synthetic. They are
never pooled with production rows.

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
bid/ask midpoints. It is descriptive only and is **not** an executable price. This probability
baseline is separate from paper execution, which uses the side the engine chose (see Costs).

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
4. Production rows carry the pinned feature recipe above (`feature_recipe_mismatch`).
5. Explicit venue `event_id`; no fallback to market id or contract label (`missing_event_id`).
6. Explicit `observation_end_at`; `observation_start_at` null or exactly 60 s before it
   (`missing_observation_provenance`).
7. Decision before averaging begins and at least 300 s before `min(expires_at, observation_end_at)`
   (`timing_ineligible`).
8. YES ask present, finite, in `[0, 1]`, with displayed size ≥ 1 (`no_displayed_ask`).
9. Consistent quotes (`yes_bid <= yes_ask`, `no_bid <= no_ask`, all finite in `[0, 1]`) and valid
   structural, blend, and midpoint probabilities with `lower <= probability <= upper`
   (`invalid_quote` / `invalid_arm`). A bad arm excludes the whole row.
10. Decision time inside the source kind's window (`outside_window`).
11. Duplicate `observation_id`, checked across every input row (adapter-excluded rows included)
    before any other rule: if every copy is identical (`content_id` equal), keep exactly one and
    exclude the rest (`duplicate_observation`). If any copies disagree, exclude **all** copies
    (`duplicate_conflict`). The result never depends on input order.
12. Every recommendation state, including WATCH, is eligible. No selection on entry or edge.

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
  are descriptive only: point metrics with row, event, and date counts. **No** regime intervals or
  bootstrap, because regimes are a few categories and not independent clusters.

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

## Costs and execution (Phase 1C only, after 1B)

`decision_costs` mirror the engine defaults with a 0.07 quadratic fee coefficient and `fee_rate`
0.0. Scenarios:

| Scenario | Slippage (bps) | Latency (s) | Fee multiplier | Resolution haircut |
| --- | --- | --- | --- | --- |
| base | 25 | 30 | 1.0 | 0.01 |
| adverse | 50 | 60 | 1.5 | 0.02 |
| severe | 100 | 120 | 2.0 | 0.03 |

The `execution` object in the JSON is authoritative. In summary:

- **Entry and side.** Only rows whose recorded state is ENTER YES or ENTER NO create a paper order.
  The order is on `opportunity.side`, which the engine chose. YES uses `yes_ask`/`yes_ask_size`; NO
  uses `no_ask`/`no_ask_size`. WATCH rows create no order but stay in probability evidence.
- **Fill snapshot.** `executable_after` is decision time plus the scenario's latency. The fill uses
  the snapshot for the same market with the **earliest** timestamp that meets all of these:
  - observed at or after `executable_after`;
  - observed no more than 120 s (`maximum_input_age_seconds`) after `executable_after`;
  - observed no later than `min(expires_at, observation_end_at) − 300 s`, so at least 300 s of
    entry horizon remain at execution;
  - observed at or before the report as-of time.

  Identical snapshots at that timestamp count once. If snapshots at that timestamp differ, the order
  does not fill (`execution_snapshot_conflict`). Later snapshots are never searched for a better
  price. The order also does not fill if no snapshot qualifies (`no_execution_snapshot`), or if the
  fill snapshot lacks a valid side ask (`side_ask_unavailable`), shows zero side size
  (`insufficient_liquidity`), has non-finite, out-of-range, or crossed quotes
  (`invalid_execution_snapshot`), or fails the price limit (`limit_exceeded`).
- **Quotes preserved.** The chosen snapshot's full set of bids, asks, sizes, and `observed_at` is
  recorded unchanged next to the decision-time quotes.
- **Price limit.** The order fills only if the conservative probability, minus the effective cost at
  the later ask, minus the haircut, is at least `min_conservative_edge`. The conservative
  probability is `lower_probability_yes` for YES and `1 − upper_probability_yes` for NO. The
  effective cost is the engine's one-contract all-in cost at the later ask:
  `ask·(1 + fee_rate + slippage/10000)` plus the fee `fee_multiplier·0.07·ask·(1 − ask)` rounded up
  to the cent, as `CryptoThresholdEngine` prices whole-contract (Kalshi) candidates.
- **Quantity.** Intended units are the most whole contracts whose all-in cost fits
  `suggested_max_exposure`, using the engine's cent-rounded fees. Filled units are capped at 10 % of
  the later side size and by the remaining per-event exposure, processed in
  `(decision time, observation_id)` order. Zero filled units is a nonfill
  (`insufficient_liquidity`); fewer than intended is a partial fill.
- **Settlement.** Each filled contract pays 1 if its side wins, minus the haircut. Rows unresolved
  at the report as-of time stay open and unscored.

Every scenario is reported and conclusions use the most adverse one. The top-level
`minimum_ask_size` (1) gates baseline eligibility. `decision_costs.minimum_ask_size` is recorded
engine policy and does not re-gate recorded states. Costs never change probability metrics or
eligibility.

## Replacement

This file and its JSON are immutable once any analysis has used them. Changes to eligibility,
metrics, windows, or constants need a new `protocol_version`, a new file path, a new `protocol_id`,
and an explicit trial record. v1 and v2 both stay unchanged.
