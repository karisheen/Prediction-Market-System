# Independent prospective forward study v1

## Registration and boundaries

The standalone registration is `experiments/forward/protocol-v1.json`, frozen at
**2026-11-05T13:40:12.844226+00:00**, before this study's observations exist. Its identity is
`sha256:e7f440c05ac51bdc8a13456bfa8185e15cc6420c9bffa60a6053cd1103e0056f`.
The identity is `sha256:` plus `evidence.content_id` of every document field except
`protocol_id`. The validator accepts only that immutable identity: recomputing a hash
of an edited document cannot authorize a different study. A new study needs a new
prospective registration, version and path.

This is **independent research**, not supersession of Phase 1, not managed model
approval, and not authorization for Discord or trading. The original paper-validation
campaign and both Phase 1 protocol JSON/documentation pairs remain unchanged. The
v2 model, feature-selection recipe, scoring, bootstrap and cost semantics are reused;
this registration supplies a genuinely prospective population and auditable capture
rules. It does not claim that earlier unrelated evidence has never been inspected.

The exact engine configuration is frozen, including quadratic fee coefficient 0.07,
structural weight 0.5, fixed uncertainty margin 0.05 and all sizing/freshness limits.
SHA-256 pins cover engine, domain, recipe, research, calibration, Kalshi parser and
`uv.lock`. Validation rejects current model-source or dependency-lock drift. Evolving
study storage/reporting code is deliberately not a probability-model source pin.

## Population and schedule

The half-open UTC study window is **2026-11-06 00:00 through 2027-01-05 00:00**.
Schedule one slot each hour at minute **45**, giving **1,440 expected slots**.
The two folds split at **2026-12-06 00:00**, with **720 expected slots each**.
The immutable final report/outcome cutoff is **2027-01-07 00:00 UTC**. Before that
cutoff reports expose operational coverage/status only, never interim outcomes,
probability scores or execution performance. A later invocation does not move the
cutoff or admit later evidence.

For each slot, discover the **complete paginated KXBTC universe**. Retain every
candidate's venue event/market identities, original metadata, supported-contract
classification and exclusion reason, plus contemporaneous spot and its source clocks.
Eligible contracts are explicit 60-second arithmetic-average terminal ranges whose
minimum expiry/observation-end is strictly after slot+300 seconds and no later than
slot+3,600 seconds. Choose the earliest supported event, breaking ties by event ID.
Within it choose at most three ranges by absolute range-midpoint distance to spot,
breaking ties by market ticker. Persist the full selection inputs and their content
hash **before requesting selected books**. Quotes, recommendations, results and
performance never choose the population.

Empty discovery/selection and source failures are no-market/failed slots, not an
excuse to remove expected slots. This study uses the stronger **first persisted
attempt only** rule: no replacement attempts, including after failures or crashes.
A better later book can never replace the originally selected attempt.

## Timing and retained evidence

An eligible attempt starts within 60 seconds after its scheduled slot; it cannot start
early. Decisions must occur by slot+120 seconds. Required source availability and raw
book receipt must be no later than decision time. The original decision book remains
separate from the logical market snapshot stamped with the decision clock.

Forecast generation uses decision time; actual storage persistence must satisfy
`decision_at <= forecast_persisted_at <= decision_at + 5 seconds` and occur **before
observation_start_at**. This intentionally allows persistence after generation rather
than requiring the impossible converse. Source/context freshness remains 120 seconds.
A late capture is retained but never made retrospectively eligible.

Compact, immutable evidence retains:

- Registration and actual persistence clocks, exact frozen protocol and source pins.
- Slot attempts, full prequote selection metadata/spot/hash, and slot results.
- Per-market raw decision books, request/receipt/decision/persistence clocks, fees,
  original rules, forecast inputs, full research contexts and content identities.
- A per-market `observation` with exactly the existing experiment row fields:
  `observation_id`, `event_id`, `market_id`, `observed_at`, `opportunity`, `inputs`,
  `run_id`, `run_manifest`, `run_manifest_sha256`, `input_id`, `research_context`,
  `resolution`. The last field is null at capture; outcomes join separately.
- Independent delayed execution books and explicit failures, not reconstructed books.
- Actual public resolution response receipts and raw revisions; never expiry-backdated
  availability. Conflicting outcomes exclude the market.

The record schema is `pms-forward-evidence-v1`. Each stored record has `record_type`,
`protocol_id`, `record_id`, `persisted_at`, plus its typed payload. `record_id` is the
content hash excluding itself. Natural-key conflicts cannot be hidden by assigning a
new content hash. Registration precedes study start. Selection precedes book requests;
store-generated persistence clocks cannot be caller-backdated. Existing ordinary-alert
or retrospective rows cannot be promoted into eligible prospective evidence.

## Probability and missingness gates

Every selected valid **WATCH and entry** forecast contributes to probability evidence.
No calibration profile is supplied in this independent study. Raw structural/blended
probabilities do not require calibration; the fixed-margin execution analysis is
explicitly research-only and provides no calibrated/live-approval claim. Probability
baseline eligibility requires displayed YES ask size at least one; the separate engine
entry policy retains minimum ask size ten.

Preserve v2 event-weighted Brier and log loss, paired structural/blend/YES-ask
comparisons, and the descriptive recorded-midpoint anchor. Preserve feature-selection
branch identities, descriptive regimes and calibration tables. The primary event
bootstrap and secondary UTC event-date block bootstrap use **2,000 paired replicates,
seed 1729, 95% percentile intervals**. Both intervals must exclude zero in the same
direction for **both metrics** for a directional paired comparison.

Additional minimum gates are **20 resolved events**, **20 distinct event dates**, and
**two qualifying folds with at least 10 resolved events each**. Assign each event to
its earliest-decision fold. Duplicate/conflicting identities and inconsistent event
dates fail closed; unresolved observations remain visible and unscored.

Directional probability statements additionally require **at least 95% valid capture
coverage overall and in each fold**. A valid slot requires complete discovery,
nonempty deterministic selection, and **all** selected-market captures valid and on
time. Missing, late, partial, failed, interrupted and no-market slots stay in the
denominator. The schedule determines absent slots without needing a stored placeholder.
Execution completeness is disclosed separately and never filters probability metrics.
Any permitted probability statement is **conditional on the captured population**.
There is no unconditional policy-profit claim when scheduled slots are missing.

## Execution: observable-book simulation, not fills

Independently request books at decision+30, +60 and +120 seconds for base, adverse and
severe scenarios. Use the earliest returned response for each scenario, received no
later than target+120 seconds and still at least 300 seconds before minimum
expiry/observation-end. Never reuse a decision book or search later for a better price.
Preserve failures for every selected forecast, including WATCH.

Only recorded entries create simulated orders, on the engine's selected YES/NO side.
Reuse v2 base/adverse/severe fees, slippage and resolution haircuts, cent-rounded fees,
whole contracts, 10% displayed-depth participation and fixed market/event exposure
caps. Preserve all-in-cost and conservative price limits. Report every scenario;
conclusions use the most adverse. Do not infer queue priority or actual fills.

Missing execution is **unavailable evidence**, not a zero-P&L non-loss. A valid
observed book failing price/liquidity limits is a distinct simulated nonfill. Unresolved
simulated fills remain open and unscored. An execution-direction claim has the stronger
preregistered gate of **100% valid slot coverage**, admissible execution captures in
**all scenarios for every entry**, and resolutions for **all simulated fills** by the
cutoff. Otherwise execution output is descriptive and explicitly incomplete. Even a
complete result is an **after-cost book-observable simulation**, not deployable profit
or managed approval.

## Public protocol API

`prediction_market_system.forward_protocol` exposes:

- `DEFAULT_FORWARD_PROTOCOL_PATH: Path`
- `load_forward_protocol(path: Path) -> dict[str, Any]`
- `validate_forward_protocol(document: dict[str, Any]) -> dict[str, Any]`
- `forward_protocol_id(document: dict[str, Any]) -> str`
- `scheduled_slots(document, start: datetime | None = None,
  end: datetime | None = None) -> list[datetime]`

Schedule bounds intersect the frozen half-open window and never change its hourly
origin. Naive or inverted bounds are rejected. Empty/disjoint intersections yield an
empty list. Loading rejects duplicate JSON keys, nonfinite values, changed content,
unsupported rehashed registrations and source drift. The API grants no authorization.
