# AGENTS.md

## Purpose

This file defines repository-level instructions for autonomous coding agents working on **Prediction Market System**.

Read this file before making changes.

Also read the relevant portions of:

* `README.md`
* `docs/venue-decision.md`
* `docs/operations-hardening.md`
* `pyproject.toml`

Do not treat this file as a substitute for understanding the implementation. Inspect the code paths affected by your work before modifying them.

---

# 1. Project mission

Prediction Market System is a quantitative prediction-market **research, calibration, backtesting, and decision-support system**.

Its purpose is to:

1. ingest auditable point-in-time market and research data;
2. estimate probabilities using explicit structural models;
3. calibrate those forecasts against independent historical evidence;
4. evaluate opportunities using executable prices and conservative costs;
5. validate models using no-look-ahead walk-forward testing;
6. gather untouched forward shadow evidence;
7. preserve the complete decision trail for later analysis; and
8. surface appropriately approved opportunities for manual review.

This repository is **not** a generic prediction-market application and is **not** an automated trading bot.

Correctness, temporal integrity, reproducibility, auditability, and falsifiability take priority over:

* producing more signals;
* making backtests look profitable;
* increasing apparent model sophistication;
* adding features for their own sake; or
* maximizing short-term development velocity.

A rigorous `NO-GO`, rejection, unsupported result, or `WATCH` decision is preferable to a misleading positive result.

---

# 2. Non-negotiable system boundary

The system is read-only with respect to trading venues.

Do not add, without an explicit repository-level change in scope:

* automated order placement;
* exchange authentication for trading;
* wallet signing;
* custody;
* fund transfers;
* live position management;
* automated portfolio execution;
* automatic response to Discord interactions;
* an internal exchange or matching engine.

Market-data authentication may be introduced in the future when required for legitimate read-only data access, but trading authority must remain outside this system unless the repository's architecture and documented mission are deliberately changed.

Discord is a **notification/manual-review surface**, not an execution surface.

Entry candidates are research observations, not permission to trade.

If future real-money execution is desired, prefer a separate execution and position-management system with an explicit interface to this research system.

---

# 3. Research integrity comes first

Never alter research methodology merely to improve reported performance.

Do not:

* weaken validation thresholds because too few models pass;
* reduce uncertainty margins to create more entries;
* modify execution assumptions to make backtests more profitable;
* select favorable historical samples after observing outcomes;
* introduce look-ahead information;
* silently remove losing observations;
* treat correlated contracts as independent evidence;
* optimize parameters on the held-out test window;
* repeatedly tune against the same held-out evidence and continue calling it held out;
* infer edge from a small or selectively observed sample;
* present passing unit tests as evidence of predictive alpha.

Model quality must be established empirically.

Implementation correctness and predictive validity are separate questions.

Tests can demonstrate the former. They do not establish the latter.

---

# 4. Temporal integrity

Point-in-time correctness is a core invariant.

Any historical evaluation must use only information that would have been available at the evaluation timestamp.

For code involving historical data, backtesting, calibration, features, or research context:

* respect the requested `as_of` boundary;
* never select source records with timestamps after that boundary;
* use only completed market or research intervals when required by the existing methodology;
* account for publication/availability time when it differs from observation time;
* preserve provider timestamps;
* preserve raw/source metadata needed to audit temporal availability;
* ensure trailing-window features contain a complete historical window;
* fail closed when required historical inputs are missing or stale.

A value existing in the database does **not** mean it was available historically.

Forward-fetched snapshots must never leak backward into historical contexts.

When adding a new research feature, ask:

> Could this exact value have been reconstructed using only data available at the historical decision time?

If not, it must not be used in a historical backtest as though it were point-in-time available.

Add tests specifically designed to detect temporal leakage.

---

# 5. Outcome leakage is prohibited

Resolved outcomes may be used as labels after they become historically available.

They must not influence historical feature selection, contract selection, sampling, or model inputs before resolution.

In particular:

* do not choose historical contracts because they later resolved YES or NO;
* do not select the most profitable contract from a historical ladder;
* do not choose representative samples using future outcomes;
* do not use settlement metadata before its historical availability;
* do not use final event information to reconstruct earlier market state.

Dense mutually exclusive ladders must not masquerade as many independent predictive observations.

Preserve event-level independence assumptions in calibration and validation.

---

# 6. Fail closed

When correctness cannot be established, prefer rejection over inference.

This applies especially to:

* ambiguous resolution rules;
* unsupported contract structures;
* stale required data;
* missing calibration profiles;
* incompatible model versions;
* incomplete historical windows;
* malformed venue metadata;
* unknown benchmark semantics;
* impossible or inconsistent price states.

Never silently route an ambiguous contract through the nearest mathematical model.

An unsupported market is a valid system result.

Preserve the reason for the rejection in the audit trail whenever practical.

---

# 7. Contract semantics determine the model

Do not model all binary contracts identically.

The existing architecture distinguishes structural contract types such as:

* fixed-time terminal ranges;
* fixed-time terminal thresholds;
* explicitly defined touch/first-passage barriers.

Terminal and path-dependent contracts represent different probability questions.

A first-passage model may only be used when contract metadata and resolution rules clearly establish touch semantics and direction.

Preserve:

* original venue rule text;
* stated resolution benchmark;
* structural-model classification;
* classification rationale where available.

When introducing new contract types:

1. define their semantics explicitly;
2. document the mathematical assumptions;
3. implement a deliberate parser/classifier;
4. fail closed on ambiguity;
5. add representative positive and negative classification tests.

Do not broaden support using fragile keyword matching without safeguards.

---

# 8. Calibration is versioned evidence

Calibration profiles are evidence tied to a specific modeling configuration.

Do not silently reuse calibration evidence across incompatible:

* symbols;
* structural models;
* model versions;
* materially changed feature definitions;
* materially changed probability-generation logic.

If a model change can alter forecast behavior, evaluate whether the model version must change.

A new model version generally requires new validation evidence.

Do not manipulate version identifiers to reuse favorable old calibration.

Live or forward evaluation should use calibration compatible with the exact forecast configuration.

Missing qualifying calibration should fail closed unless the user has explicitly chosen an existing research-only uncalibrated override.

Uncalibrated research behavior must not silently become approved alert behavior.

---

# 9. Preserve held-out validation

Walk-forward validation must preserve chronological separation between training/calibration data and test evidence.

Maintain the intent of:

* training cutoffs;
* non-overlapping test windows;
* independent-event sample requirements;
* minimum fold requirements;
* held-out evaluation;
* deployment approval gates;
* return-on-cost criteria;
* event-weighted calibration/scoring evidence.

Do not expose future test outcomes to training logic.

Do not move test observations into training merely to satisfy minimum sample requirements.

If evidence is insufficient, report insufficient evidence.

That is a valid outcome.

---

# 10. Conservative execution modeling

Backtests should approximate executable reality, not idealized mid-price trading.

Preserve or strengthen assumptions around:

* executable bid/ask prices;
* latency;
* adverse price movement;
* fees;
* fee rounding;
* slippage;
* volume participation;
* partial fills;
* whole-contract constraints where applicable;
* minimum liquidity;
* resolution risk;
* market/event exposure.

Never synthesize liquidity that was not observed.

Do not treat candle volume as exact historical order-book depth.

Do not imply exact queue position or fill probability when the historical data cannot establish it.

If improved historical execution data becomes available, use it explicitly and document the increased fidelity.

Prefer conservative uncertainty over invented precision.

---

# 11. Probability and numerical correctness

Probability values must remain within valid bounds.

Financial and probability calculations should be deterministic and numerically stable.

Take special care with:

* extreme tails;
* near-expiry contracts;
* already-crossed barriers;
* zero or near-zero volatility;
* very large or small strikes;
* complement probabilities;
* bid/ask conversion;
* fee calculations;
* contract rounding;
* Kelly sizing.

Do not introduce mathematically convenient approximations without documenting their limitations.

Where numerical behavior is non-trivial, add tests for:

* normal cases;
* boundary cases;
* pathological inputs;
* symmetry/complement relationships where applicable.

---

# 12. Risk controls are controls, not suggestions

Do not bypass risk constraints simply because a forecast has large estimated edge.

Preserve the separation between:

* probability estimation;
* uncertainty;
* executable cost;
* liquidity;
* resolution risk;
* sizing;
* per-market exposure;
* aggregate event exposure;
* approval state.

Risk decisions should remain deterministic and auditable.

Changes to defaults that materially increase allowed exposure require explicit justification and tests.

Do not loosen conservative defaults merely to create larger hypothetical returns.

---

# 13. Venue architecture

Venue-specific behavior belongs behind clear venue/data-source boundaries.

Current direction:

* Kalshi is the primary implemented market-data and historical venue.
* Polymarket is a planned second **read-only** venue and cross-venue signal source.

When adding another venue:

* keep provider-specific payload handling isolated;
* preserve raw identifiers and timestamps;
* normalize only concepts that are genuinely equivalent;
* preserve venue-specific resolution semantics;
* preserve venue-specific fee rules;
* preserve benchmark differences;
* preserve source provenance.

Do not force different venues into a common abstraction by discarding important semantics.

Cross-venue contract matching must be conservative.

Two markets are not equivalent merely because their titles look similar.

Matching should account for, where relevant:

* underlying event;
* outcome definition;
* strike/bounds;
* observation time;
* expiration;
* resolution source;
* benchmark;
* path dependence;
* venue-specific exceptions.

If equivalence is uncertain, classify the pair as unmatched rather than generating misleading cross-venue signals.

---

# 14. External research sources

Current research inputs include Coinbase and Deribit data.

When adding or changing data providers:

* preserve source provenance;
* retain provider timestamps;
* distinguish historical from current-only availability;
* define freshness requirements;
* handle provider outages explicitly;
* avoid silently substituting semantically different data;
* make benchmark mismatches auditable.

A convenient proxy is not necessarily the venue's actual resolution benchmark.

Document material proxy risk.

Optional research inputs may degrade gracefully where the architecture allows it.

Required inputs must fail closed when unavailable or stale.

---

# 15. Architecture guide

The main package is:

`src/prediction_market_system/`

Important areas currently include:

* `domain.py` — shared domain concepts;
* `engine.py` — probability/opportunity evaluation logic;
* `calibration.py` — calibration logic and uncertainty evidence;
* `backtest.py` — walk-forward historical replay and validation;
* `research.py` — research-context construction and regime logic;
* `research_storage.py` — persistence for research inputs;
* `storage.py` — primary SQLite persistence/audit behavior;
* `paper_alerts.py` — forward shadow evaluation and campaign behavior;
* `validation.py` — validation/approval-related logic;
* `discord.py` — manual-review notification integration;
* `transport.py` — shared transport concerns;
* `sources/` — external research/data-source clients;
* `venues/` — prediction-market venue integrations;
* `cli.py` — Typer CLI surface.

Tests live under:

`tests/`

Before introducing a new abstraction, determine whether it belongs in one of the existing boundaries.

Avoid turning `cli.py` or another orchestration module into a second domain layer.

Keep provider-specific logic out of core probability and calibration code when practical.

Keep core model logic independently testable.

---

# 16. SQLite is part of the research record

SQLite is the authoritative audit store for the current system.

Treat persisted records as research evidence, not disposable application state.

Changes to persistence must preserve:

* auditability;
* provenance;
* idempotency where expected;
* historical reconstruction;
* model evidence;
* resolution evidence;
* forward validation evidence.

Do not casually delete or rewrite historical evidence.

Schema changes should be migration-safe.

For migrations:

* preserve existing data;
* make reruns safe where the current migration system expects it;
* test upgrades from existing schema states when practical;
* distinguish immutable source records from mutable operational state;
* preserve stable external identifiers.

Maintenance/compaction behavior must remain narrowly scoped.

Do not broaden retention cleanup in a way that can remove:

* entry candidates;
* deliveries;
* failures;
* unsupported-market evidence;
* calibration/model evidence;
* regime history;
* source history;
* venue history;
* resolutions

unless the change is explicitly designed, documented, and tested.

---

# 17. Idempotency matters

External synchronization, archival, validation, maintenance, and notification workflows should be safe to retry where practical.

Do not introduce duplicate immutable source records when stable identifiers exist.

Do not allow repeated settlement/archive/validation operations to double-count evidence.

Discord notification behavior should remain idempotent according to its existing delivery semantics.

When implementing scheduled work, assume retries and process restarts will occur.

---

# 18. Shadow-first operation

The intended operating stage is forward shadow observation.

The managed system should continue to prioritize:

* research collection;
* supported-market evaluation;
* audit persistence;
* model validation;
* regime coverage;
* operational reliability.

Do not enable Discord delivery automatically when a model passes validation.

Enabling approved-only Discord delivery is a deliberate deployment-policy decision.

Do not introduce automatic trading as the next step after Discord approval.

Forward observations gathered after a model/design decision are especially valuable because they have not been retrospectively selected.

Protect that evidence.

---

# 19. Operational behavior

The current scheduled design intentionally uses observable one-shot processes rather than an opaque forever-loop.

Preserve that model unless there is a strong operational reason to change it.

Important concerns include:

* overlapping runs;
* stale research;
* recovery after host downtime;
* transport failures;
* rate limits;
* SQLite writer contention;
* cycle gaps;
* bounded storage growth;
* deployment rollback.

Operational improvements should increase observability and recoverability without weakening research controls.

Future infrastructure work may include:

* always-on hosting;
* database/disk telemetry;
* cycle-gap monitoring;
* research freshness monitoring;
* archive-failure monitoring;
* rate-limit telemetry;
* controlled SQLite maintenance;
* off-machine backups;
* restore drills.

Infrastructure changes must not silently alter the research methodology.

---

# 20. Secrets and credentials

Never commit:

* `.env`;
* Discord webhook URLs;
* API secrets;
* private keys;
* wallet credentials;
* tokens;
* credentials embedded in example commands or fixtures.

Use environment/configuration mechanisms already established by the repository.

Tests should use fake credentials or mocked transports.

Do not log secrets.

Do not place credentials in SQLite research records unless explicitly required and securely designed.

---

# 21. Configuration changes

Settings use the existing configuration layer and `PMS_` environment-variable convention.

When adding configuration:

* provide a conservative default where a safe universal default exists;
* document the setting;
* expose it through `.env.example` when appropriate;
* validate invalid combinations;
* avoid configuration that silently disables important safeguards.

Do not turn core statistical safeguards into hidden convenience toggles.

Overrides that intentionally bypass evidence requirements should remain explicit, difficult to invoke accidentally, and clearly marked as research/manual behavior.

---

# 22. Testing expectations

Every meaningful behavior change should include or update tests.

Match tests to the risk of the change.

High-priority areas include:

* probability models;
* contract classification;
* calibration;
* temporal boundaries;
* historical reconstruction;
* backtesting;
* execution assumptions;
* fee calculations;
* risk constraints;
* venue normalization;
* source freshness;
* storage/migrations;
* maintenance;
* validation gates;
* alert approval;
* operational recovery.

Regression tests are expected for bug fixes whenever practical.

For research logic, test invariants rather than only individual examples.

Examples of valuable invariants include:

* no historical source timestamp exceeds `as_of`;
* probabilities remain in `[0, 1]`;
* ambiguous contracts fail closed;
* missing qualifying calibration cannot produce approved delivery;
* future outcomes cannot enter training data;
* dense contracts from one event do not become independent evidence;
* unavailable liquidity is not synthesized;
* repeated idempotent operations do not duplicate evidence.

---

# 23. Required quality checks

Before considering repository-wide work complete, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The repository uses:

* Python 3.11+
* strict Mypy;
* Ruff linting/formatting;
* Pytest.

Do not suppress type errors broadly to make checks pass.

Do not add blanket `# type: ignore`, lint exclusions, or test skips without a specific documented reason.

During development, targeted tests are encouraged.

Before completion of substantial work, run the full quality suite.

If a check cannot be run, explicitly report which check was not run and why.

---

# 24. Dependencies

Avoid unnecessary dependencies.

Before adding one, determine whether:

* the standard library is sufficient;
* the existing stack already solves the problem;
* the dependency materially improves correctness or maintainability.

Prefer small, well-maintained dependencies with clear purpose.

Do not add large frameworks for a narrow feature.

Do not introduce ML/AI libraries merely to make the project appear more sophisticated.

Additional modeling complexity must earn its place through testability and empirical evaluation.

---

# 25. Quantitative/model changes

When introducing or materially modifying a model:

1. state the hypothesis;
2. define the required inputs;
3. verify historical point-in-time availability;
4. document mathematical assumptions;
5. implement deterministic tests;
6. assign an appropriate model/version identity;
7. evaluate it out of sample;
8. compare it with the existing baseline;
9. persist sufficient evidence to audit the comparison.

Useful sophistication may include better:

* calibration;
* feature construction;
* regime modeling;
* volatility modeling;
* model combination;
* scoring;
* sensitivity analysis;
* uncertainty estimation;
* market-microstructure analysis.

Sophistication alone is not evidence of improvement.

Do not replace a simpler model solely because a more complex model is available.

---

# 26. Backtest changes require extra scrutiny

Changes to `backtest.py`, calibration, historical storage, source reconstruction, or outcome handling can invalidate research conclusions even when tests pass.

When modifying these areas, explicitly review for:

* look-ahead bias;
* survivorship bias;
* sample-selection bias;
* event dependence;
* fee leakage;
* resolution leakage;
* timestamp errors;
* unrealistic execution;
* repeated tuning against held-out data.

A backtest becoming substantially more profitable after a seemingly innocuous change is a reason to investigate, not celebrate automatically.

---

# 27. Cross-venue research

Cross-venue functionality should initially be treated as **research intelligence**, not an execution/arbitrage system.

Potential valid outputs include:

* normalized comparable markets;
* executable probability differences;
* spread differences;
* liquidity comparisons;
* cross-venue disagreement;
* historical disagreement features;
* venue-specific calibration diagnostics.

Do not label a difference "arbitrage" unless simultaneous executable economics, contract equivalence, fees, settlement semantics, and operational constraints actually justify that claim.

Preserve uncertainty when venue contracts are only approximately equivalent.

---

# 28. UI and reporting

A frontend/dashboard is not required merely because the project lacks one.

Add user-facing visualization only when it materially improves research or operations.

Useful surfaces could include:

* calibration diagnostics;
* model comparisons;
* walk-forward results;
* regime coverage;
* shadow observations;
* opportunity history;
* data freshness;
* operational health.

Do not build a generic SaaS dashboard.

Do not fabricate metrics or demo data and present them as research evidence.

Keep SQLite/source evidence authoritative.

---

# 29. Documentation must remain truthful

Update documentation when implementation behavior changes.

Potentially affected files include:

* `README.md`;
* `docs/venue-decision.md`;
* `docs/operations-hardening.md`;
* `.env.example`;
* CLI help;
* architecture/design documentation.

If future work becomes implemented, update the relevant future-work section.

Do not leave documentation claiming a capability is unimplemented after implementing it.

Do not claim a feature is production-ready unless its behavior and validation support that description.

Distinguish clearly between:

* implemented;
* tested;
* backtested;
* forward shadow-tested;
* empirically validated;
* approved for manual review;
* appropriate for real-money use.

These are not interchangeable.

---

# 30. Preserve useful limitations

Do not "fix" a conservative limitation until you understand why it exists.

Examples include:

* unsupported ambiguous contracts;
* strict stale-data rejection;
* minimum calibration samples;
* held-out approval gates;
* adverse execution assumptions;
* bounded Discord behavior;
* shadow-only managed schedules;
* bounded WATCH retention.

Some friction is intentional.

Investigate before removing it.

---

# 31. Avoid unnecessary rewrites

Preserve good existing architecture.

Refactor when it materially improves:

* correctness;
* testability;
* maintainability;
* extensibility;
* observability;
* performance.

Do not rewrite modules merely to impose a preferred style.

Do not introduce microservices, queues, caches, distributed systems, or event buses without a demonstrated need.

This project currently benefits from a relatively small, inspectable Python + SQLite architecture.

Complexity must justify itself.

---

# 32. Working with autonomous subagents

When delegating work to subagents:

* give each subagent a bounded responsibility;
* prevent overlapping architectural rewrites;
* require findings to be grounded in repository code;
* retain one coherent final design.

Good parallel investigations include:

* data-source integration;
* quantitative methodology;
* storage/migration impact;
* operational reliability;
* tests;
* documentation.

Do not allow separate agents to independently modify shared research invariants without coordination.

The primary agent remains responsible for integration and correctness.

---

# 33. Change discipline

Before implementing a substantial change:

1. inspect the relevant implementation;
2. inspect associated tests;
3. identify the invariants affected;
4. determine whether historical/model evidence becomes incompatible;
5. implement the smallest coherent design that solves the problem;
6. add tests;
7. run targeted verification;
8. run the full quality suite before completion;
9. update documentation.

For large features, prefer complete vertical slices over many disconnected scaffolds.

Do not create dozens of speculative abstractions for future work that does not yet exist.

---

# 34. Definition of done

A change is not complete merely because code was generated.

For substantial work, completion means:

* the implementation is coherent with existing architecture;
* relevant invariants are preserved;
* edge cases are handled;
* new behavior is tested;
* temporal integrity is preserved;
* persistence remains auditable;
* documentation matches reality;
* Ruff passes;
* formatting passes;
* strict Mypy passes;
* Pytest passes.

For research/model work, also report:

* what hypothesis was tested;
* what evidence was used;
* what data boundaries were enforced;
* whether model/version compatibility changed;
* whether results are in-sample, held-out, or forward;
* whether the result passed or failed existing evidence gates.

A failed experiment can still be successfully completed engineering work.

---

# 35. Final principle

When choosing between:

**more impressive output**

and

**more defensible evidence**

choose the defensible evidence.

When choosing between:

**assuming a market can be modeled**

and

**rejecting ambiguous semantics**

reject the ambiguity.

When choosing between:

**a more profitable backtest**

and

**a more realistic backtest**

choose realism.

When choosing between:

**more signals**

and

**better-calibrated signals**

choose calibration.

The goal is not to make this system look like it has an edge.

The goal is to build a system capable of determining, as rigorously as practical, whether an edge actually exists.
