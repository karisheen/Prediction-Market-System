# Operational hardening change record

## Implemented changes

### Shadow-only production schedule

The managed `paper-alerts` launch agent uses neither `--send-discord` nor
`--allow-unapproved-discord`. It continues calibrated shadow evaluations and SQLite audit
writes while the validation campaign accumulates evidence, but no entry candidate can reach
Discord from the managed schedule. The system remains read-only with respect to trading
venues.

The CLI keeps both delivery flags for an explicitly initiated local/manual review. After the
exact live calibration profiles pass the persisted approval gates, enabling approved-only
Discord delivery is a deliberate deployment-policy change rather than an automatic side
effect of model validation.

### Compact evidence storage

The `compact_evidence_v1` storage format removes repeated evaluation payloads
without changing forecasts, IDs, timestamps, classifications, or approval rules.
The canonical evaluation remains in `forecast_ledger`; inputs and research
contexts remain content-addressed in `input_objects`. Operational forecast and
opportunity rows reference the ledger only when their complete logical payloads
match exactly. Market checks reference an identical saved opportunity instead
of embedding it again. Differing, orphaned, and pre-ledger historical records
stay inline and retain their original classification.

`run_manifests` shares full build/source identity through `code_identities`.
Hydration restores the original manifest and verifies the shared identity hash;
the original `manifest_sha256` remains unchanged.

The `payload_json` columns in `input_objects`, `forecast_ledger`, `run_manifests`,
`forecasts`, `opportunities`, and `paper_alert_market_checks` now contain
zlib-compressed canonical JSON BLOBs. `code_identities` uses the same encoding.
Use repository readers or `storage_payloads.load_payload` for logical JSON;
direct `json_extract(payload_json, ...)` is not valid for these columns.
Market checks have a scalar `opportunity_id` for maintenance joins.

Existing populated stores are not converted at startup. With scheduled writers
stopped, run `pms db-migrate-storage --source SOURCE --destination NEW_PATH`.
It copies bounded batches to a new file, preserves original rowids and logical
content, verifies per-table counts and hydrated content hashes, checks integrity,
and refuses introduced foreign-key violations. Existing violations are reported,
not silently repaired. Unsupported source schema changes and corrupted input fail
closed. The verification manifest is published last; an interrupted destination
without a successful manifest must not be activated. Never overwrite the source.
The command does not switch the application, enable alerts, or restart jobs.

It verifies a consistent read snapshot, not permanent writer exclusion. With
writers still stopped, run `pms db-verify-storage --manifest NEW_PATH.migration.json`
immediately before activation. A changed source requires a new migration; keep
writers stopped through the actual switch. Do not treat successful copying as
proof that a concurrently running source has no newer records.

Retention policy: immutable ledger, input, and provenance evidence is retained
losslessly compressed, not deleted by age. The 14-day policy below applies to
transient detailed WATCH checks and safely redundant operational indexes.
Compression reduces the storage rate, but does not impose a fixed lifetime
database-size cap. Capacity and backup space still require monitoring; archival
to another store must preserve replay access and be verified before any deletion.

#### November 4 recovery and migration

The production database reported corruption after its last successful collection
on October 31. The original damaged file is retained; a timed-out recovery attempt
did not establish that its newer records are unrecoverable. The operator approved
restoring the verified October 28 backup and resuming shadow collection.
Records collected after that backup and before collection stopped are not present
in the restored active store. Do not treat that interval as complete evidence.

The October 28 backup's separate compact migration completed on November 4:
44,263,919,616 bytes became 14,832,197,632 bytes, a 66.5% reduction.
Both databases passed integrity checks, every original table's logical content
hash and row count matched, and no foreign-key violations were introduced.
These figures compare the same backup contents, not the larger damaged database.
Retained recovery copies and backups consume additional disk space.

The format cutover requires a coordinated database and code switch with all
agents stopped. Stage and verify the committed release first, preserve the old
database and sidecars, check migration-source freshness, then activate the new
database and release before restarting agents. If activation fails, restore
both database and application pointers before any restart; rolling back only
the code would pair the old reader with the new BLOB format.

Release `63f8a0e` was activated with the compact database at 13:18 UTC on
November 4. The staged release passed 489 tests, Ruff lint/format checks, and
strict Mypy. A production-sized research refresh and shadow scan evaluated
318 markets with zero runtime failures and zero Discord entries. All remained
WATCH-only because calibration was missing; the CLI's exit code 1 reflected
that existing eligibility gate, which was not bypassed.

The fresh online backup, `prediction_markets-20261104T131854Z.db.gz`, passed
`quick_check` and occupies 3.71 GiB. This is larger than the older October 28
gzip archive: compact live storage does not guarantee smaller compressed
archives. All three older archives were retained for this cutover; the existing
scheduled three-backup rotation remains unchanged. The heartbeat subsequently
reported live scans, research refresh, validation archives, and backup health OK.

After that backup succeeded, the operator-approved expanded backup and temporary
recovery clone were removed. Cleanup reclaimed approximately 41.24 GiB and left
214.17 GiB free at observation time. The 51.8 GiB damaged original remains
preserved, along with the compressed restore points. The operational receipt at
`data/storage-recovery-20261104.json` records the exact retained paths, recovery
gap, verification evidence, and decision rationale. The migration manifest is
now a historical verification record, not a reusable activation authorization:
its temporary source was removed and its destination became the active store.

### Bounded detailed WATCH retention

`pms paper-alert-maintain` compacts detailed `WATCH` evaluations older than a configured UTC
cutoff. The command defaults to a preview and requires `--apply` before it changes data.
The managed launch agent applies a 14-day retention window daily at 03:15 local time and
commits at most 5,000 evaluations per transaction.

Before deletion, the command writes daily counts to `paper_alert_watch_rollups`. It then
removes only the linked detailed records:

- `paper_alert_market_checks` rows whose status is exactly `watch`;
- their unalerted `WATCH` opportunity rows when the payload is a canonical-ledger
  reference and no remaining check depends on the opportunity;
- canonical-ledger forecast references left without an opportunity; and
- cycles left without any detailed checks.

Entry candidates, delivered alerts, failures, unsupported-market records, missing-calibration
records, unapproved-model records, Discord delivery state, model evidence, market regimes,
research inputs, venue history, and resolutions are not pruned. Each bounded batch is
transactional and releases the SQLite writer lock before the next batch. Compaction refuses
to proceed if a WATCH check cannot be linked to its persisted evaluation. Re-running a
completed compaction does not double-count its rollup.

Inline historical forecasts and opportunities are retained: a daily count rollup
is not a substitute for their unique audit evidence. Canonical ledger, input, and
manifest records remain available after operational WATCH rows are removed.

Preview or apply maintenance manually:

```bash
uv run pms paper-alert-maintain \
  --series KXBTC \
  --watch-retention-days 14

uv run pms paper-alert-maintain \
  --series KXBTC \
  --watch-retention-days 14 \
  --batch-size 5000 \
  --apply
```

SQLite does not return deleted pages to the filesystem automatically. Schedule a separate,
explicit maintenance window before running `VACUUM`; it requires free disk space and an
exclusive rewrite of the database.

### Atomic, reproducible macOS deployment

`ops/deploy_macos.py` replaces manual source copying. It:

1. Refuses to deploy a dirty Git working tree.
2. Snapshots prior launch-agent definitions and loaded membership.
3. Moves runtime data and `.env` to shared paths outside a code release on the first run.
4. Copies source into a revision-labelled release directory without secrets, data, caches, or
   Git metadata.
5. Runs `uv sync --frozen`, the full test suite, Ruff lint and formatting checks, and strict
   Mypy checks inside the staged release, matching the GitHub quality gate. Existing jobs
   continue while these gates run.
6. Stops the jobs requiring replacement, then switches the stable `app` symlink.
   An identical loaded forward-study job remains loaded throughout: an in-flight process
   keeps its old release while the next invocation uses the new application pointer.
7. Initializes the current-format schema and, for KXBTC/BTC, preregisters the frozen study.
8. Installs the alert, research, archive, validation, maintenance, backup, heartbeat and
   applicable forward-study jobs without changing the original seven schedules.
9. On activation failure, restores the previous application pointer, exact plist contents
   and prior loaded membership; newly introduced plists are removed.

A changed loaded forward-job definition is refused before activation rather than killing
an irreversible capture attempt. Switching away from KXBTC/BTC removes the obsolete forward
job only when it is not running. A rollback failure is reported explicitly with the original
deployment error retained; it is never presented as a successful recovery.

Scheduled commands still initialize the store on startup, but they compare `sqlite_master` with
the tables and indexes the schema defines and run DDL only when an object is missing. Applied
migrations are likewise detected read-only, so routine job starts never request the SQLite
writer lock.

The deployment script does not perform the offline compact-storage conversion.
Migrate and verify a separate sound database before activating this format;
an unconverted populated database is rejected by initialization.

Run from a clean, committed checkout:

```bash
uv run python ops/deploy_macos.py
```

Runtime layout after the first atomic deployment:

```text
~/Library/Application Support/PredictionMarketSystem/
├── .env                 # shared secret configuration
├── data/                # shared SQLite database and logs
├── backups/             # verified, compressed database backups and manifests
├── app -> releases/...  # stable atomic application pointer
└── releases/            # immutable revision-labelled code releases
```

Old releases are deliberately retained for rollback. Remove them manually only after the
active revision and rollback target are known.

### Research release activation: 2026-10-18

Release `20261018T072504Z-4a51f60` activated the merged research baseline. The previous
operational release, `20261018T064901Z-b42104b`, remains on disk. All seven launch agents
were reloaded; entry delivery remains disabled, while operational and validation-status
notifications retain their existing configuration. Downturn backtests were not run.

The staged release passed 402 tests (five existing Pydantic serialization warnings), Ruff
lint, and strict Mypy. Before adding the evidence schema, initialization made
`data/backups/prediction_markets-pre-v2-20261018T072546555071Z.sqlite3` using the verified
backup/restore-drill path. The daily compressed backup remains separate in `backups/`.
Legacy counts were unchanged: 1,877 alert events, 11 model validations, and 19 calibration
profiles. The `v2_evidence_foundation` marker was recorded at 08:50:29 UTC.

The deployment took 5,126 seconds, with collection stopped during the full integrity,
content-fingerprint, backup, and restore verification. This is not the runtime of the
daily compressed backup. The first resumed scan recorded 269 forward forecasts but
rejected results when the decision price expired during recovery/evaluation. A subsequent
warm scan evaluated all 318 markets in about 43 seconds with zero runtime failures.
All remained WATCH-only without qualifying calibration; that condition intentionally
returns CLI exit code 1 and does not imply a collector crash.

At 08:54 UTC, `shadow-report --series KXBTC` reported 587 forward WATCH forecasts,
none resolved yet. Every ledger row had its input object and run manifest;
`explain-forecast` returned the recorded inputs, manifest, and research context. The
heartbeat reported scans, research refresh, archive failures, and daily backup freshness
healthy. Historical records were not relabelled as new forward evidence, and this
deployment did not approve any model or establish a trading edge.

Follow-up at 13:43 UTC confirmed 15,851 forward WATCH forecasts, with no resolved labels
yet, and another scheduled scan of 318 markets with zero runtime failures. All heartbeat
checks remained healthy. The scheduled 09:15 UTC backup had also completed successfully,
producing a verified compressed backup of approximately 0.84 GiB.

Run reports from the deployed application directory, not the development checkout,
whose relative database path still refers to its separate development database.
Legacy archive success also does not certify the new interval-completeness requirements;
the archive/validation commands must establish eligibility rather than assuming an
approval date from the old day count.

### Verified daily database backups

`pms db-backup` runs daily at 04:15 local time, after the archive, weekly validation, and
maintenance jobs. It copies the database with SQLite's online backup API from a read-only
connection in a single step, so the whole copy comes from one WAL read snapshot. Scheduled
writers keep committing during the copy; a paged backup was rejected because SQLite restarts
it from the first page whenever another connection commits between steps. `VACUUM INTO`
would also hold a single read snapshot and would drop the roughly 20% of free pages, but it
rebuilds every table and index and was no faster in the measurements below; the page copy
was kept as the simpler operation.

The copy is written under a hidden `.partial` name in `backups/` beside `data/`. It must
pass `PRAGMA quick_check` before it is gzip-compressed (level 6), fsynced, and atomically
renamed to `prediction_markets-<UTC timestamp>.db.gz`. Each backup has a `.json` manifest
with its creation time, source and backup sizes, SHA-256, check result, and duration.
`db-backup-last-attempt.json` records whether the latest attempt succeeded. The command
refuses to start unless free space exceeds twice the database size plus a 10 GiB margin.
It keeps the newest three backups and prunes older ones only after a new backup succeeds.
A failed attempt removes its partial files and leaves earlier backups untouched.

On a 512 MiB sample read from the live 11.1 GB database, gzip level 6 compressed at about
95 MB/s with a 13.9x ratio. Level 1 was about twice as fast but produced archives 50% larger.
On a synthetic 1.1 GB WAL database with a concurrent writer, the single-step copy took
29 s with a cold page cache and 4.6 s warm (`VACUUM INTO`: 27 s and 10.3 s). `quick_check` ran
at 17 MB/s cold and 58 MB/s warm. No write failed; the slowest write took 1.2 s during the
cold copy, from disk contention rather than locking. A paged copy with a writer committing
every 0.5 s never got past a quarter of the pages. This host has 8 GiB of RAM, so an 11 GB
copy is mostly uncached: plan for a full backup of 10–20 minutes (up to about 5 minutes
copying, which is the only time the read snapshot is held, up to about 11 minutes checking,
and about 2 minutes compressing), producing about 0.8 GB.

`paper-alert-heartbeat` reports a `Database backup` problem if the newest verified backup is
older than 36 hours (`--max-backup-age-hours`), the last attempt failed, or no backup exists
yet. Run `uv run pms db-backup` once after the first deployment so the heartbeat does not
report a missing backup until the first scheduled run.

Backups stay on the same disk as the database, so they protect against corruption and
mistaken writes but not against disk loss. Restoring is manual:

1. Stop every `com.karisheen.prediction-market-*` launch agent with `launchctl bootout`.
2. Compare `shasum -a 256 BACKUP.db.gz` with `backup_sha256` in its manifest.
3. Move `data/prediction_markets.db` and its `-wal` and `-shm` files aside; do not delete
   them until the restore is verified.
4. Run `gunzip -c BACKUP.db.gz > data/prediction_markets.db`, then
   `sqlite3 data/prediction_markets.db 'PRAGMA integrity_check'`.
5. Redeploy or bootstrap the agents again. Audit records written after the backup's
   `created_at` are not in the restored database.

### Validation campaign continuity

The existing daily archive and weekly validation schedules remain active. The archive gathers
completed UTC days; validation remains in `COLLECTING EVIDENCE` until the configured
chronological window is complete. Deployment does not weaken the 90-day training, two 30-day
held-out windows, minimum sample/event/fold, return-on-cost, or Brier-score gates.
Ordinary research backtests may persist profiles and gate results, but only an
active frozen campaign plus a matching deployment-policy fingerprint can make a
profile eligible for managed delivery. Changing fees, slippage, sizing, or other
operational policy after approval cannot reuse that approval.

### Historical archive discovery and completeness repair

Kalshi keeps old events on the events endpoint, but removes their nested markets
after moving them into its historical tier. The collector now merges live event
markets with paginated historical markets for that same event before selecting
contracts. It keeps selected contracts explicit, respects market opening times,
and does not request zero-length history for contracts closing at the left
window boundary. Exhausted discovery at the exact event limit is distinguished
from a genuinely truncated universe.

An incomplete window is no longer reported or stored as a successful archive.
Recovered source rows and diagnostic counts are retained, while the error identifies
missing/duplicate/unexpected intervals and their provider or ticker. Zero newly
inserted rows on a retry do not imply missing fetched data. Only current-version
certificates (`coverage_version=2`, `coverage_complete=1`, successful status)
advance contiguous campaign coverage. Legacy certificates remain in the database
but must be rechecked; no old records are silently promoted.

The bounded work queue prioritizes never-attempted days and then the oldest
attempts, with chronological ties. This avoids repeatedly spending every daily
budget on the same unavailable historical intervals. Heartbeat checks missing
and uncertified days for the active campaign, not merely recent exceptions.
A newly completed UTC day has a 24-hour scheduling grace period, allowing the
daily local-time archive job to run; it remains uncertified during that grace.
Explicit failures are unhealthy immediately, and older gaps are never excused.

Real-provider reproduction on November 5 found zero live nested markets but 188
historical markets for event `KXBTC-26AUG2020`. The old CLI incorrectly certified
August 20 with zero market candles. The repaired CLI retrieved the historical
universe and reported genuine gaps instead. For example, the historical endpoint
returned no candle ending at August 20 23:04 UTC for
`KXBTC-26AUG2020-T73299.99`, even with an exact-minute request; the corresponding
live endpoint returned 404. Complete Coinbase spot data did not repair that
independent venue gap. Such windows remain blocked: no candles are synthesized,
historical receipt times are preserved, and reconstructed data is not forward
evidence. The frozen campaign and its statistical thresholds remain unchanged.

References: [Kalshi historical partitioning](https://docs.kalshi.com/getting_started/historical_data)
and [historical candle semantics](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks).

#### November 5 production recheck

Release `a1f96b6` passed 548 tests, Ruff lint/format checks, strict Mypy, and the
GitHub quality gate, then was deployed with the unchanged shadow-only schedule.
A subsequent recorded forecast replay matched the deployed source fingerprint.

The production sweep rechecked every completed campaign day from August 7
through November 4 in bounded batches of 31, 31, and 28. All 90 event universes
were discovered, recovering 386,946 market snapshot rows and fetching 334,404
market candles. Source timestamps and receipt provenance were preserved.
Every day still had missing required Kalshi minute intervals; none received a
version-2 completeness certificate. This was not a transport failure: all 90
attempts produced interval diagnostics.

Coinbase supplied complete minute spot coverage for 88 days. Three minute closes
remained absent on October 2 (16:48 UTC) and October 30 (19:56 and 19:57 UTC);
narrow requests returned their neighboring candles but not those missing minutes.
No synthetic or alternate-source observations were inserted to hide either
provider's gaps.

The campaign was refreshed without Discord delivery and correctly remains
`COLLECTING EVIDENCE`, with 0/150 certified contiguous days. Previous unversioned
certificates do not override the new source checks. The original frozen campaign
was not reset, and statistical gates were not weakened. The calendar-only January
checkpoint is not a reliable readiness date while these data requirements remain
unsatisfied.

The daily archive agent was restored after the manual sweep. Its fair bounded
queue continues collecting newer days and revisiting older gaps without starving
either. Exact per-window counts and missing-interval diagnostics are retained in
`data/archive-backfill-20261105.json` and the corresponding `.log`. These are
historical recovery records, not independent forward forecasts or evidence of edge.

#### Verified free Coinbase trade reconstruction

The subsequent November 5 investigation recovered continuous public Coinbase
Exchange trade tapes covering the missing minutes. All three absent minutes had
trades, and four neighboring native candles represented only partial trade
coverage. Four outer control candles matched full trade aggregation exactly.
This rules out inactivity for these Coinbase gaps; it does not establish the
provider's internal failure cause.

`coinbase-repair` refetches the retained audit's bounded source windows before
every dry-run or apply. It verifies trade-ID continuity, timestamp ordering,
minute-boundary witnesses, exact-decimal OHLCV, and unchanged native controls.
The seven derived rows are stored separately with actual receipt-time
availability, retained native observations, and one compressed proof per batch.
Reads validate proof integrity and reject unrecognized native revisions.
Snapshots carry the same repair evidence; old native-only snapshots remain valid.
Repeated apply is idempotent, including when some native candles have subsequently
been corrected by Coinbase.

A real-data dry-run created no database. An isolated CLI apply preserved all
2,877 original candle values and restored 1,440 effective spot minutes on each of
October 2 and October 30. All seven OHLCV results matched the independent trade
audit. A second apply added zero repairs and no duplicate proof. The full suite
passed 612 tests, Ruff lint/format checks, and strict Mypy.
The actual archive CLI then refetched October 2: 1,439 native spot candles became
1,440 effective spot candles through the verified repairs. Kalshi discovery
completed, but missing Kalshi intervals still produced a failed archive attempt
with `coverage_complete=0`, as required.

This does not clear the Kalshi requirement. Sampled missing Kalshi intervals had
no trades, but a zero-volume interval can still carry quote OHLC used by research
and probability scoring. No-trade evidence alone therefore does not justify
inventing quotes or exempting intervals. Initial Predexon and Oddpool archive access
required authenticated accounts. A subsequently supplied free Predexon key was verified:
the sampled August 7 gap had snapshots in only 5 of 51 missing minutes, and the sampled
August 20 gap had none in its three missing minutes. A complete Kalshi control hour also
disagreed with diagnostic snapshot-derived OHLC in 29 of 54 comparable minutes.
Those snapshots are not a verified replacement for venue candles. No subscription was
purchased, no synthetic Kalshi candles were inserted, and no provider gap was waived.

Kalshi's published candle documentation does not specify the whole-candle omission rule.
Observed no-trade minutes can have quoted, zero-volume candles, so inactivity alone does
not explain all omissions. A precise inquiry and reproducible public URLs are retained in
`data/kalshi-omission-inquiry-20261105.txt`; the support-access audit is
`data/kalshi-support-access-20261105.json`. No authenticated support channel was accessible,
so the inquiry was **not submitted** and no authoritative provider answer is claimed.

The original campaign and its clock policy remain unchanged. Newly received
historical data cannot become evidence available to past decisions. Any separate
retrospective protocol would need explicit preregistration and separation from
forward evidence; neither a protocol replacement nor a campaign restart is part
of this repair.


### Independent prospective study operations

`experiments/forward/protocol-v1.json` is a separate immutable registration, not a repair
certificate, restarted campaign or replacement for Phase 1. Its identity is
`sha256:e7f440c05ac51bdc8a13456bfa8185e15cc6420c9bffa60a6053cd1103e0056f`.
The existing campaign, original protocol files and pinned model sources remain unchanged.

The KXBTC/BTC deployment adds `com.karisheen.prediction-market-forward`, scheduled hourly
at minute 45 without `RunAtLoad`, logging to `data/paper-alert-forward.log`. The study has
1,440 slots from November 6 through January 4 inclusive, split into two fixed 30-day folds.
The first capture is **2026-11-06 00:45 UTC** (November 5, 18:45 Central). The deployment
must register before study start; later deployment accepts only an identical existing
registration. The hourly local minute-45 trigger matches UTC minute 45 in the current
Central timezone. A move to a timezone with a fractional-hour offset requires explicit
schedule review, not an assumed UTC match.

Keep the host awake and connected for scheduled captures. A missed launch cannot be
backfilled as prospective evidence. The first persisted attempt owns its slot, including
failure or interruption; later invocations cannot replace it. `forward-report` reports
missing, late, partial, no-market and failed slots separately, with future slots excluded
from elapsed coverage. These study-specific checks are not an extension of the existing
campaign heartbeat; use the forward report and its dedicated log to inspect collection.

Each attempt retains complete market discovery and deterministic selection before books,
actual request/receipt/decision/persistence clocks, forecasts, original metadata and fees,
and separately requested +30/+60/+120-second books. Source proofs survive ordinary WATCH
cleanup. Objects use compressed content-addressed storage; hourly collection looks up only
its own attempt, and outcome resolution streams indexed candidates rather than repeatedly
hydrating all prior study payloads. This avoids a growing full-history load on each capture.
It does not promise a fixed total database size.

```bash
cd "$HOME/Library/Application Support/PredictionMarketSystem/app"
uv run pms forward-report
uv run pms forward-capture --dry-run
uv run pms forward-resolve
```

The diagnostic is explicitly ineligible, uses temporary storage and actual current clocks,
and does not rewrite the protocol or register a study. A November 5 live CLI smoke captured
three valid decisions and all nine latency books with zero capture/execution failures;
production had no forward tables afterward. Temporary-database CLI smoke also verified
idempotent registration, pre-start no-op capture, empty resolution, embargoed coverage,
future-`--as-of` refusal and database-output overwrite refusal; reporting left that
database byte-identical.

Before **2027-01-07 00:00 UTC**, reports expose no outcomes, scores, P&L or directional
conclusions. Later reports always use that same fixed evidence cutoff. A directional
probability result needs at least 95% valid slots overall and in each fold plus the
preregistered event/date and paired-bootstrap gates. A directional execution simulation
also needs 100% valid slots, every entry's scenario books and every fill's resolution.
Missing execution is unavailable evidence, never a zero loss. Results remain research-only;
there is no trade execution, calibration approval or Discord entry authorization.

### Self-healing research inputs after host downtime

`paper-alerts` depends on hourly research inputs written by the separate research launch
agent. After the host is shut down or asleep, the alert schedule resumes before the hourly
sync does and every 5-minute cycle fails with a stale-candle error until the sync catches
up; before this change the gap lasted until the next successful hourly run, and one
missed run during a wake-time DNS failure extended it further.

Now, when stored research is stale or missing, `paper-alerts` performs the same bounded
refresh the research agent does (Coinbase candles, Deribit DVOL, funding, ticker, and one
regime snapshot) and retries the evaluation once. Each recovery is recorded in
`research_data_sync_runs` with `purpose = "paper-alerts-recovery"` so it is distinguishable
from the scheduled hourly sync. The scheduled research agent remains the primary source;
the recovery path only runs when a cycle would otherwise fail.

All three HTTP data clients (Coinbase, Deribit, Kalshi) also retry transport-level failures
that never produced a response (DNS resolution, connection refused, connect/read timeouts)
up to three times with 1, 2, and 4 second delays. HTTP status errors keep their existing
handling; Kalshi 429 backoff is unchanged and applies after transport retries.

## Operational verification

After each deployment:

```bash
readlink "$HOME/Library/Application Support/PredictionMarketSystem/app"
launchctl print "gui/$(id -u)/com.karisheen.prediction-market-alerts"
launchctl print "gui/$(id -u)/com.karisheen.prediction-market-maintenance"
uv run pms paper-alerts \
  --series KXBTC --symbol BTC \
  --interval 60 --realized-window-days 30
```

The final command is intentionally shadow-only. A healthy cycle reports current research,
calibrated evaluations, and zero delivery attempts.

## Future work

1. **Move the scheduler to an always-on host.** LaunchAgents cannot run while this Mac is
   asleep or logged out. Provisioning requires the target host, operating system, access
   method, secret store, backup destination, and service manager. Once available, deploy the
   same committed revision and shared-data policy under a system service rather than a GUI
   LaunchAgent.
2. **Alert on database capacity telemetry.** `pms doctor` now reports file size, free disk,
   WAL size, integrity, cycle gaps, archive failures, error categories, unresolved Discord
   deliveries, and the age of the latest research/regime observation. Wiring those readings
   into an external alerting channel remains future work.
3. **Plan an offline SQLite rewrite.** The historical database already contains substantial
   append-only data. After WATCH compaction has created free pages, back it up, verify the
   backup, stop all agents, run `VACUUM`, and verify integrity before restart.
4. **Add off-machine backups.** Daily `pms db-backup` archives are still on the same disk.
   `pms db-snapshot` makes a one-off copy with an integrity check, SHA256 fingerprint, and
   an exercised restore drill; `pms db-restore` restores to a new path only. Copying
   verified backups off the machine and protecting `.env` separately from SQLite remain
   future work.
5. **Evaluate rollup granularity after one month.** Daily WATCH counts preserve operational
   volume but not every old forecast payload. Add model/regime dimensions only if a concrete
   analysis needs them; do not restore unbounded payload retention.
6. **Wait for independent validation evidence.** Do not infer an edge from repeated Discord
   updates or a small resolved sample. Approval still requires held-out events, folds, return
   on cost, and event-weighted calibration evidence.
