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

### Bounded detailed WATCH retention

`pms paper-alert-maintain` compacts detailed `WATCH` evaluations older than a configured UTC
cutoff. The command defaults to a preview and requires `--apply` before it changes data.
The managed launch agent applies a 14-day retention window daily at 03:15 local time and
commits at most 5,000 evaluations per transaction.

Before deletion, the command writes daily counts to `paper_alert_watch_rollups`. It then
removes only the linked detailed records:

- `paper_alert_market_checks` rows whose status is exactly `watch`;
- their `WATCH` opportunities;
- forecasts left without an opportunity; and
- cycles left without any detailed checks.

Entry candidates, delivered alerts, failures, unsupported-market records, missing-calibration
records, unapproved-model records, Discord delivery state, model evidence, market regimes,
research inputs, venue history, and resolutions are not pruned. Each bounded batch is
transactional and releases the SQLite writer lock before the next batch. Compaction refuses
to proceed if a WATCH check cannot be linked to its persisted evaluation. Re-running a
completed compaction does not double-count its rollup.

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
2. Stops the managed launch agents.
3. Moves runtime data and `.env` to shared paths outside a code release on the first run.
4. Copies source into a revision-labelled release directory without secrets, data, caches, or
   Git metadata.
5. Runs `uv sync --frozen`, the full test suite, Ruff, and strict Mypy checks inside the staged
   release.
6. Switches the stable `app` symlink to the verified release.
7. Runs schema migrations once while all managed agents remain stopped.
8. Rewrites and reloads the alert, research, archive, validation, maintenance, backup, and
   heartbeat launch agents.
9. Restores the previous application pointer and agents if activation fails.

Scheduled commands still initialize the store on startup, but they compare `sqlite_master` with
the tables and indexes the schema defines and run DDL only when an object is missing. Applied
migrations are likewise detected read-only, so routine job starts never request the SQLite
writer lock.

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
