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
8. Rewrites and reloads the alert, research, archive, validation, and maintenance launch agents.
9. Restores the previous application pointer and agents if activation fails.

Run from a clean, committed checkout:

```bash
uv run python ops/deploy_macos.py
```

Runtime layout after the first atomic deployment:

```text
~/Library/Application Support/PredictionMarketSystem/
├── .env                 # shared secret configuration
├── data/                # shared SQLite database and logs
├── app -> releases/...  # stable atomic application pointer
└── releases/            # immutable revision-labelled code releases
```

Old releases are deliberately retained for rollback. Remove them manually only after the
active revision and rollback target are known.

### Validation campaign continuity

The existing daily archive and weekly validation schedules remain active. The archive gathers
completed UTC days; validation remains in `COLLECTING EVIDENCE` until the configured
chronological window is complete. Deployment does not weaken the 90-day training, two 30-day
held-out windows, minimum sample/event/fold, return-on-cost, or Brier-score gates.

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
2. **Add database capacity telemetry.** Alert on file size, free disk, cycle gaps, archive
   failures, rate-limit responses, and the age of the latest research/regime observation.
3. **Plan an offline SQLite rewrite.** The historical database already contains substantial
   append-only data. After WATCH compaction has created free pages, back it up, verify the
   backup, stop all agents, run `VACUUM`, and verify integrity before restart.
4. **Add off-machine backups and restore drills.** Protect `.env` separately from SQLite.
   Exercise point-in-time restoration before relying on the long validation campaign.
5. **Evaluate rollup granularity after one month.** Daily WATCH counts preserve operational
   volume but not every old forecast payload. Add model/regime dimensions only if a concrete
   analysis needs them; do not restore unbounded payload retention.
6. **Wait for independent validation evidence.** Do not infer an edge from repeated Discord
   updates or a small resolved sample. Approval still requires held-out events, folds, return
   on cost, and event-weighted calibration evidence.
