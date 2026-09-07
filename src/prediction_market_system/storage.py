from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from prediction_market_system.backtest import BacktestResult, HistoricalMarketData
from prediction_market_system.calibration import UncertaintyCalibrationProfile
from prediction_market_system.domain import MarketRegimeSnapshot, Opportunity, ProbabilityForecast
from prediction_market_system.engine import deployment_policy_id
from prediction_market_system.evidence import (
    EVIDENCE_SCHEMA,
    EvidenceRepositoryMixin,
    canonical_json,
    content_id,
    load_input_object,
    save_ledger_evaluation,
    save_manifest,
    save_research_context,
    save_venue_revision,
)
from prediction_market_system.operations import backup_database
from prediction_market_system.recipe import MODEL_VERSION
from prediction_market_system.redaction import redact_payload, redact_secrets
from prediction_market_system.research import (
    ResearchContext,
    ResearchDataUnavailable,
    calculate_realized_volatility,
    research_payload_hash,
)
from prediction_market_system.research_storage import RESEARCH_SCHEMA, ResearchRepositoryMixin
from prediction_market_system.validation import campaign_matches_backtest
from prediction_market_system.venues.kalshi import (
    CandlestickPeriod,
    KalshiCandlestick,
    KalshiEventFeeChange,
    KalshiMarket,
    KalshiSeriesFeeChange,
)


class AlertStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    FAILED = "failed"
    SENDING = "sending"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class MarketCheckStatus(StrEnum):
    UNSUPPORTED = "unsupported"
    MISSING_CALIBRATION = "missing_calibration"
    UNAPPROVED_MODEL = "unapproved_model"
    FAILED = "failed"
    WATCH = "watch"
    ENTRY_CANDIDATE = "entry_candidate"
    DELIVERED = "delivered"


@dataclass(frozen=True)
class AlertRecord:
    opportunity_id: str
    status: AlertStatus
    discord_message_id: str | None


@dataclass(frozen=True)
class UnresolvedAlertAttempt:
    """A remote delivery whose outcome is not known; blocks further sends for the market."""

    opportunity_id: str
    market_id: str
    status: AlertStatus
    attempts: int
    error: str | None
    updated_at: datetime


@dataclass(frozen=True)
class PaperAlertStatusSummary:
    previous_requested_at: datetime | None
    requested_at: datetime
    cycles: int
    delivered_alerts: int
    resolved_alerts: int
    profitable_alerts: int

    @property
    def unresolved_alerts(self) -> int:
        return self.delivered_alerts - self.resolved_alerts


@dataclass(frozen=True)
class WatchCompactionResult:
    cutoff_at: datetime
    eligible_checks: int
    rolled_up_checks: int
    deleted_checks: int
    deleted_opportunities: int
    deleted_forecasts: int
    deleted_cycles: int
    batches: int
    applied: bool


@dataclass(frozen=True)
class KalshiHistoryWriteResult:
    market_snapshots: int = 0
    candlesticks: int = 0
    rule_snapshots: int = 0
    resolutions: int = 0
    series_fee_changes: int = 0
    event_fee_changes: int = 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteRepository(ResearchRepositoryMixin, EvidenceRepositoryMixin):
    """Append-oriented forecast, alert, and venue-history audit storage."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_before_v2_migration()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_forecasts_market_generated
                ON forecasts (market_id, generated_at);

                CREATE TABLE IF NOT EXISTS opportunities (
                    opportunity_id TEXT PRIMARY KEY,
                    forecast_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (forecast_id) REFERENCES forecasts (forecast_id)
                );

                CREATE INDEX IF NOT EXISTS idx_opportunities_market_created
                ON opportunities (market_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_opportunities_forecast
                ON opportunities (forecast_id);

                CREATE TABLE IF NOT EXISTS alert_events (
                    opportunity_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    discord_message_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (opportunity_id)
                        REFERENCES opportunities (opportunity_id)
                );

                CREATE TABLE IF NOT EXISTS discord_deliveries (
                    market_id TEXT PRIMARY KEY,
                    discord_message_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kalshi_market_snapshots (
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    event_ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_updated_at TEXT,
                    close_time TEXT NOT NULL,
                    result TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (ticker, observed_at)
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_markets_series_close
                ON kalshi_market_snapshots (series_ticker, close_time);

                CREATE TABLE IF NOT EXISTS kalshi_candlesticks (
                    ticker TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    period_interval_minutes INTEGER NOT NULL,
                    end_period_ts INTEGER NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (ticker, period_interval_minutes, end_period_ts)
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_candles_series_period
                ON kalshi_candlesticks (
                    series_ticker, period_interval_minutes, end_period_ts
                );

                CREATE TABLE IF NOT EXISTS kalshi_rule_snapshots (
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    rules_primary TEXT NOT NULL,
                    rules_secondary TEXT NOT NULL,
                    PRIMARY KEY (ticker, observed_at)
                );

                CREATE TABLE IF NOT EXISTS kalshi_resolutions (
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    result TEXT NOT NULL,
                    settlement_value_dollars TEXT,
                    settlement_ts TEXT,
                    expiration_value TEXT NOT NULL,
                    PRIMARY KEY (ticker, observed_at)
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_resolutions_settlement
                ON kalshi_resolutions (settlement_ts);

                CREATE TABLE IF NOT EXISTS kalshi_series_fee_changes (
                    change_id TEXT PRIMARY KEY,
                    series_ticker TEXT NOT NULL,
                    fee_type TEXT NOT NULL,
                    fee_multiplier REAL NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_series_fees_schedule
                ON kalshi_series_fee_changes (series_ticker, scheduled_at);

                CREATE TABLE IF NOT EXISTS kalshi_event_fee_changes (
                    change_id TEXT PRIMARY KEY,
                    event_ticker TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    fee_type_override TEXT,
                    fee_multiplier_override REAL,
                    scheduled_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_event_fees_schedule
                ON kalshi_event_fee_changes (event_ticker, scheduled_at);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    series_ticker TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_backtest_runs_series_generated
                ON backtest_runs (series_ticker, generated_at);

                CREATE TABLE IF NOT EXISTS uncertainty_calibrations (
                    profile_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    cutoff_at TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_calibrations_model_cutoff
                ON uncertainty_calibrations (
                    symbol, model_name, model_version, cutoff_at
                );

                CREATE TABLE IF NOT EXISTS paper_model_validations (
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    calibration_profile_id TEXT,
                    accepted INTEGER NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, model_name),
                    FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_model_validations_profile
                ON paper_model_validations (
                    calibration_profile_id, accepted, generated_at
                );

                CREATE TABLE IF NOT EXISTS paper_validation_archive_runs (
                    series_ticker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    period_interval_minutes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    counts_json TEXT,
                    error TEXT,
                    PRIMARY KEY (
                        series_ticker, symbol, start_at, end_at,
                        period_interval_minutes
                    )
                );

                CREATE TABLE IF NOT EXISTS paper_validation_campaigns (
                    series_ticker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    discord_message_id TEXT,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (series_ticker, symbol)
                );

                CREATE TABLE IF NOT EXISTS market_regime_snapshots (
                    series_ticker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (series_ticker, symbol, observed_at)
                );

                CREATE INDEX IF NOT EXISTS idx_market_regimes_series_observed
                ON market_regime_snapshots (series_ticker, symbol, observed_at);

                CREATE TABLE IF NOT EXISTS paper_alert_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    series_ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_paper_cycles_series_observed
                ON paper_alert_cycles (series_ticker, observed_at);

                CREATE TABLE IF NOT EXISTS paper_alert_market_checks (
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    event_ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (cycle_id, market_id)
                );

                CREATE INDEX IF NOT EXISTS idx_paper_checks_series_observed
                ON paper_alert_market_checks (series_ticker, observed_at);

                CREATE INDEX IF NOT EXISTS idx_paper_checks_compaction
                ON paper_alert_market_checks (series_ticker, status, observed_at);

                CREATE TABLE IF NOT EXISTS paper_alert_status_requests (
                    series_ticker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    PRIMARY KEY (series_ticker, symbol)
                );

                CREATE TABLE IF NOT EXISTS paper_alert_watch_rollups (
                    series_ticker TEXT NOT NULL,
                    observed_day TEXT NOT NULL,
                    evaluation_count INTEGER NOT NULL,
                    first_observed_at TEXT NOT NULL,
                    last_observed_at TEXT NOT NULL,
                    compacted_at TEXT NOT NULL,
                    PRIMARY KEY (series_ticker, observed_day)
                );

                """
            )
            connection.executescript(RESEARCH_SCHEMA)
            connection.executescript(EVIDENCE_SCHEMA)
            self._apply_migrations(connection)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)",
                ("v2_evidence_foundation", _utc_now()),
            )

    def _backup_before_v2_migration(self) -> None:
        if not self.database_path.exists() or self.database_path.stat().st_size == 0:
            return
        uri = self.database_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            has_schema = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='schema_migrations'"
            ).fetchone()
            if (
                has_schema is not None
                and connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE migration_name='v2_evidence_foundation'"
                ).fetchone()
                is not None
            ):
                return
        destination = (
            self.database_path.parent
            / "backups"
            / (f"{self.database_path.stem}-pre-v2-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.sqlite3")
        )
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        backup_database(self.database_path, destination)

    @staticmethod
    def _apply_migrations(connection: sqlite3.Connection) -> None:
        migration_name = "paper_alert_cycles_backfill_v1"
        applied = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE migration_name = ?",
            (migration_name,),
        ).fetchone()
        if applied is not None:
            return

        connection.execute("BEGIN IMMEDIATE")
        try:
            applied = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE migration_name = ?",
                (migration_name,),
            ).fetchone()
            if applied is None:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO paper_alert_cycles (
                        cycle_id, series_ticker, observed_at
                    )
                    SELECT cycle_id, series_ticker, MIN(observed_at)
                    FROM paper_alert_market_checks
                    GROUP BY cycle_id, series_ticker
                    """
                )
                connection.execute(
                    """
                    INSERT INTO schema_migrations (migration_name, applied_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                    """,
                    (migration_name,),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def save_research_context(self, context: ResearchContext) -> str:
        """Persist the exact research context once; forecasts reference it by content ID."""
        with self._connect() as connection:
            return save_research_context(connection, context)

    def research_context_by_id(self, research_context_id: str) -> ResearchContext | None:
        """Rehydrate a persisted context and re-derive its realized volatility from the store.

        The realized observation carries only a digest of its source candle revisions, so
        the immutable point-in-time spot history must reproduce it exactly. Missing or
        revised history fails closed instead of trusting the recorded number.
        """
        with self._connect() as connection:
            payload = load_input_object(connection, research_context_id)
        if payload is None:
            return None
        context = ResearchContext.model_validate(payload)
        if context.content_id() != research_context_id:
            raise ValueError("persisted research context does not match its identity")
        realized = context.realized_volatility
        candles = self.spot_candles_as_of(
            symbol=context.symbol,
            as_of=context.as_of,
            interval_seconds=int(realized.raw_payload["interval_seconds"]),
            window_seconds=realized.window_seconds,
        )
        reproduced = calculate_realized_volatility(
            candles,
            symbol=context.symbol,
            as_of=context.as_of,
            window_seconds=realized.window_seconds,
        )
        if research_payload_hash(reproduced) != research_payload_hash(realized):
            raise ResearchDataUnavailable(
                "stored spot history does not reproduce the recorded realized volatility"
            )
        return context

    def save_evaluation(
        self,
        forecast: ProbabilityForecast,
        opportunity: Opportunity,
        *,
        ledger_kind: str | None = None,
    ) -> None:
        created_at = forecast.generated_at.isoformat()
        with self._connect() as connection:
            save_ledger_evaluation(connection, forecast, opportunity, ledger_kind=ledger_kind)
            connection.execute(
                """
                INSERT OR IGNORE INTO forecasts (
                    forecast_id, market_id, generated_at, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(forecast.forecast_id),
                    forecast.market_id,
                    created_at,
                    forecast.model_dump_json(),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO opportunities (
                    opportunity_id, forecast_id, market_id, state,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(opportunity.opportunity_id),
                    str(forecast.forecast_id),
                    opportunity.market.market_id,
                    opportunity.state.value,
                    created_at,
                    opportunity.model_dump_json(),
                ),
            )

    def save_paper_alert_cycle(
        self,
        *,
        cycle_id: str,
        series_ticker: str,
        observed_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_alert_cycles (
                    cycle_id, series_ticker, observed_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(cycle_id) DO UPDATE SET
                    observed_at = MIN(paper_alert_cycles.observed_at, excluded.observed_at)
                """,
                (cycle_id, series_ticker.upper(), observed_at.isoformat()),
            )

    def save_paper_market_check(
        self,
        *,
        cycle_id: str,
        market_id: str,
        series_ticker: str,
        event_ticker: str,
        observed_at: datetime,
        status: MarketCheckStatus,
        reason: str | None,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_alert_cycles (
                    cycle_id, series_ticker, observed_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(cycle_id) DO UPDATE SET
                    observed_at = MIN(paper_alert_cycles.observed_at, excluded.observed_at)
                """,
                (cycle_id, series_ticker.upper(), observed_at.isoformat()),
            )
            connection.execute(
                """
                INSERT INTO paper_alert_market_checks (
                    cycle_id, market_id, series_ticker, event_ticker,
                    observed_at, status, reason, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cycle_id, market_id) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    status = excluded.status,
                    reason = excluded.reason,
                    payload_json = excluded.payload_json
                """,
                (
                    cycle_id,
                    market_id,
                    series_ticker.upper(),
                    event_ticker,
                    observed_at.isoformat(),
                    status.value,
                    None if reason is None else redact_secrets(reason)[:1_000],
                    canonical_json(redact_payload(payload)),
                ),
            )

    def paper_market_checks(self, cycle_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT cycle_id, market_id, series_ticker, event_ticker,
                       observed_at, status, reason, payload_json
                FROM paper_alert_market_checks
                WHERE cycle_id = ?
                ORDER BY market_id
                """,
                (cycle_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(str(row["payload_json"])),
            }
            for row in rows
        ]

    def compact_watch_history(
        self,
        *,
        series_ticker: str,
        cutoff_at: datetime,
        apply: bool = False,
        compacted_at: datetime | None = None,
        batch_size: int = 5_000,
    ) -> WatchCompactionResult:
        if cutoff_at.tzinfo is None or cutoff_at.utcoffset() is None:
            raise ValueError("WATCH retention cutoff must be timezone-aware")
        if batch_size < 1:
            raise ValueError("WATCH compaction batch size must be positive")
        normalized_series = series_ticker.upper()
        cutoff = cutoff_at.astimezone(UTC)
        cutoff_text = cutoff.isoformat()
        compacted = compacted_at or datetime.now(UTC)
        if compacted.tzinfo is None or compacted.utcoffset() is None:
            raise ValueError("WATCH compaction timestamp must be timezone-aware")
        compacted_text = compacted.astimezone(UTC).isoformat()

        with self._connect() as connection:
            eligible_row = connection.execute(
                """
                SELECT COUNT(*) AS eligible_checks
                FROM paper_alert_market_checks
                WHERE series_ticker = ? AND status = ? AND observed_at < ?
                """,
                (normalized_series, MarketCheckStatus.WATCH.value, cutoff_text),
            ).fetchone()
        eligible_checks = int(eligible_row["eligible_checks"])
        if not apply or eligible_checks == 0:
            return WatchCompactionResult(
                cutoff_at=cutoff,
                eligible_checks=eligible_checks,
                rolled_up_checks=0,
                deleted_checks=0,
                deleted_opportunities=0,
                deleted_forecasts=0,
                deleted_cycles=0,
                batches=0,
                applied=apply,
            )

        deleted_checks = 0
        deleted_opportunities = 0
        deleted_forecasts = 0
        deleted_cycles = 0
        batches = 0
        while deleted_checks < eligible_checks:
            batch = self._compact_watch_batch(
                series_ticker=normalized_series,
                cutoff_text=cutoff_text,
                compacted_text=compacted_text,
                batch_size=min(batch_size, eligible_checks - deleted_checks),
            )
            batch_checks, batch_opportunities, batch_forecasts, batch_cycles = batch
            if batch_checks == 0:
                raise RuntimeError(
                    "WATCH compaction made no progress before all eligible checks were removed"
                )
            deleted_checks += batch_checks
            deleted_opportunities += batch_opportunities
            deleted_forecasts += batch_forecasts
            deleted_cycles += batch_cycles
            batches += 1

        return WatchCompactionResult(
            cutoff_at=cutoff,
            eligible_checks=eligible_checks,
            rolled_up_checks=deleted_checks,
            deleted_checks=deleted_checks,
            deleted_opportunities=deleted_opportunities,
            deleted_forecasts=deleted_forecasts,
            deleted_cycles=deleted_cycles,
            batches=batches,
            applied=True,
        )

    def _compact_watch_batch(
        self,
        *,
        series_ticker: str,
        cutoff_text: str,
        compacted_text: str,
        batch_size: int,
    ) -> tuple[int, int, int, int]:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TEMP TABLE watch_compaction_targets (
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    opportunity_id TEXT,
                    forecast_id TEXT,
                    PRIMARY KEY (cycle_id, market_id)
                )
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO watch_compaction_targets (
                    cycle_id, market_id, series_ticker, observed_at,
                    opportunity_id, forecast_id
                )
                SELECT checks.cycle_id, checks.market_id, checks.series_ticker,
                       checks.observed_at,
                       json_extract(
                           checks.payload_json, '$.opportunity.opportunity_id'
                       ),
                       opportunities.forecast_id
                FROM paper_alert_market_checks AS checks
                LEFT JOIN opportunities
                  ON opportunities.opportunity_id = json_extract(
                      checks.payload_json, '$.opportunity.opportunity_id'
                  )
                WHERE checks.series_ticker = ?
                  AND checks.status = ?
                  AND checks.observed_at < ?
                ORDER BY checks.observed_at, checks.cycle_id, checks.market_id
                LIMIT ?
                """,
                (
                    series_ticker,
                    MarketCheckStatus.WATCH.value,
                    cutoff_text,
                    batch_size,
                ),
            )
            target_row = connection.execute(
                """
                SELECT COUNT(*) AS target_count,
                       SUM(forecast_id IS NULL) AS missing_count
                FROM watch_compaction_targets
                """
            ).fetchone()
            target_count = int(target_row["target_count"])
            missing_count = int(target_row["missing_count"] or 0)
            if missing_count:
                raise RuntimeError(
                    "refusing to compact WATCH history: "
                    f"{missing_count} checks lack linked evaluations"
                )

            connection.execute(
                """
                INSERT INTO paper_alert_watch_rollups (
                    series_ticker, observed_day, evaluation_count,
                    first_observed_at, last_observed_at, compacted_at
                )
                SELECT series_ticker, substr(observed_at, 1, 10), COUNT(*),
                       MIN(observed_at), MAX(observed_at), ?
                FROM watch_compaction_targets
                GROUP BY series_ticker, substr(observed_at, 1, 10)
                ON CONFLICT(series_ticker, observed_day) DO UPDATE SET
                    evaluation_count = (
                        paper_alert_watch_rollups.evaluation_count
                        + excluded.evaluation_count
                    ),
                    first_observed_at = MIN(
                        paper_alert_watch_rollups.first_observed_at,
                        excluded.first_observed_at
                    ),
                    last_observed_at = MAX(
                        paper_alert_watch_rollups.last_observed_at,
                        excluded.last_observed_at
                    ),
                    compacted_at = excluded.compacted_at
                """,
                (compacted_text,),
            )
            deleted_checks = connection.execute(
                """
                DELETE FROM paper_alert_market_checks
                WHERE (cycle_id, market_id) IN (
                    SELECT cycle_id, market_id
                    FROM watch_compaction_targets
                )
                """
            ).rowcount
            if deleted_checks != target_count:
                raise RuntimeError(
                    "WATCH compaction target changed during deletion: "
                    f"expected {target_count}, deleted {deleted_checks}"
                )
            deleted_opportunities = connection.execute(
                """
                DELETE FROM opportunities
                WHERE opportunity_id IN (
                    SELECT opportunity_id FROM watch_compaction_targets
                )
                  AND state = 'WATCH'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM alert_events
                      WHERE alert_events.opportunity_id = opportunities.opportunity_id
                  )
                """
            ).rowcount
            deleted_forecasts = connection.execute(
                """
                DELETE FROM forecasts
                WHERE forecast_id IN (
                    SELECT forecast_id FROM watch_compaction_targets
                )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM opportunities
                      WHERE opportunities.forecast_id = forecasts.forecast_id
                  )
                """
            ).rowcount
            deleted_cycles = connection.execute(
                """
                DELETE FROM paper_alert_cycles
                WHERE cycle_id IN (
                    SELECT cycle_id FROM watch_compaction_targets
                )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM paper_alert_market_checks
                      WHERE paper_alert_market_checks.cycle_id = paper_alert_cycles.cycle_id
                  )
                """
            ).rowcount

        return deleted_checks, deleted_opportunities, deleted_forecasts, deleted_cycles

    def paper_watch_rollups(self, series_ticker: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT series_ticker, observed_day, evaluation_count,
                       first_observed_at, last_observed_at, compacted_at
                FROM paper_alert_watch_rollups
                WHERE series_ticker = ?
                ORDER BY observed_day
                """,
                (series_ticker.upper(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def paper_alert_status_since_last_request(
        self,
        *,
        series_ticker: str,
        symbol: str,
        requested_at: datetime | None = None,
    ) -> PaperAlertStatusSummary:
        requested = requested_at or datetime.now(UTC)
        if requested.tzinfo is None or requested.utcoffset() is None:
            raise ValueError("status request timestamp must be timezone-aware")
        requested = requested.astimezone(UTC)
        requested_text = requested.isoformat()
        normalized_series = series_ticker.upper()
        normalized_symbol = symbol.upper()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            checkpoint = connection.execute(
                """
                SELECT requested_at
                FROM paper_alert_status_requests
                WHERE series_ticker = ? AND symbol = ?
                """,
                (normalized_series, normalized_symbol),
            ).fetchone()
            previous = (
                None
                if checkpoint is None
                else datetime.fromisoformat(str(checkpoint["requested_at"])).astimezone(UTC)
            )
            previous_text = None if previous is None else previous.isoformat()
            cycle_row = connection.execute(
                """
                SELECT COUNT(*) AS cycle_count
                FROM paper_alert_cycles
                WHERE series_ticker = ?
                  AND observed_at <= ?
                  AND (? IS NULL OR observed_at > ?)
                """,
                (
                    normalized_series,
                    requested_text,
                    previous_text,
                    previous_text,
                ),
            ).fetchone()
            alert_rows = connection.execute(
                """
                WITH latest_resolutions AS (
                    SELECT ticker, result,
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker
                               ORDER BY COALESCE(settlement_ts, observed_at) DESC,
                                        observed_at DESC
                           ) AS resolution_rank
                    FROM kalshi_resolutions
                    WHERE observed_at <= ?
                      AND (settlement_ts IS NULL OR settlement_ts <= ?)
                )
                SELECT checks.payload_json, resolutions.result
                FROM paper_alert_market_checks AS checks
                LEFT JOIN latest_resolutions AS resolutions
                  ON resolutions.ticker = checks.market_id
                 AND resolutions.resolution_rank = 1
                WHERE checks.series_ticker = ?
                  AND checks.status = ?
                  AND checks.observed_at <= ?
                """,
                (
                    requested_text,
                    requested_text,
                    normalized_series,
                    MarketCheckStatus.DELIVERED.value,
                    requested_text,
                ),
            ).fetchall()
            connection.execute(
                """
                INSERT INTO paper_alert_status_requests (
                    series_ticker, symbol, requested_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(series_ticker, symbol) DO UPDATE SET
                    requested_at = excluded.requested_at
                """,
                (normalized_series, normalized_symbol, requested_text),
            )

        resolved_alerts = 0
        profitable_alerts = 0
        for row in alert_rows:
            result = row["result"]
            if result is None:
                continue
            resolved_alerts += 1
            payload = json.loads(str(row["payload_json"]))
            side = payload.get("opportunity", {}).get("side")
            if isinstance(side, str) and side.casefold() == str(result).casefold():
                profitable_alerts += 1

        return PaperAlertStatusSummary(
            previous_requested_at=previous,
            requested_at=requested,
            cycles=int(cycle_row["cycle_count"]),
            delivered_alerts=len(alert_rows),
            resolved_alerts=resolved_alerts,
            profitable_alerts=profitable_alerts,
        )

    def save_kalshi_history(
        self,
        *,
        series_ticker: str,
        observed_at: datetime,
        markets: list[KalshiMarket],
        candlesticks: dict[str, list[KalshiCandlestick]],
        period_interval: CandlestickPeriod,
        series_fee_changes: list[KalshiSeriesFeeChange],
        event_fee_changes: list[KalshiEventFeeChange],
    ) -> KalshiHistoryWriteResult:
        market_tickers = {market.ticker for market in markets}
        unexpected_tickers = candlesticks.keys() - market_tickers
        if unexpected_tickers:
            unexpected = ", ".join(sorted(unexpected_tickers))
            raise ValueError(f"candlesticks supplied for unknown markets: {unexpected}")

        observed = observed_at.isoformat()
        inserted = {
            "market_snapshots": 0,
            "candlesticks": 0,
            "rule_snapshots": 0,
            "resolutions": 0,
            "series_fee_changes": 0,
            "event_fee_changes": 0,
        }
        with self._connect() as connection:
            for market in markets:
                save_venue_revision(
                    connection,
                    kind="market",
                    source_key=market.ticker,
                    series_ticker=series_ticker,
                    available_at=observed_at,
                    payload=market.model_dump(mode="json"),
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_market_snapshots (
                        ticker, observed_at, series_ticker, event_ticker, status,
                        source_updated_at, close_time, result, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market.ticker,
                        observed,
                        series_ticker,
                        market.event_ticker,
                        market.status,
                        market.updated_time.isoformat() if market.updated_time else None,
                        market.close_time.isoformat(),
                        market.result,
                        market.model_dump_json(),
                    ),
                )
                inserted["market_snapshots"] += cursor.rowcount

                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_rule_snapshots (
                        ticker, observed_at, rules_primary, rules_secondary
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        market.ticker,
                        observed,
                        market.rules_primary,
                        market.rules_secondary,
                    ),
                )
                inserted["rule_snapshots"] += cursor.rowcount

                if market.result:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO kalshi_resolutions (
                            ticker, observed_at, result, settlement_value_dollars,
                            settlement_ts, expiration_value
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            market.ticker,
                            observed,
                            market.result,
                            (
                                str(market.settlement_value_dollars)
                                if market.settlement_value_dollars is not None
                                else None
                            ),
                            market.settlement_ts.isoformat() if market.settlement_ts else None,
                            market.expiration_value,
                        ),
                    )
                    inserted["resolutions"] += cursor.rowcount

                for candle in candlesticks.get(market.ticker, []):
                    save_venue_revision(
                        connection,
                        kind="candle",
                        source_key=f"{market.ticker}:{period_interval}:{candle.end_period_ts}",
                        series_ticker=series_ticker,
                        available_at=observed_at,
                        payload=candle.model_dump(mode="json"),
                    )
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO kalshi_candlesticks (
                            ticker, series_ticker, period_interval_minutes,
                            end_period_ts, retrieved_at, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            market.ticker,
                            series_ticker,
                            period_interval,
                            candle.end_period_ts,
                            observed,
                            candle.model_dump_json(),
                        ),
                    )
                    inserted["candlesticks"] += cursor.rowcount

            for change in series_fee_changes:
                save_venue_revision(
                    connection,
                    kind="series_fee",
                    source_key=change.id,
                    series_ticker=series_ticker,
                    available_at=observed_at,
                    payload=change.model_dump(mode="json"),
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_series_fee_changes (
                        change_id, series_ticker, fee_type, fee_multiplier,
                        scheduled_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        change.id,
                        change.series_ticker,
                        change.fee_type,
                        change.fee_multiplier,
                        change.scheduled_ts.isoformat(),
                        observed,
                        change.model_dump_json(),
                    ),
                )
                inserted["series_fee_changes"] += cursor.rowcount

            for event_change in event_fee_changes:
                save_venue_revision(
                    connection,
                    kind="event_fee",
                    source_key=event_change.id,
                    series_ticker=series_ticker,
                    available_at=observed_at,
                    payload=event_change.model_dump(mode="json"),
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_event_fee_changes (
                        change_id, event_ticker, series_ticker, fee_type_override,
                        fee_multiplier_override, scheduled_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_change.id,
                        event_change.event_ticker,
                        event_change.series_ticker,
                        event_change.fee_type_override,
                        event_change.fee_multiplier_override,
                        event_change.scheduled_ts.isoformat(),
                        observed,
                        event_change.model_dump_json(),
                    ),
                )
                inserted["event_fee_changes"] += cursor.rowcount

        return KalshiHistoryWriteResult(**inserted)

    def load_kalshi_backtest_data(
        self,
        *,
        series_ticker: str,
        start: datetime,
        end: datetime,
        period_interval: CandlestickPeriod,
        max_events: int,
    ) -> tuple[HistoricalMarketData, ...]:
        with self._connect() as connection:
            market_rows = connection.execute(
                """
                WITH ranked_snapshots AS (
                    SELECT series_ticker, event_ticker, ticker, close_time, payload_json,
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker ORDER BY observed_at ASC
                           ) AS snapshot_rank
                    FROM kalshi_market_snapshots
                    WHERE series_ticker = ? AND close_time >= ? AND close_time <= ?
                ),
                latest_snapshots AS (
                    SELECT series_ticker, event_ticker, ticker, close_time, payload_json
                    FROM ranked_snapshots
                    WHERE snapshot_rank = 1
                ),
                selected_events AS (
                    SELECT event_ticker, MIN(close_time) AS event_close
                    FROM latest_snapshots
                    GROUP BY event_ticker
                    ORDER BY event_close ASC
                    LIMIT ?
                )
                SELECT latest.series_ticker, latest.ticker, latest.close_time,
                       latest.payload_json
                FROM latest_snapshots AS latest
                JOIN selected_events USING (event_ticker)
                ORDER BY latest.close_time ASC, latest.ticker ASC
                """,
                (
                    series_ticker.upper(),
                    start.isoformat(),
                    end.isoformat(),
                    max_events,
                ),
            ).fetchall()
            series_fee_rows = connection.execute(
                """
                SELECT retrieved_at AS available_at, payload_json
                FROM kalshi_series_fee_changes
                WHERE series_ticker = ?
                ORDER BY scheduled_at ASC
                """,
                (series_ticker.upper(),),
            ).fetchall()
            event_fee_rows = connection.execute(
                """
                SELECT retrieved_at AS available_at, payload_json
                FROM kalshi_event_fee_changes
                WHERE series_ticker = ?
                ORDER BY scheduled_at ASC
                """,
                (series_ticker.upper(),),
            ).fetchall()
            series_fee_rows = [
                *series_fee_rows,
                *connection.execute(
                    "SELECT available_at,payload_json FROM venue_revision_history "
                    "WHERE series_ticker=? AND kind='series_fee' ORDER BY available_at",
                    (series_ticker.upper(),),
                ).fetchall(),
            ]
            event_fee_rows = [
                *event_fee_rows,
                *connection.execute(
                    "SELECT available_at,payload_json FROM venue_revision_history "
                    "WHERE series_ticker=? AND kind='event_fee' ORDER BY available_at",
                    (series_ticker.upper(),),
                ).fetchall(),
            ]

            series_fees = tuple(
                {
                    change.id: change
                    for change in (
                        KalshiSeriesFeeChange.model_validate_json(str(item["payload_json"]))
                        for item in sorted(series_fee_rows, key=lambda value: value["available_at"])
                    )
                }.values()
            )
            event_fees = tuple(
                {
                    change.id: change
                    for change in (
                        KalshiEventFeeChange.model_validate_json(str(item["payload_json"]))
                        for item in sorted(event_fee_rows, key=lambda value: value["available_at"])
                    )
                }.values()
            )
            markets: list[HistoricalMarketData] = []
            for row in market_rows:
                market = KalshiMarket.model_validate_json(str(row["payload_json"]))
                if market.open_time is not None and market.open_time >= end:
                    continue
                resolution = connection.execute(
                    """
                    SELECT result, settlement_value_dollars, settlement_ts,
                           expiration_value, observed_at
                    FROM kalshi_resolutions
                    WHERE ticker = ?
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    (market.ticker,),
                ).fetchone()
                outcome_available_at = None
                if resolution is not None:
                    outcome_available_at = datetime.fromisoformat(str(resolution["observed_at"]))
                    if resolution["settlement_ts"] is not None:
                        outcome_available_at = max(
                            outcome_available_at,
                            datetime.fromisoformat(str(resolution["settlement_ts"])),
                        )
                    payload = market.model_dump()
                    payload.update(
                        {
                            "result": str(resolution["result"]),
                            "settlement_value_dollars": resolution["settlement_value_dollars"],
                            "settlement_ts": resolution["settlement_ts"],
                            "expiration_value": str(resolution["expiration_value"]),
                        }
                    )
                    market = KalshiMarket.model_validate(payload)

                candle_rows = connection.execute(
                    """
                    SELECT retrieved_at AS available_at, payload_json
                    FROM kalshi_candlesticks
                    WHERE ticker = ? AND period_interval_minutes = ?
                      AND end_period_ts >= ? AND end_period_ts <= ?
                    ORDER BY end_period_ts ASC
                    """,
                    (
                        market.ticker,
                        period_interval,
                        int(start.timestamp()),
                        int(market.close_time.timestamp()),
                    ),
                ).fetchall()
                candle_prefix = f"{market.ticker}:{period_interval}:"
                candle_rows = [
                    *candle_rows,
                    *connection.execute(
                        "SELECT available_at,payload_json FROM venue_revision_history "
                        "WHERE kind='candle' AND substr(source_key,1,?)=? ORDER BY available_at",
                        (len(candle_prefix), candle_prefix),
                    ).fetchall(),
                ]
                metadata_rows = connection.execute(
                    "SELECT observed_at AS available_at,payload_json "
                    "FROM kalshi_market_snapshots WHERE ticker=? "
                    "UNION SELECT available_at,payload_json FROM venue_revision_history "
                    "WHERE kind='market' AND source_key=? ORDER BY available_at",
                    (market.ticker, market.ticker),
                ).fetchall()
                metadata = tuple(
                    (
                        datetime.fromisoformat(str(item["available_at"])),
                        KalshiMarket.model_validate_json(str(item["payload_json"])),
                    )
                    for item in metadata_rows
                )
                candle_revisions = tuple(
                    (
                        datetime.fromisoformat(str(item["available_at"])),
                        KalshiCandlestick.model_validate_json(str(item["payload_json"])),
                    )
                    for item in candle_rows
                )
                latest_candles = {
                    item.end_period_ts: item
                    for _, item in sorted(candle_revisions, key=lambda revision: revision[0])
                }
                markets.append(
                    HistoricalMarketData(
                        series_ticker=str(row["series_ticker"]),
                        market=market,
                        metadata_observed_at=metadata[0][0] if metadata else None,
                        metadata_snapshots=metadata,
                        outcome_available_at=outcome_available_at,
                        candlesticks=tuple(latest_candles[key] for key in sorted(latest_candles)),
                        candlestick_revisions=candle_revisions,
                        series_fee_revisions=tuple(
                            (
                                datetime.fromisoformat(str(item["available_at"])),
                                KalshiSeriesFeeChange.model_validate_json(
                                    str(item["payload_json"])
                                ),
                            )
                            for item in series_fee_rows
                        ),
                        event_fee_revisions=tuple(
                            (
                                datetime.fromisoformat(str(item["available_at"])),
                                KalshiEventFeeChange.model_validate_json(str(item["payload_json"])),
                            )
                            for item in event_fee_rows
                            if json.loads(str(item["payload_json"]))["event_ticker"]
                            == market.event_ticker
                        ),
                        series_fee_changes=series_fees,
                        event_fee_changes=tuple(
                            change
                            for change in event_fees
                            if change.event_ticker == market.event_ticker
                        ),
                    )
                )
        return tuple(markets)

    def save_backtest_result(
        self, result: BacktestResult, *, campaign_id: str | None = None
    ) -> BacktestResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT result_json FROM backtest_runs WHERE run_id=?", (str(result.run_id),)
            ).fetchone()
            if previous is not None:
                if str(previous[0]) != result.model_dump_json():
                    raise ValueError("backtest result is immutable; use a new run ID")
                return result
            population = tuple(item for fold in result.folds for item in fold.forecasts)
            consumed = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT event_id FROM holdout_usage WHERE series_ticker=?",
                    (result.config.series_ticker,),
                ).fetchall()
            }
            reused = {item.event_ticker for item in population} & consumed
            if reused:
                result = result.model_copy(
                    update={
                        "model_validations": tuple(
                            item.model_copy(
                                update={
                                    "accepted_for_paper_alerts": False,
                                    "rejection_reasons": (
                                        *item.rejection_reasons,
                                        f"holdout already consumed: {len(reused)} events; "
                                        "descriptive rerun only",
                                    ),
                                }
                            )
                            for item in result.model_validations
                        ),
                    }
                )
            policy_id = (
                None if result.engine_config is None else deployment_policy_id(result.engine_config)
            )
            if campaign_id is not None:
                if result.engine_config is None:
                    raise ValueError("campaign backtests require the engine configuration")
                registration = connection.execute(
                    """SELECT series_ticker, symbol, superseded_at, configuration_json
                       FROM validation_campaign_registrations WHERE campaign_id=?""",
                    (campaign_id,),
                ).fetchone()
                if registration is None:
                    raise ValueError("validation campaign is not registered")
                if registration["superseded_at"] is not None:
                    raise ValueError("validation campaign is no longer active")
                if (
                    str(registration["series_ticker"]) != result.config.series_ticker
                    or str(registration["symbol"]) != result.config.symbol
                ):
                    raise ValueError("validation campaign does not match this backtest series")
                if not campaign_matches_backtest(
                    json.loads(str(registration["configuration_json"])),
                    config=result.config,
                    engine=result.engine_config,
                ):
                    raise ValueError("backtest does not match the frozen validation campaign")
                result = result.model_copy(
                    update={
                        "model_validations": tuple(
                            item.model_copy(
                                update={
                                    "campaign_id": campaign_id,
                                    "deployment_policy_id": policy_id,
                                }
                            )
                            for item in result.model_validations
                        )
                    }
                )
            save_manifest(
                connection,
                run_id=str(result.run_id),
                kind="walk-forward",
                recorded_at=result.generated_at,
                configuration={
                    "backtest": result.config.model_dump(mode="json"),
                    "engine": None
                    if result.engine_config is None
                    else result.engine_config.model_dump(mode="json"),
                    "campaign_id": campaign_id,
                    "deployment_policy_id": policy_id,
                },
                inputs=result.input_manifest,
            )
            for item in population:
                connection.execute(
                    "INSERT OR IGNORE INTO holdout_usage VALUES (?, ?, ?, ?, ?)",
                    (
                        str(result.run_id),
                        result.config.series_ticker,
                        item.event_ticker,
                        item.recipe_id or "legacy",
                        item.observed_at.isoformat(),
                    ),
                )
            connection.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, series_ticker, generated_at, start_at, end_at,
                    config_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(result.run_id),
                    result.config.series_ticker,
                    result.generated_at.isoformat(),
                    result.config.start.isoformat(),
                    result.config.end.isoformat(),
                    result.config.model_dump_json(),
                    result.model_dump_json(),
                ),
            )
            profiles = (
                *(profile for fold in result.folds for profile in fold.calibration_profiles),
                *result.deployment_profiles,
            )
            for profile in profiles:
                previous_profile = connection.execute(
                    "SELECT payload_json FROM uncertainty_calibrations WHERE profile_id=?",
                    (str(profile.profile_id),),
                ).fetchone()
                if (
                    previous_profile is not None
                    and str(previous_profile[0]) != profile.model_dump_json()
                ):
                    raise ValueError("calibration profile ID cannot overwrite existing evidence")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO uncertainty_calibrations (
                        profile_id, symbol, model_name, model_version,
                        cutoff_at, generated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(profile.profile_id),
                        profile.symbol,
                        profile.model_name,
                        profile.model_version,
                        profile.cutoff_at.isoformat(),
                        profile.generated_at.isoformat(),
                        profile.model_dump_json(),
                    ),
                )
            for validation in result.model_validations:
                connection.execute(
                    """
                    INSERT INTO paper_model_validations_v2 (
                        run_id, model_name, model_version, recipe_id, calibration_profile_id,
                        accepted, generated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(result.run_id),
                        validation.model_name,
                        validation.model_version,
                        validation.recipe_id or "legacy",
                        (
                            None
                            if validation.calibration_profile_id is None
                            else str(validation.calibration_profile_id)
                        ),
                        int(validation.accepted_for_paper_alerts),
                        result.generated_at.isoformat(),
                        validation.model_dump_json(),
                    ),
                )
        return result

    def latest_uncertainty_calibration(
        self,
        *,
        symbol: str,
        model_name: str,
        model_version: str,
        as_of: datetime,
        recipe_id: str | None = None,
    ) -> UncertaintyCalibrationProfile | None:
        if recipe_id is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM uncertainty_calibrations
                WHERE symbol=? AND model_name=? AND model_version=?
                  AND cutoff_at<=? AND generated_at<=?
                  AND json_extract(payload_json,'$.recipe_id')=?
                  AND json_extract(payload_json,'$.research_only')=0
                ORDER BY cutoff_at DESC, generated_at DESC LIMIT 1
                """,
                (
                    symbol.upper(),
                    model_name,
                    model_version,
                    as_of.isoformat(),
                    as_of.isoformat(),
                    recipe_id,
                ),
            ).fetchone()
        return (
            None
            if row is None
            else UncertaintyCalibrationProfile.model_validate_json(str(row["payload_json"]))
        )

    def uncertainty_calibration(self, profile_id: UUID) -> UncertaintyCalibrationProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM uncertainty_calibrations WHERE profile_id=?",
                (str(profile_id),),
            ).fetchone()
        return (
            None
            if row is None
            else UncertaintyCalibrationProfile.model_validate_json(str(row["payload_json"]))
        )

    def is_calibration_approved(
        self,
        profile_id: UUID,
        *,
        as_of: datetime | None = None,
        deployment_policy_id: str | None = None,
    ) -> bool:
        """True only for currently active frozen-campaign evidence under this policy.

        ``accepted_for_paper_alerts`` remains the research-gate result. Deployability
        additionally requires an active campaign identity and a matching operational
        policy fingerprint. Missing campaign or policy metadata fails closed.
        """
        if not deployment_policy_id:
            return False
        boundary = as_of or datetime.now(UTC)
        profile = self.uncertainty_calibration(profile_id)
        if profile is None or profile.recipe_id is None or profile.research_only:
            return False
        if (
            profile.model_version != MODEL_VERSION
            or max(profile.generated_at, profile.cutoff_at) > boundary
        ):
            return False
        with self._connect() as connection:
            row = connection.execute(
                """SELECT v.accepted, v.payload_json, b.config_json
                   FROM paper_model_validations_v2 v
                   JOIN backtest_runs b USING(run_id)
                   WHERE v.calibration_profile_id=? AND v.generated_at<=?
                   ORDER BY v.generated_at DESC LIMIT 1""",
                (str(profile_id), boundary.isoformat()),
            ).fetchone()
        if (
            row is None
            or not bool(row["accepted"])
            or json.loads(str(row["config_json"])).get("require_calibration") is not True
        ):
            return False
        payload = json.loads(str(row["payload_json"]))
        campaign_id = payload.get("campaign_id")
        stored_policy = payload.get("deployment_policy_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            return False
        if not isinstance(stored_policy, str) or stored_policy != deployment_policy_id:
            return False
        return self.campaign_is_active(campaign_id)

    def validation_archive_succeeded(
        self,
        *,
        series_ticker: str,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        period_interval: CandlestickPeriod,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, json_extract(counts_json,'$.coverage_complete') AS complete
                FROM paper_validation_archive_runs
                WHERE series_ticker = ? AND symbol = ?
                  AND start_at = ? AND end_at = ?
                  AND period_interval_minutes = ?
                """,
                (
                    series_ticker.upper(),
                    symbol.upper(),
                    start_at.isoformat(),
                    end_at.isoformat(),
                    period_interval,
                ),
            ).fetchone()
        return row is not None and str(row["status"]) == "succeeded" and row["complete"] == 1

    def begin_validation_archive(
        self,
        *,
        series_ticker: str,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        period_interval: CandlestickPeriod,
    ) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_validation_archive_runs (
                    series_ticker, symbol, start_at, end_at,
                    period_interval_minutes, status, started_at,
                    completed_at, counts_json, error
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, NULL, NULL, NULL)
                ON CONFLICT (
                    series_ticker, symbol, start_at, end_at,
                    period_interval_minutes
                ) DO UPDATE SET
                    status = 'running',
                    started_at = excluded.started_at,
                    completed_at = NULL,
                    counts_json = NULL,
                    error = NULL
                """,
                (
                    series_ticker.upper(),
                    symbol.upper(),
                    start_at.isoformat(),
                    end_at.isoformat(),
                    period_interval,
                    now,
                ),
            )

    def complete_validation_archive(
        self,
        *,
        series_ticker: str,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        period_interval: CandlestickPeriod,
        counts: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE paper_validation_archive_runs
                SET status = ?, completed_at = ?, counts_json = ?, error = ?
                WHERE series_ticker = ? AND symbol = ?
                  AND start_at = ? AND end_at = ?
                  AND period_interval_minutes = ?
                """,
                (
                    "failed" if error is not None else "succeeded",
                    _utc_now(),
                    None if counts is None else json.dumps(counts, sort_keys=True),
                    None if error is None else redact_secrets(error),
                    series_ticker.upper(),
                    symbol.upper(),
                    start_at.isoformat(),
                    end_at.isoformat(),
                    period_interval,
                ),
            )

    def validation_archive_coverage(
        self,
        *,
        series_ticker: str,
        symbol: str,
        period_interval: CandlestickPeriod,
        campaign_start: datetime,
    ) -> tuple[int, datetime]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT start_at,end_at FROM paper_validation_archive_runs
                   WHERE series_ticker=? AND symbol=? AND period_interval_minutes=?
                     AND start_at>=? AND status='succeeded'
                     AND json_extract(counts_json,'$.coverage_complete')=1
                   ORDER BY start_at""",
                (
                    series_ticker.upper(),
                    symbol.upper(),
                    period_interval,
                    campaign_start.isoformat(),
                ),
            ).fetchall()
        coverage_end = campaign_start
        count = 0
        for row in rows:
            start_at = datetime.fromisoformat(str(row["start_at"]))
            end_at = datetime.fromisoformat(str(row["end_at"]))
            if start_at != coverage_end or end_at - start_at != timedelta(days=1):
                break
            count += 1
            coverage_end = end_at
        return count, coverage_end

    def validation_campaign_message_id(
        self,
        *,
        series_ticker: str,
        symbol: str,
    ) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT discord_message_id
                FROM paper_validation_campaigns
                WHERE series_ticker = ? AND symbol = ?
                """,
                (series_ticker.upper(), symbol.upper()),
            ).fetchone()
        if row is None or row["discord_message_id"] is None:
            return None
        return str(row["discord_message_id"])

    def save_validation_campaign(
        self,
        *,
        series_ticker: str,
        symbol: str,
        state: str,
        payload: dict[str, Any],
        discord_message_id: str | None,
    ) -> None:
        with self._connect() as connection:
            report_payload = canonical_json(redact_payload(payload))
            report_id = content_id(
                {
                    "series": series_ticker.upper(),
                    "symbol": symbol.upper(),
                    "payload": payload,
                }
            )
            connection.execute(
                "INSERT OR IGNORE INTO campaign_history VALUES (?, ?, ?, ?, ?)",
                (report_id, series_ticker.upper(), symbol.upper(), _utc_now(), report_payload),
            )
            connection.execute(
                """
                INSERT INTO paper_validation_campaigns (
                    series_ticker, symbol, discord_message_id,
                    state, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_ticker, symbol) DO UPDATE SET
                    discord_message_id = COALESCE(
                        excluded.discord_message_id,
                        paper_validation_campaigns.discord_message_id
                    ),
                    state = excluded.state,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    series_ticker.upper(),
                    symbol.upper(),
                    discord_message_id,
                    state,
                    _utc_now(),
                    report_payload,
                ),
            )

    def save_market_regime(
        self,
        *,
        series_ticker: str,
        regime: MarketRegimeSnapshot,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO market_regime_snapshots (
                    series_ticker, symbol, observed_at, regime, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    series_ticker.upper(),
                    regime.symbol.upper(),
                    regime.observed_at.isoformat(),
                    regime.label,
                    regime.model_dump_json(),
                ),
            )

    def latest_market_regime(
        self,
        *,
        series_ticker: str,
        symbol: str,
        as_of: datetime,
    ) -> MarketRegimeSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM market_regime_snapshots
                WHERE series_ticker = ? AND symbol = ? AND observed_at <= ?
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (
                    series_ticker.upper(),
                    symbol.upper(),
                    as_of.isoformat(),
                ),
            ).fetchone()
        if row is None:
            return None
        return MarketRegimeSnapshot.model_validate_json(str(row["payload_json"]))

    def market_regime_coverage(
        self,
        *,
        series_ticker: str,
        symbol: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT regime, COUNT(*) AS observation_count,
                       MIN(observed_at) AS first_observed_at,
                       MAX(observed_at) AS last_observed_at
                FROM market_regime_snapshots
                WHERE series_ticker = ? AND symbol = ?
                GROUP BY regime
                ORDER BY regime
                """,
                (series_ticker.upper(), symbol.upper()),
            ).fetchall()
        return [dict(row) for row in rows]

    def queue_alert(self, opportunity: Opportunity) -> AlertRecord:
        now = _utc_now()
        opportunity_id = str(opportunity.opportunity_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO alert_events (
                    opportunity_id, market_id, state, status, created_at,
                    updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    opportunity_id,
                    opportunity.market.market_id,
                    opportunity.state.value,
                    AlertStatus.QUEUED.value,
                    now,
                    now,
                    opportunity.model_dump_json(),
                ),
            )
            row = connection.execute(
                """
                SELECT status, discord_message_id
                FROM alert_events
                WHERE opportunity_id = ?
                """,
                (opportunity_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError("failed to queue alert")
        return AlertRecord(
            opportunity_id=opportunity_id,
            status=AlertStatus(str(row["status"])),
            discord_message_id=row["discord_message_id"],
        )

    def claim_alert_attempt(self, opportunity: Opportunity) -> None:
        """Persist a per-market claim before network I/O; crashes remain unresolved."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            blocked = connection.execute(
                "SELECT 1 FROM alert_events WHERE market_id=? "
                "AND status IN ('sending','uncertain','failed') LIMIT 1",
                (opportunity.market.market_id,),
            ).fetchone()
            if blocked is not None:
                raise RuntimeError(
                    "remote delivery outcome unresolved; manual reconciliation required"
                )
            cursor = connection.execute(
                "UPDATE alert_events SET status='sending', attempts=attempts+1, updated_at=? "
                "WHERE opportunity_id=? AND status IN ('queued','rejected')",
                (_utc_now(), str(opportunity.opportunity_id)),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("delivery attempt already claimed or completed")

    def unresolved_alert_attempts(self) -> tuple[UnresolvedAlertAttempt, ...]:
        """Deliveries that were claimed but never confirmed sent or rejected by Discord."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT opportunity_id, market_id, status, attempts, error, updated_at "
                "FROM alert_events WHERE status IN ('sending','uncertain','failed') "
                "ORDER BY updated_at, opportunity_id"
            ).fetchall()
        return tuple(
            UnresolvedAlertAttempt(
                opportunity_id=str(row["opportunity_id"]),
                market_id=str(row["market_id"]),
                status=AlertStatus(str(row["status"])),
                attempts=int(row["attempts"]),
                error=None if row["error"] is None else str(row["error"]),
                updated_at=datetime.fromisoformat(str(row["updated_at"])).astimezone(UTC),
            )
            for row in rows
        )

    def resolve_alert_attempt(
        self,
        opportunity_id: str,
        *,
        delivered: bool,
        discord_message_id: str | None,
        note: str,
    ) -> AlertRecord:
        """Record the operator-observed outcome of an unresolved remote delivery.

        The system cannot learn the truth on its own, so the operator must state
        what Discord shows. A delivered outcome requires the observed message ID and
        becomes the market's update target; a not-delivered outcome returns the
        attempt to ``rejected`` so a later cycle may claim it again. Nothing here
        sends, edits, or deletes remote messages.
        """
        note = redact_secrets(note.strip())
        if not note:
            raise ValueError("reconciliation requires a non-empty operator note")
        if delivered and not (discord_message_id and discord_message_id.strip()):
            raise ValueError("a delivered reconciliation requires the observed Discord message ID")
        if not delivered and discord_message_id:
            raise ValueError("a not-delivered reconciliation cannot carry a message ID")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT market_id, status FROM alert_events WHERE opportunity_id=?",
                (opportunity_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown alert opportunity")
            if str(row["status"]) not in {"sending", "uncertain", "failed"}:
                raise ValueError(f"alert is {row['status']}, not awaiting reconciliation")
            status = AlertStatus.DELIVERED if delivered else AlertStatus.REJECTED
            connection.execute(
                "UPDATE alert_events SET status=?, discord_message_id=?, error=?, updated_at=? "
                "WHERE opportunity_id=?",
                (
                    status.value,
                    discord_message_id.strip() if discord_message_id else None,
                    f"manual reconciliation ({now}): {note}"[:1000],
                    now,
                    opportunity_id,
                ),
            )
            if delivered and discord_message_id:
                connection.execute(
                    """
                    INSERT INTO discord_deliveries (market_id, discord_message_id, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(market_id) DO UPDATE SET
                        discord_message_id = excluded.discord_message_id,
                        updated_at = excluded.updated_at
                    """,
                    (str(row["market_id"]), discord_message_id.strip(), now),
                )
        return AlertRecord(
            opportunity_id=opportunity_id,
            status=status,
            discord_message_id=discord_message_id.strip() if discord_message_id else None,
        )

    def mark_alert_outcome(self, opportunity_id: str, error: str, *, uncertain: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE alert_events SET status=?, error=?, updated_at=? "
                "WHERE opportunity_id=? AND status='sending'",
                (
                    AlertStatus.UNCERTAIN.value if uncertain else AlertStatus.REJECTED.value,
                    redact_secrets(error)[:1000],
                    _utc_now(),
                    opportunity_id,
                ),
            )

    def get_discord_delivery(self, market_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT discord_message_id
                FROM discord_deliveries
                WHERE market_id = ?
                """,
                (market_id,),
            ).fetchone()
        return None if row is None else str(row["discord_message_id"])

    def mark_alert_delivered(
        self,
        opportunity: Opportunity,
        discord_message_id: str,
    ) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE alert_events
                SET status = ?,
                    discord_message_id = ?, error = NULL, updated_at = ?
                WHERE opportunity_id = ?
                """,
                (
                    AlertStatus.DELIVERED.value,
                    discord_message_id,
                    now,
                    str(opportunity.opportunity_id),
                ),
            )
            connection.execute(
                """
                INSERT INTO discord_deliveries (
                    market_id, discord_message_id, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    discord_message_id = excluded.discord_message_id,
                    updated_at = excluded.updated_at
                """,
                (opportunity.market.market_id, discord_message_id, now),
            )

    def opportunity_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM opportunities
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=120.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 120000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection
