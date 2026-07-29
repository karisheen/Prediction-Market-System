from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from prediction_market_system.domain import Opportunity, ProbabilityForecast


class AlertStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass(frozen=True)
class AlertRecord:
    opportunity_id: str
    status: AlertStatus
    discord_message_id: str | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteRepository:
    """Append-oriented forecast and alert audit storage."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
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
                """
            )

    def save_evaluation(
        self,
        forecast: ProbabilityForecast,
        opportunity: Opportunity,
    ) -> None:
        created_at = forecast.generated_at.isoformat()
        with self._connect() as connection:
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
                SET status = ?, attempts = attempts + 1,
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

    def mark_alert_failed(self, opportunity_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE alert_events
                SET status = ?, attempts = attempts + 1,
                    error = ?, updated_at = ?
                WHERE opportunity_id = ?
                """,
                (
                    AlertStatus.FAILED.value,
                    error[:1_000],
                    _utc_now(),
                    opportunity_id,
                ),
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
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection
