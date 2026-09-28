"""Audit and recovery commands, separated from collection/evaluation orchestration."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from prediction_market_system.config import Settings
from prediction_market_system.experiment_cli import register_experiment_commands
from prediction_market_system.operations import backup_database, database_health, restore_database
from prediction_market_system.redaction import redact_payload, redact_secrets
from prediction_market_system.storage import SQLiteRepository


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(redact_payload(payload), sort_keys=True, indent=2, allow_nan=False))


def _repository() -> SQLiteRepository:
    path = Settings().database_path
    if not path.is_file():
        raise typer.BadParameter("database does not exist; run init-db explicitly")
    return SQLiteRepository(path)


def register_audit_commands(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor() -> None:
        """Read-only database integrity, capacity, freshness and operational gaps (JSON)."""
        try:
            report = database_health(Settings().database_path)
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        _emit(report)
        if report.get("integrity") == "failed":
            raise typer.Exit(code=1)

    @app.command("alerts-unresolved")
    def alerts_unresolved() -> None:
        """List claimed Discord deliveries whose remote outcome is unknown (JSON)."""
        repository = _repository()
        attempts = repository.unresolved_alert_attempts()
        _emit(
            {
                "count": len(attempts),
                "attempts": [
                    {
                        "opportunity_id": attempt.opportunity_id,
                        "market_id": attempt.market_id,
                        "status": attempt.status.value,
                        "attempts": attempt.attempts,
                        "error": attempt.error,
                        "updated_at": attempt.updated_at.isoformat(),
                    }
                    for attempt in attempts
                ],
            }
        )

    @app.command("alerts-reconcile")
    def alerts_reconcile(
        opportunity_id: Annotated[str, typer.Argument(help="Unresolved alert opportunity ID.")],
        note: Annotated[str, typer.Option(help="What the operator observed in Discord.")],
        delivered: Annotated[
            bool, typer.Option("--delivered/--not-delivered", help="Observed remote outcome.")
        ] = False,
        discord_message_id: Annotated[
            str | None, typer.Option(help="Observed message ID (required when delivered).")
        ] = None,
    ) -> None:
        """Record the operator-observed outcome of an uncertain delivery; sends nothing."""
        repository = _repository()
        try:
            record = repository.resolve_alert_attempt(
                opportunity_id,
                delivered=delivered,
                discord_message_id=discord_message_id,
                note=note,
            )
        except ValueError as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        _emit(
            {
                "opportunity_id": record.opportunity_id,
                "status": record.status.value,
                "discord_message_id": record.discord_message_id,
            }
        )

    @app.command("db-backup")
    def db_backup(destination: Annotated[Path, typer.Option(help="New backup file path.")]) -> None:
        """Online SQLite backup with integrity, SHA256 and an exercised restore drill."""
        try:
            report = backup_database(Settings().database_path, destination)
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        _emit(report)

    @app.command("db-restore")
    def db_restore(
        source: Annotated[Path, typer.Option(help="Verified backup database.")],
        destination: Annotated[Path, typer.Option(help="Unused restore path; never overwrites.")],
    ) -> None:
        """Restore to a NEW path; does not switch configuration or restart services."""
        try:
            report = restore_database(source, destination)
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        _emit(report)

    @app.command("explain-forecast")
    def explain_forecast(forecast_id: Annotated[str, typer.Argument()]) -> None:
        """Show exact recorded inputs, recipe, calibration and immutable run manifest."""
        try:
            _emit(_repository().explain_forecast(forecast_id))
        except (ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None

    @app.command("shadow-report")
    def shadow_report(series: Annotated[str, typer.Option()]) -> None:
        """Score every resolved recorded forecast, including WATCH, independently of Discord."""
        try:
            _emit(_repository().shadow_report(series_ticker=series, as_of=datetime.now(UTC)))
        except (ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None

    @app.command("campaign-report")
    def campaign_report(
        series: Annotated[str, typer.Option()], symbol: Annotated[str, typer.Option()] = "BTC"
    ) -> None:
        """Report archive windows, gaps/failures and previously consumed holdout events."""
        try:
            _emit(_repository().campaign_report(series_ticker=series, symbol=symbol))
        except (ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None

    @app.command("compare-runs")
    def compare_runs(
        first: Annotated[str, typer.Argument()], second: Annotated[str, typer.Argument()]
    ) -> None:
        """Compare Brier scores only on identical input revisions and resolved observations."""
        try:
            _emit(_repository().compare_runs(first, second))
        except (ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None

    register_experiment_commands(app)
