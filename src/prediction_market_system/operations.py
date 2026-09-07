"""Explicit, non-destructive SQLite recovery and read-only operational inspection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prediction_market_system.redaction import redact_secrets


def _existing_database(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        raise ValueError("Database paths must not be symlinks")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Database source must be an existing regular file")
    return path.resolve(strict=True)


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    return connection


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _integrity(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        # SQLite diagnostics can include persisted values; do not disclose them.
        raise ValueError("Database integrity check failed")


def _encoded_row(row: tuple[Any, ...]) -> bytes:
    values = [
        ["blob", value.hex()]
        if isinstance(value, bytes)
        else ["float", value.hex()]
        if isinstance(value, float)
        else [type(value).__name__, value]
        for value in row
    ]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"


def _fingerprint(connection: sqlite3.Connection) -> dict[str, Any]:
    """Stream canonical schema/data hashes without loading the research store into memory."""
    schema_rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name, tbl_name"
    ).fetchall()
    schema_hash = hashlib.sha256()
    for row in schema_rows:
        schema_hash.update(_encoded_row(row))
    tables: dict[str, Any] = {}
    for kind, name, _, sql in schema_rows:
        if kind != "table":
            continue
        if sql and str(sql).lstrip().upper().startswith("CREATE VIRTUAL TABLE"):
            raise ValueError("Recovery verification does not support virtual tables")
        columns = connection.execute(f"PRAGMA table_info({_identifier(name)})").fetchall()
        order = ", ".join(str(index + 1) for index in range(len(columns)))
        digest = hashlib.sha256()
        count = 0
        for row in connection.execute(f"SELECT * FROM {_identifier(name)} ORDER BY {order}"):
            digest.update(_encoded_row(row))
            count += 1
        tables[name] = {"rows": count, "sha256": digest.hexdigest()}
    return {
        "schema_sha256": schema_hash.hexdigest(),
        "tables": tables,
        "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
        "application_id": connection.execute("PRAGMA application_id").fetchone()[0],
    }


def _copy_snapshot(source: sqlite3.Connection, destination: Path) -> dict[str, Any]:
    # TemporaryDirectory is private; pre-create restrictive files before SQLite opens them.
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    with closing(sqlite3.connect(destination)) as target:
        source.backup(target, pages=1024)
        # Publish a standalone database, never a file dependent on temporary WAL sidecars.
        target.execute("PRAGMA journal_mode = DELETE")
        _integrity(target)
        return _fingerprint(target)


def _verified_copy(source: Path, destination: Path, operation: str) -> dict[str, Any]:
    source = _existing_database(source)
    destination = Path(os.path.abspath(destination))
    if destination.is_symlink():
        raise ValueError("Database paths must not be symlinks")
    destination = destination.parent.resolve(strict=True) / destination.name
    if source == destination:
        raise ValueError("Source and destination must differ")
    if destination.exists():
        raise FileExistsError("Database destination already exists")
    if not destination.parent.is_dir():
        raise FileNotFoundError("Database destination directory does not exist")
    for suffix in ("-wal", "-shm", "-journal"):
        if os.path.lexists(str(destination) + suffix):
            raise FileExistsError("Database destination has existing SQLite sidecars")

    with tempfile.TemporaryDirectory(prefix=".pms-recovery-", dir=destination.parent) as directory:
        staged = Path(directory) / "snapshot.sqlite3"
        drill = Path(directory) / "restored.sqlite3"
        with closing(_readonly(source)) as original:
            # Pin the fingerprint and backup to the same committed snapshot while writers run.
            original.execute("BEGIN")
            _integrity(original)
            evidence = _fingerprint(original)
            if _copy_snapshot(original, staged) != evidence:
                raise ValueError("Backup schema/data fingerprint does not match source snapshot")
        # A backup is not accepted until it has been reopened and actually restored.
        with closing(_readonly(staged)) as snapshot:
            if _copy_snapshot(snapshot, drill) != evidence:
                raise ValueError("Restored schema/data fingerprint does not match backup")
        with staged.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            os.fsync(handle.fileno())
        size = staged.stat().st_size
        # link is an atomic no-replace publication, unlike rename/replace. A concurrent
        # destination creator wins safely; TemporaryDirectory only removes our own files.
        os.link(staged, destination, follow_symlinks=False)

    return {
        "operation": operation,
        "completed_at": datetime.now(UTC).isoformat(),
        "source": redact_secrets(str(source)),
        "destination": redact_secrets(str(destination)),
        "database_bytes": size,
        "sha256": sha256,
        "integrity": "ok",
        "restore_verified": True,
        "fingerprint": evidence,
    }


def backup_database(source: Path, destination: Path) -> dict[str, Any]:
    """Online backup plus a verified restore drill; never overwrite an existing target."""
    return _verified_copy(source, destination, "backup")


def restore_database(source: Path, destination: Path) -> dict[str, Any]:
    """Restore into a new database path only; callers must stop services before cutover."""
    return _verified_copy(source, destination, "restore")


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _error_category(message: str) -> str:
    lowered = message.lower()
    if "429" in lowered or "rate limit" in lowered:
        return "rate_limit"
    if "locked" in lowered or "busy" in lowered:
        return "database_contention"
    if "stale" in lowered or "missing" in lowered or "unavailable" in lowered:
        return "input_unavailable"
    if any(word in lowered for word in ("timeout", "connection", "dns", "network")):
        return "transport"
    return "other"


_STATUS_TABLES = {
    "research_sync": ("research_data_sync_runs", "started_at"),
    "archive": ("paper_validation_archive_runs", "started_at"),
    "validation": ("paper_model_validations_v2", "generated_at"),
    "legacy_validation": ("paper_model_validations", "generated_at"),
    "validation_campaign": ("paper_validation_campaigns", "updated_at"),
    "cycle": ("paper_alert_cycles", "observed_at"),
    "market_check": ("paper_alert_market_checks", "observed_at"),
    "regime": ("market_regime_snapshots", "observed_at"),
}
_STATUS_FIELDS = (
    "status",
    "state",
    "accepted",
    "regime",
    "symbol",
    "series_ticker",
    "model_name",
    "model_version",
    "started_at",
    "completed_at",
    "generated_at",
    "updated_at",
    "observed_at",
    "start_at",
    "end_at",
    "error",
    "reason",
)
_FRESHNESS_TABLES = {
    "spot_candles": ("crypto_spot_candles", "end_at"),
    "volatility": ("crypto_volatility_observations", "observed_at"),
    "funding": ("crypto_funding_observations", "observed_at"),
    "derivatives": ("crypto_derivatives_snapshots", "observed_at"),
    "regime": ("market_regime_snapshots", "observed_at"),
    "venue_markets": ("kalshi_market_snapshots", "observed_at"),
}


def _columns(connection: sqlite3.Connection, table: str, tables: set[str]) -> set[str]:
    if table not in tables:
        return set()
    return {row[1] for row in connection.execute(f"PRAGMA table_info({_identifier(table)})")}


def _latest_status(
    connection: sqlite3.Connection, table: str, clock: str, columns: set[str], now: datetime
) -> dict[str, Any] | None:
    if clock not in columns:
        return None
    selected = [name for name in _STATUS_FIELDS if name in columns]
    rows = connection.execute(
        f"SELECT {', '.join(_identifier(name) for name in selected)} "
        f"FROM {_identifier(table)} ORDER BY julianday({_identifier(clock)}) DESC LIMIT 1"
    ).fetchall()
    if not rows:
        return None
    result = {
        name: redact_secrets(value) if isinstance(value, str) else value
        for name, value in zip(selected, rows[0], strict=True)
    }
    timestamp = _timestamp(result.get(clock))
    result["age_seconds"] = (now - timestamp).total_seconds() if timestamp else None
    for name in ("error", "reason"):
        if result.get(name):
            result["error_category"] = _error_category(str(result[name]))
            break
    return result


def _cycle_gaps(connection: sqlite3.Connection, columns: set[str], now: datetime) -> dict[str, Any]:
    if not {"observed_at", "series_ticker"} <= columns:
        return {"sample_limit": 1000, "sampled_rows": 0, "series": {}}
    rows = connection.execute(
        "SELECT series_ticker, observed_at FROM paper_alert_cycles "
        "ORDER BY julianday(observed_at) DESC LIMIT 1000"
    ).fetchall()
    groups: dict[str, list[datetime]] = {}
    for series, observed_at in rows:
        timestamp = _timestamp(observed_at)
        if timestamp is not None:
            groups.setdefault(redact_secrets(str(series)), []).append(timestamp)
    result: dict[str, Any] = {}
    for series, timestamps in groups.items():
        gaps = [
            (left - right).total_seconds()
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        ]
        result[series] = {
            "latest_at": timestamps[0].isoformat(),
            "age_seconds": (now - timestamps[0]).total_seconds(),
            "latest_gap_seconds": gaps[0] if gaps else None,
            "maximum_gap_seconds": max(gaps) if gaps else None,
            "sampled_cycles": len(timestamps),
        }
    return {"sample_limit": 1000, "sampled_rows": len(rows), "series": result}


def _unresolved_deliveries(connection: sqlite3.Connection, tables: set[str]) -> dict[str, Any]:
    """Count remote deliveries whose outcome is unknown; each blocks its market's alerts."""
    if "alert_events" not in tables:
        return {"count": 0, "by_status": {}, "markets": []}
    rows = connection.execute(
        "SELECT status, market_id FROM alert_events "
        "WHERE status IN ('sending','uncertain','failed') ORDER BY updated_at LIMIT 100"
    ).fetchall()
    by_status: dict[str, int] = {}
    markets: list[str] = []
    for status, market_id in rows:
        by_status[str(status)] = by_status.get(str(status), 0) + 1
        if str(market_id) not in markets:
            markets.append(str(market_id))
    return {"count": len(rows), "by_status": by_status, "markets": markets[:20]}


def database_health(path: Path) -> dict[str, Any]:
    """Inspect an existing database without migrations, audit writes, or maintenance.

    Missing paths raise rather than creating a database. Corrupt stores return a
    generic failure, never SQLite diagnostics containing persisted secret values.
    Freshness ages are descriptive; they do not approve inputs for evaluation.
    """
    path = _existing_database(path)
    now = datetime.now(UTC)
    report: dict[str, Any] = {
        "checked_at": now.isoformat(),
        "path": redact_secrets(str(path)),
        "database_bytes": path.stat().st_size,
        "free_disk_bytes": shutil.disk_usage(path.parent).free,
        "wal_bytes": 0,
        "integrity": "unknown",
        "latest_operations": {},
        "freshness": {},
        "error_categories": {},
    }
    wal = Path(str(path) + "-wal")
    if wal.is_file() and not wal.is_symlink():
        report["wal_bytes"] = wal.stat().st_size
    try:
        with closing(_readonly(path)) as connection:
            connection.execute("BEGIN")
            _integrity(connection)
            report["integrity"] = "ok"
            for field, pragma in (
                ("page_size", "page_size"),
                ("page_count", "page_count"),
                ("freelist_count", "freelist_count"),
            ):
                report[field] = connection.execute(f"PRAGMA {pragma}").fetchone()[0]
            report["free_page_bytes"] = report["freelist_count"] * report["page_size"]
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")
            }
            for kind, (table, clock) in _STATUS_TABLES.items():
                columns = _columns(connection, table, tables)
                report["latest_operations"][kind] = _latest_status(
                    connection, table, clock, columns, now
                )
                if "error" in columns:
                    # Bound scanning while reporting historical failures beyond the latest run.
                    ordering = (
                        f"ORDER BY julianday({_identifier(clock)}) DESC" if clock in columns else ""
                    )
                    rows = connection.execute(
                        f"SELECT error FROM {_identifier(table)} WHERE error IS NOT NULL "
                        f"{ordering} LIMIT 100"
                    )
                    categories: dict[str, int] = {}
                    for (error,) in rows:
                        category = _error_category(str(error))
                        categories[category] = categories.get(category, 0) + 1
                    report["error_categories"][kind] = {"sample_limit": 100, "counts": categories}
            for kind, (table, clock) in _FRESHNESS_TABLES.items():
                columns = _columns(connection, table, tables)
                timestamp = None
                if clock in columns:
                    row = connection.execute(
                        f"SELECT {_identifier(clock)} FROM {_identifier(table)} "
                        f"ORDER BY julianday({_identifier(clock)}) DESC LIMIT 1"
                    ).fetchone()
                    timestamp = _timestamp(row[0]) if row else None
                report["freshness"][kind] = {
                    "latest_at": timestamp.isoformat() if timestamp else None,
                    "age_seconds": (now - timestamp).total_seconds() if timestamp else None,
                }
            report["cycle_gaps"] = _cycle_gaps(
                connection, _columns(connection, "paper_alert_cycles", tables), now
            )
            report["unresolved_deliveries"] = _unresolved_deliveries(connection, tables)
    except (sqlite3.DatabaseError, ValueError):
        report["integrity"] = "failed"
        report["error"] = "Database integrity or schema inspection failed"
    return report
