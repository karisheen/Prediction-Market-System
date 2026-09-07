import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from prediction_market_system.operations import (
    backup_database,
    database_health,
    restore_database,
)


def _create_database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            "CREATE TABLE evidence (id INTEGER PRIMARY KEY, value BLOB, note TEXT);"
            "CREATE INDEX evidence_note ON evidence(note);"
            "INSERT INTO evidence VALUES (1, X'00FF', 'untouched evidence');"
            "PRAGMA user_version = 7;"
        )


def test_online_wal_backup_restores_committed_data_while_writer_connected(tmp_path: Path) -> None:
    source, backup, restored = (tmp_path / name for name in ("live.db", "backup.db", "restore.db"))
    _create_database(source)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("INSERT INTO evidence VALUES (2, X'0102', 'committed WAL data')")
        writer.commit()
        writer.execute("INSERT INTO evidence VALUES (3, NULL, 'uncommitted data')")
        before = source.read_bytes()
        backup_evidence = backup_database(source, backup)
        restore_evidence = restore_database(backup, restored)
        assert source.read_bytes() == before
        assert writer.execute("SELECT COUNT(*) FROM evidence").fetchone() == (3,)
        assert backup_evidence["fingerprint"] == restore_evidence["fingerprint"]
        assert backup_evidence["restore_verified"] is True
        assert backup_evidence["sha256"] == hashlib.sha256(backup.read_bytes()).hexdigest()
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
        assert stat.S_IMODE(restored.stat().st_mode) == 0o600
        with closing(sqlite3.connect(restored)) as recovered:
            assert recovered.execute("SELECT * FROM evidence ORDER BY id").fetchall() == [
                (1, b"\x00\xff", "untouched evidence"),
                (2, b"\x01\x02", "committed WAL data"),
            ]
            assert recovered.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert recovered.execute("PRAGMA user_version").fetchone() == (7,)
            assert recovered.execute(
                "SELECT name FROM sqlite_schema WHERE type='index'"
            ).fetchall() == [("evidence_note",)]
        writer.rollback()
    assert not Path(str(backup) + "-wal").exists()
    assert not Path(str(restored) + "-wal").exists()


@pytest.mark.parametrize("operation", [backup_database, restore_database])
def test_recovery_refuses_overwrite_self_and_hardlinks(
    tmp_path: Path, operation: Callable[[Path, Path], dict[str, Any]]
) -> None:
    source, destination, alias = (tmp_path / name for name in ("source.db", "other.db", "alias.db"))
    _create_database(source)
    destination.write_bytes(b"preexisting destination")
    before = source.read_bytes()
    with pytest.raises(ValueError):
        operation(source, source)
    with pytest.raises(FileExistsError):
        operation(source, destination)
    os.link(source, alias)
    with pytest.raises(FileExistsError):
        operation(source, alias)
    assert source.read_bytes() == before
    assert destination.read_bytes() == b"preexisting destination"
    assert alias.read_bytes() == before


@pytest.mark.parametrize("operation", [backup_database, restore_database])
def test_recovery_rejects_corruption_without_partial_artifacts(
    tmp_path: Path, operation: Callable[[Path, Path], dict[str, Any]]
) -> None:
    source, destination = tmp_path / "broken.db", tmp_path / "recovered.db"
    source.write_bytes(b"this is not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError):
        operation(source, destination)
    assert source.read_bytes() == b"this is not a SQLite database"
    assert not destination.exists()
    assert set(tmp_path.iterdir()) == {source}
    destination.write_bytes(b"do not truncate")
    with pytest.raises(FileExistsError):
        operation(source, destination)
    assert destination.read_bytes() == b"do not truncate"


def test_recovery_refuses_symlinks_and_orphan_sidecars(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "backup.db"
    _create_database(source)
    alias = tmp_path / "alias.db"
    alias.symlink_to(source)
    with pytest.raises(ValueError):
        backup_database(alias, destination)
    destination.symlink_to(tmp_path / "absent.db")
    with pytest.raises(ValueError):
        restore_database(source, destination)
    assert destination.is_symlink()
    destination.unlink()
    sidecar = Path(str(destination) + "-wal")
    sidecar.write_bytes(b"unowned WAL")
    with pytest.raises(FileExistsError):
        backup_database(source, destination)
    assert sidecar.read_bytes() == b"unowned WAL"
    assert not destination.exists()


def test_missing_database_is_never_created(tmp_path: Path) -> None:
    missing, destination = tmp_path / "missing.db", tmp_path / "backup.db"
    with pytest.raises(FileNotFoundError):
        database_health(missing)
    with pytest.raises(FileNotFoundError):
        backup_database(missing, destination)
    with pytest.raises(FileNotFoundError):
        restore_database(missing, destination)
    assert not missing.exists()
    assert not destination.exists()


def test_health_preserves_legacy_database_and_reports_corruption(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _create_database(path)
    before = path.read_bytes()
    report = database_health(path)
    assert report["integrity"] == "ok"
    assert report["latest_operations"]["research_sync"] is None
    assert report["freshness"]["regime"]["latest_at"] is None
    assert report["database_bytes"] == len(before)
    assert report["page_count"] * report["page_size"] == len(before)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}
    path.write_bytes(b"corrupt bytes")
    report = database_health(path)
    assert report["integrity"] == "failed"
    assert path.read_bytes() == b"corrupt bytes"


def test_health_reports_gaps_status_freshness_and_redacted_failures(tmp_path: Path) -> None:
    path = tmp_path / "health.db"
    now = datetime.now(UTC)
    latest = now - timedelta(seconds=60)
    earlier = latest - timedelta(minutes=20)
    fake_token = "synthetic-webhook-token"
    error = f"HTTP 429 https://discord.com/api/webhooks/123/{fake_token}"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            "CREATE TABLE research_data_sync_runs "
            "(status TEXT, started_at TEXT, completed_at TEXT, error TEXT, request_json TEXT);"
            "CREATE TABLE paper_alert_cycles (series_ticker TEXT, observed_at TEXT);"
            "CREATE TABLE market_regime_snapshots "
            "(series_ticker TEXT, symbol TEXT, observed_at TEXT, regime TEXT, payload_json TEXT);"
        )
        connection.execute(
            "INSERT INTO research_data_sync_runs VALUES (?, ?, ?, ?, ?)",
            ("failed", earlier.isoformat(), latest.isoformat(), error, "secret raw payload"),
        )
        connection.executemany(
            "INSERT INTO paper_alert_cycles VALUES (?, ?)",
            [("KXBTC", earlier.isoformat()), ("KXBTC", latest.isoformat())],
        )
        connection.execute(
            "INSERT INTO market_regime_snapshots VALUES (?, ?, ?, ?, ?)",
            ("KXBTC", "BTC", latest.isoformat(), "elevated", "secret raw payload"),
        )
        connection.commit()
    before = path.read_bytes()
    report = database_health(path)
    assert report["integrity"] == "ok"
    assert report["latest_operations"]["research_sync"]["status"] == "failed"
    assert report["latest_operations"]["regime"]["regime"] == "elevated"
    assert report["freshness"]["regime"]["latest_at"] == latest.isoformat()
    assert report["cycle_gaps"]["series"]["KXBTC"]["maximum_gap_seconds"] == 1200
    assert report["error_categories"]["research_sync"]["counts"] == {"rate_limit": 1}
    assert fake_token not in json.dumps(report)
    assert "secret raw payload" not in json.dumps(report)
    assert path.read_bytes() == before


def test_restore_drill_rejects_intact_but_changed_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market_system import operations

    source, destination = tmp_path / "source.db", tmp_path / "backup.db"
    _create_database(source)
    before = source.read_bytes()
    copy_snapshot = operations._copy_snapshot
    copies = 0

    def faulty_restore(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
        nonlocal copies
        evidence = copy_snapshot(connection, path)
        copies += 1
        if copies == 2:
            with closing(sqlite3.connect(path)) as recovered:
                recovered.execute("UPDATE evidence SET note = 'changed during restore'")
                recovered.commit()
                assert recovered.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                evidence = operations._fingerprint(recovered)
        return evidence

    monkeypatch.setattr(operations, "_copy_snapshot", faulty_restore)
    with pytest.raises(ValueError, match="Restored schema/data fingerprint"):
        backup_database(source, destination)
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {source}
