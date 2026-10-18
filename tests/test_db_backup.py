import fcntl
import gzip
import hashlib
import json
import shutil
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prediction_market_system import db_backup
from prediction_market_system.cli import app
from prediction_market_system.db_backup import (
    LAST_ATTEMPT_NAME,
    BackupError,
    BackupInProgressError,
    create_backup,
    default_backup_dir,
    last_attempt,
    latest_manifest,
)
from prediction_market_system.heartbeat import backup_check

NOW = datetime(2026, 10, 18, 9, 15, tzinfo=UTC)

runner = CliRunner()


def source_database(tmp_path: Path, rows: int = 2_000) -> Path:
    path = tmp_path / "data" / "prediction_markets.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            "CREATE TABLE ledger (id INTEGER PRIMARY KEY, pair INTEGER NOT NULL, payload TEXT)"
        )
        connection.execute("CREATE INDEX ledger_pair ON ledger (pair)")
        connection.executemany(
            "INSERT INTO ledger (pair, payload) VALUES (?, ?)",
            [(index // 2, "x" * 500) for index in range(rows)],
        )
    return path


def restore(archive: Path, destination: Path) -> Path:
    with gzip.open(archive, "rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
    return destination


def backup(source: Path, backup_dir: Path, now: datetime = NOW, keep: int = 3) -> Path:
    return create_backup(source, backup_dir, keep=keep, free_space_margin_bytes=0, now=now).path


def test_backup_is_consistent_while_another_connection_writes(tmp_path: Path) -> None:
    source = source_database(tmp_path, rows=20_000)
    stop = threading.Event()
    started = threading.Event()
    errors: list[BaseException] = []
    commits = 0

    def write_pairs() -> None:
        nonlocal commits
        with closing(sqlite3.connect(source, timeout=120.0)) as connection:
            connection.execute("PRAGMA busy_timeout = 120000")
            pair = 1_000_000
            while not stop.is_set():
                try:
                    with connection:
                        connection.executemany(
                            "INSERT INTO ledger (pair, payload) VALUES (?, ?)",
                            [(pair, "y" * 500), (pair, "y" * 500)],
                        )
                except sqlite3.Error as exc:
                    errors.append(exc)
                    return
                pair += 1
                commits += 1
                started.set()

    writer = threading.Thread(target=write_pairs)
    writer.start()
    assert started.wait(10)
    try:
        result = create_backup(
            source, tmp_path / "backups", keep=3, free_space_margin_bytes=0, now=NOW
        )
    finally:
        stop.set()
        writer.join()

    assert errors == []
    assert commits > 0
    restored = restore(result.path, tmp_path / "restored.db")
    with closing(sqlite3.connect(restored)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        unpaired = connection.execute(
            "SELECT COUNT(*) FROM (SELECT pair FROM ledger GROUP BY pair HAVING COUNT(*) != 2)"
        ).fetchone()[0]
        backed_up_rows = connection.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    assert unpaired == 0
    assert backed_up_rows >= 20_000
    with closing(sqlite3.connect(source)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_manifest_records_verified_backup(tmp_path: Path) -> None:
    source = source_database(tmp_path)
    backup_dir = tmp_path / "backups"

    result = create_backup(source, backup_dir, keep=3, free_space_margin_bytes=0, now=NOW)

    assert result.path == backup_dir / "prediction_markets-20261018T091500Z.db.gz"
    manifest = json.loads((backup_dir / f"{result.path.name}.json").read_text())
    assert manifest == result.manifest
    assert manifest["status"] == "succeeded"
    assert manifest["created_at"] == NOW.isoformat()
    assert manifest["source_path"] == str(source.resolve())
    assert manifest["source_bytes"] >= source.stat().st_size
    assert manifest["snapshot_bytes"] > manifest["backup_bytes"] == result.path.stat().st_size
    assert manifest["backup_sha256"] == hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert manifest["check"] == "quick_check"
    assert manifest["check_result"] == "ok"
    assert manifest["duration_seconds"] >= 0
    assert latest_manifest(backup_dir) == manifest
    assert last_attempt(backup_dir) == {
        "status": "succeeded",
        "started_at": NOW.isoformat(),
        "completed_at": manifest["completed_at"],
        "backup_file": result.path.name,
    }
    assert sorted(path.name for path in backup_dir.iterdir()) == [
        ".db-backup.lock",
        LAST_ATTEMPT_NAME,
        result.path.name,
        f"{result.path.name}.json",
    ]
    with gzip.open(result.path) as archive:
        assert archive.read(16) == b"SQLite format 3\x00"
    check = backup_check(backup_dir, now=NOW + timedelta(hours=1), max_age=timedelta(hours=36))
    assert check.healthy


def test_retention_keeps_newest_backups_after_success(tmp_path: Path) -> None:
    source = source_database(tmp_path)
    backup_dir = tmp_path / "backups"
    for day in range(3):
        backup(source, backup_dir, now=NOW + timedelta(days=day), keep=2)

    result = create_backup(
        source, backup_dir, keep=2, free_space_margin_bytes=0, now=NOW + timedelta(days=3)
    )

    assert [path.name for path in result.pruned] == ["prediction_markets-20261019T091500Z.db.gz"]
    assert sorted(path.name for path in backup_dir.glob("*.db.gz")) == [
        "prediction_markets-20261020T091500Z.db.gz",
        "prediction_markets-20261021T091500Z.db.gz",
    ]
    assert sorted(path.name for path in backup_dir.glob("*.db.gz.json")) == [
        "prediction_markets-20261020T091500Z.db.gz.json",
        "prediction_markets-20261021T091500Z.db.gz.json",
    ]


def test_low_free_space_refuses_without_touching_existing_backups(tmp_path: Path) -> None:
    source = source_database(tmp_path)
    backup_dir = tmp_path / "backups"
    existing = backup(source, backup_dir)

    with pytest.raises(BackupError, match="refusing to back up"):
        create_backup(
            source,
            backup_dir,
            keep=1,
            free_space_margin_bytes=1 << 60,
            now=NOW + timedelta(days=1),
        )

    assert list(backup_dir.glob("*.db.gz")) == [existing]
    attempt = last_attempt(backup_dir)
    assert attempt is not None
    assert attempt["status"] == "failed"
    assert "refusing to back up" in attempt["error"]
    check = backup_check(backup_dir, now=NOW + timedelta(days=1), max_age=timedelta(hours=36))
    assert not check.healthy
    assert "refusing to back up" in check.detail


def test_failed_integrity_check_keeps_previous_backups_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = source_database(tmp_path)
    backup_dir = tmp_path / "backups"
    existing = backup(source, backup_dir)
    snapshot = db_backup._snapshot

    def corrupt_snapshot(database_path: Path, snapshot_path: Path) -> None:
        snapshot(database_path, snapshot_path)
        with snapshot_path.open("r+b") as handle:
            handle.seek(4096)
            handle.write(b"\xff" * 4096 * 4)

    monkeypatch.setattr(db_backup, "_snapshot", corrupt_snapshot)

    with pytest.raises(BackupError, match="quick_check failed"):
        create_backup(
            source, backup_dir, keep=1, free_space_margin_bytes=0, now=NOW + timedelta(days=1)
        )

    assert list(backup_dir.glob("*.db.gz")) == [existing]
    assert list(backup_dir.glob(".*.partial*")) == []
    attempt = last_attempt(backup_dir)
    assert attempt is not None
    assert attempt["status"] == "failed"
    assert "quick_check failed" in attempt["error"]


def test_concurrent_backup_is_refused_without_recording_failure(tmp_path: Path) -> None:
    source = source_database(tmp_path)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    (backup_dir / ".stale.db.partial").write_bytes(b"in progress")

    with (backup_dir / ".db-backup.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(BackupInProgressError):
            backup(source, backup_dir)

    assert (backup_dir / ".stale.db.partial").exists()
    assert last_attempt(backup_dir) is None


def test_default_backup_dir_is_beside_the_data_directory(tmp_path: Path) -> None:
    source = source_database(tmp_path)
    link = tmp_path / "app"
    link.mkdir()
    (link / "data").symlink_to(source.parent, target_is_directory=True)

    assert default_backup_dir(link / "data" / source.name) == tmp_path.resolve() / "backups"


def test_command_reports_backup_and_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = source_database(tmp_path)
    backup_dir = tmp_path / "backups"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(source))

    succeeded = runner.invoke(app, ["db-backup", "--backup-dir", str(backup_dir), "--keep", "1"])
    refused = runner.invoke(
        app,
        ["db-backup", "--backup-dir", str(backup_dir), "--free-space-margin-gb", "1e9"],
    )

    assert succeeded.exit_code == 0, succeeded.output
    assert "quick_check: ok" in succeeded.output
    assert len(list(backup_dir.glob("*.db.gz"))) == 1
    assert refused.exit_code == 1
    assert "Database backup failed: refusing to back up" in refused.output
