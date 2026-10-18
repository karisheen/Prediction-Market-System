"""Verified, compressed online backups of the SQLite audit database."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAST_ATTEMPT_NAME = "db-backup-last-attempt.json"
COMPRESSION_LEVEL = 6
_LOCK_NAME = ".db-backup.lock"
_COPY_CHUNK_BYTES = 8 << 20
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class BackupError(RuntimeError):
    pass


class BackupInProgressError(BackupError):
    pass


@dataclass(frozen=True)
class BackupResult:
    path: Path
    manifest: dict[str, Any]
    pruned: list[Path]


def default_backup_dir(database_path: Path) -> Path:
    return database_path.resolve().parent.parent / "backups"


def backup_files(backup_dir: Path, stem: str) -> list[Path]:
    return sorted(backup_dir.glob(f"{stem}-*.db.gz"), reverse=True)


def manifest_path(backup_path: Path) -> Path:
    return backup_path.with_name(f"{backup_path.name}.json")


def latest_manifest(backup_dir: Path) -> dict[str, Any] | None:
    for path in sorted(backup_dir.glob("*.db.gz.json"), reverse=True):
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "succeeded"
            and (backup_dir / manifest["backup_file"]).exists()
        ):
            return manifest
    return None


def last_attempt(backup_dir: Path) -> dict[str, Any] | None:
    path = backup_dir / LAST_ATTEMPT_NAME
    if not path.exists():
        return None
    attempt: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return attempt


def create_backup(
    database_path: Path,
    backup_dir: Path,
    *,
    keep: int,
    free_space_margin_bytes: int,
    now: datetime | None = None,
) -> BackupResult:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    started_at = (now or datetime.now(UTC)).astimezone(UTC)
    backup_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(backup_dir):
        _remove_partials(backup_dir)
        try:
            result = _create_verified_backup(
                database_path,
                backup_dir,
                keep=keep,
                free_space_margin_bytes=free_space_margin_bytes,
                started_at=started_at,
            )
        except Exception as exc:
            _remove_partials(backup_dir)
            _write_json(
                backup_dir / LAST_ATTEMPT_NAME,
                {
                    "status": "failed",
                    "started_at": started_at.isoformat(),
                    "completed_at": datetime.now(UTC).isoformat(),
                    "error": str(exc) or type(exc).__name__,
                },
            )
            raise
        _write_json(
            backup_dir / LAST_ATTEMPT_NAME,
            {
                "status": "succeeded",
                "started_at": started_at.isoformat(),
                "completed_at": result.manifest["completed_at"],
                "backup_file": result.path.name,
            },
        )
        return result


def _create_verified_backup(
    database_path: Path,
    backup_dir: Path,
    *,
    keep: int,
    free_space_margin_bytes: int,
    started_at: datetime,
) -> BackupResult:
    clock = time.monotonic()
    if not database_path.exists():
        raise BackupError(f"database not found: {database_path}")
    wal_path = database_path.with_name(f"{database_path.name}-wal")
    source_bytes = database_path.stat().st_size + (
        wal_path.stat().st_size if wal_path.exists() else 0
    )
    required = 2 * source_bytes + free_space_margin_bytes
    free = shutil.disk_usage(backup_dir).free
    if free < required:
        raise BackupError(
            f"refusing to back up: {_gib(free)} free in {backup_dir}, need {_gib(required)} "
            f"(2 x {_gib(source_bytes)} database + {_gib(free_space_margin_bytes)} margin)"
        )

    stem = database_path.stem
    final_path = backup_dir / f"{stem}-{started_at.strftime(_TIMESTAMP_FORMAT)}.db.gz"
    if final_path.exists():
        raise BackupError(f"backup already exists: {final_path}")
    snapshot_path = backup_dir / f".{final_path.name.removesuffix('.gz')}.partial"
    compressed_path = backup_dir / f".{final_path.name}.partial"

    _snapshot(database_path, snapshot_path)
    snapshot_bytes = snapshot_path.stat().st_size
    check_result = _quick_check(snapshot_path)
    if check_result != "ok":
        raise BackupError(f"quick_check failed on backup copy: {check_result}")
    _compress(snapshot_path, compressed_path, archive_name=final_path.name.removesuffix(".gz"))
    os.replace(compressed_path, final_path)
    _fsync_directory(backup_dir)
    snapshot_path.unlink()

    manifest: dict[str, Any] = {
        "status": "succeeded",
        "created_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.monotonic() - clock, 1),
        "source_path": str(database_path.resolve()),
        "source_bytes": source_bytes,
        "snapshot_bytes": snapshot_bytes,
        "backup_file": final_path.name,
        "backup_bytes": final_path.stat().st_size,
        "backup_sha256": _sha256(final_path),
        "method": "sqlite3 online backup (read-only source, single step)",
        "compression": f"gzip level {COMPRESSION_LEVEL}",
        "check": "quick_check",
        "check_result": check_result,
        "sqlite_version": sqlite3.sqlite_version,
    }
    _write_json(manifest_path(final_path), manifest)
    pruned = _prune(backup_dir, stem, keep=keep)
    return BackupResult(final_path, manifest, pruned)


def _snapshot(database_path: Path, snapshot_path: Path) -> None:
    source_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as source,
        closing(sqlite3.connect(snapshot_path)) as target,
    ):
        source.execute("PRAGMA busy_timeout = 120000")
        # One step copies the whole file inside a single WAL read snapshot. Paged steps would
        # restart from page one whenever the live writer commits between steps.
        source.backup(target, pages=-1)
        target.execute("PRAGMA journal_mode = DELETE")


def _quick_check(path: Path) -> str:
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    except sqlite3.DatabaseError as exc:
        return f"error: {exc}"
    return "; ".join(rows[:5]) if rows else "no result"


def _compress(source: Path, destination: Path, *, archive_name: str) -> None:
    with source.open("rb") as reader, destination.open("wb") as raw:
        with gzip.GzipFile(
            filename=archive_name,
            mode="wb",
            compresslevel=COMPRESSION_LEVEL,
            fileobj=raw,
        ) as writer:
            shutil.copyfileobj(reader, writer, _COPY_CHUNK_BYTES)
        raw.flush()
        os.fsync(raw.fileno())


def _prune(backup_dir: Path, stem: str, *, keep: int) -> list[Path]:
    pruned: list[Path] = []
    for path in backup_files(backup_dir, stem)[keep:]:
        manifest_path(path).unlink(missing_ok=True)
        path.unlink()
        pruned.append(path)
    if pruned:
        _fsync_directory(backup_dir)
    return pruned


def _remove_partials(backup_dir: Path) -> None:
    for path in backup_dir.glob(".*.partial*"):
        path.unlink()


@contextmanager
def _exclusive_lock(backup_dir: Path) -> Iterator[None]:
    with (backup_dir / _LOCK_NAME).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupInProgressError(f"another backup is running in {backup_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.new")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.1f} GiB"
