"""Offline, verified copy of SQLite evidence into the compact storage format."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.storage_payloads import (
    STORAGE_SCHEMA_VERSION,
    compact_payload,
    load_payload,
)

_PAYLOAD_TABLES = frozenset(
    {
        "input_objects",
        "run_manifests",
        "forecast_ledger",
        "forecasts",
        "opportunities",
        "paper_alert_market_checks",
    }
)
_LOGICAL_DEPENDENCIES = {
    "run_manifests": {"code_identities"},
    "forecast_ledger": {"input_objects", "run_manifests"},
    "forecasts": {"forecast_ledger"},
    "opportunities": {"forecast_ledger", "forecasts"},
    "paper_alert_market_checks": {"opportunities"},
}
_SQLITE_CACHE_KIB = 32 * 1024


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(name): str(sql)
        for name, sql in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({_quote(table)})")]


def _integrity(connection: sqlite3.Connection) -> list[str]:
    result = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
    if result != ["ok"]:
        raise ValueError(f"SQLite integrity check failed: {result!r}")
    return result


def _foreign_keys(connection: sqlite3.Connection) -> list[list[Any]]:
    return [list(row) for row in connection.execute("PRAGMA foreign_key_check")]


def _file_state(path: Path) -> dict[str, tuple[int, int] | None]:
    result: dict[str, tuple[int, int] | None] = {}
    for suffix in ("", "-wal"):
        candidate = Path(str(path) + suffix)
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            result[suffix] = None
        else:
            # A read-only SQLite connection can create empty WAL coordination
            # sidecars. Only a WAL containing frames represents source evidence.
            result[suffix] = (
                None if suffix == "-wal" and stat.st_size == 0 else (stat.st_size, stat.st_mtime_ns)
            )
    return result


def _copy_order(connection: sqlite3.Connection, tables: dict[str, str]) -> list[str]:
    dependencies = {
        table: (
            {str(row[2]) for row in connection.execute(f"PRAGMA foreign_key_list({_quote(table)})")}
            | _LOGICAL_DEPENDENCIES.get(table, set())
        )
        & tables.keys() - {table}
        for table in tables
    }
    ordered: list[str] = []
    while dependencies:
        ready = sorted(table for table, parents in dependencies.items() if not parents)
        if not ready:
            raise ValueError("Unsupported cyclic source table dependencies")
        for table in ready:
            ordered.append(table)
            del dependencies[table]
        for parents in dependencies.values():
            parents.difference_update(ready)
    return ordered


def _sql_tokens(sql: str) -> tuple[str, ...]:
    """Ignore formatting whitespace, never whitespace within quoted SQL values."""
    return tuple(
        re.findall(
            r"""'(?:''|[^'])*'|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\]|\w+|[^\s]""",
            sql,
        )
    )


def _index_definitions(connection: sqlite3.Connection, table: str) -> dict[str, tuple[Any, ...]]:
    definitions: dict[str, tuple[Any, ...]] = {}
    for _, name, unique, origin, partial in connection.execute(
        f"PRAGMA index_list({_quote(table)})"
    ):
        columns = tuple(row[2] for row in connection.execute(f"PRAGMA index_info({_quote(name)})"))
        ordering = tuple(
            (row[2], row[3], row[4], row[5])
            for row in connection.execute(f"PRAGMA index_xinfo({_quote(name)})")
        )
        definitions[str(name)] = (unique, origin, partial, columns, ordering)
    return definitions


def _schema_plan(
    source: sqlite3.Connection, target: sqlite3.Connection
) -> tuple[dict[str, list[str]], list[str]]:
    tables = _tables(source)
    unknown = tables.keys() - _tables(target).keys()
    if unknown:
        raise ValueError(f"Unsupported source tables: {sorted(unknown)!r}")
    columns: dict[str, list[str]] = {}
    for table in tables:
        columns[table] = _columns(source, table)
        extra = set(columns[table]) - set(_columns(target, table))
        if extra:
            raise ValueError(f"Unsupported source columns in {table}: {sorted(extra)!r}")
        if any(name.lower() in {"rowid", "_rowid_", "oid"} for name in columns[table]):
            raise ValueError(f"Unsupported explicit rowid column in {table}")
        try:
            source.execute(f"SELECT rowid FROM {_quote(table)} LIMIT 0")
        except sqlite3.DatabaseError as error:
            raise ValueError(f"Unsupported WITHOUT ROWID source table: {table}") from error
        hidden = source.execute(f"PRAGMA table_xinfo({_quote(table)})").fetchall()
        if any(row[6] for row in hidden):
            raise ValueError(f"Unsupported generated source columns in {table}")
        source_indexes = _index_definitions(source, table)
        target_indexes = _index_definitions(target, table)
        source_constraints = Counter(
            definition for definition in source_indexes.values() if definition[1] != "c"
        )
        target_constraints = Counter(
            definition for definition in target_indexes.values() if definition[1] != "c"
        )
        if source_constraints != target_constraints:
            raise ValueError(f"Unsupported source index constraints in {table}")
        for name in source_indexes.keys() & target_indexes.keys():
            if source_indexes[name] != target_indexes[name]:
                raise ValueError(f"Unsupported source index definition: {name}")
    target_objects = {
        (str(kind), str(name)): sql
        for kind, name, sql in target.execute("SELECT type, name, sql FROM sqlite_master")
    }
    extra_indexes: list[str] = []
    for kind, name, sql in source.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE type IN ('view', 'trigger', 'index')"
    ):
        if kind == "trigger":
            raise ValueError(f"Unsupported source trigger: {name}")
        if sql is None:
            continue  # Automatic constraints were compared structurally above.
        if (kind, name) in target_objects:
            target_sql = target_objects[(kind, name)]
            if target_sql is None or _sql_tokens(str(sql)) != _sql_tokens(str(target_sql)):
                raise ValueError(f"Unsupported source {kind} definition: {name}")
        elif kind == "index":
            extra_indexes.append(str(sql))
        else:
            raise ValueError(f"Unsupported source {kind}: {name}")
    return columns, extra_indexes


def _logical_row(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    row: tuple[Any, ...],
    logical_payload: dict[str, Any] | None = None,
) -> bytes:
    values = list(row)
    if table in _PAYLOAD_TABLES:
        position = columns.index("payload_json") + 1
        values[position] = (
            logical_payload
            if logical_payload is not None
            else load_payload(connection, table, values[position])
        )
    # Tag SQL values so BLOBs cannot collide with JSON strings or objects.
    tagged = [
        ["blob", value.hex()] if isinstance(value, bytes) else [type(value).__name__, value]
        for value in values
    ]
    return (
        json.dumps(tagged, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _copy_table(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    columns: list[str],
    batch_size: int,
    *,
    commit_batches: bool = True,
) -> dict[str, Any]:
    projection = "rowid," + ",".join(map(_quote, columns))
    cursor = source.execute(f"SELECT {projection} FROM {_quote(table)} ORDER BY rowid")
    insert_columns = ["rowid", *columns]
    derive_opportunity = table == "paper_alert_market_checks" and "opportunity_id" not in columns
    if derive_opportunity:
        insert_columns.append("opportunity_id")
    insert = (
        f"INSERT INTO {_quote(table)} ({','.join(map(_quote, insert_columns))}) "
        f"VALUES ({','.join('?' for _ in insert_columns)})"
    )
    digest = hashlib.sha256()
    count = 0
    while rows := cursor.fetchmany(batch_size):
        output: list[list[Any]] = []
        for row in rows:
            values = list(row)
            logical = None
            if table in _PAYLOAD_TABLES:
                position = columns.index("payload_json") + 1
                logical = load_payload(source, table, values[position])
                values[position] = compact_payload(target, table, logical)
                if derive_opportunity:
                    opportunity = logical.get("opportunity")
                    values.append(
                        opportunity.get("opportunity_id") if isinstance(opportunity, dict) else None
                    )
            digest.update(_logical_row(source, table, columns, row, logical))
            output.append(values)
        target.executemany(insert, output)
        if commit_batches:
            target.commit()
        count += len(rows)
    return {"source_rows": count, "source_logical_sha256": digest.hexdigest()}


def _verify_table(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    columns: list[str],
    batch_size: int,
    expected: dict[str, Any],
) -> None:
    # Select only original rowids: the initializer adds migration markers and
    # manifest normalization can add shared code identities, but never replaces evidence.
    ids = source.execute(f"SELECT rowid FROM {_quote(table)} ORDER BY rowid")
    projection = "rowid," + ",".join(map(_quote, columns))
    digest = hashlib.sha256()
    count = 0
    while batch := ids.fetchmany(min(batch_size, 900)):
        rows = target.execute(
            f"SELECT {projection} FROM {_quote(table)} "
            f"WHERE rowid IN ({','.join('?' for _ in batch)}) ORDER BY rowid",
            [row[0] for row in batch],
        )
        for row in rows:
            digest.update(_logical_row(target, table, columns, row))
            count += 1
    if count != expected["source_rows"] or digest.hexdigest() != expected["source_logical_sha256"]:
        raise ValueError(f"Logical content verification failed for {table}")
    expected["destination_logical_sha256"] = digest.hexdigest()
    expected["verified_original_rows"] = count


def _publish_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Exclusive atomic publication: never replace another attempt's manifest.
        os.link(temporary, path)
    finally:
        temporary.unlink()


def assert_migration_source_unchanged(manifest: dict[str, Any]) -> None:
    """Check snapshot freshness immediately before activation with writers stopped.

    This is a point-in-time check, not a writer lock or proof of quiescence.
    Writers must remain stopped until activation has finished. The destination
    and its manifest binding must be verified separately by the activation caller.
    """
    if (
        manifest.get("status") != "success"
        or manifest.get("storage_schema_version") != STORAGE_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported or unsuccessful storage migration manifest")
    snapshot = manifest.get("source_snapshot")
    files = snapshot.get("files") if isinstance(snapshot, dict) else None
    if not isinstance(files, dict) or set(files) != {"", "-wal"}:
        raise ValueError("Migration manifest has invalid recorded source file states")
    expected: dict[str, tuple[int, int] | None] = {}
    for suffix, state in files.items():
        if state is None and suffix == "-wal":
            expected[suffix] = None
        elif (
            isinstance(state, (list, tuple))
            and len(state) == 2
            and all(type(value) is int and value >= 0 for value in state)
        ):
            expected[suffix] = (state[0], state[1])
        else:
            raise ValueError("Migration manifest has invalid recorded source file states")
    name = manifest.get("source")
    if not isinstance(name, str) or not name or not Path(name).is_absolute():
        raise ValueError("Migration manifest has invalid source path")
    source = Path(name)
    try:
        if not source.is_file():
            raise ValueError("Migration source no longer exists as a regular file")
        actual = _file_state(source)
    except OSError as error:
        raise ValueError("Migration source file states are unavailable") from error
    if actual != expected:
        raise ValueError(
            "Source changed since the migration snapshot; stop writers and migrate again"
        )


def migrate_storage(source: Path, destination: Path, *, batch_size: int = 1000) -> dict[str, Any]:
    """Copy an offline source without rewriting it, publishing verification last.

    Failures remove only the destination created by this call. An interrupted
    process may leave an incomplete destination; it is never overwritten on retry.
    The source read transaction spans copying, verification, and publication.
    Success verifies a consistent snapshot, not continued source quiescence:
    WAL writers are not excluded. Detected concurrent changes fail the copy;
    activation still requires stopped writers and a fresh source-state check.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    manifest_path = destination.with_name(destination.name + ".migration.json")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    artifacts = [
        destination,
        Path(str(destination) + "-wal"),
        Path(str(destination) + "-shm"),
        Path(str(destination) + "-journal"),
        manifest_path,
        temporary_manifest,
    ]
    if any(path.exists() or path.is_symlink() for path in artifacts):
        raise FileExistsError(f"Migration destination or sidecar already exists: {destination}")
    before = _file_state(source)
    owned = False
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
            original.execute("PRAGMA query_only=ON")
            original.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
            version = original.execute("PRAGMA data_version").fetchone()[0]
            original.execute("BEGIN")
            source_integrity = _integrity(original)
            source_foreign_keys = _foreign_keys(original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb"):
                owned = True
            SQLiteRepository(destination).initialize()
            with closing(sqlite3.connect(destination)) as target:
                target.execute("PRAGMA foreign_keys=OFF")
                target.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
                target.execute("PRAGMA synchronous=FULL")
                columns, extra_indexes = _schema_plan(original, target)
                markers = target.execute("SELECT * FROM schema_migrations").fetchall()
                indexes = [
                    (str(name), str(sql))
                    for name, sql in target.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
                    )
                    if "UNIQUE" not in str(sql).upper()
                ]
                target.execute("BEGIN IMMEDIATE")
                target.execute("DELETE FROM schema_migrations")
                for name, _ in indexes:
                    target.execute(f"DROP INDEX {_quote(name)}")
                target.commit()
                results = {
                    table: _copy_table(original, target, table, columns[table], batch_size)
                    for table in _copy_order(original, _tables(original))
                    if table != "schema_migrations"
                }
                # Only tiny marker metadata remains uncommitted during final
                # verification; incomplete populated copies remain unmarked.
                for _, sql in indexes:
                    target.execute(sql)
                for sql in extra_indexes:
                    target.execute(sql)
                target.execute("BEGIN IMMEDIATE")
                if "schema_migrations" in columns:
                    results["schema_migrations"] = _copy_table(
                        original,
                        target,
                        "schema_migrations",
                        columns["schema_migrations"],
                        batch_size,
                        commit_batches=False,
                    )
                added_markers = 0
                for marker in markers:
                    added_markers += target.execute(
                        "INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)", marker
                    ).rowcount
                for pragma in ("application_id", "user_version"):
                    value = int(original.execute(f"PRAGMA {pragma}").fetchone()[0])
                    target.execute(f"PRAGMA {pragma}={value}")
                for table, names in columns.items():
                    _verify_table(original, target, table, names, batch_size, results[table])
                for table in _tables(target):
                    result = results.setdefault(table, {"source_rows": 0})
                    count = int(
                        target.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
                    )
                    result["destination_rows"] = count
                    extra = count - result["source_rows"]
                    if table == "schema_migrations":
                        expected_extra = added_markers
                    elif table == "code_identities":
                        expected_extra = extra
                    else:
                        expected_extra = 0
                    if extra != expected_extra or extra < 0:
                        raise ValueError(f"Row count verification failed for {table}")
                    result["added_rows"] = extra
                destination_integrity = _integrity(target)
                destination_foreign_keys = _foreign_keys(target)
                introduced = Counter(map(tuple, destination_foreign_keys)) - Counter(
                    map(tuple, source_foreign_keys)
                )
                if introduced:
                    raise ValueError(
                        f"Migration introduces foreign-key violations: {dict(introduced)!r}"
                    )
                if _file_state(source) != before:
                    raise ValueError(
                        "Source files changed during migration; stop all writers and retry"
                    )
                target.commit()
                target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                target.execute("PRAGMA journal_mode=DELETE")
            manifest: dict[str, Any] = {
                "status": "success",
                "storage_schema_version": STORAGE_SCHEMA_VERSION,
                "completed_at": datetime.now(UTC).isoformat(),
                "source": str(source),
                "destination": str(destination),
                "manifest_path": str(manifest_path),
                "batch_size": batch_size,
                "source_snapshot": {
                    "files": {
                        suffix: list(state) if state is not None else None
                        for suffix, state in before.items()
                    },
                    "read_transaction": True,
                    "consistency": "verified-read-snapshot",
                    "writer_exclusion": False,
                },
                "activation": {
                    "requires_stopped_writers": True,
                    "requires_source_freshness_check": True,
                    "source_freshness_guard": "assert_migration_source_unchanged",
                    "copy_success_proves_source_quiescence": False,
                },
                "before_bytes": sum(state[0] for state in before.values() if state is not None),
                "after_bytes": destination.stat().st_size,
                "tables": results,
                "integrity": {"source": source_integrity, "destination": destination_integrity},
                "foreign_keys": {
                    "source": source_foreign_keys,
                    "destination": destination_foreign_keys,
                    "introduced": [],
                    "status": "preexisting-violations" if source_foreign_keys else "ok",
                },
            }
            assert_migration_source_unchanged(manifest)
            _publish_manifest(manifest_path, manifest)
            original.rollback()
            if original.execute("PRAGMA data_version").fetchone()[0] != version:
                raise ValueError("Source changed during migration; stop all writers and retry")
            assert_migration_source_unchanged(manifest)
        return manifest
    except BaseException:
        if owned:
            for path in artifacts:
                path.unlink(missing_ok=True)
        raise
