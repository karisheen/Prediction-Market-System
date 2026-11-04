import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_storage import build_evaluation

from prediction_market_system import storage_migration
from prediction_market_system.domain import Opportunity, ProbabilityForecast
from prediction_market_system.evidence import EVIDENCE_FORWARD_SHADOW, EVIDENCE_HISTORICAL
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.storage_migration import migrate_storage
from prediction_market_system.storage_payloads import (
    STORAGE_SCHEMA_VERSION,
    decode_payload,
    encode_payload,
    load_payload,
)

PAYLOAD_TABLES = (
    "input_objects",
    "run_manifests",
    "forecast_ledger",
    "forecasts",
    "opportunities",
    "paper_alert_market_checks",
)


def _history(path: Path) -> list[str]:
    repository = SQLiteRepository(path)
    repository.initialize()
    ids = []
    for kind in (EVIDENCE_FORWARD_SHADOW, EVIDENCE_HISTORICAL):
        forecast, opportunity = build_evaluation()
        assert isinstance(forecast, ProbabilityForecast)
        assert isinstance(opportunity, Opportunity)
        repository.save_evaluation(forecast, opportunity, ledger_kind=kind)
        ids.append(str(forecast.forecast_id))
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "INSERT INTO paper_alert_market_checks "
                "(cycle_id, market_id, series_ticker, event_ticker, observed_at, status, "
                "reason, payload_json, opportunity_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    opportunity.market.market_id,
                    "BTC",
                    "event",
                    forecast.generated_at.isoformat(),
                    opportunity.state.value,
                    None,
                    encode_payload({"opportunity": opportunity.model_dump(mode="json")}),
                    str(opportunity.opportunity_id),
                ),
            )
    with closing(sqlite3.connect(path)) as connection, connection:
        known = (forecast.generated_at + timedelta(days=31)).isoformat()
        connection.execute(
            "INSERT INTO kalshi_resolutions VALUES (?, ?, 'yes', '1', ?, '110')",
            (opportunity.market.market_id, known, known),
        )
        connection.execute(
            "INSERT INTO kalshi_rule_snapshots VALUES (?, ?, ?, ?)",
            (opportunity.market.market_id, known, "Original benchmark", "Original correction"),
        )
        connection.execute(
            "INSERT INTO schema_migrations VALUES ('original-migration', '2000-01-01')"
        )
        connection.execute("PRAGMA application_id=1234")
        connection.execute("PRAGMA user_version=17")
    _checkpoint(path)
    return ids


def _checkpoint(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")


def _make_text_source(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        for table in PAYLOAD_TABLES:
            rows = connection.execute(f"SELECT rowid, payload_json FROM {table}").fetchall()
            for rowid, payload in rows:
                logical = load_payload(connection, table, payload)
                connection.execute(
                    f"UPDATE {table} SET payload_json=? WHERE rowid=?",
                    (json.dumps(logical), rowid),
                )
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_name=?", (STORAGE_SCHEMA_VERSION,)
        )
        connection.execute("DROP TABLE code_identities")
        connection.execute("DROP INDEX idx_paper_checks_opportunity")
        connection.execute("ALTER TABLE paper_alert_market_checks DROP COLUMN opportunity_id")
    _checkpoint(path)


def _logical_history(path: Path) -> dict[str, list[Any]]:
    with closing(sqlite3.connect(path)) as connection:
        return {
            table: [
                load_payload(connection, table, row[0])
                for row in connection.execute(f"SELECT payload_json FROM {table} ORDER BY rowid")
            ]
            for table in PAYLOAD_TABLES
        }


@pytest.mark.parametrize("legacy_text", [False, True])
def test_migration_preserves_real_forecasts_outcomes_hashes_and_history(
    tmp_path: Path, legacy_text: bool
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    forecast_ids = _history(source)
    if legacy_text:
        _make_text_source(source)
    logical_before = _logical_history(source)
    source_bytes = source.read_bytes()
    with closing(sqlite3.connect(source)) as connection:
        manifest_hashes = connection.execute(
            "SELECT run_id, kind, manifest_sha256 FROM run_manifests ORDER BY run_id"
        ).fetchall()
        outcomes = connection.execute("SELECT * FROM kalshi_resolutions").fetchall()
    result = migrate_storage(source, destination, batch_size=1)
    assert source.read_bytes() == source_bytes
    assert _logical_history(destination) == logical_before
    assert result["status"] == "success"
    assert result["integrity"] == {"source": ["ok"], "destination": ["ok"]}
    assert result["foreign_keys"]["status"] == "ok"
    assert json.loads(Path(result["manifest_path"]).read_text())["status"] == "success"
    with closing(sqlite3.connect(destination)) as connection:
        assert (
            connection.execute(
                "SELECT run_id, kind, manifest_sha256 FROM run_manifests ORDER BY run_id"
            ).fetchall()
            == manifest_hashes
        )
        assert connection.execute("SELECT * FROM kalshi_resolutions").fetchall() == outcomes
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 1234
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 17
        assert (
            connection.execute(
                "SELECT applied_at FROM schema_migrations WHERE migration_name='original-migration'"
            ).fetchone()[0]
            == "2000-01-01"
        )
        assert connection.execute("SELECT COUNT(*) FROM code_identities").fetchone()[0] == 1
        for table in PAYLOAD_TABLES:
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE typeof(payload_json) != 'blob'"
                ).fetchone()[0]
                == 0
            )
            evidence = result["tables"][table]
            assert evidence["source_logical_sha256"] == evidence["destination_logical_sha256"]
            assert evidence["source_rows"] == evidence["destination_rows"]
        check = connection.execute(
            "SELECT opportunity_id, payload_json FROM paper_alert_market_checks LIMIT 1"
        ).fetchone()
        assert (
            check[0]
            == load_payload(connection, "paper_alert_market_checks", check[1])["opportunity"][
                "opportunity_id"
            ]
        )
        assert decode_payload(check[1])["kind"] == "opportunity"
    repository = SQLiteRepository(destination)
    repository.initialize()
    assert len(repository.opportunity_history()) == 2
    for forecast_id in forecast_ids:
        assert repository.explain_forecast(forecast_id)["manifest"]["kind"] in {
            EVIDENCE_FORWARD_SHADOW,
            EVIDENCE_HISTORICAL,
        }


def test_legacy_and_differing_orphan_history_is_not_relabelled(tmp_path: Path) -> None:
    source, destination = tmp_path / "legacy.db", tmp_path / "compact.db"
    _history(source)
    _make_text_source(source)
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("DELETE FROM forecast_ledger")
        connection.execute("DELETE FROM run_manifests")
        connection.execute("DELETE FROM input_objects")
        connection.execute(
            "UPDATE forecasts SET payload_json=json_set(payload_json, '$.historical_note', ?) "
            "WHERE rowid=1",
            ("Preserve differing historical forecast",),
        )
    before = _logical_history(source)
    result = migrate_storage(source, destination, batch_size=1)
    assert _logical_history(destination) == before
    assert result["tables"]["forecast_ledger"]["destination_rows"] == 0
    assert result["tables"]["run_manifests"]["destination_rows"] == 0
    with closing(sqlite3.connect(destination)) as connection:
        assert all(
            decode_payload(row[0])["kind"] == "inline"
            for row in connection.execute("SELECT payload_json FROM forecasts")
        )


def test_differing_payload_with_existing_ledger_is_preserved(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    _make_text_source(source)
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute(
            "UPDATE forecasts SET payload_json="
            "json_set(payload_json, '$.historical_note', 'different')"
        )
    before = _logical_history(source)
    migrate_storage(source, destination)
    assert _logical_history(destination) == before
    with closing(sqlite3.connect(destination)) as connection:
        assert (
            decode_payload(connection.execute("SELECT payload_json FROM forecasts").fetchone()[0])[
                "kind"
            ]
            == "inline"
        )


def test_corrupt_source_fails_without_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "corrupt.db", tmp_path / "compact.db"
    source.write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        migrate_storage(source, destination)
    assert source.read_bytes() == b"not a sqlite database"
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".migration.json").exists()


def test_malformed_payload_failure_cleans_up_and_explicit_retry_succeeds(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    _make_text_source(source)
    with closing(sqlite3.connect(source)) as connection, connection:
        old = connection.execute("SELECT payload_json FROM forecasts WHERE rowid=1").fetchone()[0]
        connection.execute("UPDATE forecasts SET payload_json='not-json' WHERE rowid=1")
    before = source.read_bytes()
    with pytest.raises(ValueError):
        migrate_storage(source, destination, batch_size=1)
    assert source.read_bytes() == before
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("UPDATE forecasts SET payload_json=? WHERE rowid=1", (old,))
    assert migrate_storage(source, destination, batch_size=1)["status"] == "success"


def test_existing_destination_and_interrupted_artifacts_are_never_overwritten(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    destination.write_bytes(b"interrupted or unrelated file")
    with pytest.raises(FileExistsError):
        migrate_storage(source, destination)
    assert destination.read_bytes() == b"interrupted or unrelated file"
    destination.unlink()
    sidecar = Path(str(destination) + ".migration.json")
    sidecar.write_text('{"status":"old"}')
    with pytest.raises(FileExistsError):
        migrate_storage(source, destination)
    assert sidecar.read_text() == '{"status":"old"}'
    assert not destination.exists()
    with pytest.raises(FileExistsError):
        migrate_storage(source, source)


def test_unknown_tables_are_rejected_not_discarded(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE irreplaceable_research (data BLOB)")
        connection.execute("INSERT INTO irreplaceable_research VALUES (x'0012')")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="Unsupported source tables"):
        migrate_storage(source, destination)
    assert source.read_bytes() == before
    assert not destination.exists()


def test_existing_foreign_key_violation_is_reported_and_not_repaired(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute(
            "INSERT INTO opportunities VALUES "
            "('orphan', 'missing', 'old-market', 'WATCH', 'old', ?)",
            (encode_payload({"opportunity_id": "orphan", "forecast": {"forecast_id": "missing"}}),),
        )
    result = migrate_storage(source, destination)
    assert result["foreign_keys"]["status"] == "preexisting-violations"
    assert result["foreign_keys"]["source"] == result["foreign_keys"]["destination"]
    assert result["foreign_keys"]["introduced"] == []
    assert result["tables"]["opportunities"]["source_rows"] == 3


def test_new_schema_must_not_introduce_foreign_key_violations(tmp_path: Path) -> None:
    source, destination = tmp_path / "legacy.db", tmp_path / "compact.db"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute(
            "CREATE TABLE opportunities (opportunity_id TEXT PRIMARY KEY, forecast_id TEXT, "
            "market_id TEXT, state TEXT, created_at TEXT, payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO opportunities VALUES ('orphan', 'missing', 'm', 'WATCH', 'old', '{}')"
        )
    with pytest.raises(ValueError, match="introduces foreign-key violations"):
        migrate_storage(source, destination)
    assert not destination.exists()


def test_logical_digest_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    original = storage_migration.compact_payload

    def damaged(connection: sqlite3.Connection, table: str, value: dict[str, Any]) -> bytes:
        if table == "forecasts":
            return encode_payload({**value, "lost_evidence": True})
        return original(connection, table, value)

    monkeypatch.setattr(storage_migration, "compact_payload", damaged)
    with pytest.raises(ValueError, match="Logical content verification failed"):
        migrate_storage(source, destination, batch_size=1)
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_invalid_batch_size_has_no_output(tmp_path: Path, batch_size: int) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    source.touch()
    with pytest.raises(ValueError, match="positive integer"):
        migrate_storage(source, destination, batch_size=batch_size)
    assert not destination.exists()


def test_minimal_legacy_forecast_stays_legacy_without_ledger(tmp_path: Path) -> None:
    source, destination = tmp_path / "legacy.db", tmp_path / "compact.db"
    payload = {"forecast_id": "legacy-id", "probability_yes": 0.25, "original": ["history"]}
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute(
            "CREATE TABLE forecasts (forecast_id TEXT PRIMARY KEY, market_id TEXT NOT NULL, "
            "generated_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO forecasts VALUES ('legacy-id', 'old-market', 'old', ?)",
            (json.dumps(payload),),
        )
    result = migrate_storage(source, destination)
    assert result["tables"]["forecasts"]["source_rows"] == 1
    assert result["tables"]["forecast_ledger"]["destination_rows"] == 0
    with closing(sqlite3.connect(destination)) as connection:
        stored = connection.execute("SELECT payload_json FROM forecasts").fetchone()[0]
        assert load_payload(connection, "forecasts", stored) == payload
        assert decode_payload(stored)["kind"] == "inline"


def test_concurrent_source_writer_prevents_success_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    original = storage_migration._verify_table
    changed = False

    def verify_with_writer(
        source_connection: sqlite3.Connection,
        target: sqlite3.Connection,
        table: str,
        columns: list[str],
        batch_size: int,
        expected: dict[str, Any],
    ) -> None:
        nonlocal changed
        if not changed:
            with closing(sqlite3.connect(source)) as writer, writer:
                writer.execute("INSERT INTO schema_migrations VALUES ('concurrent-writer', 'now')")
            changed = True
        original(source_connection, target, table, columns, batch_size, expected)

    monkeypatch.setattr(storage_migration, "_verify_table", verify_with_writer)
    with pytest.raises(ValueError, match="Source .*changed during migration"):
        migrate_storage(source, destination, batch_size=1)
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()


def test_manifest_publication_failure_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)

    def fail_publication(path: Path, manifest: dict[str, Any]) -> None:
        raise OSError("manifest publication unavailable")

    monkeypatch.setattr(storage_migration, "_publish_manifest", fail_publication)
    with pytest.raises(OSError, match="publication unavailable"):
        migrate_storage(source, destination)
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()


def test_batches_commit_without_marking_partial_storage_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    original = storage_migration.compact_payload
    copied_forecasts = 0

    def interrupt_after_batch(
        connection: sqlite3.Connection, table: str, value: dict[str, Any]
    ) -> bytes:
        nonlocal copied_forecasts
        if table == "forecasts":
            copied_forecasts += 1
            if copied_forecasts == 2:
                with closing(sqlite3.connect(destination)) as observer:
                    assert observer.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
                    assert (
                        observer.execute(
                            "SELECT COUNT(*) FROM schema_migrations WHERE migration_name=?",
                            (STORAGE_SCHEMA_VERSION,),
                        ).fetchone()[0]
                        == 0
                    )
                raise RuntimeError("interrupted after committed batch")
        return original(connection, table, value)

    monkeypatch.setattr(storage_migration, "compact_payload", interrupt_after_batch)
    with pytest.raises(RuntimeError, match="committed batch"):
        migrate_storage(source, destination, batch_size=1)
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()


def test_read_only_wal_source_without_pending_frames_is_supported(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    before = source.read_bytes()
    source.chmod(0o444)
    try:
        result = migrate_storage(source, destination, batch_size=1)
    finally:
        source.chmod(0o644)
    assert result["status"] == "success"
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "CREATE VIEW venue_revision_history AS "
            "SELECT kind, source_key, series_ticker, available_at, payload_json "
            "FROM venue_source_revisions WHERE kind='omitted'",
            "Unsupported source view definition",
        ),
        (
            "CREATE INDEX idx_forecasts_market_generated ON forecasts (generated_at, market_id)",
            "Unsupported source index definition",
        ),
        (
            "CREATE INDEX idx_forecasts_market_generated ON forecasts (market_id, generated_at) "
            "WHERE market_id != 'excluded'",
            "Unsupported source index definition",
        ),
    ],
)
def test_altered_known_schema_objects_are_rejected(
    tmp_path: Path, statement: str, message: str
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    with closing(sqlite3.connect(source)) as connection, connection:
        if statement.startswith("CREATE VIEW"):
            connection.execute("DROP VIEW venue_revision_history")
        else:
            connection.execute("DROP INDEX idx_forecasts_market_generated")
        connection.execute(statement)
    before = source.read_bytes()
    with pytest.raises(ValueError, match=message):
        migrate_storage(source, destination)
    assert source.read_bytes() == before
    assert not destination.exists()


@pytest.mark.parametrize("constraint", ["UNIQUE", "COLLATE NOCASE UNIQUE"])
def test_unknown_automatic_unique_constraints_are_rejected(tmp_path: Path, constraint: str) -> None:
    source, destination = tmp_path / "legacy.db", tmp_path / "compact.db"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute(
            "CREATE TABLE forecasts (forecast_id TEXT PRIMARY KEY, "
            f"market_id TEXT NOT NULL {constraint}, generated_at TEXT NOT NULL, "
            "payload_json TEXT NOT NULL)"
        )
    with pytest.raises(ValueError, match="Unsupported source index constraints"):
        migrate_storage(source, destination)
    assert not destination.exists()


def test_schema_whitespace_and_extra_explicit_indexes_are_preserved(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    extra_index = (
        "CREATE UNIQUE INDEX additional_forecast_order "
        "ON forecasts (generated_at DESC, forecast_id COLLATE NOCASE)"
    )
    with closing(sqlite3.connect(source)) as connection, connection:
        for kind, name in (
            ("view", "venue_revision_history"),
            ("index", "idx_forecasts_market_generated"),
        ):
            sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (name,)
            ).fetchone()[0]
            connection.execute(f"DROP {kind} {name}")
            connection.execute(sql.replace(" ", " \n "))
        connection.execute(extra_index)
    migrate_storage(source, destination)
    with closing(sqlite3.connect(destination)) as connection:
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='additional_forecast_order'"
            ).fetchone()[0]
            == extra_index
        )


def test_wal_commit_at_manifest_publication_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    publish = storage_migration._publish_manifest

    def publish_with_writer(path: Path, manifest: dict[str, Any]) -> None:
        publish(path, manifest)
        with closing(sqlite3.connect(source)) as writer, writer:
            writer.execute("INSERT INTO schema_migrations VALUES ('publication-writer', 'now')")

    monkeypatch.setattr(storage_migration, "_publish_manifest", publish_with_writer)
    with pytest.raises(ValueError, match="Source .*changed"):
        migrate_storage(source, destination)
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()


def test_activation_guard_rejects_source_changes_after_success(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    manifest = migrate_storage(source, destination)
    recorded = json.loads(Path(manifest["manifest_path"]).read_text())
    storage_migration.assert_migration_source_unchanged(recorded)
    assert recorded["source_snapshot"]["writer_exclusion"] is False
    assert recorded["activation"]["requires_stopped_writers"] is True
    assert recorded["activation"]["copy_success_proves_source_quiescence"] is False
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("INSERT INTO schema_migrations VALUES ('post-copy-writer', 'now')")
    with pytest.raises(ValueError, match="Source changed"):
        storage_migration.assert_migration_source_unchanged(recorded)


@pytest.mark.parametrize(
    "replacement",
    [
        {"status": "failed"},
        {"storage_schema_version": "unknown"},
        {"source_snapshot": {}},
        {"source_snapshot": {"files": {"": None, "-wal": None}}},
        {"source_snapshot": {"files": {"": [True, 1], "-wal": None}}},
        {"source": "relative.db"},
    ],
)
def test_activation_guard_rejects_malformed_manifest(
    tmp_path: Path, replacement: dict[str, Any]
) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    manifest = migrate_storage(source, destination)
    with pytest.raises(ValueError):
        storage_migration.assert_migration_source_unchanged({**manifest, **replacement})


def test_activation_guard_requires_existing_source(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.db", tmp_path / "compact.db"
    _history(source)
    manifest = migrate_storage(source, destination)
    source.unlink()
    with pytest.raises(ValueError, match="no longer exists"):
        storage_migration.assert_migration_source_unchanged(manifest)
