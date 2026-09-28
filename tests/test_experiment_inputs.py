import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest
from test_research import research_candles, research_context

from prediction_market_system import experiment_inputs
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSnapshot,
    Opportunity,
    TerminalRangeContract,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import (
    EVIDENCE_FORWARD_SHADOW,
    EVIDENCE_HISTORICAL,
    EVIDENCE_MANUAL_RESEARCH,
    content_id,
)
from prediction_market_system.experiment_inputs import load_fixture, load_local_database
from prediction_market_system.research import ResearchContext
from prediction_market_system.storage import SQLiteRepository

FIXTURE = Path(__file__).resolve().parents[1] / "experiments" / "phase1" / "fixture-v1.json"
PROTOCOL: dict[str, Any] = {
    "protocol_id": "sha256:test-protocol",
    "series": "KXBTC",
    "symbol": "BTC",
    "report_as_of": "2026-09-26T00:00:00+00:00",
    "archive_start": "2026-07-01T00:00:00+00:00",
    "archive_end": "2026-09-26T00:00:00+00:00",
    "structural_weight": 0.5,
    "maximum_input_age_seconds": 120,
    "decision_costs": {
        "fee_coefficient": 0.07,
        "fee_type": "quadratic",
        "slippage_bps": 25,
        "latency_seconds": 30,
        "resolution_haircut": 0.01,
        "uncertainty_margin": 0.05,
        "minimum_ask_size": 10,
        "fractional_kelly": 0.25,
        "paper_bankroll": 10000,
        "max_bankroll_fraction": 0.02,
        "max_event_bankroll_fraction": 0.02,
        "min_conservative_edge": 0.03,
        "minimum_seconds_to_expiry": 300,
    },
    "cost_scenarios": [
        {"name": "base", "latency_seconds": 30},
        {"name": "adverse", "latency_seconds": 60},
        {"name": "severe", "latency_seconds": 120},
    ],
}
EXCLUDED_KEYS = {"observation_id", "event_id", "market_id", "observed_at", "exclusion_reason"}
AS_OF = datetime(2026, 8, 3, 12, tzinfo=UTC)
DECISION = AS_OF + timedelta(seconds=30)
END = AS_OF + timedelta(hours=1)
TICKER = "KXBTC-26AUG0313-B124"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def test_fixture_cases_land_where_declared_and_replay_through_the_engine() -> None:
    observations, metadata = load_fixture(FIXTURE, PROTOCOL)
    rows = {row["observation_id"]: row for row in observations}
    for case in metadata["fixture"]["cases"]:
        row = rows.get(case["observation_id"])
        if case["expected"] == "outside-window":
            assert row is None, case["case_id"]
            continue
        assert row is not None, case["case_id"]
        if case["expected"] == "adapter-excluded":
            assert set(row) == EXCLUDED_KEYS, case["case_id"]
            continue
        assert "exclusion_reason" not in row, case["case_id"]
        if case["expected"] == "scored":
            assert row["resolution"] is not None, case["case_id"]
        if case["expected"] in {"unresolved", "resolution-withheld"}:
            assert row["resolution"] is None, case["case_id"]
        expected_reason = (
            "outcome-after-cutoff" if case["expected"] == "resolution-withheld" else None
        )
        assert row["resolution_unavailable_reason"] == expected_reason, case["case_id"]
        inputs = row["inputs"]
        assert row["input_id"] == content_id(inputs)
        assert row["run_manifest_sha256"] == content_id(row["run_manifest"])
        assert row["run_manifest"]["kind"] == "synthetic-fixture"
        assert "input_manifest" not in row["opportunity"]["forecast"]
        replayed, _ = CryptoThresholdEngine(
            EngineConfig.model_validate(inputs["engine_config"])
        ).evaluate(
            MarketSnapshot.model_validate(inputs["market"]),
            CryptoSnapshot.model_validate(inputs["crypto"]),
            TerminalRangeContract.model_validate(inputs["contract"]),
        )
        recorded = row["opportunity"]["forecast"]
        assert replayed.structural_probability_yes == recorded["structural_probability_yes"]
        assert replayed.probability_yes == recorded["probability_yes"]
        assert replayed.market_probability_yes == recorded["market_probability_yes"]

    scored = [
        rows[case["observation_id"]]
        for case in metadata["fixture"]["cases"]
        if case["expected"] == "scored"
    ]
    events = {row["event_id"] for row in scored}
    dates = {_utc(row["inputs"]["market"]["observation_end_at"]).date() for row in scored}
    assert len(events) >= 6 and len(dates) >= 5 and len(scored) > len(events)
    assert len({(row["event_id"], row["observed_at"]) for row in scored}) < len(scored)
    assert len({row["market_id"] for row in scored}) < len(scored)
    failures = set(metadata["fixture"]["failure_examples"])
    assert len(failures) >= 2 and failures <= {row["observation_id"] for row in scored}


def test_fixture_identities_are_deterministic_and_content_addressed() -> None:
    first, first_metadata = load_fixture(FIXTURE, PROTOCOL)
    second, second_metadata = load_fixture(FIXTURE, PROTOCOL)
    assert first == second
    assert first_metadata["dataset_sha256"] == second_metadata["dataset_sha256"]
    fixture = json.loads(FIXTURE.read_text())
    namespace = UUID(fixture["uuid_namespace"])
    assert first_metadata["source"]["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    for case in first_metadata["fixture"]["cases"]:
        assert case["observation_id"] == str(uuid5(namespace, f"forecast:{case['case_id']}"))
    for row in first:
        if "exclusion_reason" not in row:
            assert row["opportunity"]["forecast"]["forecast_id"] == row["observation_id"]
            assert row["run_id"] == row["observation_id"]


@pytest.mark.parametrize(
    ("report_as_of", "expected"),
    [
        ("2026-09-26T00:00:00+00:00", None),
        # Settled but not yet recorded locally: settlement alone is not availability.
        ("2026-09-26T01:03:00+00:00", None),
        ("2026-09-26T01:04:00+00:00", "yes"),
    ],
)
def test_fixture_outcome_is_withheld_until_recorded_by_report_cutoff(
    report_as_of: str, expected: str | None
) -> None:
    observations, metadata = load_fixture(FIXTURE, {**PROTOCOL, "report_as_of": report_as_of})
    case = next(c for c in metadata["fixture"]["cases"] if c["case_id"] == "F1-A-T1")
    row = next(row for row in observations if row["observation_id"] == case["observation_id"])
    assert (row["resolution"] or {}).get("result") == expected
    assert row["resolution_unavailable_reason"] == (
        "outcome-after-cutoff" if expected is None else None
    )
    if expected is None:
        assert "2026-09-26T01:04:00" not in json.dumps(row)
    withheld = metadata["inventory"].get("resolutions_withheld_after_report_as_of", 0)
    assert withheld == (0 if expected else 1)


def test_fixture_execution_snapshots_stay_inside_the_fill_horizon() -> None:
    observations, metadata = load_fixture(FIXTURE, PROTOCOL)
    horizon = timedelta(seconds=120 + PROTOCOL["maximum_input_age_seconds"])
    attached = 0
    for row in observations:
        for snapshot in row.get("execution", []):
            observed = _utc(snapshot["observed_at"])
            assert _utc(row["observed_at"]) < observed <= _utc(row["observed_at"]) + horizon
            assert "result" not in snapshot and "settlement_ts" not in snapshot
            attached += 1
    assert attached == metadata["inventory"]["execution_snapshots_attached"] > 0
    assert metadata["inventory"]["execution_snapshots_outside_horizon"] == 1


def _record_database(
    path: Path,
    *,
    kinds: tuple[str, ...] = (EVIDENCE_FORWARD_SHADOW,),
    history: bool = True,
) -> tuple[list[Opportunity], ResearchContext]:
    repository = SQLiteRepository(path)
    repository.initialize()
    context = research_context(AS_OF)
    if history:
        repository.save_research_data(spot_candles=research_candles(AS_OF))
    repository.save_research_context(context)
    engine = CryptoThresholdEngine()
    contract = TerminalRangeContract(
        lower_bound=120.0, upper_bound=128.0, settlement_window_seconds=60
    )
    market = MarketSnapshot(
        market_id=TICKER,
        question="Bitcoin range at 13:00 UTC?",
        venue="Kalshi",
        observed_at=DECISION,
        expires_at=END,
        observation_end_at=END,
        observation_start_at=END - timedelta(seconds=60),
        yes_bid=0.40,
        yes_ask=0.42,
        no_bid=0.58,
        no_ask=0.60,
        yes_ask_size=50,
        no_ask_size=50,
        resolution_rule="Average of the sixty seconds before 1 PM UTC on August 3, 2026.",
        series_id="KXBTC",
        event_id="KXBTC-26AUG0313",
    )
    crypto = context.to_crypto_snapshot(strike_price=engine.reference_price(contract))
    opportunities = []
    for kind in kinds:
        forecast, opportunity = engine.evaluate(market, crypto, contract)
        repository.save_evaluation(forecast, opportunity, ledger_kind=kind)
        opportunities.append(opportunity)
    return opportunities, context


def _execute(path: Path, sql: str, parameters: tuple[Any, ...] = ()) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(sql, parameters)


def _insert_resolution(
    path: Path, observed_at: datetime, result: str = "yes", settled_at: datetime | None = None
) -> None:
    settled = settled_at or END + timedelta(minutes=2)
    _execute(
        path,
        "INSERT INTO kalshi_resolutions VALUES (?, ?, ?, '1', ?, '')",
        (TICKER, observed_at.isoformat(), result, settled.isoformat()),
    )


def _insert_snapshot(path: Path, observed_at: datetime) -> None:
    payload = {
        "ticker": TICKER,
        "yes_bid_dollars": "0.4100",
        "yes_ask_dollars": "0.4300",
        "no_bid_dollars": "0.5700",
        "no_ask_dollars": "0.5900",
        "yes_bid_size_fp": "12.00",
        "yes_ask_size_fp": "7.00",
        "result": "yes",
        "settlement_value_dollars": "1.0000",
    }
    _execute(
        path,
        "INSERT INTO kalshi_market_snapshots VALUES "
        "(?, ?, 'KXBTC', 'KXBTC-26AUG0313', 'active', NULL, ?, '', ?)",
        (TICKER, observed_at.isoformat(), END.isoformat(), json.dumps(payload)),
    )


def test_database_emits_only_forward_rows_and_never_reclassifies_other_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.db"
    kinds = (
        EVIDENCE_FORWARD_SHADOW,
        EVIDENCE_HISTORICAL,
        EVIDENCE_MANUAL_RESEARCH,
        EVIDENCE_FORWARD_SHADOW,
    )
    opportunities, context = _record_database(path, kinds=kinds)
    ambiguous = str(opportunities[3].forecast.forecast_id)
    _execute(path, "DELETE FROM run_manifests WHERE run_id = ?", (ambiguous,))
    _insert_resolution(path, END + timedelta(minutes=4))
    for offset in (-10, 30, 200, 300):
        _insert_snapshot(path, DECISION + timedelta(seconds=offset))

    observations, metadata = load_local_database(path, PROTOCOL)

    assert [row["observation_id"] for row in observations] == [
        str(opportunities[0].forecast.forecast_id)
    ]
    row = observations[0]
    assert row["event_id"] == "KXBTC-26AUG0313"
    assert row["research_context"] == context.model_dump(mode="json")
    assert row["resolution"] == {
        "result": "yes",
        "observed_at": (END + timedelta(minutes=4)).isoformat(),
        "settlement_ts": (END + timedelta(minutes=2)).isoformat(),
    }
    assert row["resolution_unavailable_reason"] is None
    assert [_utc(snapshot["observed_at"]) - DECISION for snapshot in row["execution"]] == [
        timedelta(seconds=30),
        timedelta(seconds=200),
    ]
    assert row["execution"][0]["yes_ask_size"] == 7.0
    assert row["execution"][0]["no_ask_size"] == 12.0
    assert "result" not in row["execution"][0]
    assert metadata["source_kind"] == "retrospective-local"
    assert metadata["inventory"]["ledger_rows_by_evidence_class"] == {
        "ambiguous-legacy": 1,
        "forward-evaluation": 1,
        "historical-evaluation": 1,
        "manual-evaluation": 1,
    }
    counted = {(item["category"], item["reason"]): item["count"] for item in metadata["exclusions"]}
    assert (
        counted[("legacy", "ledger row lacks a classifiable run manifest (ambiguous-legacy)")] == 1
    )
    assert sum(count for (category, _), count in counted.items() if category == "population") == 2
    assert metadata["consumed"]["research_context_ids"] == [context.content_id()]


def test_database_is_read_only_and_missing_schema_is_an_empty_inventoried_population(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE forecasts (forecast_id TEXT PRIMARY KEY, market_id TEXT NOT NULL, "
            "generated_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO forecasts VALUES (?, 'KXBTC-LEGACY', '2026-08-01T00:00:00+00:00', '{}')",
            [("legacy-1",), ("legacy-2",)],
        )
    before = path.read_bytes()
    path.chmod(0o444)
    try:
        observations, metadata = load_local_database(path, PROTOCOL)
    finally:
        path.chmod(0o644)

    assert observations == []
    assert metadata["schema"]["status"] == "missing-required"
    assert {item["table"] for item in metadata["schema"]["missing_required"]} == {
        "forecast_ledger",
        "run_manifests",
        "input_objects",
    }
    assert metadata["inventory"]["legacy_forecasts_without_ledger"] == 2
    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master").fetchall()
    assert tables == [("forecasts",), ("sqlite_autoindex_forecasts_1",)]

    missing = tmp_path / "absent.db"
    with pytest.raises(FileNotFoundError):
        load_local_database(missing, PROTOCOL)
    assert not missing.exists()


def test_database_hash_and_extraction_share_a_frozen_committed_wal_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "live.db"
    (opportunity,), context = _record_database(path)
    frozen_paths: list[Path] = []
    frozen_bytes: list[bytes] = []
    original_hash = experiment_inputs._sha256_file
    resolution_observed = END + timedelta(minutes=4)
    resolution_settled = END + timedelta(minutes=2)

    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        main_before = path.read_bytes()
        writer.execute(
            "INSERT INTO kalshi_resolutions VALUES (?, ?, 'yes', '1', ?, '')",
            (TICKER, resolution_observed.isoformat(), resolution_settled.isoformat()),
        )
        writer.commit()
        assert path.read_bytes() == main_before
        assert path.with_name(path.name + "-wal").stat().st_size > 0

        def hash_then_commit_new_source_state(snapshot: Path) -> str:
            digest = original_hash(snapshot)
            frozen_paths.append(snapshot)
            frozen_bytes.append(snapshot.read_bytes())
            # A real WAL commit between hashing and extraction must not change either
            # the outcome or the point-in-time research reconstructed by the adapter.
            with writer:
                writer.execute("UPDATE kalshi_resolutions SET result = 'no'")
                writer.execute("DELETE FROM crypto_spot_candles")
            return digest

        with monkeypatch.context() as patch:
            patch.setattr(experiment_inputs, "_sha256_file", hash_then_commit_new_source_state)
            observations, metadata = load_local_database(path, PROTOCOL)
        assert writer.execute("SELECT result FROM kalshi_resolutions").fetchone() == ("no",)
        assert writer.execute("SELECT COUNT(*) FROM crypto_spot_candles").fetchone() == (0,)

    (row,) = observations
    assert row["observation_id"] == str(opportunity.forecast.forecast_id)
    assert row["resolution"] == {
        "result": "yes",
        "observed_at": resolution_observed.isoformat(),
        "settlement_ts": resolution_settled.isoformat(),
    }
    assert row["resolution_unavailable_reason"] is None
    assert row["research_context"] == context.model_dump(mode="json")
    (snapshot_path,) = frozen_paths
    (snapshot_bytes,) = frozen_bytes
    assert snapshot_path != path
    assert not snapshot_path.parent.exists()
    assert metadata["source"]["sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
    assert metadata["source"]["bytes"] == len(snapshot_bytes)
    assert metadata["source"]["wal_sha256"] is None
    assert str(snapshot_path.parent) not in json.dumps(metadata)

    # Replaying the exact hashed bytes reproduces the extracted rows. The live source
    # has a new outcome, while immutable research revisions still reproduce its context.
    replay_path = tmp_path / "captured.db"
    replay_path.write_bytes(snapshot_bytes)
    replay, replay_metadata = load_local_database(replay_path, PROTOCOL)
    repeated, repeated_metadata = load_local_database(replay_path, PROTOCOL)
    assert replay == repeated == observations
    assert replay_metadata == repeated_metadata
    assert replay_metadata["dataset_sha256"] == metadata["dataset_sha256"]
    (changed,), changed_metadata = load_local_database(path, PROTOCOL)
    assert changed["observation_id"] == row["observation_id"]
    assert changed["resolution"] == {**row["resolution"], "result": "no"}
    assert changed["resolution_unavailable_reason"] is None
    assert changed["research_context"] == context.model_dump(mode="json")
    assert changed_metadata["consumed"]["research_context_ids"] == [context.content_id()]


def test_database_snapshot_is_removed_when_extraction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.db"
    _record_database(path)
    snapshots: list[Path] = []

    def fail_hash(snapshot: Path) -> str:
        snapshots.append(snapshot)
        raise OSError("snapshot hash failed")

    monkeypatch.setattr(experiment_inputs, "_sha256_file", fail_hash)
    with pytest.raises(OSError, match="snapshot hash failed"):
        load_local_database(path, PROTOCOL)
    (snapshot,) = snapshots
    assert not snapshot.parent.exists()


def test_database_missing_outcomes_are_not_marked_as_withheld(tmp_path: Path) -> None:
    path = tmp_path / "unresolved.db"
    _record_database(path)

    (row,), metadata = load_local_database(path, PROTOCOL)

    assert row["resolution"] is None
    assert row["resolution_unavailable_reason"] is None
    assert metadata["inventory"]["unresolved_at_report_as_of"] == 1
    assert metadata["inventory"].get("resolutions_withheld_after_report_as_of", 0) == 0


@pytest.mark.parametrize(
    ("protocol_update", "result", "execution_offsets", "window_reason"),
    [
        ({"report_as_of": (DECISION + timedelta(seconds=100)).isoformat()}, None, [30], None),
        # Settled at END+2m but not recorded locally until END+4m.
        ({"report_as_of": (END + timedelta(minutes=3)).isoformat()}, None, [30, 200], None),
        ({"report_as_of": (END + timedelta(minutes=4)).isoformat()}, "yes", [30, 200], None),
        (
            {"archive_start": "2026-08-04T00:00:00+00:00"},
            None,
            None,
            "observed before archive_start",
        ),
        ({"report_as_of": AS_OF.isoformat()}, None, None, "observed after report_as_of"),
    ],
)
def test_database_outcomes_quotes_and_rows_respect_report_boundaries(
    tmp_path: Path,
    protocol_update: dict[str, str],
    result: str | None,
    execution_offsets: list[int] | None,
    window_reason: str | None,
) -> None:
    path = tmp_path / "audit.db"
    _record_database(path)
    _insert_resolution(path, END + timedelta(minutes=4))
    for offset in (30, 200):
        _insert_snapshot(path, DECISION + timedelta(seconds=offset))

    observations, metadata = load_local_database(path, {**PROTOCOL, **protocol_update})

    if window_reason is not None:
        assert observations == []
        assert {"category": "window", "reason": window_reason, "count": 1} in metadata["exclusions"]
        return
    (row,) = observations
    assert (row["resolution"] or {}).get("result") == result
    assert row["resolution_unavailable_reason"] == (
        "outcome-after-cutoff" if result is None else None
    )
    if result is None:
        assert (END + timedelta(minutes=4)).isoformat() not in json.dumps(row)
    assert [
        (_utc(snapshot["observed_at"]) - DECISION).total_seconds() for snapshot in row["execution"]
    ] == execution_offsets


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("payload", "malformed ledger payload"),
        ("input", "recorded input object does not match its identity"),
        ("history", "research context not reproducible"),
        ("conflict", "conflicting resolution records"),
        ("leak", "resolution was recorded at or before the forecast"),
    ],
)
def test_database_defects_become_explicit_exclusions_not_silent_drops(
    tmp_path: Path, defect: str, reason: str
) -> None:
    path = tmp_path / "audit.db"
    (opportunity,), _ = _record_database(path, history=defect != "history")
    forecast_id = str(opportunity.forecast.forecast_id)
    if defect == "payload":
        _execute(
            path,
            "UPDATE forecast_ledger SET payload_json = '{\"truncated\"' WHERE forecast_id = ?",
            (forecast_id,),
        )
    elif defect == "input":
        _execute(
            path,
            "UPDATE input_objects SET payload_json = '{}' WHERE input_id = "
            "(SELECT input_id FROM forecast_ledger WHERE forecast_id = ?)",
            (forecast_id,),
        )
    elif defect == "conflict":
        _insert_resolution(path, END + timedelta(minutes=4), "yes")
        _insert_resolution(path, END + timedelta(minutes=5), "no")
    elif defect == "leak":
        _insert_resolution(path, DECISION - timedelta(seconds=1), settled_at=AS_OF)

    observations, metadata = load_local_database(path, PROTOCOL)

    (row,) = observations
    assert set(row) == EXCLUDED_KEYS
    assert row["observation_id"] == forecast_id
    assert row["exclusion_reason"].startswith(reason)
    assert metadata["inventory"]["observations_excluded_by_adapter"] == 1
    assert metadata["consumed"]["run_ids"] == []


@pytest.mark.parametrize(
    "missing_tables",
    [
        ("input_objects",),
        ("run_manifests",),
        ("input_objects", "run_manifests"),
    ],
)
def test_partial_schema_counts_each_existing_ledger_row_once(
    tmp_path: Path, missing_tables: tuple[str, ...]
) -> None:
    path = tmp_path / "partial.db"
    _record_database(path, kinds=(EVIDENCE_FORWARD_SHADOW, EVIDENCE_HISTORICAL))
    with closing(sqlite3.connect(path)) as connection, connection:
        for table in missing_tables:
            connection.execute(f"DROP TABLE {table}")

    observations, metadata = load_local_database(path, PROTOCOL)

    assert observations == []
    assert metadata["inventory"]["ledger_rows_total"] == 2
    assert metadata["inventory"]["legacy_forecasts_without_ledger"] == 0
    assert {item["table"] for item in metadata["schema"]["missing_required"]} == set(missing_tables)
    (exclusion,) = metadata["exclusions"]
    assert exclusion["category"] == "schema"
    assert exclusion["count"] == 2
    assert all(table in exclusion["reason"] for table in missing_tables)


def test_incomplete_ledger_columns_keep_schema_and_legacy_counts_distinct(tmp_path: Path) -> None:
    path = tmp_path / "partial-ledger.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE forecast_ledger (forecast_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE forecasts (forecast_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO forecast_ledger VALUES (?)", [("ledger-1",), ("ledger-2",)]
        )
        connection.executemany(
            "INSERT INTO forecasts VALUES (?)", [("ledger-1",), ("ledger-2",), ("legacy-1",)]
        )

    observations, metadata = load_local_database(path, PROTOCOL)

    assert observations == []
    assert metadata["inventory"]["ledger_rows_total"] == 2
    assert metadata["inventory"]["legacy_forecasts_without_ledger"] == 1
    assert {item["category"]: item["count"] for item in metadata["exclusions"]} == {
        "schema": 2,
        "legacy": 1,
    }


@pytest.mark.parametrize(
    ("column_recorded_at", "payload_recorded_at", "reason"),
    [
        (
            (DECISION + timedelta(seconds=1)).isoformat(),
            (DECISION + timedelta(seconds=1)).isoformat(),
            "run manifest was not recorded at the forecast observation time",
        ),
        (
            (END + timedelta(minutes=5)).isoformat(),
            (END + timedelta(minutes=5)).isoformat(),
            "run manifest was not recorded at the forecast observation time",
        ),
        (
            DECISION.isoformat(),
            (DECISION + timedelta(seconds=1)).isoformat(),
            "run manifest recording time disagrees with its recorded payload",
        ),
        (
            (DECISION + timedelta(seconds=1)).isoformat(),
            DECISION.isoformat(),
            "run manifest recording time disagrees with its recorded payload",
        ),
        (
            DECISION.replace(tzinfo=None).isoformat(),
            DECISION.isoformat(),
            "malformed run manifest or input object",
        ),
        (
            DECISION.isoformat(),
            DECISION.replace(tzinfo=None).isoformat(),
            "malformed run manifest or input object",
        ),
        (
            DECISION.isoformat(),
            "invalid",
            "malformed run manifest or input object",
        ),
    ],
)
def test_manifest_recording_time_must_match_payload_and_forecast(
    tmp_path: Path,
    column_recorded_at: str,
    payload_recorded_at: str,
    reason: str,
) -> None:
    path = tmp_path / "manifest-time.db"
    (opportunity,), _ = _record_database(path)
    _insert_resolution(path, END + timedelta(minutes=4))
    forecast_id = str(opportunity.forecast.forecast_id)
    with closing(sqlite3.connect(path)) as connection, connection:
        (payload_json,) = connection.execute(
            "SELECT payload_json FROM run_manifests WHERE run_id = ?", (forecast_id,)
        ).fetchone()
        manifest = json.loads(payload_json)
        manifest["recorded_at"] = payload_recorded_at
        connection.execute(
            "UPDATE run_manifests SET recorded_at = ?, payload_json = ?, manifest_sha256 = ? "
            "WHERE run_id = ?",
            (column_recorded_at, json.dumps(manifest), content_id(manifest), forecast_id),
        )

    (row,), metadata = load_local_database(path, PROTOCOL)

    assert set(row) == EXCLUDED_KEYS
    assert row["observation_id"] == forecast_id
    assert row["exclusion_reason"].startswith(reason)
    assert metadata["inventory"]["observations_excluded_by_adapter"] == 1
    assert metadata["consumed"]["run_ids"] == []
