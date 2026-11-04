import copy
import json
import sqlite3
import zlib
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from test_storage import build_evaluation

from prediction_market_system.backtest import EffectiveFee
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSide,
    MarketSnapshot,
    Opportunity,
    ProbabilityForecast,
    RecommendationState,
    ThresholdContract,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import (
    EVIDENCE_FORWARD_SHADOW,
    content_id,
    load_input_object,
)
from prediction_market_system.storage import MarketCheckStatus, SQLiteRepository
from prediction_market_system.storage_payloads import (
    compact_payload,
    decode_payload,
    encode_payload,
    load_payload,
)

COMPACT_TABLES = (
    "input_objects",
    "forecast_ledger",
    "run_manifests",
    "forecasts",
    "opportunities",
    "paper_alert_market_checks",
    "code_identities",
)
IMMUTABLE_TABLES = ("input_objects", "forecast_ledger", "run_manifests", "code_identities")


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "compact.db")
    repository.initialize()
    return repository


@pytest.fixture
def evaluation() -> tuple[ProbabilityForecast, Opportunity]:
    forecast, opportunity = build_evaluation()
    assert isinstance(forecast, ProbabilityForecast)
    assert isinstance(opportunity, Opportunity)
    market = opportunity.market.model_copy(update={"series_id": "KXBTC", "event_id": "EVENT"})
    opportunity = opportunity.model_copy(
        update={
            "market": market,
            "state": RecommendationState.ENTER_YES,
            "side": MarketSide.YES,
            "executable_price": 0.42,
            "conservative_probability": 0.75,
            "conservative_net_edge": 0.30,
            "suggested_max_exposure": 25.0,
        }
    )
    return forecast, opportunity


def save_check(
    repository: SQLiteRepository,
    opportunity: Opportunity,
    *,
    cycle_id: str = "cycle",
    status: MarketCheckStatus = MarketCheckStatus.DELIVERED,
) -> dict[str, Any]:
    payload = {
        "opportunity": opportunity.model_dump(mode="json"),
        "diagnostics": {"source": "fixture", "warnings": ["retained detail"]},
    }
    repository.save_paper_market_check(
        cycle_id=cycle_id,
        market_id=opportunity.market.market_id,
        series_ticker="KXBTC",
        event_ticker="EVENT",
        observed_at=opportunity.forecast.generated_at,
        status=status,
        reason="retained reason",
        payload=payload,
    )
    return payload


def stored_payloads(repository: SQLiteRepository, tables: tuple[str, ...]) -> dict[str, list[Any]]:
    with sqlite3.connect(repository.database_path) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }


def test_compressed_payload_codec_preserves_canonical_logical_json() -> None:
    payload = {"unicode": "caf\u00e9", "nested": {"null": None, "list": [1, True, 0.25]}}
    reordered = {"nested": payload["nested"], "unicode": payload["unicode"]}
    encoded = encode_payload(payload)
    assert isinstance(encoded, bytes)
    assert encoded == encode_payload(reordered)
    assert decode_payload(encoded) == payload
    assert decode_payload(json.dumps(payload)) == payload
    assert json.loads(zlib.decompress(encoded)) == payload


@pytest.mark.parametrize("table", COMPACT_TABLES)
def test_compact_payload_helpers_reject_unapproved_table_names(
    repository: SQLiteRepository, table: str
) -> None:
    with sqlite3.connect(repository.database_path) as connection:
        invalid_table = f"{table}; DROP TABLE forecasts"
        with pytest.raises(ValueError, match="unsupported"):
            compact_payload(connection, invalid_table, {})
        with pytest.raises(ValueError, match="unsupported"):
            load_payload(connection, invalid_table, encode_payload({}))


def test_all_compact_tables_roundtrip_through_repository_consumers(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    check_payload = save_check(repository, opportunity)
    expected = opportunity.model_dump(mode="json")
    ledger_expected = copy.deepcopy(expected)
    ledger_expected["forecast"].pop("input_manifest")
    explained = repository.explain_forecast(str(forecast.forecast_id))

    assert repository.opportunity_history() == [expected]
    assert repository.recorded_opportunity(str(forecast.forecast_id)) == opportunity
    check = repository.paper_market_checks("cycle")[0]
    assert check["payload"] == check_payload
    assert check["reason"] == "retained reason"
    assert check["status"] == MarketCheckStatus.DELIVERED.value
    assert explained["opportunity"] == ledger_expected
    assert explained["inputs"] == forecast.input_manifest
    assert explained["manifest"]["inputs"]["input_id"] == content_id(forecast.input_manifest)
    assert explained["evidence_class"] == EVIDENCE_FORWARD_SHADOW
    assert repository.forward_event_entry_exposure(
        event_id="EVENT", exclude_forecast_id=""
    ) == pytest.approx(25.0)
    assert (
        repository.forward_event_entry_exposure(
            event_id="EVENT", exclude_forecast_id=str(forecast.forecast_id)
        )
        == 0.0
    )

    with sqlite3.connect(repository.database_path) as connection:
        for table in COMPACT_TABLES:
            rows = connection.execute(f"SELECT payload_json FROM {table}").fetchall()
            assert rows, table
            assert all(isinstance(row[0], bytes) for row in rows), table
            for (raw,) in rows:
                assert isinstance(decode_payload(raw), dict)
                assert isinstance(load_payload(connection, table, raw), dict)
        forecast_raw = connection.execute("SELECT payload_json FROM forecasts").fetchone()[0]
        opportunity_raw = connection.execute("SELECT payload_json FROM opportunities").fetchone()[0]
        check_raw = connection.execute(
            "SELECT payload_json FROM paper_alert_market_checks"
        ).fetchone()[0]
        assert load_payload(connection, "forecasts", forecast_raw) == forecast.model_dump(
            mode="json"
        )
        assert load_payload(connection, "opportunities", opportunity_raw) == expected
        assert load_payload(connection, "paper_alert_market_checks", check_raw) == check_payload
        assert "input_manifest" not in zlib.decompress(forecast_raw).decode()
        assert "suggested_max_exposure" not in zlib.decompress(opportunity_raw).decode()
        assert "suggested_max_exposure" not in zlib.decompress(check_raw).decode()
        assert load_input_object(connection, content_id(forecast.input_manifest)) == (
            forecast.input_manifest
        )

    inputs = explained["inputs"]
    reproduced, _ = CryptoThresholdEngine(
        EngineConfig.model_validate(inputs["engine_config"])
    ).evaluate(
        MarketSnapshot.model_validate(inputs["market"]),
        CryptoSnapshot.model_validate(inputs["crypto"]),
        ThresholdContract.model_validate(inputs["contract"]),
    )
    assert reproduced.probability_yes == forecast.probability_yes
    assert reproduced.recipe_id == forecast.recipe_id


def test_status_and_paper_pnl_hydrate_compressed_checks(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    save_check(repository, opportunity)
    settled = forecast.generated_at + timedelta(hours=1)
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "INSERT INTO kalshi_resolutions VALUES (?, ?, 'yes', '1', ?, '')",
            (opportunity.market.market_id, settled.isoformat(), settled.isoformat()),
        )
    fee = EffectiveFee(fee_type="quadratic", multiplier=0.07)
    before = repository.paper_alert_status_since_last_request(
        series_ticker="KXBTC", symbol="BTC", requested_at=settled - timedelta(seconds=1)
    )
    assert before.cycles == before.delivered_alerts == 1
    assert before.outcomes(fee=fee, lot_contracts=1).unresolved == 1
    after = repository.paper_alert_status_since_last_request(
        series_ticker="KXBTC", symbol="BTC", requested_at=settled
    )
    assert after.cycles == 0
    assert after.delivered_alerts == 1
    assert after.scorable_alerts[0].opportunity_id == str(opportunity.opportunity_id)
    assert after.scorable_alerts[0].probability_yes == forecast.probability_yes
    outcomes = after.outcomes(fee=fee, lot_contracts=1)
    assert outcomes.resolved == outcomes.wins == 1
    assert outcomes.unresolved == 0
    assert outcomes.total_cost == Decimal("0.44")
    assert outcomes.total_pnl == Decimal("0.56")


def test_repeated_writes_are_idempotent_and_code_identity_is_shared(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    save_check(repository, opportunity)
    before = stored_payloads(repository, COMPACT_TABLES)
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    save_check(repository, opportunity)
    assert stored_payloads(repository, COMPACT_TABLES) == before

    second_forecast = forecast.model_copy(update={"forecast_id": uuid4()})
    second = opportunity.model_copy(update={"opportunity_id": uuid4(), "forecast": second_forecast})
    repository.save_evaluation(second_forecast, second, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    first_manifest = repository.explain_forecast(str(forecast.forecast_id))["manifest"]
    second_manifest = repository.explain_forecast(str(second_forecast.forecast_id))["manifest"]
    assert first_manifest["code"] == second_manifest["code"]
    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM code_identities").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM input_objects").fetchone() == (1,)
        for digest, raw in connection.execute(
            "SELECT manifest_sha256, payload_json FROM run_manifests"
        ):
            manifest = load_payload(connection, "run_manifests", raw)
            assert content_id(manifest) == digest
            assert "source_files" not in zlib.decompress(raw).decode()


@pytest.mark.parametrize("changed", ["forecast", "allocation", "inputs"])
def test_compaction_does_not_weaken_immutable_evidence(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    changed: str,
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    before = stored_payloads(repository, COMPACT_TABLES)
    if changed == "forecast":
        forecast = forecast.model_copy(update={"probability_yes": 0.01})
        opportunity = opportunity.model_copy(update={"forecast": forecast})
    elif changed == "allocation":
        opportunity = opportunity.model_copy(update={"suggested_max_exposure": 26.0})
    else:
        inputs = copy.deepcopy(forecast.input_manifest)
        inputs["changed_evidence"] = True
        forecast = forecast.model_copy(update={"input_manifest": inputs})
        opportunity = opportunity.model_copy(update={"forecast": forecast})
    with pytest.raises(ValueError, match="immutable"):
        repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    assert stored_payloads(repository, COMPACT_TABLES) == before


def test_watch_pruning_preserves_immutable_inputs_manifests_and_explanation(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    opportunity = opportunity.model_copy(
        update={"state": RecommendationState.WATCH, "suggested_max_exposure": 0.0}
    )
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    save_check(repository, opportunity, status=MarketCheckStatus.WATCH)
    explained = repository.explain_forecast(str(forecast.forecast_id))
    before = stored_payloads(repository, IMMUTABLE_TABLES)
    cutoff = forecast.generated_at + timedelta(days=1)
    preview = repository.compact_watch_history(series_ticker="KXBTC", cutoff_at=cutoff)
    assert preview.eligible_checks == 1
    assert preview.deleted_checks == 0
    assert repository.paper_market_checks("cycle")
    result = repository.compact_watch_history(
        series_ticker="KXBTC", cutoff_at=cutoff, apply=True, batch_size=1
    )
    assert result.deleted_checks == result.deleted_opportunities == result.deleted_forecasts == 1
    assert result.deleted_cycles == result.rolled_up_checks == 1
    assert repository.paper_market_checks("cycle") == []
    assert repository.opportunity_history() == []
    assert repository.recorded_opportunity(str(forecast.forecast_id)) is None
    assert repository.explain_forecast(str(forecast.forecast_id)) == explained
    assert stored_payloads(repository, IMMUTABLE_TABLES) == before
    report = repository.shadow_report(series_ticker="KXBTC", as_of=cutoff)
    assert report["recipes"][0]["watch_forecasts"] == 1
    rollups = repository.paper_watch_rollups("KXBTC")
    assert rollups[0]["evaluation_count"] == 1
    repeated = repository.compact_watch_history(series_ticker="KXBTC", cutoff_at=cutoff, apply=True)
    assert repeated.deleted_checks == repeated.rolled_up_checks == 0
    assert repository.paper_watch_rollups("KXBTC") == rollups


@pytest.mark.parametrize("table", ["forecasts", "opportunities", "paper_alert_market_checks"])
@pytest.mark.parametrize("variant", ["exact", "differing", "orphan"])
def test_legacy_payloads_deduplicate_only_exact_matches(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    table: str,
    variant: str,
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    logical = opportunity.model_dump(mode="json")
    if table == "forecasts":
        logical = logical["forecast"]
        if variant == "differing":
            logical["probability_yes"] = 0.01
        elif variant == "orphan":
            logical["forecast_id"] = str(uuid4())
    else:
        if variant == "differing":
            logical["suggested_max_exposure"] = 26.0
        elif variant == "orphan":
            logical["opportunity_id"] = str(uuid4())
            logical["forecast"]["forecast_id"] = str(uuid4())
        if table == "paper_alert_market_checks":
            logical = {"opportunity": logical, "legacy_detail": ["keep", None]}
    legacy = json.dumps(logical, indent=2)
    with sqlite3.connect(repository.database_path) as connection:
        assert load_payload(connection, table, legacy) == logical
        compacted = compact_payload(connection, table, json.loads(legacy))
        assert isinstance(compacted, bytes)
        assert load_payload(connection, table, compacted) == logical
        if variant == "exact":
            duplicated_field = (
                "probability_yes" if table == "forecasts" else "suggested_max_exposure"
            )
            assert duplicated_field not in zlib.decompress(compacted).decode()
        else:
            # Inline fallbacks must remain readable without either possible reference target.
            connection.execute("DELETE FROM opportunities")
            connection.execute("DELETE FROM forecasts")
            connection.execute("DELETE FROM forecast_ledger")
            assert load_payload(connection, table, compacted) == logical


@pytest.mark.parametrize(
    ("deleted_table", "payload_table"),
    [
        ("code_identities", "run_manifests"),
        ("forecast_ledger", "forecasts"),
        ("forecast_ledger", "opportunities"),
        ("input_objects", "forecasts"),
        ("input_objects", "opportunities"),
        ("opportunities", "paper_alert_market_checks"),
    ],
)
def test_dangling_references_fail_closed(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    deleted_table: str,
    payload_table: str,
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    save_check(repository, opportunity)
    with sqlite3.connect(repository.database_path) as connection:
        raw = connection.execute(f"SELECT payload_json FROM {payload_table}").fetchone()[0]
        connection.execute(f"DELETE FROM {deleted_table}")
        with pytest.raises(ValueError, match="dangling"):
            load_payload(connection, payload_table, raw)
    with pytest.raises(ValueError, match="dangling"):
        if payload_table == "run_manifests":
            repository.explain_forecast(str(forecast.forecast_id))
        elif payload_table == "paper_alert_market_checks":
            repository.paper_market_checks("cycle")
        else:
            repository.recorded_opportunity(str(forecast.forecast_id))


def test_empty_initialization_records_marker_and_is_repeatable(
    repository: SQLiteRepository,
) -> None:
    with sqlite3.connect(repository.database_path) as connection:
        marker = connection.execute(
            "SELECT applied_at FROM schema_migrations WHERE migration_name = 'compact_evidence_v1'"
        ).fetchall()
        assert len(marker) == 1
        columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(paper_alert_market_checks)")
        }
        assert columns["opportunity_id"][3] == 0
    repository.initialize()
    with sqlite3.connect(repository.database_path) as connection:
        assert (
            connection.execute(
                "SELECT applied_at FROM schema_migrations "
                "WHERE migration_name = 'compact_evidence_v1'"
            ).fetchall()
            == marker
        )


def test_populated_unmarked_database_requires_explicit_migration(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_name = 'compact_evidence_v1'"
        )
        connection.execute(
            "UPDATE forecasts SET payload_json = ? WHERE forecast_id = ?",
            (forecast.model_dump_json(), str(forecast.forecast_id)),
        )
    before = stored_payloads(repository, COMPACT_TABLES)
    with pytest.raises((ValueError, RuntimeError), match="migrat"):
        repository.initialize()
    assert stored_payloads(repository, COMPACT_TABLES) == before
    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM schema_migrations WHERE migration_name = 'compact_evidence_v1'"
        ).fetchone() == (0,)


@pytest.mark.parametrize("raw", [b"not-zlib", zlib.compress(b"[]"), "[]"])
def test_corrupt_or_non_object_payloads_fail_closed(raw: str | bytes) -> None:
    with pytest.raises(ValueError):
        decode_payload(raw)


@pytest.mark.parametrize("table", ["forecasts", "opportunities"])
def test_legacy_missing_input_manifest_is_not_synthesized(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    table: str,
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    logical = opportunity.model_dump(mode="json")
    logical["forecast"].pop("input_manifest")
    if table == "forecasts":
        logical = logical["forecast"]
    with sqlite3.connect(repository.database_path) as connection:
        compacted = compact_payload(connection, table, logical)
        assert load_payload(connection, table, compacted) == logical


def test_checks_without_an_opportunity_keep_nullable_scalar_link(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    payload = {"unsupported": {"reason": "ambiguous benchmark", "metadata": [None, 7]}}
    repository.save_paper_market_check(
        cycle_id="unsupported",
        market_id=opportunity.market.market_id,
        series_ticker="KXBTC",
        event_ticker="EVENT",
        observed_at=forecast.generated_at,
        status=MarketCheckStatus.UNSUPPORTED,
        reason="ambiguous benchmark",
        payload=payload,
    )
    assert repository.paper_market_checks("unsupported")[0]["payload"] == payload
    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute(
            "SELECT opportunity_id, typeof(payload_json) FROM paper_alert_market_checks"
        ).fetchone() == (None, "blob")


def test_legacy_evaluation_without_ledger_cannot_be_reclassified(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "INSERT INTO forecasts (forecast_id, market_id, generated_at, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (
                str(forecast.forecast_id),
                forecast.market_id,
                forecast.generated_at.isoformat(),
                encode_payload(forecast.model_dump(mode="json")),
            ),
        )
    before = stored_payloads(repository, COMPACT_TABLES)
    with pytest.raises(ValueError, match="legacy evaluation lacks ledger evidence"):
        repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    assert stored_payloads(repository, COMPACT_TABLES) == before
    explained = repository.explain_forecast(str(forecast.forecast_id))
    assert explained["classification"] == "legacy-unreconstructible"
    assert explained["forecast"] == forecast.model_dump(mode="json")


@pytest.mark.parametrize(
    ("table", "field", "invalid"),
    [
        ("forecasts", "forecast_id", None),
        ("opportunities", "forecast_id", None),
        ("forecasts", "with_inputs", "false"),
        ("opportunities", "with_inputs", 1),
        ("opportunities", "with_inputs", None),
        ("run_manifests", "code_id", None),
        ("run_manifests", "value", None),
    ],
)
def test_malformed_reference_envelopes_fail_closed(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    table: str,
    field: str,
    invalid: Any,
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    with sqlite3.connect(repository.database_path) as connection:
        raw = connection.execute(f"SELECT payload_json FROM {table}").fetchone()[0]
        malformed = decode_payload(raw)
        if invalid is None:
            malformed.pop(field)
        else:
            malformed[field] = invalid
        with pytest.raises(ValueError):
            load_payload(connection, table, encode_payload(malformed))


def test_missing_code_identity_table_fails_closed(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    with sqlite3.connect(repository.database_path) as connection:
        raw = connection.execute("SELECT payload_json FROM run_manifests").fetchone()[0]
        connection.execute("DROP TABLE code_identities")
        with pytest.raises(ValueError, match="dangling"):
            load_payload(connection, "run_manifests", raw)
    with pytest.raises(ValueError, match="dangling"):
        repository.explain_forecast(str(forecast.forecast_id))


def test_explanation_rejects_tampered_shared_code_identity(
    repository: SQLiteRepository, evaluation: tuple[ProbabilityForecast, Opportunity]
) -> None:
    forecast, opportunity = evaluation
    repository.save_evaluation(forecast, opportunity)
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE code_identities SET payload_json = ?",
            (encode_payload({"source_sha256": "tampered"}),),
        )
    with pytest.raises(ValueError, match="code identity.*hash"):
        repository.explain_forecast(str(forecast.forecast_id))


@pytest.mark.parametrize("variant", ["differing", "orphan"])
def test_watch_pruning_preserves_inline_historical_evaluations(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    variant: str,
) -> None:
    forecast, opportunity = evaluation
    opportunity = opportunity.model_copy(
        update={"state": RecommendationState.WATCH, "suggested_max_exposure": 0.0}
    )
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    if variant == "differing":
        forecast = forecast.model_copy(
            update={"supporting_evidence": ("distinct historical forecast",)}
        )
        opportunity = opportunity.model_copy(
            update={"forecast": forecast, "warnings": ("distinct historical evidence",)}
        )
    with sqlite3.connect(repository.database_path) as connection:
        if variant == "orphan":
            connection.execute("DELETE FROM forecast_ledger")
        connection.execute(
            "UPDATE forecasts SET payload_json = ?",
            (compact_payload(connection, "forecasts", forecast.model_dump(mode="json")),),
        )
        connection.execute(
            "UPDATE opportunities SET payload_json = ?",
            (compact_payload(connection, "opportunities", opportunity.model_dump(mode="json")),),
        )
    save_check(repository, opportunity, status=MarketCheckStatus.WATCH)
    operational_before = stored_payloads(repository, ("forecasts", "opportunities"))
    immutable_before = stored_payloads(repository, IMMUTABLE_TABLES)
    explanation = repository.explain_forecast(str(forecast.forecast_id))
    result = repository.compact_watch_history(
        series_ticker="KXBTC",
        cutoff_at=forecast.generated_at + timedelta(days=1),
        apply=True,
    )
    assert result.deleted_checks == result.rolled_up_checks == result.deleted_cycles == 1
    assert result.deleted_opportunities == result.deleted_forecasts == 0
    assert repository.paper_market_checks("cycle") == []
    assert stored_payloads(repository, ("forecasts", "opportunities")) == operational_before
    assert stored_payloads(repository, IMMUTABLE_TABLES) == immutable_before
    assert repository.opportunity_history() == [opportunity.model_dump(mode="json")]
    assert repository.recorded_opportunity(str(forecast.forecast_id)) == opportunity
    assert repository.explain_forecast(str(forecast.forecast_id)) == explanation
    assert repository.paper_watch_rollups("KXBTC")[0]["evaluation_count"] == 1


@pytest.mark.parametrize(
    ("remaining_status", "remaining_days"),
    [(MarketCheckStatus.WATCH, 2), (MarketCheckStatus.DELIVERED, 0)],
)
def test_watch_pruning_preserves_evaluation_referenced_by_remaining_check(
    repository: SQLiteRepository,
    evaluation: tuple[ProbabilityForecast, Opportunity],
    remaining_status: MarketCheckStatus,
    remaining_days: int,
) -> None:
    forecast, opportunity = evaluation
    opportunity = opportunity.model_copy(
        update={"state": RecommendationState.WATCH, "suggested_max_exposure": 0.0}
    )
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    payload = save_check(repository, opportunity, status=MarketCheckStatus.WATCH)
    repository.save_paper_market_check(
        cycle_id="remaining",
        market_id=opportunity.market.market_id,
        series_ticker="KXBTC",
        event_ticker="EVENT",
        observed_at=forecast.generated_at + timedelta(days=remaining_days),
        status=remaining_status,
        reason="retained reason",
        payload=payload,
    )
    before = stored_payloads(repository, IMMUTABLE_TABLES + ("forecasts", "opportunities"))
    result = repository.compact_watch_history(
        series_ticker="KXBTC",
        cutoff_at=forecast.generated_at + timedelta(days=1),
        apply=True,
    )
    assert result.deleted_checks == result.rolled_up_checks == result.deleted_cycles == 1
    assert result.deleted_opportunities == result.deleted_forecasts == 0
    assert repository.paper_market_checks("cycle") == []
    remaining = repository.paper_market_checks("remaining")
    assert len(remaining) == 1
    assert remaining[0]["payload"] == payload
    assert remaining[0]["status"] == remaining_status.value
    assert repository.recorded_opportunity(str(forecast.forecast_id)) == opportunity
    assert stored_payloads(repository, IMMUTABLE_TABLES + ("forecasts", "opportunities")) == before
