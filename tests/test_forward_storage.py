from __future__ import annotations

import copy
import gzip
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from prediction_market_system import forward_capture, forward_storage
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import canonical_json, content_id, save_input_object
from prediction_market_system.forward_protocol import (
    DEFAULT_FORWARD_PROTOCOL_PATH,
    load_forward_protocol,
)
from prediction_market_system.forward_storage import ForwardStudyStore
from prediction_market_system.research import (
    ResearchContext,
    SpotCandle,
    calculate_realized_volatility,
)
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import (
    KalshiAPIError,
    KalshiMarket,
    KalshiOrderBook,
    to_market_snapshot,
)

REGISTERED = datetime(2026, 11, 5, 20, tzinfo=UTC)
SLOT = datetime(2026, 11, 6, 0, 45, tzinfo=UTC)


@pytest.fixture
def protocol() -> dict[str, Any]:
    return load_forward_protocol(DEFAULT_FORWARD_PROTOCOL_PATH)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    current = [REGISTERED]
    monkeypatch.setattr(forward_storage, "_now", lambda: current[0])
    return current


def attempt(
    protocol: dict[str, Any],
    *,
    started: datetime = SLOT,
    scheduled: datetime = SLOT,
    attempt_id: str = "attempt-1",
) -> dict[str, Any]:
    return {
        "schema": "pms-forward-evidence-v1",
        "record_type": "slot_attempt",
        "protocol_id": protocol["protocol_id"],
        "slot_id": content_id(
            {"protocol_id": protocol["protocol_id"], "scheduled_at": scheduled.isoformat()}
        ),
        "scheduled_at": scheduled.isoformat(),
        "attempt_id": attempt_id,
        "started_at": started.isoformat(),
    }


def candle(end: datetime, close: Decimal, interval: int) -> SpotCandle:
    return SpotCandle(
        provider="coinbase",
        product_id="BTC-USD",
        interval_seconds=interval,
        start_at=end - timedelta(seconds=interval),
        end_at=end,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=Decimal(10),
        retrieved_at=end,
        raw_payload={"fixture": "synthetic source, not study evidence"},
    )


def captured_store(
    tmp_path: Path, protocol: dict[str, Any], clock: list[datetime]
) -> tuple[ForwardStudyStore, dict[str, Any], dict[str, Any]]:
    """Real models and proofs, with only the wall clock controlled for prospective tests."""
    store = ForwardStudyStore(tmp_path / "forward.sqlite")
    store.register(protocol)
    clock[0] = SLOT
    first = attempt(protocol)
    store.append(first)
    end = SLOT.replace(hour=1, minute=0)
    market = KalshiMarket.model_validate(
        {
            "ticker": "KXBTC-FIXTURE-RANGE",
            "event_ticker": "KXBTC-FIXTURE",
            "series_ticker": "KXBTC",
            "market_type": "binary",
            "title": "Bitcoin range",
            "yes_sub_title": "59500 to 60500",
            "no_sub_title": "Outside range",
            "close_time": end.isoformat(),
            "latest_expiration_time": end.isoformat(),
            "status": "active",
            "notional_value_dollars": "1",
            "can_close_early": False,
            "strike_type": "between",
            "floor_strike": 59500,
            "cap_strike": 60500,
            "rules_primary": (
                "Resolves YES if the simple average of the sixty seconds before 1 AM UTC "
                "is between 59500 and 60500 at 1 AM UTC on Nov 6, 2026."
            ),
            "yes_bid_dollars": "0.28",
            "yes_ask_dollars": "0.30",
            "no_bid_dollars": "0.70",
            "no_ask_dollars": "0.72",
            "yes_bid_size_fp": "25",
            "yes_ask_size_fp": "25",
        }
    )
    contract = market.price_contract()
    spot = candle(SLOT, Decimal(60000), 60)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TABLE input_objects (input_id TEXT PRIMARY KEY, payload_json BLOB)"
        )
        spot_id = save_input_object(connection, spot.model_dump(mode="json"))
    selected = {
        "schema": first["schema"],
        "record_type": "selection",
        "protocol_id": first["protocol_id"],
        "slot_id": first["slot_id"],
        "attempt_id": first["attempt_id"],
        "scheduled_at": first["scheduled_at"],
        "discovery_started_at": SLOT.isoformat(),
        "discovery_received_at": SLOT.isoformat(),
        "selection_at": SLOT.isoformat(),
        "discovery_complete": True,
        "discovered_candidates": [
            {
                "event_id": market.event_ticker,
                "market_id": market.ticker,
                "market": market.model_dump(mode="json"),
                "contract": contract.model_dump(mode="json"),
                "eligible": True,
                "exclusion_reason": None,
            }
        ],
        "spot": {
            "price": 60000.0,
            "observed_at": SLOT.isoformat(),
            "received_at": SLOT.isoformat(),
            "source": "coinbase",
            "input_id": spot_id,
        },
        "selected_event_id": market.event_ticker,
        "selected_market_ids": [market.ticker],
        "error": None,
    }
    selected["selection_inputs_id"] = content_id(
        {
            key: selected[key]
            for key in ("discovered_candidates", "spot", "selected_event_id", "selected_market_ids")
        }
    )
    store.append(selected)
    decision = SLOT + timedelta(seconds=2)
    # Hourly history ends at the last completed UTC hour, not the :45 decision slot.
    hourly_end = SLOT.replace(minute=0, second=0, microsecond=0)
    hourly = [
        candle(
            hourly_end - timedelta(hours=720 - index),
            Decimal(f"{60000 + 40 * math.sin(index):.2f}"),
            3600,
        )
        for index in range(721)
    ]
    context = ResearchContext(
        symbol="BTC",
        event_ticker=market.event_ticker,
        as_of=decision,
        spot=spot,
        realized_volatility=calculate_realized_volatility(
            hourly,
            symbol="BTC",
            as_of=SLOT,
            window_seconds=2592000,
        ),
    )
    raw_book = {"yes_dollars": [["0.28", "25"]], "no_dollars": [["0.70", "25"]]}
    snapshot = to_market_snapshot(
        market, KalshiOrderBook.model_validate(raw_book), observed_at=decision
    )
    crypto = context.to_crypto_snapshot(strike_price=60000, expected_annual_return=0.0)
    forecast, opportunity = CryptoThresholdEngine(
        EngineConfig(**protocol["engine_config"])
    ).evaluate(
        snapshot,
        crypto,
        contract,
        None,
    )
    inputs = forecast.input_manifest
    input_id = content_id(inputs)
    run_id = str(uuid4())
    manifest = {
        "schema": "pms-run-manifest-v2",
        "kind": "forward-evaluation",
        "run_id": run_id,
        "recorded_at": decision.isoformat(),
        "code": {
            "source_files": protocol["model_source_sha256"],
            "lock_sha256": protocol["dependency_lock_sha256"],
        },
        "configuration": {"recipe_id": forecast.recipe_id, "engine": protocol["engine_config"]},
        "inputs": {"input_id": input_id},
    }
    observation = {
        "observation_id": str(forecast.forecast_id),
        "event_id": market.event_ticker,
        "market_id": market.ticker,
        "observed_at": decision.isoformat(),
        "opportunity": opportunity.model_dump(mode="json"),
        "inputs": inputs,
        "run_id": run_id,
        "run_manifest": manifest,
        "run_manifest_sha256": content_id(manifest),
        "input_id": input_id,
        "research_context": context.model_dump(mode="json"),
        "resolution": None,
    }
    capture = {
        "schema": first["schema"],
        "record_type": "market_capture",
        "protocol_id": first["protocol_id"],
        "slot_id": first["slot_id"],
        "attempt_id": first["attempt_id"],
        "scheduled_at": first["scheduled_at"],
        "selection_inputs_id": selected["selection_inputs_id"],
        "market_id": market.ticker,
        "event_id": market.event_ticker,
        "status": "valid",
        "error": None,
        "book_requested_at": (SLOT + timedelta(seconds=1)).isoformat(),
        "book_received_at": decision.isoformat(),
        "raw_book": raw_book,
        "raw_book_id": content_id(raw_book),
        "decision_at": decision.isoformat(),
        "forecast_persisted_at": None,
        "observation": observation,
        "fee_context": {
            "series_id": "KXBTC",
            "fee_type": "quadratic",
            "fee_multiplier": 1.0,
            "received_at": SLOT.isoformat(),
            "raw_payload": {
                "series": {
                    "series": {"ticker": "KXBTC", "fee_type": "quadratic", "fee_multiplier": 1.0}
                },
                "event": {"event": {"event_ticker": market.event_ticker}},
                "series_fee_changes": [],
                "event_fee_changes": [],
            },
        },
    }
    clock[0] = decision + timedelta(seconds=1)
    return store, capture, selected


def test_registration_is_prospective_and_idempotent(tmp_path, protocol, clock):
    store = ForwardStudyStore(tmp_path / "study.sqlite")
    registered = store.register(protocol)
    assert registered["registered_at"] == registered["persisted_at"] == REGISTERED.isoformat()
    clock[0] = SLOT
    assert store.register(protocol) == registered
    with pytest.raises(ValueError, match="prospective"):
        ForwardStudyStore(tmp_path / "late.sqlite").register(protocol)


def test_records_is_read_only_for_missing_and_unrelated_database(tmp_path, protocol):
    path = tmp_path / "missing" / "study.sqlite"
    assert ForwardStudyStore(path).records(protocol["protocol_id"]) == []
    assert not path.parent.exists()
    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    before = path.read_bytes()
    assert ForwardStudyStore(path).records(protocol["protocol_id"]) == []
    assert path.read_bytes() == before


def test_atomic_first_attempt_concurrency_and_conflicts(tmp_path, protocol, clock):
    store = ForwardStudyStore(tmp_path / "study.sqlite")
    store.register(protocol)
    clock[0] = SLOT
    original = attempt(protocol)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.append(original), range(16)))
    assert results.count(True) == 1
    assert results.count(False) == 15
    with pytest.raises(ValueError, match="conflicting"):
        store.append(attempt(protocol, attempt_id="replacement"))
    assert len(store.records(protocol["protocol_id"])) == 2


def test_reused_attempt_identity_and_future_clock_fail_closed(tmp_path, protocol, clock):
    store = ForwardStudyStore(tmp_path / "study.sqlite")
    store.register(protocol)
    clock[0] = SLOT
    with pytest.raises(ValueError, match="future"):
        store.append(attempt(protocol, started=SLOT + timedelta(seconds=1)))
    store.append(attempt(protocol))
    clock[0] = SLOT + timedelta(hours=1)
    with pytest.raises(ValueError, match="reused"):
        store.append(attempt(protocol, started=clock[0], scheduled=clock[0]))


@pytest.mark.parametrize(
    "extra", [{"persisted_at": REGISTERED.isoformat()}, {"record_id": "forged"}, {"unknown": True}]
)
def test_caller_cannot_supply_persistence_or_unknown_fields(tmp_path, protocol, clock, extra):
    store = ForwardStudyStore(tmp_path / "study.sqlite")
    store.register(protocol)
    clock[0] = SLOT
    with pytest.raises(ValueError, match="fields"):
        store.append(attempt(protocol) | extra)


def test_orphan_attempt_and_selection_are_rejected(tmp_path, protocol, clock):
    store = ForwardStudyStore(tmp_path / "study.sqlite")
    clock[0] = SLOT
    with pytest.raises(ValueError, match="registration"):
        store.append(attempt(protocol))
    clock[0] = REGISTERED
    store.register(protocol)
    clock[0] = SLOT
    row = attempt(protocol)
    selection = {key: value for key, value in row.items() if key != "started_at"}
    selection.update(
        record_type="selection",
        discovery_started_at=SLOT.isoformat(),
        discovery_received_at=SLOT.isoformat(),
        selection_at=SLOT.isoformat(),
        discovery_complete=True,
        discovered_candidates=[],
        spot=None,
        selected_event_id=None,
        selected_market_ids=[],
        error=None,
    )
    selection["selection_inputs_id"] = content_id(
        {
            key: selection[key]
            for key in ("discovered_candidates", "spot", "selected_event_id", "selected_market_ids")
        }
    )
    with pytest.raises(ValueError, match="slot_attempt"):
        store.append(selection)


def test_capture_persistence_is_actual_and_identical_retry_is_idempotent(tmp_path, protocol, clock):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    assert store.append(capture)
    clock[0] += timedelta(minutes=1)
    assert not store.append(capture)
    saved = store.records(protocol["protocol_id"])[-1]
    assert saved["forecast_persisted_at"] == saved["persisted_at"]
    assert saved["persisted_at"] == (SLOT + timedelta(seconds=3)).isoformat()
    assert capture["forecast_persisted_at"] is None
    changed = copy.deepcopy(capture)
    changed["error"] = "replacement"
    with pytest.raises(ValueError, match="conflicting"):
        store.append(changed)


def test_late_forecast_cannot_claim_earlier_persistence(tmp_path, protocol, clock):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    clock[0] = SLOT + timedelta(seconds=8)
    with pytest.raises(ValueError, match="timing"):
        store.append(capture)
    capture["forecast_persisted_at"] = (SLOT + timedelta(seconds=3)).isoformat()
    with pytest.raises(ValueError, match="storage-owned"):
        store.append(capture)


@pytest.mark.parametrize("mutation", ["book", "event", "status", "selection", "future-context"])
def test_valid_capture_requires_original_proofs(tmp_path, protocol, clock, mutation):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    if mutation == "book":
        capture["raw_book"]["yes_dollars"][0][0] = "0.01"
    elif mutation == "event":
        capture["event_id"] = "different-event"
    elif mutation == "status":
        capture["observation"]["opportunity"]["suggested_max_exposure"] += 1
    elif mutation == "selection":
        capture["selection_inputs_id"] = "unknown"
    else:
        capture["observation"]["research_context"]["spot"]["retrieved_at"] = (
            SLOT + timedelta(hours=1)
        ).isoformat()
    with pytest.raises(ValueError):
        store.append(capture)


@pytest.mark.parametrize("corruption", ["gzip", "hash", "missing", "source"])
def test_corrupt_and_missing_payload_proofs_fail_closed(tmp_path, protocol, clock, corruption):
    store, _, selected = captured_store(tmp_path, protocol, clock)
    with sqlite3.connect(store.path) as connection:
        if corruption == "source":
            connection.execute(
                "DELETE FROM forward_objects WHERE object_id = ?", (selected["spot"]["input_id"],)
            )
        else:
            object_id = connection.execute(
                "SELECT payload_id FROM forward_records WHERE record_type = 'selection'"
            ).fetchone()[0]
            if corruption == "missing":
                connection.execute("DELETE FROM forward_objects WHERE object_id = ?", (object_id,))
            else:
                payload = (
                    b"not gzip"
                    if corruption == "gzip"
                    else gzip.compress(canonical_json({"forged": True}).encode())
                )
                connection.execute(
                    "UPDATE forward_objects SET payload_gzip = ? WHERE object_id = ?",
                    (payload, object_id),
                )
    with pytest.raises(ValueError):
        store.records(protocol["protocol_id"])


def test_source_proof_is_self_contained_after_original_inputs_removed(tmp_path, protocol, clock):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    store.append(capture)
    before = store.records(protocol["protocol_id"])
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM input_objects")
    assert store.records(protocol["protocol_id"]) == before


def test_resolution_revisions_are_retained_without_choosing_a_result(tmp_path, protocol, clock):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    store.append(capture)
    candidates = list(store.resolution_candidates(protocol["protocol_id"]))
    assert len(candidates) == 1
    assert candidates[0][0]["market_id"] == capture["market_id"]
    assert candidates[0][1] == set()
    clock[0] = SLOT.replace(hour=2, minute=0)
    resolution = {
        "schema": capture["schema"],
        "record_type": "resolution",
        "protocol_id": protocol["protocol_id"],
        "market_id": capture["market_id"],
        "event_id": capture["event_id"],
        "result": "yes",
        "observed_at": clock[0].isoformat(),
        "received_at": clock[0].isoformat(),
        "raw_payload": {
            "market": {
                "ticker": capture["market_id"],
                "event_ticker": capture["event_id"],
                "result": "yes",
            }
        },
    }
    assert store.append(resolution)
    assert not store.append(resolution)
    assert next(store.resolution_candidates(protocol["protocol_id"]))[1] == {"yes"}
    conflict = copy.deepcopy(resolution)
    conflict["result"] = conflict["raw_payload"]["market"]["result"] = "no"
    with pytest.raises(ValueError, match="conflicting"):
        store.append(conflict)
    clock[0] += timedelta(minutes=1)
    conflict["observed_at"] = conflict["received_at"] = clock[0].isoformat()
    assert store.append(conflict)
    assert [
        row["result"]
        for row in store.records(protocol["protocol_id"])
        if row["record_type"] == "resolution"
    ] == ["yes", "no"]
    assert next(store.resolution_candidates(protocol["protocol_id"]))[1] == {"yes", "no"}


def test_forward_records_survive_watch_cleanup_without_wiring(tmp_path, protocol, clock):
    path = tmp_path / "existing.sqlite"
    repository = SQLiteRepository(path)
    repository.initialize()
    store = ForwardStudyStore(path)
    store.register(protocol)
    clock[0] = SLOT
    store.append(attempt(protocol))
    before = store.records(protocol["protocol_id"])
    repository.compact_watch_history(
        series_ticker="KXBTC",
        cutoff_at=SLOT + timedelta(days=30),
        apply=True,
        compacted_at=SLOT + timedelta(days=31),
    )
    assert store.records(protocol["protocol_id"]) == before


def test_valid_execution_requires_independent_later_request(tmp_path, protocol, clock):
    store, capture, selected = captured_store(tmp_path, protocol, clock)
    store.append(capture)
    target = datetime.fromisoformat(capture["decision_at"]) + timedelta(seconds=30)
    clock[0] = target + timedelta(seconds=1)
    market = KalshiMarket.model_validate(selected["discovered_candidates"][0]["market"])
    snapshot = to_market_snapshot(
        market, KalshiOrderBook.model_validate(capture["raw_book"]), observed_at=clock[0]
    )
    execution = {
        "schema": capture["schema"],
        "record_type": "execution_capture",
        "protocol_id": protocol["protocol_id"],
        "slot_id": capture["slot_id"],
        "attempt_id": capture["attempt_id"],
        "market_id": capture["market_id"],
        "event_id": capture["event_id"],
        "observation_id": capture["observation"]["observation_id"],
        "scenario": "base",
        "target_at": target.isoformat(),
        "requested_at": capture["book_requested_at"],
        "received_at": clock[0].isoformat(),
        "status": "valid",
        "error": None,
        "raw_book": capture["raw_book"],
        "raw_book_id": capture["raw_book_id"],
        "snapshot": snapshot.model_dump(mode="json"),
    }
    with pytest.raises(ValueError, match="request"):
        store.append(execution)
    execution["requested_at"] = target.isoformat()
    assert store.append(execution)
    assert not store.append(execution)
    execution["snapshot"]["yes_ask_size"] += 1
    with pytest.raises(ValueError, match="conflicting"):
        store.append(execution)


def test_slot_status_cannot_claim_valid_without_all_captures(tmp_path, protocol, clock):
    store, capture, selected = captured_store(tmp_path, protocol, clock)
    result = {
        "schema": capture["schema"],
        "record_type": "slot_result",
        "protocol_id": protocol["protocol_id"],
        "slot_id": capture["slot_id"],
        "attempt_id": capture["attempt_id"],
        "scheduled_at": capture["scheduled_at"],
        "completed_at": clock[0].isoformat(),
        "status": "valid",
        "selected_market_ids": selected["selected_market_ids"],
        "captured_market_ids": [],
        "error": None,
    }
    with pytest.raises(ValueError, match="coverage"):
        store.append(result)
    store.append(capture)
    result["captured_market_ids"] = selected["selected_market_ids"]
    assert store.append(result)


def test_missing_source_reference_prevents_selection_append(tmp_path, protocol, clock):
    store, _, selected = captured_store(tmp_path, protocol, clock)
    # A second prospective slot cannot borrow a missing source proof from the old ledger.
    later = SLOT + timedelta(hours=1)
    clock[0] = later
    first = attempt(protocol, scheduled=later, started=later, attempt_id="attempt-2")
    store.append(first)
    selected.update({key: first[key] for key in ("slot_id", "attempt_id", "scheduled_at")})
    selected.update(
        discovery_started_at=later.isoformat(),
        discovery_received_at=later.isoformat(),
        selection_at=later.isoformat(),
        discovery_complete=False,
        discovered_candidates=[],
        selected_event_id=None,
        selected_market_ids=[],
        error="source unavailable",
    )
    selected["spot"]["input_id"] = "missing-content-id"
    selected["selection_inputs_id"] = content_id(
        {
            key: selected[key]
            for key in ("discovered_candidates", "spot", "selected_event_id", "selected_market_ids")
        }
    )
    with pytest.raises(ValueError, match="source proof"):
        store.append(selected)


def test_late_attempt_keeps_explicit_result_without_selection(tmp_path, protocol, clock):
    store = ForwardStudyStore(tmp_path / "study.sqlite")
    store.register(protocol)
    clock[0] = SLOT + timedelta(minutes=5)
    first = attempt(protocol, started=clock[0])
    store.append(first)
    result = {key: value for key, value in first.items() if key != "started_at"}
    result.update(
        record_type="slot_result",
        completed_at=clock[0].isoformat(),
        status="late",
        selected_market_ids=[],
        captured_market_ids=[],
        error="missed start deadline",
    )
    assert store.append(result)
    assert store.records(protocol["protocol_id"])[-1]["status"] == "late"


def test_slot_reads_validate_only_selected_dependencies(tmp_path, protocol, clock):
    store, _, _ = captured_store(tmp_path, protocol, clock)
    later = SLOT + timedelta(hours=1)
    clock[0] = later
    newer = attempt(protocol, started=later, scheduled=later, attempt_id="attempt-2")
    store.append(newer)
    with sqlite3.connect(store.path) as connection:
        object_id = connection.execute(
            "SELECT payload_id FROM forward_records WHERE record_type = 'selection'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE forward_objects SET payload_gzip = ? WHERE object_id = ?",
            (b"corrupt old selection", object_id),
        )

    # A new slot neither depends on nor silently repairs the unrelated corrupt discovery.
    saved = store.slot_attempt(protocol["protocol_id"], later)
    assert saved is not None and saved["attempt_id"] == "attempt-2"
    assert store.slot_attempt(protocol["protocol_id"], later + timedelta(hours=1)) is None
    assert store.attempt_records(protocol["protocol_id"], "attempt-2") == [saved]
    with pytest.raises(ValueError, match="corrupt"):
        store.attempt_records(protocol["protocol_id"], "attempt-1")
    with pytest.raises(ValueError, match="corrupt"):
        store.records(protocol["protocol_id"])


def test_slot_reads_are_read_only_for_absent_studies(tmp_path, protocol):
    path = tmp_path / "absent" / "study.sqlite"
    store = ForwardStudyStore(path)
    assert store.slot_attempt(protocol["protocol_id"], SLOT) is None
    assert store.attempt_records(protocol["protocol_id"], "attempt-1") == []
    assert list(store.resolution_candidates(protocol["protocol_id"])) == []
    assert not path.parent.exists()
    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    before = path.read_bytes()
    store = ForwardStudyStore(path)
    assert store.slot_attempt(protocol["protocol_id"], SLOT) is None
    assert store.attempt_records(protocol["protocol_id"], "attempt-1") == []
    assert list(store.resolution_candidates(protocol["protocol_id"])) == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", [None, [], {}, "unknown"])
def test_unknown_record_types_fail_before_dictionary_indexing(tmp_path, protocol, kind):
    record = {"record_type": kind}
    with pytest.raises(ValueError, match="unknown forward record type"):
        forward_storage.validate_forward_record(record, protocol, [])
    with pytest.raises(ValueError, match="unknown forward record type"):
        ForwardStudyStore(tmp_path / "study.sqlite").append(record)


def test_resolution_candidates_validate_only_original_market_dependencies(
    tmp_path, protocol, clock
):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    store.append(capture)
    later = SLOT + timedelta(hours=1)
    clock[0] = later
    store.append(attempt(protocol, started=later, scheduled=later, attempt_id="unrelated-attempt"))
    with sqlite3.connect(store.path) as connection:
        object_id = connection.execute(
            "SELECT payload_id FROM forward_records WHERE attempt_id = 'unrelated-attempt'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE forward_objects SET payload_gzip = ? WHERE object_id = ?",
            (b"corrupt unrelated attempt", object_id),
        )
    candidates = list(store.resolution_candidates(protocol["protocol_id"]))
    assert len(candidates) == 1
    assert candidates[0][0]["market_id"] == capture["market_id"]
    assert candidates[0][1] == set()
    with pytest.raises(ValueError, match="corrupt"):
        store.records(protocol["protocol_id"])
    with sqlite3.connect(store.path) as connection:
        object_id = connection.execute(
            "SELECT payload_id FROM forward_records WHERE record_type = 'selection'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE forward_objects SET payload_gzip = ? WHERE object_id = ?",
            (b"corrupt required selection", object_id),
        )
    with pytest.raises(ValueError, match="corrupt"):
        list(store.resolution_candidates(protocol["protocol_id"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current_status", "historical_status"),
    [(200, None), (404, 200), (503, None), (404, 404), (404, 503)],
)
async def test_resolution_payload_uses_historical_endpoint_only_after_not_found(
    current_status, historical_status
):
    requested = []
    payload = {"market": {"ticker": "KXBTC-ARCHIVED", "result": "yes", "status": "settled"}}

    def respond(request):
        requested.append(request.url.path)
        if request.url.path == "/markets/KXBTC-ARCHIVED":
            return httpx.Response(current_status, json=payload)
        assert request.url.path == "/historical/markets/KXBTC-ARCHIVED"
        assert historical_status is not None
        return httpx.Response(historical_status, json=payload)

    async with httpx.AsyncClient(
        base_url="https://kalshi.test", transport=httpx.MockTransport(respond)
    ) as client:
        kalshi = forward_capture._PublicKalshi(
            client=client, max_rate_limit_retries=0, max_transient_retries=0
        )
        if current_status == 200 or historical_status == 200:
            assert await kalshi.resolution_payload("KXBTC-ARCHIVED") == payload
        else:
            with pytest.raises(KalshiAPIError):
                await kalshi.resolution_payload("KXBTC-ARCHIVED")
    assert requested == ["/markets/KXBTC-ARCHIVED"] + (
        ["/historical/markets/KXBTC-ARCHIVED"] if current_status == 404 else []
    )


@pytest.mark.parametrize(
    "series_multiplier,historical_multiplier,clear,accepted",
    [(1, 2, True, True), (2, 1, True, False), (1, 2, False, False)],
)
def test_fee_proof_distinguishes_current_null_from_missing_override(
    tmp_path, protocol, clock, series_multiplier, historical_multiplier, clear, accepted
):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    raw = capture["fee_context"]["raw_payload"]
    raw["series"]["series"]["fee_multiplier"] = series_multiplier
    raw["event_fee_changes"] = [
        {
            "event_ticker": capture["event_id"],
            "series_ticker": "KXBTC",
            "scheduled_ts": (SLOT - timedelta(hours=1)).isoformat(),
            "fee_type_override": "quadratic",
            "fee_multiplier_override": historical_multiplier,
        }
    ]
    if clear:
        raw["event"]["event"].update(fee_type_override=None, fee_multiplier_override=None)
    if accepted:
        assert store.append(capture)
        assert store.records(protocol["protocol_id"])[-1]["fee_context"]["fee_multiplier"] == 1
    else:
        with pytest.raises(ValueError, match="fee proof"):
            store.append(capture)


@pytest.mark.parametrize(
    "requested_offset,received_offset,persisted_offset,accepted",
    [
        (-1, None, 1, False),
        (-1, 1, 1, False),
        (1, 0, 1, False),
        (0, 2, 1, False),
        (2, None, 1, False),
        (0, None, 1, True),
        (0, 121, 121, True),
        (0, 900, 900, True),
    ],
)
def test_failed_execution_requires_causal_clocks_but_retains_late_receipts(
    tmp_path, protocol, clock, requested_offset, received_offset, persisted_offset, accepted
):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    store.append(capture)
    target = datetime.fromisoformat(capture["decision_at"]) + timedelta(seconds=30)
    clock[0] = target + timedelta(seconds=persisted_offset)
    raw = capture["raw_book"] if received_offset is not None else None
    execution = {
        "schema": capture["schema"],
        "record_type": "execution_capture",
        "protocol_id": protocol["protocol_id"],
        "slot_id": capture["slot_id"],
        "attempt_id": capture["attempt_id"],
        "market_id": capture["market_id"],
        "event_id": capture["event_id"],
        "observation_id": capture["observation"]["observation_id"],
        "scenario": "base",
        "target_at": target.isoformat(),
        "requested_at": (target + timedelta(seconds=requested_offset)).isoformat(),
        "received_at": (
            None
            if received_offset is None
            else (target + timedelta(seconds=received_offset)).isoformat()
        ),
        "status": "failed",
        "error": "source unavailable or receipt too late",
        "raw_book": raw,
        "raw_book_id": None if raw is None else content_id(raw),
        "snapshot": None,
    }
    if accepted:
        assert store.append(execution)
        stored = store.records(protocol["protocol_id"])[-1]
        assert stored["status"] == "failed" and stored["snapshot"] is None
        assert stored["received_at"] == execution["received_at"]
    else:
        with pytest.raises(ValueError, match="execution|future forward clock"):
            store.append(execution)


@pytest.mark.parametrize(
    "current,reset_offset,accepted",
    [
        ({"fee_type_override": None, "fee_multiplier_override": None}, None, True),
        ({"fee_type_override": None}, None, False),
        ({"fee_multiplier_override": None}, None, False),
        ({}, -1, True),
        ({}, 0, True),
        ({}, 1, False),
    ],
)
def test_fee_proof_replays_field_resets_only_when_effective(
    tmp_path, protocol, clock, current, reset_offset, accepted
):
    store, capture, _ = captured_store(tmp_path, protocol, clock)
    raw = capture["fee_context"]["raw_payload"]
    previous = {
        "id": "event-fee-previous",
        "event_ticker": capture["event_id"],
        "series_ticker": "KXBTC",
        "scheduled_ts": (SLOT - timedelta(hours=1)).isoformat(),
        "fee_type_override": "flat",
        "fee_multiplier_override": 2,
    }
    raw["event_fee_changes"] = [previous]
    if reset_offset is not None:
        raw["event_fee_changes"].append(
            {
                **previous,
                "id": "event-fee-reset",
                "scheduled_ts": (SLOT + timedelta(seconds=reset_offset)).isoformat(),
                "fee_type_override": None,
                "fee_multiplier_override": None,
            }
        )
    raw["event"]["event"].update(current)
    if accepted:
        assert store.append(capture)
        assert store.records(protocol["protocol_id"])[-1]["fee_context"] == capture["fee_context"]
    else:
        with pytest.raises(ValueError, match="fee proof"):
            store.append(capture)
