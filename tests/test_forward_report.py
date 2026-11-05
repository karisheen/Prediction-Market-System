"""Behavioral embargo, provenance, population and observable-book simulation tests."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import pytest
from test_experiment import kalshi_market, kalshi_rule, research_context

from prediction_market_system import forward_report
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import content_id
from prediction_market_system.forward_protocol import (
    DEFAULT_FORWARD_PROTOCOL_PATH,
    load_forward_protocol,
)
from prediction_market_system.forward_report import build_forward_report
from prediction_market_system.venues.kalshi import KalshiOrderBook, to_market_snapshot

SLOT = datetime(2026, 11, 6, 0, 45, tzinfo=UTC)
CUTOFF = datetime(2027, 1, 7, tzinfo=UTC)


class MemoryStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def records(self, protocol_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self.rows)


@pytest.fixture
def protocol() -> dict[str, Any]:
    return load_forward_protocol(DEFAULT_FORWARD_PROTOCOL_PATH)


@pytest.fixture(autouse=True)
def wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forward_report, "_utcnow", lambda: CUTOFF + timedelta(days=10))


def sealed(record: dict[str, Any]) -> dict[str, Any]:
    record["record_id"] = content_id(
        {key: value for key, value in record.items() if key != "record_id"}
    )
    return record


def record(
    document: dict[str, Any], kind: str, persisted: datetime, **fields: Any
) -> dict[str, Any]:
    return sealed(
        {
            "schema": "pms-forward-evidence-v1",
            "record_type": kind,
            "protocol_id": document["protocol_id"],
            "persisted_at": persisted.isoformat(),
            **fields,
        }
    )


def registration(protocol: dict[str, Any]) -> dict[str, Any]:
    clock = datetime.fromisoformat(protocol["frozen_at"]) + timedelta(seconds=1)
    return record(
        protocol,
        "registration",
        clock,
        registered_at=clock.isoformat(),
        protocol=protocol,
        model_source_sha256=protocol["model_source_sha256"],
    )


@lru_cache(maxsize=64)
def context_at(slot: datetime) -> Any:
    return research_context(slot, implied=True)


def captured_slot(
    protocol: dict[str, Any],
    slot: datetime = SLOT,
    *,
    lower: float = 59_500.0,
    result: str | None = "yes",
    execution: bool = True,
    book: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build actual parser/engine/context evidence, never mocked forecast validation."""
    decision = slot + timedelta(seconds=5)
    end = slot.replace(minute=0) + timedelta(hours=1)
    event_id = f"KXBTC-{end:%Y%m%d%H}"
    market_id = f"{event_id}-B{int(lower)}"
    venue = kalshi_market(
        market_id,
        event_id,
        end,
        lower,
        lower + 1000,
        kalshi_rule(end, lower, lower + 1000),
    )
    contract = venue.price_contract()
    book = book or {"yes_dollars": [["0.28", "1000"]], "no_dollars": [["0.70", "1000"]]}
    snapshot = to_market_snapshot(
        venue,
        KalshiOrderBook.model_validate(book),
        observed_at=decision,
    )
    context = context_at(slot)
    crypto = context.to_crypto_snapshot(strike_price=lower + 500, expected_annual_return=0)
    engine = CryptoThresholdEngine(EngineConfig.model_validate(protocol["engine_config"]))
    forecast, opportunity = engine.evaluate(snapshot, crypto, contract)
    inputs = forecast.input_manifest
    run_id = str(forecast.forecast_id)
    manifest = {
        "schema": "pms-run-manifest-v2",
        "kind": "forward-evaluation",
        "run_id": run_id,
        "recorded_at": decision.isoformat(),
        "configuration": {"recipe_id": forecast.recipe_id, "engine": protocol["engine_config"]},
        "inputs": {"input_id": content_id(inputs)},
        "code": {
            "source_files": protocol["model_source_sha256"],
            "source_sha256": content_id(protocol["model_source_sha256"]),
            "lock_sha256": protocol["dependency_lock_sha256"],
        },
    }
    opportunity_payload = opportunity.model_dump(mode="json")
    opportunity_payload["forecast"].pop("input_manifest")
    observation = {
        "observation_id": run_id,
        "event_id": event_id,
        "market_id": market_id,
        "observed_at": decision.isoformat(),
        "opportunity": opportunity_payload,
        "inputs": inputs,
        "run_id": run_id,
        "run_manifest": manifest,
        "run_manifest_sha256": content_id(manifest),
        "input_id": content_id(inputs),
        "research_context": context.model_dump(mode="json"),
        "resolution": None,
    }
    slot_id = content_id({"protocol_id": protocol["protocol_id"], "scheduled_at": slot.isoformat()})
    attempt_id = f"attempt-{slot_id}"
    common = {"slot_id": slot_id, "attempt_id": attempt_id, "scheduled_at": slot.isoformat()}
    selection_inputs = {
        "discovered_candidates": [
            {
                "event_id": event_id,
                "market_id": market_id,
                "market": venue.model_dump(mode="json"),
                "contract": contract.model_dump(mode="json"),
                "eligible": True,
                "exclusion_reason": None,
            }
        ],
        "spot": {
            "price": float(context.spot.close),
            "observed_at": context.spot.end_at.isoformat(),
            "received_at": context.spot.retrieved_at.isoformat(),
            "source": context.spot.provider,
            "input_id": context.content_id(),
        },
        "selected_event_id": event_id,
        "selected_market_ids": [market_id],
    }
    selection_id = content_id(selection_inputs)
    rows = [
        record(protocol, "slot_attempt", slot, **common, started_at=slot.isoformat()),
        record(
            protocol,
            "selection",
            slot + timedelta(seconds=2),
            **common,
            discovery_started_at=slot.isoformat(),
            discovery_received_at=(slot + timedelta(seconds=1)).isoformat(),
            selection_at=(slot + timedelta(seconds=2)).isoformat(),
            discovery_complete=True,
            **selection_inputs,
            selection_inputs_id=selection_id,
            error=None,
        ),
        record(
            protocol,
            "market_capture",
            decision + timedelta(seconds=1),
            **common,
            selection_inputs_id=selection_id,
            market_id=market_id,
            event_id=event_id,
            status="valid",
            error=None,
            book_requested_at=(slot + timedelta(seconds=3)).isoformat(),
            book_received_at=(slot + timedelta(seconds=4)).isoformat(),
            raw_book=book,
            raw_book_id=content_id(book),
            decision_at=decision.isoformat(),
            forecast_persisted_at=(decision + timedelta(seconds=1)).isoformat(),
            observation=observation,
            fee_context={
                "series_id": "KXBTC",
                "fee_type": "quadratic",
                "fee_multiplier": 1.0,
                "received_at": (slot + timedelta(seconds=3)).isoformat(),
                "raw_payload": {
                    "series": {
                        "series": {
                            "ticker": "KXBTC",
                            "fee_type": "quadratic",
                            "fee_multiplier": 1.0,
                        }
                    },
                    "event": {"event": {"event_ticker": event_id}},
                    "series_fee_changes": [],
                    "event_fee_changes": [],
                },
            },
        ),
    ]
    if execution:
        for scenario in protocol["cost_scenarios"]:
            target = decision + timedelta(seconds=scenario["latency_seconds"])
            received = target + timedelta(seconds=1)
            execution_snapshot = to_market_snapshot(
                venue,
                KalshiOrderBook.model_validate(book),
                observed_at=received,
            )
            rows.append(
                record(
                    protocol,
                    "execution_capture",
                    received + timedelta(seconds=1),
                    slot_id=slot_id,
                    attempt_id=attempt_id,
                    market_id=market_id,
                    event_id=event_id,
                    observation_id=run_id,
                    scenario=scenario["name"],
                    target_at=target.isoformat(),
                    requested_at=target.isoformat(),
                    received_at=received.isoformat(),
                    status="valid",
                    error=None,
                    raw_book=book,
                    raw_book_id=content_id(book),
                    snapshot=execution_snapshot.model_dump(mode="json"),
                )
            )
    rows.append(
        record(
            protocol,
            "slot_result",
            decision + timedelta(seconds=125),
            **common,
            completed_at=(decision + timedelta(seconds=125)).isoformat(),
            status="valid",
            selected_market_ids=[market_id],
            captured_market_ids=[market_id],
            error=None,
        )
    )
    if result is not None:
        resolved = end + timedelta(minutes=5)
        rows.append(
            record(
                protocol,
                "resolution",
                resolved + timedelta(seconds=1),
                market_id=market_id,
                event_id=event_id,
                result=result,
                observed_at=resolved.isoformat(),
                received_at=resolved.isoformat(),
                raw_payload={"ticker": market_id, "event_ticker": event_id, "result": result},
            )
        )
    return rows


def report(
    protocol: dict[str, Any], rows: list[dict[str, Any]], at: datetime = CUTOFF
) -> dict[str, Any]:
    return build_forward_report(store=MemoryStore(rows), protocol=protocol, as_of=at)


def test_embargo_has_only_elapsed_operational_coverage(
    protocol: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("interim scoring was invoked")

    monkeypatch.setattr(forward_report, "compare_forward_observations", forbidden)
    monkeypatch.setattr(forward_report, "assess_forward_decisions", forbidden)
    rows = [registration(protocol), *captured_slot(protocol)]
    first = report(protocol, rows, SLOT - timedelta(seconds=1))
    assert first["status"] == "coverage-only"
    assert first["coverage"]["expected_slots"] == 1440
    assert first["coverage"]["elapsed_slots"] == first["coverage"]["missing_slots"] == 0
    assert first["coverage"]["future_slots"] == 1440
    during = report(protocol, rows, SLOT + timedelta(minutes=30))
    assert during["coverage"]["elapsed_slots"] == 1
    assert during["coverage"]["valid_slots"] == 1
    assert during["coverage"]["future_slots"] == 1439
    assert "probability" not in during and "execution" not in during
    assert "outcome_exclusions" not in during and "conclusion" not in during
    assert during["coverage"]["folds"][1]["elapsed_slots"] == 0


def test_naive_and_future_report_clocks_are_rejected(protocol: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="timezone"):
        report(protocol, [], datetime(2026, 11, 6))
    with pytest.raises(ValueError, match="future"):
        report(protocol, [], CUTOFF + timedelta(days=11))


def test_empty_final_denominators_include_every_slot_and_fold(protocol: dict[str, Any]) -> None:
    result = report(protocol, [registration(protocol)])
    assert result["coverage"]["missing_slots"] == 1440
    assert result["coverage"]["future_slots"] == 0
    assert [fold["missing_slots"] for fold in result["coverage"]["folds"]] == [720, 720]
    assert result["probability"]["conclusion"]["status"] == "inconclusive"
    assert result["execution"]["conclusion"]["status"] == "descriptive-incomplete"


def test_valid_capture_reproduces_and_scores_without_reclassifying_campaign(
    protocol: dict[str, Any],
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    before = copy.deepcopy(rows)
    result = report(protocol, rows)
    assert rows == before
    assert result["rejected_records"] == []
    assert result["coverage"]["valid_slots"] == 1
    assert result["probability"]["counts"]["eligible"] == 1
    assert result["probability"]["counts"]["resolved"] == 1
    assert result["probability"]["source_kind"] == "prospective-forward"
    assert result["probability"]["protocol_id"] == protocol["protocol_id"]
    assert not result["probability"]["conclusion"]["gates"]["capture_coverage"]
    assert [row["name"] for row in result["execution"]["scenarios"]] == [
        "base",
        "adverse",
        "severe",
    ]


@pytest.mark.parametrize(
    "mutation", ["hash", "config", "pins", "persistence", "selection", "old-ledger"]
)
def test_corrupt_or_forged_capture_cannot_count_as_valid(
    protocol: dict[str, Any],
    mutation: str,
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    capture = next(row for row in rows if row["record_type"] == "market_capture")
    if mutation == "hash":
        capture["raw_book"]["yes_dollars"][0][0] = "0.01"
    elif mutation == "config":
        capture["observation"]["inputs"]["engine_config"]["uncertainty_margin"] = 0
        sealed(capture)
    elif mutation == "pins":
        capture["observation"]["run_manifest"]["code"]["source_files"] = {}
        sealed(capture)
    elif mutation == "persistence":
        capture["persisted_at"] = (SLOT + timedelta(minutes=3)).isoformat()
        capture["forecast_persisted_at"] = capture["persisted_at"]
        sealed(capture)
    elif mutation == "selection":
        rows = [row for row in rows if row["record_type"] != "selection"]
    else:
        rows = [row for row in rows if row["record_type"] not in {"slot_attempt", "registration"}]
    result = report(protocol, rows)
    assert result["coverage"]["valid_slots"] == 0
    assert result["probability"]["counts"]["eligible"] == 0
    assert result["rejected_records"]


def test_fixed_cutoff_excludes_late_persistence_even_when_observed_earlier(
    protocol: dict[str, Any],
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    resolution = next(row for row in rows if row["record_type"] == "resolution")
    resolution["persisted_at"] = (CUTOFF + timedelta(seconds=1)).isoformat()
    sealed(resolution)
    at_cutoff = report(protocol, rows)
    later = report(protocol, rows, CUTOFF + timedelta(days=5))
    assert at_cutoff == later
    assert later["probability"]["counts"]["resolved"] == 0
    assert later["probability"]["counts"]["unresolved"] == 1
    for scenario in later["execution"]["scenarios"]:
        assert scenario["summary"]["haircut_adjusted_pnl_dollars"] is None
        assert scenario["summary"]["filled_unresolved_orders"] == 1
        assert scenario["decisions"][0]["status"] == "filled"
        assert not scenario["summary"]["all_fills_resolved"]


def test_duplicate_and_conflicting_outcomes_fail_closed(protocol: dict[str, Any]) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    resolution = next(row for row in rows if row["record_type"] == "resolution")
    duplicate = report(protocol, [*rows, copy.deepcopy(resolution)])
    assert duplicate["probability"]["counts"]["resolved"] == 1
    revision = copy.deepcopy(resolution)
    revision["result"] = "no"
    revision["raw_payload"]["result"] = "no"
    revision["observed_at"] = (SLOT + timedelta(hours=1)).isoformat()
    revision["received_at"] = revision["observed_at"]
    revision["persisted_at"] = revision["observed_at"]
    sealed(revision)
    conflicted = report(protocol, [*rows, revision])
    assert conflicted["probability"]["counts"]["eligible"] == 0
    assert conflicted["outcome_exclusions"][0]["reason"] == "conflicting_or_invalid_resolution"
    assert not conflicted["execution"]["conclusion"]["directional_gate"]


def test_missing_execution_is_unavailable_not_zero_or_nonloss(protocol: dict[str, Any]) -> None:
    result = report(protocol, [registration(protocol), *captured_slot(protocol, execution=False)])
    assert result["probability"]["counts"]["eligible"] == 1
    for scenario in result["execution"]["scenarios"]:
        assert scenario["summary"]["unavailable_orders"] == 1
        assert scenario["summary"]["nonfill_orders"] == 0
        assert scenario["summary"]["haircut_adjusted_pnl_dollars"] is None
        assert scenario["summary"]["total_cost_dollars"] is None
        assert not scenario["summary"]["all_fills_resolved"]
        assert not scenario["summary"]["complete"]
        assert scenario["decisions"][0]["status"] == "unavailable"
        assert scenario["decisions"][0]["cost_dollars"] is None


def test_no_entry_uses_actual_no_ask_and_depth_in_all_scenarios(protocol: dict[str, Any]) -> None:
    rows = [registration(protocol), *captured_slot(protocol, lower=61_000, result="no")]
    result = report(protocol, rows)
    captures = {row["scenario"]: row for row in rows if row["record_type"] == "execution_capture"}
    costs = []
    pnl = []
    for scenario in result["execution"]["scenarios"]:
        decision = scenario["decisions"][0]
        capture = captures[scenario["name"]]
        snapshot = decision["execution_snapshot"]
        assert snapshot == {
            **capture["snapshot"],
            "snapshot_id": capture["record_id"],
            "source": "prospective-forward",
            "status": capture["snapshot"]["source_metadata"]["status"],
        }
        assert "status" not in capture["snapshot"]
        assert decision["execution_snapshot_ids"] == [capture["record_id"]]
        assert decision["state"] == "ENTER NO"
        assert decision["execution_price"] == 0.72
        assert decision["execution_snapshot"]["yes_bid"] == 0.28
        assert decision["execution_snapshot"]["no_ask_size"] == 1000
        assert decision["filled_contracts"] == 100
        assert decision["execution_latency_seconds"] >= scenario["configuration"]["latency_seconds"]
        costs.append(decision["cost_dollars"])
        pnl.append(decision["haircut_adjusted_pnl_dollars"])
    assert costs == sorted(costs)
    assert pnl == sorted(pnl, reverse=True)


def test_early_latency_capture_is_unavailable_without_borrowing_other_scenario(
    protocol: dict[str, Any],
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    adverse = next(
        row
        for row in rows
        if row["record_type"] == "execution_capture" and row["scenario"] == "adverse"
    )
    adverse["requested_at"] = (SLOT + timedelta(seconds=64)).isoformat()
    sealed(adverse)
    result = report(protocol, rows)
    scenarios = result["execution"]["scenarios"]
    assert scenarios[0]["decisions"][0]["status"] == "filled"
    assert scenarios[1]["decisions"][0]["status"] == "unavailable"
    assert scenarios[2]["decisions"][0]["status"] == "filled"


def test_each_fold_coverage_gate_cannot_be_hidden_by_overall_average(
    protocol: dict[str, Any],
) -> None:
    comparison = {
        "counts": {"resolved_events": 30, "resolved_dates": 30, "qualifying_validation_folds": 2},
        "paired_differences": [
            {
                "comparison": "blend_minus_market_yes_ask",
                "baseline_role": "primary",
                "metric": metric,
                "event_bootstrap": {"lower": -0.2, "upper": -0.1},
                "date_block_bootstrap": {"lower": -0.2, "upper": -0.1},
            }
            for metric in ("brier", "log_loss")
        ],
    }
    coverage = {
        "complete_schedule": True,
        "valid_capture_fraction": 0.96,
        "folds": [{"valid_capture_fraction": 1.0}, {"valid_capture_fraction": 0.92}],
    }
    conclusion = forward_report._probability_conclusion(comparison, coverage, protocol)
    assert conclusion["status"] == "inconclusive"
    coverage["folds"][1]["valid_capture_fraction"] = 0.95
    assert (
        forward_report._probability_conclusion(comparison, coverage, protocol)["status"]
        == "directional"
    )
    comparison["paired_differences"][1]["date_block_bootstrap"] = {"lower": -0.1, "upper": 0.1}
    assert (
        forward_report._probability_conclusion(comparison, coverage, protocol)["status"]
        == "inconclusive"
    )


def replace_execution_book(row: dict[str, Any], book: dict[str, Any]) -> None:
    from prediction_market_system.venues.kalshi import KalshiMarket

    venue = KalshiMarket.model_validate(row["snapshot"]["source_metadata"])
    row["raw_book"] = book
    row["raw_book_id"] = content_id(book)
    row["snapshot"] = to_market_snapshot(
        venue,
        KalshiOrderBook.model_validate(book),
        observed_at=datetime.fromisoformat(row["received_at"]),
    ).model_dump(mode="json")
    sealed(row)


def test_each_scenario_uses_its_own_observed_no_price_and_depth(protocol: dict[str, Any]) -> None:
    rows = [registration(protocol), *captured_slot(protocol, lower=61_000, result="no")]
    books = {
        "base": ("0.28", "1000", 0.72, 100),
        "adverse": ("0.29", "500", 0.71, 50),
        "severe": ("0.30", "250", 0.70, 25),
    }
    for row in rows:
        if row["record_type"] == "execution_capture":
            bid, depth, _, _ = books[row["scenario"]]
            replace_execution_book(
                row, {"yes_dollars": [[bid, depth]], "no_dollars": [["0.70", "1000"]]}
            )
    result = report(protocol, rows)
    assert not result["rejected_records"]
    for scenario in result["execution"]["scenarios"]:
        _, depth, price, contracts = books[scenario["name"]]
        decision = scenario["decisions"][0]
        assert decision["status"] == "filled"
        assert decision["execution_price"] == price
        assert decision["execution_snapshot"]["no_ask_size"] == int(depth)
        assert decision["filled_contracts"] == contracts


@pytest.mark.parametrize("field", ["status", "no_ask"])
def test_execution_adapter_cannot_accept_forged_snapshot_metadata(
    protocol: dict[str, Any], field: str
) -> None:
    rows = [registration(protocol), *captured_slot(protocol, lower=61_000, result="no")]
    for row in rows:
        if row["record_type"] == "execution_capture":
            if field == "status":
                row["snapshot"]["source_metadata"]["status"] = "closed"
            else:
                row["snapshot"]["no_ask"] = 0.01
            sealed(row)
    result = report(protocol, rows)
    for scenario in result["execution"]["scenarios"]:
        decision = scenario["decisions"][0]
        assert decision["status"] == "unavailable"
        assert decision["reason"] == "invalid_execution_capture"
        assert not scenario["summary"]["all_fills_resolved"]
        assert scenario["summary"]["total_cost_dollars"] is None


@pytest.mark.parametrize("reason", ["limit_exceeded", "insufficient_liquidity"])
def test_observed_price_or_depth_failure_is_explicit_nonfill(
    protocol: dict[str, Any],
    reason: str,
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    for row in rows:
        if row["record_type"] == "execution_capture":
            book = (
                {"yes_dollars": [["0.01", "1000"]], "no_dollars": [["0.01", "1000"]]}
                if reason == "limit_exceeded"
                else {"yes_dollars": [["0.28", "1000"]], "no_dollars": [["0.70", "1"]]}
            )
            replace_execution_book(row, book)
    result = report(protocol, rows)
    for scenario in result["execution"]["scenarios"]:
        decision = scenario["decisions"][0]
        assert decision["status"] == "nonfill"
        assert decision["reason"] == reason
        assert scenario["summary"]["unavailable_orders"] == 0
        assert scenario["summary"]["haircut_adjusted_pnl_dollars"] == 0


def test_watch_forecast_stays_in_same_probability_population(protocol: dict[str, Any]) -> None:
    rows = [
        registration(protocol),
        *captured_slot(
            protocol,
            book={"yes_dollars": [["0.01", "1000"]], "no_dollars": [["0.01", "1000"]]},
            execution=False,
        ),
    ]
    result = report(protocol, rows)
    assert result["probability"]["counts"]["watch"] == 1
    assert result["probability"]["counts"]["resolved"] == 1
    for scenario in result["execution"]["scenarios"]:
        assert scenario["decisions"][0]["status"] == "no-order"
        assert scenario["summary"]["unavailable_orders"] == 0


def test_multiple_ranges_are_one_independent_event_with_shared_exposure_cap(
    protocol: dict[str, Any],
) -> None:
    first = captured_slot(protocol)
    second = captured_slot(protocol, lower=61_000, result="no")
    selection = next(row for row in first if row["record_type"] == "selection")
    other = next(row for row in second if row["record_type"] == "selection")
    selection["discovered_candidates"] += other["discovered_candidates"]
    selection["selected_market_ids"] += other["selected_market_ids"]
    selection["selection_inputs_id"] = content_id(
        {
            key: selection[key]
            for key in ("discovered_candidates", "spot", "selected_event_id", "selected_market_ids")
        }
    )
    sealed(selection)
    combined = [
        *first,
        *(
            row
            for row in second
            if row["record_type"] not in {"slot_attempt", "selection", "slot_result"}
        ),
    ]
    for row in combined:
        if row["record_type"] == "market_capture":
            row["selection_inputs_id"] = selection["selection_inputs_id"]
            sealed(row)
        elif row["record_type"] == "slot_result":
            row["selected_market_ids"] = list(selection["selected_market_ids"])
            row["captured_market_ids"] = list(selection["selected_market_ids"])
            sealed(row)
    result = report(protocol, [registration(protocol), *combined])
    probability = result["probability"]
    assert probability["counts"]["eligible"] == 2
    assert probability["counts"]["resolved_events"] == 1
    assert probability["counts"]["resolved_dates"] == 1
    assert all(row["event_bootstrap"] is None for row in probability["paired_differences"])
    assert probability["conclusion"]["status"] == "inconclusive"
    for scenario in result["execution"]["scenarios"]:
        assert scenario["summary"]["total_cost_dollars"] <= 200


def test_missing_late_failed_no_market_and_interrupted_attempts_stay_in_denominator(
    protocol: dict[str, Any],
) -> None:
    rows = [registration(protocol)]
    for index, status in enumerate(("late", "failed", "no-market", "incomplete")):
        slot = SLOT + timedelta(hours=index)
        slot_id = content_id(
            {
                "protocol_id": protocol["protocol_id"],
                "scheduled_at": slot.isoformat(),
            }
        )
        common = {
            "slot_id": slot_id,
            "attempt_id": slot_id,
            "scheduled_at": slot.isoformat(),
        }
        started = slot + timedelta(seconds=61 if status == "late" else 0)
        rows.append(
            record(protocol, "slot_attempt", started, **common, started_at=started.isoformat())
        )
        if status == "no-market":
            selection = {
                "discovered_candidates": [],
                "spot": None,
                "selected_event_id": None,
                "selected_market_ids": [],
            }
            rows.append(
                record(
                    protocol,
                    "selection",
                    started + timedelta(seconds=1),
                    **common,
                    discovery_started_at=started.isoformat(),
                    discovery_received_at=started.isoformat(),
                    selection_at=started.isoformat(),
                    discovery_complete=True,
                    **selection,
                    selection_inputs_id=content_id(selection),
                    error=None,
                )
            )
        if status != "incomplete":
            rows.append(
                record(
                    protocol,
                    "slot_result",
                    started + timedelta(seconds=5),
                    **common,
                    completed_at=(started + timedelta(seconds=5)).isoformat(),
                    status=status,
                    selected_market_ids=[],
                    captured_market_ids=[],
                    error=None if status == "no-market" else "source unavailable",
                )
            )
    result = report(protocol, rows, SLOT + timedelta(hours=4, minutes=1))
    coverage = result["coverage"]
    assert coverage["elapsed_slots"] == 5
    assert coverage["future_slots"] == 1435
    assert coverage["valid_slots"] == 0
    assert coverage["missing_slots"] == 1
    assert coverage["attempted_slots"] == 4
    assert all(
        coverage[f"{state}_slots"] == 1
        for state in ("missing", "late", "failed", "no_market", "incomplete")
    )


def test_resolution_before_decision_cannot_score(protocol: dict[str, Any]) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    resolution = next(row for row in rows if row["record_type"] == "resolution")
    resolution["observed_at"] = SLOT.isoformat()
    sealed(resolution)
    result = report(protocol, rows)
    assert result["probability"]["counts"]["resolved"] == 0
    assert not result["execution"]["conclusion"]["directional_gate"]


def test_partial_slot_retains_valid_forecast_but_fails_slot_coverage(
    protocol: dict[str, Any],
) -> None:
    rows = captured_slot(protocol)
    selection = next(row for row in rows if row["record_type"] == "selection")
    other_selection = next(
        row for row in captured_slot(protocol, lower=61_000) if row["record_type"] == "selection"
    )
    selection["discovered_candidates"] += other_selection["discovered_candidates"]
    selection["selected_market_ids"] += other_selection["selected_market_ids"]
    selection["selection_inputs_id"] = content_id(
        {
            key: selection[key]
            for key in ("discovered_candidates", "spot", "selected_event_id", "selected_market_ids")
        }
    )
    sealed(selection)
    for row in rows:
        if row["record_type"] == "market_capture":
            row["selection_inputs_id"] = selection["selection_inputs_id"]
            sealed(row)
        elif row["record_type"] == "slot_result":
            row["selected_market_ids"] = list(selection["selected_market_ids"])
            row["status"] = "partial"
            row["error"] = "second selected market unavailable"
            sealed(row)
    result = report(protocol, [registration(protocol), *rows])
    assert result["coverage"]["partial_slots"] == 1
    assert result["coverage"]["valid_slots"] == 0
    assert result["probability"]["counts"]["eligible"] == 1


def test_conflicting_first_execution_books_are_unavailable_in_either_order(
    protocol: dict[str, Any],
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    original = next(
        row
        for row in rows
        if row["record_type"] == "execution_capture" and row["scenario"] == "base"
    )
    conflicting = copy.deepcopy(original)
    replace_execution_book(
        conflicting,
        {"yes_dollars": [["0.20", "1000"]], "no_dollars": [["0.70", "1000"]]},
    )
    forward = report(protocol, [*rows, conflicting])
    reverse = report(protocol, list(reversed([*rows, conflicting])))
    assert forward == reverse
    base = forward["execution"]["scenarios"][0]
    assert base["decisions"][0]["status"] == "unavailable"
    assert base["decisions"][0]["reason"] == "execution_snapshot_conflict"
    assert base["summary"]["haircut_adjusted_pnl_dollars"] is None
    assert forward["probability"]["counts"]["resolved"] == 1


def test_conflicting_capture_identity_cannot_select_surviving_good_hash(
    protocol: dict[str, Any],
) -> None:
    rows = [registration(protocol), *captured_slot(protocol)]
    capture = next(row for row in rows if row["record_type"] == "market_capture")
    corrupted = copy.deepcopy(capture)
    corrupted["raw_book"]["yes_dollars"][0][0] = "0.01"
    result = report(protocol, [*rows, corrupted])
    assert result["coverage"]["valid_slots"] == 0
    assert result["probability"]["counts"]["eligible"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("resolved_events", 19),
        ("resolved_dates", 19),
        ("qualifying_validation_folds", 1),
    ],
)
def test_sample_gates_remain_required_even_with_complete_capture_and_clear_intervals(
    protocol: dict[str, Any],
    key: str,
    value: int,
) -> None:
    counts = {"resolved_events": 20, "resolved_dates": 20, "qualifying_validation_folds": 2}
    counts[key] = value
    comparison = {
        "counts": counts,
        "paired_differences": [
            {
                "comparison": "blend_minus_market_yes_ask",
                "baseline_role": "primary",
                "metric": metric,
                "event_bootstrap": {"lower": -0.2, "upper": -0.1},
                "date_block_bootstrap": {"lower": -0.2, "upper": -0.1},
            }
            for metric in ("brier", "log_loss")
        ],
    }
    coverage = {
        "complete_schedule": True,
        "valid_capture_fraction": 1.0,
        "folds": [{"valid_capture_fraction": 1.0}, {"valid_capture_fraction": 1.0}],
    }
    conclusion = forward_report._probability_conclusion(comparison, coverage, protocol)
    assert conclusion["status"] == "inconclusive"
