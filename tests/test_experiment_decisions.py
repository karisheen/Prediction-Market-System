import copy
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

import pytest
from test_experiment import DAY, observation, rehash

from prediction_market_system.domain import CryptoSnapshot, MarketSnapshot, TerminalRangeContract
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import canonical_json, content_id
from prediction_market_system.experiment import (
    compare_observations,
    load_protocol,
    protocol_identity,
)
from prediction_market_system.experiment_decisions import assess_decisions
from prediction_market_system.experiment_inputs import load_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recorded() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = load_protocol(ROOT / "experiments/phase1/protocol-v2.json")
    observations, _ = load_fixture(ROOT / "experiments/phase1/fixture-v1.json", protocol)
    return protocol, observations


def case(recorded: tuple[dict[str, Any], list[dict[str, Any]]], name: str) -> dict[str, Any]:
    return copy.deepcopy(
        next(
            row
            for row in recorded[1]
            if row.get("run_manifest", {}).get("fixture", {}).get("case_id") == name
        )
    )


def comparison(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    return compare_observations(
        rows,
        protocol,
        source_kind="synthetic-fixture",
        as_of=datetime.fromisoformat(protocol["report_as_of"]),
    )


def assess(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    report = comparison(rows, protocol)
    assert report["counts"]["eligible"] == len({row["observation_id"] for row in rows})
    return assess_decisions(rows, report, protocol)


def snapshot(
    row: dict[str, Any], *, seconds: float = 30, ask: float = 0.1, size: float = 100
) -> dict[str, Any]:
    return {
        "snapshot_id": f"quote-{seconds}",
        "market_id": row["market_id"],
        "observed_at": (
            datetime.fromisoformat(row["observed_at"]) + timedelta(seconds=seconds)
        ).isoformat(),
        "source": "synthetic-fixture",
        "status": "active",
        "label": "research-only-boundary",
        "yes_bid": 0.0,
        "yes_ask": ask,
        "no_bid": 0.0,
        "no_ask": 0.7,
        "yes_ask_size": size,
        "no_ask_size": 100,
    }


def base(report: dict[str, Any]) -> dict[str, Any]:
    return report["scenarios"][0]["decisions"][0]


def test_real_fixture_keeps_probability_population_and_reports_both_execution_sides(
    recorded: Any,
) -> None:
    protocol, rows = recorded
    compared = comparison(rows, protocol)
    before = canonical_json(compared)
    raw_before = canonical_json(rows)
    report = assess_decisions(rows, compared, protocol)
    assert canonical_json(compared) == before
    assert canonical_json(rows) == raw_before
    assert report["comparison_sha256"] == content_id(compared)
    assert report["population_sha256"] == compared["identities"]["eligible_observations_sha256"]
    assert [scenario["name"] for scenario in report["scenarios"]] == ["base", "adverse", "severe"]
    eligible = {row["observation_id"] for row in compared["observations"]}
    for scenario in report["scenarios"]:
        assert {row["observation_id"] for row in scenario["decisions"]} == eligible
        assert scenario["configuration_id"] == content_id(scenario["configuration"])
        summary = scenario["summary"]
        assert summary["eligible_forecasts"] == 13
        assert summary["no_trade_frequency"] == (13 - summary["filled_orders"]) / 13
        for row in scenario["decisions"]:
            if row["state"] == "WATCH":
                assert row["status"] == "no-order"
                assert row["filled_contracts"] == 0
    by_id = {row["observation_id"]: row for row in report["scenarios"][0]["decisions"]}
    yes = by_id[case(recorded, "E1-A-T1")["observation_id"]]
    no = by_id[case(recorded, "E6-A-T1")["observation_id"]]
    assert (yes["side"], yes["execution_price"], yes["filled_contracts"]) == ("YES", 0.23, 4)
    assert (no["side"], no["execution_price"], no["filled_contracts"]) == ("NO", 0.6, 10)
    assert yes["fee_dollars"] == 0.05
    assert yes["slippage_dollars"] == 0.0023
    assert yes["cost_dollars"] == 0.9723
    assert no["cost_dollars"] == 6.185
    assert no["payout_dollars"] == 0
    assert no["raw_pnl_dollars"] == -6.185
    assert no["haircut_adjusted_pnl_dollars"] == -6.285
    assert report["conclusion"]["status"] == "inconclusive"


def test_small_order_fee_ceiling_and_largest_affordable_whole_quantity(recorded: Any) -> None:
    protocol = recorded[0]
    row = case(recorded, "E1-A-T1")
    row["execution"] = [snapshot(row, ask=0.1)]
    row["opportunity"]["suggested_max_exposure"] = 0.11025
    decision = base(assess([row], protocol))
    assert decision["requested_contracts"] == decision["filled_contracts"] == 1
    assert decision["fee_dollars"] == 0.01
    assert decision["cost_dollars"] == 0.11025
    assert decision["partial_fill"] is False
    row["opportunity"]["suggested_max_exposure"] = 0.110249
    decision = base(assess([row], protocol))
    assert decision["requested_contracts"] == decision["filled_contracts"] == 0
    assert decision["reason"] == "insufficient_liquidity"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"yes_ask": None}, "side_ask_unavailable"),
        ({"yes_ask_size": None}, "side_size_unavailable"),
        ({"yes_ask_size": 0}, "insufficient_liquidity"),
        ({"yes_ask_size": -1}, "invalid_execution_snapshot"),
        ({"yes_ask_size": "NaN"}, "invalid_execution_snapshot"),
        ({"yes_ask_size": "Infinity"}, "invalid_execution_snapshot"),
        ({"yes_bid": 0.4, "yes_ask": 0.2}, "invalid_execution_snapshot"),
        ({"no_bid": 0.9, "no_ask": 0.8}, "invalid_execution_snapshot"),
        ({"yes_ask": 1.1}, "invalid_execution_snapshot"),
        ({"yes_ask": True}, "invalid_execution_snapshot"),
        ({"yes_ask": 0.99}, "limit_exceeded"),
        ({"status": "closed"}, "inactive_execution_snapshot"),
        ({"status": "inactive"}, "inactive_execution_snapshot"),
    ],
)
def test_first_bad_quote_never_skips_to_favorable_later_quote(
    recorded: Any, change: Any, reason: str
) -> None:
    row = case(recorded, "E1-A-T1")
    first = {**snapshot(row), **change}
    row["execution"] = [snapshot(row, seconds=31, ask=0.01), first]
    decision = base(assess([row], recorded[0]))
    assert decision["status"] == "nonfill"
    assert decision["reason"] == reason
    assert decision["execution_snapshot"] == first
    assert decision["filled_contracts"] == 0


@pytest.mark.parametrize(
    "seconds,filled", [(29.999, False), (30, True), (150, True), (150.001, False)]
)
def test_latency_and_freshness_window_are_inclusive(
    recorded: Any, seconds: float, filled: bool
) -> None:
    row = case(recorded, "E1-A-T1")
    row["execution"] = [snapshot(row, seconds=seconds)]
    decision = base(assess([row], recorded[0]))
    assert decision["status"] == ("filled" if filled else "nonfill")
    if not filled:
        assert decision["reason"] == "no_execution_snapshot"


@pytest.mark.parametrize("seconds,filled", [(60, True), (60.001, False)])
def test_execution_cannot_cross_remaining_entry_horizon(
    recorded: Any, seconds: float, filled: bool
) -> None:
    row = observation("HORIZON", observed_at=DAY - timedelta(seconds=360))
    assert row["opportunity"]["state"] == "ENTER YES"
    row["execution"] = [snapshot(row, seconds=seconds)]
    decision = base(assess([row], recorded[0]))
    assert decision["status"] == ("filled" if filled else "nonfill")
    if not filled:
        assert decision["reason"] == "no_execution_snapshot"


@pytest.mark.parametrize("seconds,filled", [(60, True), (60.001, False)])
def test_execution_cannot_use_quote_after_report_cutoff(
    recorded: Any, seconds: float, filled: bool
) -> None:
    row = case(recorded, "E1-A-T1")
    protocol = copy.deepcopy(recorded[0])
    protocol["report_as_of"] = (
        datetime.fromisoformat(row["observed_at"]) + timedelta(seconds=60)
    ).isoformat()
    protocol["archive_end"] = protocol["report_as_of"]
    protocol["protocol_id"] = protocol_identity(protocol)
    row["execution"] = [snapshot(row, seconds=seconds)]
    decision = base(assess([row], protocol))
    assert decision["status"] == ("filled" if filled else "nonfill")
    assert decision["raw_pnl_dollars"] is None


def test_wrong_market_and_missing_snapshot_never_synthesize_fills(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    row["execution"] = [{**snapshot(row), "market_id": "another-market"}]
    decision = base(assess([row], recorded[0]))
    assert decision["reason"] == "no_execution_snapshot"
    assert decision["execution_snapshot"] is None
    row["execution"] = []
    assert base(assess([row], recorded[0]))["reason"] == "no_execution_snapshot"


def test_equal_time_conflicts_fail_closed_but_ids_alone_do_not(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    first = snapshot(row)
    second = {**first, "snapshot_id": "other-id"}
    row["execution"] = [second, first, copy.deepcopy(first)]
    decision = base(assess([row], recorded[0]))
    assert decision["status"] == "filled"
    assert decision["execution_snapshot_ids"] == ["other-id", "quote-30"]
    assert decision["execution_snapshots"] == sorted([first, second], key=canonical_json)
    second["yes_ask_size"] = 101
    decision = base(assess([row], recorded[0]))
    assert decision["reason"] == "execution_snapshot_conflict"
    row["execution"].reverse()
    assert base(assess([row], recorded[0])) == decision


@pytest.mark.parametrize("size,units", [(9.99, 0), (10, 1), (19.99, 1), (20, 2), (25.5, 2)])
def test_participation_floor_never_invents_fractional_contracts(
    recorded: Any, size: float, units: int
) -> None:
    row = case(recorded, "E1-A-T1")
    row["execution"] = [snapshot(row, size=size)]
    decision = base(assess([row], recorded[0]))
    assert decision["filled_contracts"] == units
    assert decision["partial_fill"] is (units > 0)
    if units == 0:
        assert decision["reason"] == "insufficient_liquidity"


def test_all_scenario_fee_multipliers_and_partial_sizes_are_reported(recorded: Any) -> None:
    row = case(recorded, "E3-B-T1")
    report = assess([row], recorded[0])
    assert [item["decisions"][0]["filled_contracts"] for item in report["scenarios"]] == [2, 1, 0]
    for item in report["scenarios"][:2]:
        decision = item["decisions"][0]
        units = decision["filled_contracts"]
        config = item["configuration"]
        fee = (
            Decimal(units)
            * Decimal("0.07")
            * Decimal(str(config["fee_multiplier"]))
            * Decimal("0.14")
            * Decimal("0.86")
        ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        slippage = Decimal(units) * Decimal("0.14") * Decimal(str(config["slippage_bps"])) / 10000
        assert decision["fee_dollars"] == float(fee)
        assert decision["cost_dollars"] == float(units * Decimal("0.14") + fee + slippage)
        assert decision["partial_fill"] is True


def test_event_budget_is_shared_and_processed_in_decision_id_order(recorded: Any) -> None:
    first = case(recorded, "E1-A-T1")
    first["observation_id"] = "a"
    first["inputs"]["market"]["yes_ask_size"] = 100000
    engine = CryptoThresholdEngine(EngineConfig.model_validate(first["inputs"]["engine_config"]))
    _, generated = engine.evaluate(
        MarketSnapshot.model_validate(first["inputs"]["market"]),
        CryptoSnapshot.model_validate(first["inputs"]["crypto"]),
        TerminalRangeContract.model_validate(first["inputs"]["contract"]),
    )
    assert generated.suggested_max_exposure >= 150.0
    first["opportunity"] = generated.model_dump(mode="json")
    first["inputs"] = first["opportunity"]["forecast"].pop("input_manifest")
    rehash(first)
    first["opportunity"]["suggested_max_exposure"] = 150.0
    first["execution"] = [snapshot(first, size=100000)]
    second = copy.deepcopy(first)
    second["observation_id"] = "b"
    second["market_id"] += "-other"
    second["opportunity"]["market"]["market_id"] = second["market_id"]
    second["opportunity"]["forecast"]["market_id"] = second["market_id"]
    second["inputs"]["market"]["market_id"] = second["market_id"]
    second["execution"] = [snapshot(second, size=100000)]
    rehash(second)
    rows = assess([second, first], recorded[0])["scenarios"][0]["decisions"]
    assert [row["observation_id"] for row in rows] == ["a", "b"]
    assert rows[0]["filled_contracts"] == rows[0]["requested_contracts"]
    assert 0 < rows[1]["filled_contracts"] < rows[1]["requested_contracts"]
    assert rows[1]["partial_fill"] is True
    total = sum(Decimal(str(row["cost_dollars"])) for row in rows)
    assert Decimal("199.8") < total <= Decimal("200")
    assert assess([first, second], recorded[0])["scenarios"][0]["decisions"] == rows


def test_rounded_one_contract_limit_rejects_edge_that_unrounded_fee_would_allow(
    recorded: Any,
) -> None:
    row = case(recorded, "E1-A-T1")
    probability = Decimal(str(row["opportunity"]["conservative_probability"]))
    # Solve the unrounded base cost just below the limit, then expose the cent ceiling gap.
    target = probability - Decimal("0.04")
    low, high = Decimal(0), Decimal(1)
    for _ in range(80):
        price = (low + high) / 2
        unrounded = price * Decimal("1.0025") + Decimal("0.07") * price * (1 - price)
        if unrounded < target - Decimal("0.000001"):
            low = price
        else:
            high = price
    row["execution"] = [snapshot(row, ask=float(low))]
    decision = base(assess([row], recorded[0]))
    assert probability - unrounded - Decimal("0.01") >= Decimal("0.03")
    assert decision["reason"] == "limit_exceeded"
    assert decision["effective_cost_per_contract"] > float(target)


def test_open_cost_is_separate_and_future_raw_label_never_becomes_a_loss(recorded: Any) -> None:
    resolved = case(recorded, "E1-A-T1")
    resolved["execution"] = [snapshot(resolved)]
    opened = case(recorded, "E4-A-T1")
    opened["resolution"] = None
    opened["execution"] = [snapshot(opened)]
    report = assess([opened, resolved], recorded[0])
    summary = report["scenarios"][0]["summary"]
    rows = {row["observation_id"]: row for row in report["scenarios"][0]["decisions"]}
    unresolved = rows[opened["observation_id"]]
    settled = rows[resolved["observation_id"]]
    assert unresolved["status"] == "filled"
    assert unresolved["resolution_status"] == "unresolved"
    assert unresolved["payout_dollars"] is unresolved["raw_pnl_dollars"] is None
    assert summary["open_cost_dollars"] == unresolved["cost_dollars"]
    assert summary["resolved_cost_dollars"] == settled["cost_dollars"]
    assert summary["raw_pnl_dollars"] == settled["raw_pnl_dollars"]
    assert summary["return_on_cost"] == settled["raw_pnl_dollars"] / settled["cost_dollars"]
    assert summary["total_cost_dollars"] == pytest.approx(
        summary["open_cost_dollars"] + summary["resolved_cost_dollars"]
    )
    compared = comparison([opened], recorded[0])
    opened["resolution"] = {
        "result": "no",
        "observed_at": "2026-09-26T01:01:00+00:00",
        "settlement_ts": None,
    }
    assert base(assess_decisions([opened], compared, recorded[0]))["raw_pnl_dollars"] is None


def test_zero_quote_and_zero_eligible_population_are_not_profitable_evidence(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    row["execution"] = [snapshot(row, ask=0)]
    decision = base(assess([row], recorded[0]))
    assert decision["reason"] == "unbounded_zero_price_quantity"
    assert decision["execution_snapshot"]["yes_ask"] == 0
    assert decision["filled_contracts"] == 0
    report = assess_decisions([], comparison([], recorded[0]), recorded[0])
    for scenario in report["scenarios"]:
        summary = scenario["summary"]
        assert summary["no_trade_frequency"] is None
        assert summary["recorded_watch_frequency"] is None
        assert summary["raw_pnl_dollars"] is None
        assert summary["haircut_adjusted_pnl_dollars"] is None
        assert summary["return_on_cost"] is None
        assert summary["total_cost_dollars"] == 0


@pytest.mark.parametrize(
    "field", ["input_id", "run_id", "run_manifest_sha256", "market_id", "event_id"]
)
def test_raw_population_identity_mismatch_is_rejected(recorded: Any, field: str) -> None:
    row = case(recorded, "E1-A-T1")
    compared = comparison([row], recorded[0])
    row[field] = "wrong-identity"
    with pytest.raises(ValueError):
        assess_decisions([row], compared, recorded[0])


@pytest.mark.parametrize("field", ["protocol_id", "as_of", "source_kind"])
def test_comparison_identity_mismatch_is_rejected(recorded: Any, field: str) -> None:
    row = case(recorded, "E1-A-T1")
    compared = comparison([row], recorded[0])
    compared[field] = "wrong-identity"
    with pytest.raises(ValueError):
        assess_decisions([row], compared, recorded[0])


def test_changed_inputs_quotes_and_population_hash_cannot_reuse_comparison(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    compared = comparison([row], recorded[0])
    changed = copy.deepcopy(row)
    changed["inputs"]["engine_config"]["slippage_bps"] += 1
    with pytest.raises(ValueError):
        assess_decisions([changed], compared, recorded[0])
    changed = copy.deepcopy(row)
    changed["execution"][0]["yes_ask"] += 0.01
    with pytest.raises(ValueError):
        assess_decisions([changed], compared, recorded[0])
    compared["observations"][0]["outcome_yes"] = 0
    with pytest.raises(ValueError):
        assess_decisions([row], compared, recorded[0])
    with pytest.raises(ValueError):
        assess_decisions([], {"observations": []}, recorded[0])


def test_exact_duplicate_keeps_one_and_conflicting_duplicate_cannot_reenter(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    duplicate = copy.deepcopy(row)
    compared = comparison([row, duplicate], recorded[0])
    report = assess_decisions([row, duplicate], compared, recorded[0])
    assert report["scenarios"][0]["summary"]["eligible_forecasts"] == 1
    duplicate["execution"][0]["yes_ask"] += 0.01
    with pytest.raises(ValueError):
        assess_decisions([row, duplicate], compared, recorded[0])
    conflicted = comparison([row, duplicate], recorded[0])
    assert conflicted["counts"]["eligible"] == 0
    assert (
        assess_decisions([row, duplicate], conflicted, recorded[0])["scenarios"][0]["decisions"]
        == []
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_json_finite_snapshot_cannot_reuse_a_bound_population(
    recorded: Any, value: float
) -> None:
    row = case(recorded, "E1-A-T1")
    compared = comparison([row], recorded[0])
    changed = copy.deepcopy(row)
    changed["execution"][0]["yes_ask_size"] = value
    with pytest.raises(ValueError):
        assess_decisions([changed], compared, recorded[0])


def test_recorded_exposure_cannot_exceed_reproduced_engine_allocation(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    compared = comparison([row], recorded[0])
    row["opportunity"]["suggested_max_exposure"] *= 2
    with pytest.raises(ValueError, match="reproduced engine recommendation"):
        assess_decisions([row], compared, recorded[0])


def test_engine_watch_cannot_be_promoted_to_a_recorded_entry(recorded: Any) -> None:
    row = case(recorded, "E1-B-T1")
    opportunity = row["opportunity"]
    assert opportunity["state"] == "WATCH"
    opportunity.update(
        state="ENTER YES",
        side="YES",
        executable_price=opportunity["market"]["yes_ask"],
        conservative_probability=opportunity["forecast"]["lower_probability_yes"],
        conservative_net_edge=0.0,
        suggested_max_exposure=1.0,
    )
    compared = comparison([row], recorded[0])
    assert compared["counts"]["eligible"] == 1
    with pytest.raises(ValueError, match="reproduced engine recommendation"):
        assess_decisions([row], compared, recorded[0])


def test_recorded_entry_cannot_switch_to_the_engine_suboptimal_side(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    opportunity = row["opportunity"]
    assert opportunity["state"] == "ENTER YES"
    opportunity.update(
        state="ENTER NO",
        side="NO",
        executable_price=opportunity["market"]["no_ask"],
        conservative_probability=1.0 - opportunity["forecast"]["upper_probability_yes"],
    )
    compared = comparison([row], recorded[0])
    assert compared["counts"]["eligible"] == 1
    with pytest.raises(ValueError, match="reproduced engine recommendation"):
        assess_decisions([row], compared, recorded[0])


def test_downward_recorded_allocation_remains_the_execution_budget(recorded: Any) -> None:
    row = case(recorded, "E1-A-T1")
    row["execution"] = [snapshot(row, ask=0.1)]
    assert row["opportunity"]["suggested_max_exposure"] > 0.11025
    compared = comparison([row], recorded[0])
    before = canonical_json(compared)
    row["opportunity"]["suggested_max_exposure"] = 0.11025
    report = assess_decisions([row], compared, recorded[0])
    assert canonical_json(compared) == before
    decision = base(report)
    assert decision["suggested_max_exposure"] == 0.11025
    assert decision["requested_contracts"] == decision["filled_contracts"] == 1
    assert decision["cost_dollars"] == 0.11025


@pytest.mark.parametrize("clear_side", [False, True])
def test_recorded_watch_downgrade_is_never_upgraded_by_replay(
    recorded: Any, clear_side: bool
) -> None:
    row = case(recorded, "E1-A-T1")
    row["execution"] = [snapshot(row, ask=0.01)]
    opportunity = row["opportunity"]
    assert opportunity["state"] == "ENTER YES"
    opportunity.update(state="WATCH", suggested_max_exposure=0.0)
    if clear_side:
        opportunity.update(
            side=None,
            executable_price=None,
            conservative_probability=None,
            conservative_net_edge=None,
        )
    compared = comparison([row], recorded[0])
    before = canonical_json(compared)
    report = assess_decisions([row], compared, recorded[0])
    assert canonical_json(compared) == before
    for scenario in report["scenarios"]:
        decision = scenario["decisions"][0]
        assert decision["status"] == "no-order"
        assert decision["filled_contracts"] == 0
        assert scenario["summary"]["recorded_entries"] == 0
        assert scenario["summary"]["recorded_watch_frequency"] == 1.0
