"""Recorded-order cost sensitivity, separate from probability evidence and execution."""

from __future__ import annotations

import copy
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from prediction_market_system.calibration import UncertaintyCalibrationProfile
from prediction_market_system.domain import CryptoSnapshot, Opportunity, TerminalRangeContract
from prediction_market_system.engine import (
    CryptoThresholdEngine,
    EngineConfig,
    deployment_policy_id,
    entry_horizon_at,
)
from prediction_market_system.evidence import canonical_json, content_id
from prediction_market_system.experiment import REPORT_SCHEMA, protocol_identity

_QUOTES = ("yes_bid", "yes_ask", "no_bid", "no_ask", "yes_ask_size", "no_ask_size")
_ENTRIES = {"ENTER YES": "YES", "ENTER NO": "NO"}


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("decision timestamps must be timezone-aware strings")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decision timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _population(
    observations: list[dict[str, Any]], comparison: dict[str, Any], protocol: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any], Opportunity]]:
    """Bind to the already selected rows, without selecting or scoring a new population."""
    try:
        identities = comparison["identities"]
        rows = comparison["observations"]
        source = comparison["source_kind"]
        cutoff = _timestamp(protocol["report_as_of"])
        if (
            protocol["protocol_version"] != "pms-phase1-v2"
            or protocol["protocol_id"] != protocol_identity(protocol)
            or comparison["report_schema"] != REPORT_SCHEMA
            or comparison["protocol_id"] != protocol["protocol_id"]
            or identities["protocol_id"] != protocol["protocol_id"]
            or comparison["protocol_version"] != protocol["protocol_version"]
            or identities["protocol_version"] != protocol["protocol_version"]
            or source not in {"synthetic-fixture", "retrospective-local"}
            or identities["source_kind"] != source
            or any(
                _timestamp(value) != cutoff
                for value in (
                    comparison["as_of"],
                    identities["as_of"],
                    identities["protocol_report_as_of"],
                )
            )
            or not isinstance(rows, list)
            or identities["eligible_observations_sha256"] != content_id(rows)
            or comparison["counts"]["eligible"] != len(rows)
        ):
            raise ValueError("comparison protocol, cutoff or population identity mismatch")
        for key, field in (
            ("run_ids", "run_id"),
            ("input_ids", "input_id"),
            ("run_manifest_sha256", "run_manifest_sha256"),
            ("recipe_ids", "recipe_id"),
        ):
            if identities[key] != sorted({row[field] for row in rows}):
                raise ValueError("comparison population identities mismatch")
        by_id: dict[str, list[dict[str, Any]]] = {}
        for observation in observations:
            by_id.setdefault(observation["observation_id"], []).append(observation)
        seen: set[str] = set()
        population = []
        for row in rows:
            identifier = row["observation_id"]
            if identifier in seen:
                raise ValueError("duplicate comparison observation")
            seen.add(identifier)
            copies = by_id[identifier]
            if len({content_id(item) for item in copies}) != 1:
                raise ValueError("conflicting raw observation for comparison population")
            raw = copies[0]
            inputs, manifest = raw["inputs"], raw["run_manifest"]
            opportunity = Opportunity.model_validate(raw["opportunity"])
            market, forecast = opportunity.market, opportunity.forecast
            if (
                raw.get("exclusion_reason") is not None
                or any(
                    raw[key] != row[key]
                    for key in (
                        "event_id",
                        "market_id",
                        "run_id",
                        "input_id",
                        "run_manifest_sha256",
                    )
                )
                or content_id(inputs) != row["input_id"]
                or content_id(manifest) != row["run_manifest_sha256"]
                or manifest["run_id"] != row["run_id"]
                or manifest["inputs"]["input_id"] != row["input_id"]
                or manifest["kind"]
                != ("synthetic-fixture" if source == "synthetic-fixture" else "forward-evaluation")
                or market.model_dump(mode="json") != inputs["market"]
                or market.market_id != row["market_id"]
                or market.event_id != row["event_id"]
                or _timestamp(raw["observed_at"]) != _timestamp(row["observed_at"])
                or market.observed_at != _timestamp(row["observed_at"])
                or forecast.generated_at != market.observed_at
                or _timestamp(manifest["recorded_at"]) != market.observed_at
                or market.expires_at != _timestamp(row["expires_at"])
                or market.effective_observation_end_at != _timestamp(row["observation_end_at"])
                or opportunity.state.value != row["state"]
                or forecast.recipe_id != row["recipe_id"]
                or forecast.structural_probability_yes != row["structural_probability_yes"]
                or forecast.probability_yes != row["blend_probability_yes"]
                or forecast.market_probability_yes
                != row["recorded_midpoint_anchor_probability_yes"]
                or market.yes_ask != row["market_yes_ask"]
                or market.yes_ask_size != row["market_yes_ask_size"]
                or raw.get("execution", []) != row["execution"]
            ):
                raise ValueError("raw observation does not match comparison population")
            outcome = row["outcome_yes"]
            if (outcome is not None and (type(outcome) is not int or outcome not in (0, 1))) or row[
                "resolution_status"
            ] != ("unresolved" if outcome is None else "resolved"):
                raise ValueError("invalid comparison resolution")
            if outcome is not None:
                resolved_at = _timestamp(row["resolution_observed_at"])
                if not market.observed_at < resolved_at <= cutoff or (
                    row["settlement_ts"] is not None and _timestamp(row["settlement_ts"]) > cutoff
                ):
                    raise ValueError("comparison resolution outside as-of boundary")
            if row["state"] in _ENTRIES:
                expected_side = _ENTRIES[row["state"]]
                conservative = (
                    forecast.lower_probability_yes
                    if expected_side == "YES"
                    else 1.0 - forecast.upper_probability_yes
                )
                if (
                    opportunity.side != expected_side
                    or opportunity.conservative_probability != conservative
                    or not math.isfinite(opportunity.suggested_max_exposure)
                ):
                    raise ValueError("recorded entry side or conservative probability mismatch")
                _validate_recorded_entry(opportunity, inputs, row["input_id"])
            population.append((raw, row, opportunity))
        return sorted(
            population, key=lambda item: (item[2].market.observed_at, item[1]["observation_id"])
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("malformed comparison or raw population") from error


def _validate_recorded_entry(
    opportunity: Opportunity, inputs: dict[str, Any], input_id: str
) -> None:
    """Permit recorded allocation reductions, never manufactured or upgraded entries."""
    profile_payload = inputs.get("calibration_profile")
    profile = (
        None
        if profile_payload is None
        else UncertaintyCalibrationProfile.model_validate(profile_payload)
    )
    reproduced_forecast, reproduced = CryptoThresholdEngine(
        EngineConfig.model_validate(inputs["engine_config"])
    ).evaluate(
        opportunity.market,
        CryptoSnapshot.model_validate(inputs["crypto"]),
        TerminalRangeContract.model_validate(inputs["contract"]),
        profile,
    )
    conservative = opportunity.conservative_probability
    reproduced_conservative = reproduced.conservative_probability
    if (
        content_id(reproduced_forecast.input_manifest) != input_id
        or reproduced.state != opportunity.state
        or reproduced.side != opportunity.side
        or conservative is None
        or reproduced_conservative is None
        or not math.isclose(conservative, reproduced_conservative, rel_tol=0.0, abs_tol=1e-9)
        or opportunity.suggested_max_exposure > reproduced.suggested_max_exposure + 1e-9
    ):
        raise ValueError(
            "recorded entry exceeds or disagrees with reproduced engine recommendation"
        )


def _scenario_config(protocol: dict[str, Any], scenario: dict[str, Any]) -> EngineConfig:
    costs = protocol["decision_costs"]
    return EngineConfig(
        **{
            key: costs[key]
            for key in (
                "paper_bankroll",
                "min_conservative_edge",
                "uncertainty_margin",
                "fee_rate",
                "minimum_ask_size",
                "fractional_kelly",
                "max_bankroll_fraction",
                "max_event_bankroll_fraction",
                "minimum_seconds_to_expiry",
            )
        },
        structural_weight=protocol["structural_weight"],
        maximum_input_age_seconds=protocol["maximum_input_age_seconds"],
        binary_fee_type=costs["fee_type"],
        binary_fee_coefficient=float(
            Decimal(str(costs["fee_coefficient"])) * Decimal(str(scenario["fee_multiplier"]))
        ),
        slippage_bps=scenario["slippage_bps"],
        resolution_haircut=scenario["resolution_haircut"],
    )


def _snapshots(
    raw: dict[str, Any],
    opportunity: Opportunity,
    scenario: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    start = opportunity.market.observed_at + timedelta(seconds=scenario["latency_seconds"])
    end = min(
        start + timedelta(seconds=protocol["maximum_input_age_seconds"]),
        entry_horizon_at(opportunity.market)
        - timedelta(seconds=protocol["decision_costs"]["minimum_seconds_to_expiry"]),
        _timestamp(protocol["report_as_of"]),
    )
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for snapshot in raw.get("execution", []):
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("market_id") != opportunity.market.market_id
        ):
            continue
        try:
            timestamp = _timestamp(snapshot.get("observed_at"))
        except ValueError:
            continue
        if opportunity.market.observed_at < timestamp and start <= timestamp <= end:
            candidates.append((timestamp, snapshot))
    if not candidates:
        return []
    earliest = min(timestamp for timestamp, _ in candidates)
    unique = {
        canonical_json(snapshot): snapshot
        for timestamp, snapshot in candidates
        if timestamp == earliest
    }
    return [copy.deepcopy(unique[key]) for key in sorted(unique)]


def _quote_error(snapshot: dict[str, Any], side: str) -> str | None:
    for key in _QUOTES:
        value = snapshot.get(key)
        if value is not None and (
            not _finite(value) or value < 0 or (not key.endswith("size") and value > 1)
        ):
            return "invalid_execution_snapshot"
    for prefix in ("yes", "no"):
        bid, ask = snapshot.get(f"{prefix}_bid"), snapshot.get(f"{prefix}_ask")
        if bid is not None and ask is not None and bid > ask:
            return "invalid_execution_snapshot"
    if snapshot.get("status") not in {"active", "open"}:
        return "inactive_execution_snapshot"
    if snapshot.get(f"{side}_ask") is None:
        return "side_ask_unavailable"
    if snapshot.get(f"{side}_ask_size") is None:
        return "side_size_unavailable"
    if snapshot[f"{side}_ask_size"] == 0:
        return "insufficient_liquidity"
    return None


def _affordable(engine: CryptoThresholdEngine, ask: float, budget: Decimal, upper: int) -> int:
    lower = 0
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if Decimal(str(engine.all_in_cost(ask, middle, round_fee=True))) <= budget:
            lower = middle
        else:
            upper = middle - 1
    return lower


def _decision(
    raw: dict[str, Any],
    row: dict[str, Any],
    opportunity: Opportunity,
    scenario: dict[str, Any],
    protocol: dict[str, Any],
    engine: CryptoThresholdEngine,
    event_spend: dict[str, Decimal],
    market_spend: dict[str, Decimal],
) -> dict[str, Any]:
    recorded_config = EngineConfig.model_validate(raw["inputs"]["engine_config"])
    result: dict[str, Any] = {
        key: row[key] for key in ("observation_id", "event_id", "market_id", "observed_at", "state")
    }
    result.update(
        side=opportunity.side.value if opportunity.side else None,
        status="no-order",
        reason="recorded_no_order",
        requested_contracts=0,
        filled_contracts=0,
        partial_fill=False,
        executed_at=None,
        execution_price=None,
        fee_dollars=0.0,
        slippage_dollars=0.0,
        cost_dollars=0.0,
        payout_dollars=None,
        raw_pnl_dollars=None,
        haircut_adjusted_pnl_dollars=None,
        haircut_adjusted_payout_dollars=None,
        resolution_status=row["resolution_status"],
        outcome_yes=row["outcome_yes"],
        decision_quotes={key: raw["opportunity"]["market"].get(key) for key in _QUOTES},
        execution_snapshot=None,
        execution_snapshots=[],
        execution_snapshot_ids=[],
        configured_latency_seconds=scenario["latency_seconds"],
        execution_latency_seconds=None,
        spread_dollars_per_contract=None,
        effective_cost_per_contract=None,
        conservative_probability=opportunity.conservative_probability,
        suggested_max_exposure=opportunity.suggested_max_exposure,
        resolution_haircut_per_contract=scenario["resolution_haircut"],
        recorded_engine_config=copy.deepcopy(raw["inputs"]["engine_config"]),
        recorded_engine_config_id=content_id(raw["inputs"]["engine_config"]),
        recorded_deployment_policy_id=deployment_policy_id(recorded_config),
        run_id=row["run_id"],
        input_id=row["input_id"],
        run_manifest_sha256=row["run_manifest_sha256"],
    )
    if row["state"] not in _ENTRIES:
        return result
    result.update(status="nonfill", reason="no_execution_snapshot")
    snapshots = _snapshots(raw, opportunity, scenario, protocol)
    if not snapshots:
        return result
    snapshot = snapshots[0]
    result.update(
        execution_snapshot=snapshot,
        execution_snapshots=snapshots,
        execution_snapshot_ids=sorted(
            {item["snapshot_id"] for item in snapshots if isinstance(item.get("snapshot_id"), str)}
        ),
        execution_latency_seconds=(
            _timestamp(snapshot["observed_at"]) - opportunity.market.observed_at
        ).total_seconds(),
    )
    # IDs/source labels alone cannot change economics. Status disagreements fail closed too.
    economics = {
        canonical_json({key: item.get(key) for key in (*_QUOTES, "status")}) for item in snapshots
    }
    if len(economics) != 1:
        result["reason"] = "execution_snapshot_conflict"
        return result
    side = _ENTRIES[row["state"]].lower()
    error = _quote_error(snapshot, side)
    if error:
        result["reason"] = error
        return result
    ask = float(snapshot[f"{side}_ask"])
    bid = snapshot.get(f"{side}_bid")
    result["spread_dollars_per_contract"] = None if bid is None else ask - bid
    effective_cost = engine.all_in_cost(ask, 1, round_fee=True)
    result["effective_cost_per_contract"] = effective_cost
    conservative = opportunity.conservative_probability
    assert conservative is not None
    if Decimal(str(conservative)) - Decimal(str(effective_cost)) - Decimal(
        str(scenario["resolution_haircut"])
    ) < Decimal(str(engine.config.min_conservative_edge)):
        result["reason"] = "limit_exceeded"
        return result
    if ask == 0:
        result["reason"] = "unbounded_zero_price_quantity"
        return result
    budget = Decimal(str(opportunity.suggested_max_exposure))
    intended = _affordable(engine, ask, budget, int(budget / Decimal(str(ask))))
    result["requested_contracts"] = intended
    participation = int(
        Decimal(str(protocol["decision_costs"]["max_volume_participation"]))
        * Decimal(str(snapshot[f"{side}_ask_size"]))
    )
    event_id, market_id = row["event_id"], row["market_id"]
    event_limit = Decimal(str(engine.config.paper_bankroll)) * Decimal(
        str(engine.config.max_event_bankroll_fraction)
    )
    market_limit = Decimal(str(engine.config.paper_bankroll)) * Decimal(
        str(engine.config.max_bankroll_fraction)
    )
    remaining = max(
        Decimal(0),
        min(
            budget,
            event_limit - event_spend.get(event_id, Decimal(0)),
            market_limit - market_spend.get(market_id, Decimal(0)),
        ),
    )
    filled = _affordable(engine, ask, remaining, min(intended, participation))
    if filled == 0:
        result["reason"] = "insufficient_liquidity"
        return result
    cost = Decimal(str(engine.all_in_cost(ask, filled, round_fee=True)))
    notional = Decimal(str(ask)) * filled
    slippage = notional * Decimal(str(engine.config.slippage_bps)) / 10_000
    fee = cost - notional - slippage
    event_spend[event_id] = event_spend.get(event_id, Decimal(0)) + cost
    market_spend[market_id] = market_spend.get(market_id, Decimal(0)) + cost
    result.update(
        status="filled",
        reason="partial_fill" if filled < intended else "filled",
        filled_contracts=filled,
        partial_fill=filled < intended,
        executed_at=snapshot["observed_at"],
        execution_price=ask,
        fee_dollars=float(fee),
        slippage_dollars=float(slippage),
        cost_dollars=float(cost),
    )
    outcome = row["outcome_yes"]
    if outcome is not None:
        won = outcome == (1 if side == "yes" else 0)
        payout = Decimal(filled if won else 0)
        adjusted = payout - filled * Decimal(str(scenario["resolution_haircut"]))
        result.update(
            payout_dollars=float(payout),
            haircut_adjusted_payout_dollars=float(adjusted),
            raw_pnl_dollars=float(payout - cost),
            haircut_adjusted_pnl_dollars=float(adjusted - cost),
        )
    return result


def _summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    filled = [row for row in decisions if row["status"] == "filled"]
    resolved = [row for row in filled if row["resolution_status"] == "resolved"]
    opened = [row for row in filled if row["resolution_status"] == "unresolved"]
    nonfills = [row for row in decisions if row["status"] == "nonfill"]
    count = len(decisions)

    def total(rows: list[dict[str, Any]], key: str) -> float:
        return float(sum((Decimal(str(row[key])) for row in rows), Decimal(0)))

    resolved_cost = total(resolved, "cost_dollars")
    raw_pnl = total(resolved, "raw_pnl_dollars") if resolved else None
    return {
        "eligible_forecasts": count,
        "recorded_entries": sum(row["state"] in _ENTRIES for row in decisions),
        "recorded_no_orders": sum(row["status"] == "no-order" for row in decisions),
        "filled_orders": len(filled),
        "partial_fill_orders": sum(row["partial_fill"] for row in filled),
        "nonfill_orders": len(nonfills),
        "nonfill_reasons": dict(sorted(Counter(row["reason"] for row in nonfills).items())),
        "no_trade_count": count - len(filled),
        "no_trade_frequency": (count - len(filled)) / count if count else None,
        "recorded_watch_frequency": (
            sum(row["state"] == "WATCH" for row in decisions) / count if count else None
        ),
        "filled_contracts": sum(row["filled_contracts"] for row in filled),
        "filled_resolved_orders": len(resolved),
        "filled_unresolved_orders": len(opened),
        "total_cost_dollars": total(filled, "cost_dollars"),
        "resolved_cost_dollars": resolved_cost,
        "open_cost_dollars": total(opened, "cost_dollars"),
        "fee_dollars": total(filled, "fee_dollars"),
        "slippage_dollars": total(filled, "slippage_dollars"),
        "payout_dollars": total(resolved, "payout_dollars") if resolved else None,
        "haircut_adjusted_payout_dollars": (
            total(resolved, "haircut_adjusted_payout_dollars") if resolved else None
        ),
        "raw_pnl_dollars": raw_pnl,
        "haircut_adjusted_pnl_dollars": (
            total(resolved, "haircut_adjusted_pnl_dollars") if resolved else None
        ),
        "return_on_cost": raw_pnl / resolved_cost
        if raw_pnl is not None and resolved_cost
        else None,
    }


def assess_decisions(
    observations: list[dict[str, Any]], comparison: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Assess every frozen scenario for recorded entries in the exact comparison population."""
    population = _population(observations, comparison, protocol)
    scenarios = []
    for scenario in protocol["cost_scenarios"]:
        config = _scenario_config(protocol, scenario)
        configuration = copy.deepcopy(
            {
                **scenario,
                "decision_costs": protocol["decision_costs"],
                "maximum_input_age_seconds": protocol["maximum_input_age_seconds"],
                "execution": protocol["execution"],
                "cost_rules": protocol["cost_rules"],
                "engine_config": config.model_dump(mode="json"),
                "deployment_policy_id": deployment_policy_id(config),
            }
        )
        engine = CryptoThresholdEngine(config)
        event_spend: dict[str, Decimal] = {}
        market_spend: dict[str, Decimal] = {}
        decisions = [
            _decision(raw, row, opportunity, scenario, protocol, engine, event_spend, market_spend)
            for raw, row, opportunity in population
        ]
        scenarios.append(
            {
                "name": scenario["name"],
                "configuration": configuration,
                "configuration_id": content_id(configuration),
                "summary": _summary(decisions),
                "decisions": decisions,
            }
        )
    return {
        "schema": "pms-phase1-decisions-v1",
        "protocol_id": protocol["protocol_id"],
        "source_kind": comparison["source_kind"],
        "comparison_sha256": content_id(comparison),
        "population_sha256": comparison["identities"]["eligible_observations_sha256"],
        "as_of": comparison["as_of"],
        "interpretation": (
            "paper decision sensitivity; separate from forecast-quality evidence; "
            "not execution/approval"
        ),
        "scenarios": scenarios,
        "conclusion": {
            "status": "inconclusive",
            "most_adverse_scenario": scenarios[-1]["name"] if scenarios else None,
            "basis": (
                "All frozen scenarios, including the most adverse, are descriptive recorded-order "
                "cost sensitivity only; neither profitability nor execution nor unseen-event "
                "alpha is established."
            ),
        },
        "limitations": [
            (
                "Later quotes are what-if paper execution inputs, "
                "not observed fills or an execution approval."
            ),
            (
                "Recorded WATCH rows remain in probability evidence "
                "and create no order in every scenario."
            ),
            "No fitted, recalibrated, optimized or outcome-selected decision policy is introduced.",
            (
                "Unresolved filled orders remain open and unscored; "
                "settled returns exclude their costs."
            ),
            "Resolution-haircut-adjusted PNL is stress accounting, not actual venue PNL.",
            (
                "No scenario changes the comparison population, "
                "probabilities or probability-quality metrics."
            ),
            (
                "A zero ask has no largest budget-affordable integer quantity "
                "and is an explicit nonfill."
            ),
        ],
    }
