"""Embargoed, read-only reports for the separately preregistered forward study.

Storage proves source capture; the explicit experiment adapters reuse the existing
row/reproduction, event-weighted statistics and paper-cost implementations. Nothing
here grants approval, sends notifications, or changes the original campaign.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from prediction_market_system.evidence import content_id
from prediction_market_system.experiment import compare_forward_observations
from prediction_market_system.experiment_decisions import assess_forward_decisions
from prediction_market_system.forward_protocol import scheduled_slots, validate_forward_protocol
from prediction_market_system.forward_storage import ForwardStudyStore, validate_forward_record

_STATUSES = ("valid", "missing", "late", "failed", "partial", "no-market", "incomplete")
_ORDER = {
    "registration": 0,
    "slot_attempt": 1,
    "selection": 2,
    "market_capture": 3,
    "execution_capture": 4,
    "slot_result": 5,
    "resolution": 6,
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("forward evidence clock must be an offset-aware timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("forward evidence clock must be an offset-aware timestamp")
    return parsed.astimezone(UTC)


def _identity(record: dict[str, Any]) -> tuple[Any, ...]:
    kind = record["record_type"]
    if kind == "registration":
        return (kind, record["protocol_id"])
    if kind == "slot_attempt":
        return (kind, _time(record["scheduled_at"]).isoformat())
    if kind in {"selection", "slot_result"}:
        return (kind, record["attempt_id"])
    if kind == "market_capture":
        return (kind, record["attempt_id"], record["market_id"])
    if kind == "execution_capture":
        return (kind, record["observation_id"], record["scenario"])
    if kind == "resolution":
        return (kind, record["market_id"], _time(record["observed_at"]).isoformat())
    raise ValueError("unknown forward record type")


def _admissible_records(
    records: list[dict[str, Any]], protocol: dict[str, Any], cutoff: datetime, final: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Deduplicate before dependency validation; conflicting identities never win by order."""
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    rejected: list[dict[str, Any]] = []
    resolution_conflicts: set[str] = set()
    tainted: set[tuple[Any, ...]] = set()

    def reject(record: dict[str, Any], reason: str) -> None:
        rejected.append(
            {
                "record_id": record.get("record_id"),
                "record_type": record.get("record_type"),
                "reason": reason,
                **{key: record[key] for key in ("observation_id", "scenario") if key in record},
            }
        )

    for record in records:
        # The embargo branch never accesses resolution contents, even for diagnostics.
        if not final and record.get("record_type") == "resolution":
            continue
        identity: tuple[Any, ...] | None = None
        try:
            if _time(record["persisted_at"]) > cutoff:
                continue
            if record.get("protocol_id") != protocol["protocol_id"]:
                raise ValueError("wrong forward study identity")
            identity = _identity(record)
            digest = content_id({key: value for key, value in record.items() if key != "record_id"})
            if digest != record.get("record_id"):
                raise ValueError("record hash mismatch")
            grouped[identity][digest] = record
        except (KeyError, TypeError, ValueError) as error:
            reject(record, str(error))
            if identity is not None:
                tainted.add(identity)
            if final and record.get("record_type") == "resolution":
                market_id = record.get("market_id")
                if isinstance(market_id, str):
                    resolution_conflicts.add(market_id)
    unique = []
    for identity, copies in grouped.items():
        if identity in tainted or len(copies) != 1:
            for record in copies.values():
                reject(record, "conflicting natural identity")
                if record["record_type"] == "resolution":
                    resolution_conflicts.add(record["market_id"])
        else:
            unique.append(next(iter(copies.values())))
    unique.sort(
        key=lambda row: (_time(row["persisted_at"]), _ORDER[row["record_type"]], row["record_id"])
    )
    accepted: list[dict[str, Any]] = []
    for record in unique:
        try:
            validate_forward_record(record, protocol, accepted)
        except (KeyError, TypeError, ValueError) as error:
            reject(record, str(error))
            if record["record_type"] == "resolution":
                resolution_conflicts.add(record["market_id"])
        else:
            accepted.append(record)
    rejected.sort(key=lambda row: (str(row["record_type"]), str(row["record_id"]), row["reason"]))
    return accepted, rejected, resolution_conflicts


def _coverage(
    records: list[dict[str, Any]], protocol: dict[str, Any], cutoff: datetime
) -> dict[str, Any]:
    attempts = {
        _time(row["scheduled_at"]): row for row in records if row["record_type"] == "slot_attempt"
    }
    selections = {row["attempt_id"]: row for row in records if row["record_type"] == "selection"}
    results = {row["attempt_id"]: row for row in records if row["record_type"] == "slot_result"}
    captures: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["record_type"] == "market_capture" and row["status"] == "valid":
            captures[row["attempt_id"]].add(row["market_id"])
    slots = scheduled_slots(protocol)
    slot_rows = []
    for scheduled in slots:
        if scheduled > cutoff:
            continue
        attempt = attempts.get(scheduled)
        state = "missing"
        if attempt is not None:
            identifier = attempt["attempt_id"]
            started = _time(attempt["started_at"])
            selection = selections.get(identifier)
            result = results.get(identifier)
            if not scheduled <= started <= scheduled + timedelta(seconds=60):
                state = "late"
            elif result is None:
                state = "incomplete"
            else:
                state = result["status"]
                selected = set(selection["selected_market_ids"]) if selection else set()
                if state == "valid" and not (
                    selection
                    and selection["discovery_complete"]
                    and selected
                    and captures[identifier] == selected
                    and set(result["selected_market_ids"]) == selected
                    and set(result["captured_market_ids"]) == selected
                ):
                    state = "partial" if captures[identifier] else "failed"
            if state not in _STATUSES:
                state = "failed"
        slot_rows.append(
            {
                "scheduled_at": scheduled.isoformat(),
                "status": state,
                "attempted": attempt is not None,
            }
        )

    def summarize(expected: list[datetime], elapsed: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["status"] for row in elapsed)
        count = len(elapsed)
        return {
            "expected_slots": len(expected),
            "elapsed_slots": count,
            "future_slots": len(expected) - count,
            "attempted_slots": sum(row["attempted"] for row in elapsed),
            **{f"{status.replace('-', '_')}_slots": counts[status] for status in _STATUSES},
            "valid_capture_fraction": counts["valid"] / count if count else None,
            "complete_schedule": count == len(expected),
        }

    coverage = summarize(slots, slot_rows)
    folds = []
    for fold in protocol["folds"]:
        start, end = _time(fold["start"]), _time(fold["end"])
        expected = [slot for slot in slots if start <= slot < end]
        elapsed = [row for row in slot_rows if start <= _time(row["scheduled_at"]) < end]
        folds.append({"name": fold["name"], **summarize(expected, elapsed)})
    coverage["folds"] = folds
    coverage["slots"] = slot_rows
    return coverage


def _probability_conclusion(
    comparison: dict[str, Any], coverage: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    minimum = protocol["coverage"]["minimum_valid_capture_fraction"]
    coverage_gate = coverage["complete_schedule"] and all(
        row["valid_capture_fraction"] is not None and row["valid_capture_fraction"] >= minimum
        for row in [coverage, *coverage["folds"]]
    )
    counts = comparison["counts"]
    gates = {
        "capture_coverage": coverage_gate,
        "resolved_events": counts["resolved_events"] >= protocol["minimum_resolved_events"],
        "resolved_dates": counts["resolved_dates"] >= protocol["minimum_dates"],
        "qualifying_folds": (
            counts["qualifying_validation_folds"] >= protocol["minimum_validation_folds"]
        ),
    }
    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison["paired_differences"]:
        if row["baseline_role"] == "primary":
            paired[row["comparison"]].append(row)
    conclusions = []
    for name, rows in paired.items():
        signs = []
        for row in rows:
            for key in ("event_bootstrap", "date_block_bootstrap"):
                interval = row[key]
                signs.append(
                    -1
                    if interval and interval["upper"] < 0
                    else 1
                    if interval and interval["lower"] > 0
                    else 0
                )
        same_direction = len(rows) == 2 and len(set(signs)) == 1 and signs[0] != 0
        directional = all(gates.values()) and same_direction
        conclusions.append(
            {
                "comparison": name,
                "status": "directional" if directional else "inconclusive",
                "direction": ("minuend-better" if signs[0] < 0 else "minuend-worse")
                if directional
                else None,
                "both_metrics_both_intervals_agree": same_direction,
            }
        )
    return {
        "status": "directional"
        if any(row["status"] == "directional" for row in conclusions)
        else "inconclusive",
        "population": "conditional on captured prospective WATCH and entry forecasts only",
        "unconditional_policy_profit_claim": False,
        "gates": gates,
        "comparisons": conclusions,
    }


def _final_population(
    records: list[dict[str, Any]], conflicts: set[str], cutoff: datetime
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Adapt admitted evidence to v2 scoring without changing stored venue snapshots."""
    resolutions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    executions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["record_type"] == "resolution":
            resolutions[record["market_id"]].append(record)
        elif record["record_type"] == "execution_capture":
            executions[record["scenario"]][record["observation_id"]] = {
                "snapshot": (
                    {
                        **record["snapshot"],
                        "snapshot_id": record["record_id"],
                        "source": "prospective-forward",
                        "status": record["snapshot"]["source_metadata"]["status"],
                    }
                    if record["status"] == "valid"
                    else None
                ),
                "reason": None if record["status"] == "valid" else "failed_execution_capture",
            }
    for market_id, revisions in resolutions.items():
        if len({row["result"] for row in revisions}) != 1:
            conflicts.add(market_id)
    observations = []
    excluded = []
    for record in records:
        if record["record_type"] != "market_capture" or record["status"] != "valid":
            continue
        raw = copy.deepcopy(record["observation"])
        market_id = record["market_id"]
        decision = _time(record["decision_at"])
        if market_id in conflicts:
            excluded.append(
                {
                    "observation_id": raw["observation_id"],
                    "market_id": market_id,
                    "reason": "conflicting_or_invalid_resolution",
                }
            )
            continue
        available = [
            row
            for row in resolutions[market_id]
            if decision
            < _time(row["observed_at"])
            <= _time(row["received_at"])
            <= _time(row["persisted_at"])
            <= cutoff
            and row["event_id"] == record["event_id"]
        ]
        if available:
            chosen = min(available, key=lambda row: (_time(row["observed_at"]), row["record_id"]))
            raw["resolution"] = {
                "result": chosen["result"],
                "observed_at": chosen["observed_at"],
                "settlement_ts": None,
            }
        observations.append(raw)
    return observations, dict(executions), excluded


def build_forward_report(
    *, store: ForwardStudyStore, protocol: dict[str, Any], as_of: datetime
) -> dict[str, Any]:
    """Return status before embargo; afterward score only the immutable fixed cutoff."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("report as_of must include a timezone")
    as_of = as_of.astimezone(UTC)
    if as_of > _utcnow():
        raise ValueError("report as_of cannot be in the future")
    validate_forward_protocol(protocol)
    report_cutoff = _time(protocol["report_as_of"])
    final = as_of >= report_cutoff
    cutoff = report_cutoff if final else as_of
    records, rejected, conflicts = _admissible_records(
        store.records(protocol["protocol_id"]), protocol, cutoff, final
    )
    coverage = _coverage(records, protocol, cutoff)
    report: dict[str, Any] = {
        "schema": "pms-forward-report-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "final" if final else "coverage-only",
        "as_of": cutoff.isoformat(),
        "report_as_of": report_cutoff.isoformat(),
        "research_only": True,
        "delivery_authorization": False,
        "registered": any(row["record_type"] == "registration" for row in records),
        "coverage": coverage,
        "rejected_records": rejected,
    }
    if not final:
        report["embargo"] = (
            "No outcomes, scores, execution performance or conclusions before cutoff."
        )
        return report
    observations, executions, excluded = _final_population(records, conflicts, cutoff)
    for rejection in rejected:
        if rejection["record_type"] != "execution_capture":
            continue
        scenario, observation_id = rejection.get("scenario"), rejection.get("observation_id")
        if not isinstance(scenario, str) or not isinstance(observation_id, str):
            continue
        executions.setdefault(scenario, {})[observation_id] = {
            "snapshot": None,
            "reason": (
                "execution_snapshot_conflict"
                if rejection["reason"] == "conflicting natural identity"
                else "invalid_execution_capture"
            ),
        }
    comparison = compare_forward_observations(observations, protocol)
    comparison["conclusion"] = _probability_conclusion(comparison, coverage, protocol)
    report["probability"] = comparison
    report["outcome_exclusions"] = excluded
    report["execution"] = assess_forward_decisions(
        observations,
        comparison,
        protocol,
        executions=executions,
        complete_slots=(
            coverage["valid_slots"] == coverage["expected_slots"]
            and not excluded
            and comparison["counts"]["excluded"] == 0
        ),
    )
    return report
