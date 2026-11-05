"""Independent, append-only prospective evidence; never managed approval evidence."""

from __future__ import annotations

import gzip
import json
import sqlite3
import zlib
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from prediction_market_system.domain import CryptoSnapshot, Opportunity, TerminalRangeContract
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import canonical_json, content_id, load_input_object
from prediction_market_system.experiment import validate_forward_observation
from prediction_market_system.forward_protocol import validate_forward_protocol
from prediction_market_system.research import ResearchContext, SpotCandle
from prediction_market_system.venues.kalshi import (
    KalshiMarket,
    KalshiOrderBook,
    to_market_snapshot,
)

_SCHEMA = "pms-forward-evidence-v1"
_COMMON = {"schema", "record_type", "protocol_id", "record_id", "persisted_at"}
_FIELDS = {
    "registration": "registered_at protocol model_source_sha256",
    "slot_attempt": "slot_id scheduled_at attempt_id started_at",
    "selection": (
        "slot_id attempt_id scheduled_at discovery_started_at discovery_received_at selection_at "
        "discovery_complete discovered_candidates spot selected_event_id selected_market_ids "
        "selection_inputs_id error"
    ),
    "market_capture": (
        "slot_id attempt_id scheduled_at selection_inputs_id market_id event_id status error "
        "book_requested_at book_received_at raw_book raw_book_id decision_at forecast_persisted_at "
        "observation fee_context"
    ),
    "execution_capture": (
        "slot_id attempt_id market_id event_id observation_id scenario target_at requested_at "
        "received_at status error raw_book raw_book_id snapshot"
    ),
    "slot_result": (
        "slot_id attempt_id scheduled_at completed_at status selected_market_ids "
        "captured_market_ids error"
    ),
    "resolution": "market_id event_id result observed_at received_at raw_payload",
}
_OBSERVATION = set(
    [
        "observation_id",
        "event_id",
        "market_id",
        "observed_at",
        "opportunity",
        "inputs",
        "run_id",
        "run_manifest",
        "run_manifest_sha256",
        "input_id",
        "research_context",
        "resolution",
    ]
)
_SHARED = frozenset(
    {
        "protocol",
        "market",
        "raw_book",
        "research_context",
        "inputs",
        "run_manifest",
        "opportunity",
        "raw_payload",
    }
)
_DDL = (
    "CREATE TABLE IF NOT EXISTS forward_objects "
    "(object_id TEXT PRIMARY KEY, payload_gzip BLOB NOT NULL)",
    "CREATE TABLE IF NOT EXISTS forward_records ("
    "sequence INTEGER PRIMARY KEY, protocol_id TEXT NOT NULL, record_type TEXT NOT NULL, "
    "natural_key TEXT NOT NULL, record_id TEXT NOT NULL UNIQUE, payload_id TEXT NOT NULL "
    "REFERENCES forward_objects(object_id), attempt_id TEXT, market_id TEXT, "
    "observation_id TEXT UNIQUE, run_id TEXT UNIQUE, "
    "UNIQUE(protocol_id, record_type, natural_key))",
    "CREATE INDEX IF NOT EXISTS forward_attempt ON forward_records(protocol_id, attempt_id)",
    "CREATE INDEX IF NOT EXISTS forward_market ON forward_records(protocol_id, market_id)",
    "CREATE TABLE IF NOT EXISTS forward_source_proofs ("
    "record_id TEXT NOT NULL REFERENCES forward_records(record_id), "
    "object_id TEXT NOT NULL REFERENCES forward_objects(object_id), "
    "PRIMARY KEY(record_id, object_id))",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("forward timestamp must be an ISO UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid forward timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("forward timestamp must have a UTC offset")
    return parsed.astimezone(UTC)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _keys(value: Any, keys: set[str], label: str) -> None:
    _require(isinstance(value, dict) and set(value) == keys, f"invalid {label} fields")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(_text(item) for item in value)
        and len(value) == len(set(value))
    )


def _identity(record: dict[str, Any]) -> str:
    kind = record["record_type"]
    fields = {
        "registration": ("protocol_id",),
        "slot_attempt": ("scheduled_at",),
        "selection": ("attempt_id",),
        "market_capture": ("attempt_id", "market_id"),
        "execution_capture": ("observation_id", "scenario"),
        "slot_result": ("attempt_id",),
        "resolution": ("market_id", "observed_at"),
    }[kind]
    return canonical_json([record[field] for field in fields])


def _one(records: list[dict[str, Any]], kind: str, **identity: Any) -> dict[str, Any]:
    matches = [
        row
        for row in records
        if row["record_type"] == kind
        and all(row.get(key) == value for key, value in identity.items())
    ]
    _require(len(matches) == 1, f"missing or conflicting {kind} dependency")
    return matches[0]


def _selected_market(selection: dict[str, Any], market_id: str) -> KalshiMarket:
    _require(market_id in selection["selected_market_ids"], "market was not selected")
    candidates = [
        row for row in selection["discovered_candidates"] if row["market_id"] == market_id
    ]
    _require(len(candidates) == 1, "missing or conflicting selected metadata")
    return KalshiMarket.model_validate(candidates[0]["market"])


def _validate_selection(record: dict[str, Any], attempt: dict[str, Any]) -> None:
    start = _time(record["discovery_started_at"])
    received = _time(record["discovery_received_at"])
    chosen = _time(record["selection_at"])
    _require(
        _time(attempt["persisted_at"]) <= start <= received <= chosen,
        "discovery precedes attempt or selection precedes receipt",
    )
    _require(type(record["discovery_complete"]) is bool, "invalid discovery completeness")
    _require(isinstance(record["discovered_candidates"], list), "invalid discovery candidates")
    _require(_strings(record["selected_market_ids"]), "invalid selected market identities")
    _require(
        record["selected_event_id"] is None or _text(record["selected_event_id"]),
        "invalid selected event",
    )
    source = {
        key: record[key]
        for key in ("discovered_candidates", "spot", "selected_event_id", "selected_market_ids")
    }
    _require(content_id(source) == record["selection_inputs_id"], "selection hash mismatch")
    spot = record["spot"]
    fresh_spot = False
    if spot is not None:
        _keys(spot, set(["price", "observed_at", "received_at", "source", "input_id"]), "spot")
        _require(
            type(spot["price"]) in (int, float) and spot["price"] > 0,
            "invalid selection spot price",
        )
        _require(_text(spot["source"]) and _text(spot["input_id"]), "missing spot identity")
        observed = _time(spot["observed_at"])
        _require(
            observed <= _time(spot["received_at"]) <= chosen, "selection spot is not yet available"
        )
        fresh_spot = (chosen - observed).total_seconds() <= 120
        _require(
            fresh_spot or (not record["selected_market_ids"] and _text(record["error"])),
            "stale selection spot",
        )
    eligible: list[tuple[KalshiMarket, TerminalRangeContract]] = []
    seen: set[str] = set()
    scheduled = _time(record["scheduled_at"])
    for candidate in record["discovered_candidates"]:
        _keys(
            candidate,
            set(["event_id", "market_id", "market", "contract", "eligible", "exclusion_reason"]),
            "discovered candidate",
        )
        market = KalshiMarket.model_validate(candidate["market"])
        _require(
            market.ticker == candidate["market_id"]
            and market.event_ticker == candidate["event_id"],
            "candidate identity mismatch",
        )
        _require(
            market.ticker not in seen
            or (
                not record["discovery_complete"]
                and not record["selected_market_ids"]
                and _text(record["error"])
            ),
            "duplicate discovered market",
        )
        seen.add(market.ticker)
        _require(type(candidate["eligible"]) is bool, "invalid candidate eligibility")
        contract: TerminalRangeContract | None = None
        try:
            parsed = market.price_contract()
            horizon = min(market.close_time, market.observation_end_at)
            if (
                isinstance(parsed, TerminalRangeContract)
                and parsed.settlement_window_seconds == 60
                and market.normalized_series_ticker == "KXBTC"
                and market.market_type == "binary"
                and market.notional_value_dollars == 1
                and all(
                    value is None or value <= chosen
                    for value in (market.created_time, market.updated_time, market.open_time)
                )
                and scheduled + timedelta(seconds=300)
                < horizon
                <= scheduled + timedelta(seconds=3600)
            ):
                contract = parsed
        except ValueError:
            pass
        _require(candidate["eligible"] == (contract is not None), "forged candidate eligibility")
        if candidate["contract"] is not None:
            parsed_candidate = TerminalRangeContract.model_validate(candidate["contract"])
            _require(parsed_candidate == market.price_contract(), "candidate contract mismatch")
        if contract is not None:
            _require(
                candidate["contract"] == contract.model_dump(mode="json")
                and candidate["exclusion_reason"] is None,
                "missing eligible contract",
            )
            eligible.append((market, contract))
        else:
            _require(_text(candidate["exclusion_reason"]), "missing candidate exclusion reason")
    expected: list[str] = []
    event: str | None = None
    if record["discovery_complete"] and fresh_spot and eligible:
        first = min(eligible, key=lambda item: (item[0].observation_end_at, item[0].event_ticker))
        event = first[0].event_ticker
        ranges = [
            (market, contract) for market, contract in eligible if market.event_ticker == event
        ]
        ranges.sort(
            key=lambda item: (
                abs((item[1].lower_bound + item[1].upper_bound) / 2 - spot["price"]),
                item[0].ticker,
            )
        )
        expected = [market.ticker for market, _ in ranges[:3]]
    _require(
        record["selected_market_ids"] == expected and record["selected_event_id"] == event,
        "selection does not follow frozen ordering",
    )


def _validate_observation(
    record: dict[str, Any], selection: dict[str, Any], protocol: dict[str, Any]
) -> None:
    observation = record["observation"]
    _keys(observation, _OBSERVATION, "observation")
    _require(
        observation["market_id"] == record["market_id"]
        and observation["event_id"] == record["event_id"],
        "observation identity mismatch",
    )
    decision = _time(record["decision_at"])
    _require(_time(observation["observed_at"]) == decision, "decision timestamp mismatch")
    _require(observation["resolution"] is None, "capture cannot contain an outcome")
    validate_forward_observation(observation, protocol)
    context = observation["research_context"]
    for proof in (context, context["spot"]):
        if selection["spot"]["input_id"] == content_id(proof):
            ForwardStudyStore._validate_spot(selection["spot"], proof)
    inputs = observation["inputs"]
    _keys(
        inputs,
        set(["market", "crypto", "contract", "engine_config", "recipe"])
        | ({"calibration_profile"} if "calibration_profile" in inputs else set()),
        "inputs",
    )
    _require(inputs.get("calibration_profile") is None, "calibration is forbidden in this study")
    _require(inputs["engine_config"] == protocol["engine_config"], "wrong frozen engine config")
    manifest = observation["run_manifest"]
    _keys(
        manifest,
        set(["schema", "kind", "run_id", "recorded_at", "code", "configuration", "inputs"]),
        "run manifest",
    )
    _require(manifest["schema"] == "pms-run-manifest-v2", "wrong run manifest schema")
    _require(
        manifest["configuration"].get("engine") == protocol["engine_config"],
        "manifest engine configuration mismatch",
    )
    code = manifest["code"]
    _require(
        isinstance(code, dict) and isinstance(code.get("source_files"), dict),
        "missing pinned code identity",
    )
    _require(
        all(
            code["source_files"].get(path) == digest
            for path, digest in protocol["model_source_sha256"].items()
        ),
        "model source drift",
    )
    _require(code.get("lock_sha256") == protocol["dependency_lock_sha256"], "dependency lock drift")
    market = _selected_market(selection, record["market_id"])
    reproduced_market = to_market_snapshot(
        market, KalshiOrderBook.model_validate(record["raw_book"]), observed_at=decision
    )
    _require(
        reproduced_market.model_dump(mode="json") == inputs["market"],
        "decision snapshot does not reproduce from original book and metadata",
    )
    _require(
        reproduced_market.observation_start_at is not None
        and _time(record["persisted_at"]) < reproduced_market.observation_start_at,
        "forecast persisted after observation began",
    )
    opportunity = Opportunity.model_validate(observation["opportunity"])
    _require(
        opportunity.forecast.input_manifest in ({}, inputs),
        "embedded forecast inputs conflict with observation inputs",
    )
    _, reproduced = CryptoThresholdEngine(EngineConfig(**protocol["engine_config"])).evaluate(
        reproduced_market,
        CryptoSnapshot.model_validate(inputs["crypto"]),
        TerminalRangeContract.model_validate(inputs["contract"]),
        None,
    )
    original = opportunity.model_dump(mode="json")
    expected = reproduced.model_dump(mode="json")
    for value in (original, expected):
        value.pop("opportunity_id", None)
        value["forecast"].pop("forecast_id", None)
        value["forecast"].pop("input_manifest", None)
    _require(original == expected, "recorded decision does not reproduce")
    fee = record["fee_context"]
    _keys(
        fee, set(["series_id", "fee_type", "fee_multiplier", "received_at", "raw_payload"]), "fees"
    )
    _require(
        fee["series_id"] == protocol["series"]
        and fee["fee_type"] == "quadratic"
        and fee["fee_multiplier"] == 1.0
        and isinstance(fee["raw_payload"], dict),
        "fee context incompatible with frozen policy",
    )
    raw = fee["raw_payload"]
    _keys(raw, {"series", "event", "series_fee_changes", "event_fee_changes"}, "raw fee proof")
    series = raw["series"].get("series", {})
    event = raw["event"].get("event", {})
    _require(
        series.get("ticker") == protocol["series"]
        and event.get("event_ticker") == record["event_id"],
        "fee source identity mismatch",
    )
    fee_type, multiplier = series.get("fee_type"), series.get("fee_multiplier")
    for change in sorted(raw["series_fee_changes"], key=lambda item: _time(item["scheduled_ts"])):
        _require(change["series_ticker"] == protocol["series"], "wrong series fee revision")
        if _time(change["scheduled_ts"]) <= _time(fee["received_at"]):
            fee_type, multiplier = change["fee_type"], change["fee_multiplier"]
    series_fee_type, series_multiplier = fee_type, multiplier
    for change in sorted(raw["event_fee_changes"], key=lambda item: _time(item["scheduled_ts"])):
        _require(
            change["series_ticker"] == protocol["series"]
            and change["event_ticker"] == record["event_id"],
            "wrong event fee revision",
        )
        if _time(change["scheduled_ts"]) <= _time(fee["received_at"]):
            fee_type = (
                series_fee_type
                if change["fee_type_override"] is None
                else change["fee_type_override"]
            )
            multiplier = (
                series_multiplier
                if change["fee_multiplier_override"] is None
                else change["fee_multiplier_override"]
            )
    # Present null is an authoritative reset, not an omitted current field.
    if "fee_type_override" in event:
        fee_type = (
            series_fee_type if event["fee_type_override"] is None else event["fee_type_override"]
        )
    if "fee_multiplier_override" in event:
        multiplier = (
            series_multiplier
            if event["fee_multiplier_override"] is None
            else event["fee_multiplier_override"]
        )
    _require(
        fee_type == fee["fee_type"]
        and multiplier is not None
        and float(multiplier) == fee["fee_multiplier"],
        "fee proof disagrees with policy",
    )
    _require(
        timedelta(0) <= decision - _time(fee["received_at"]) <= timedelta(seconds=120),
        "fees unavailable or stale at decision",
    )


def validate_forward_record(
    record: dict[str, Any], protocol: dict[str, Any], prior_records: list[dict[str, Any]]
) -> None:
    """Validate a hydrated record and its earlier immutable dependencies, without scoring."""
    kind = record.get("record_type")
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise ValueError("unknown forward record type")
    _keys(record, _COMMON | set(_FIELDS[kind].split()), "forward record")
    canonical_json(record)  # Reject nonfinite values, including nested raw payloads.
    _require(
        record["schema"] == _SCHEMA and record["protocol_id"] == protocol["protocol_id"],
        "wrong forward protocol or schema",
    )
    _require(
        content_id({key: value for key, value in record.items() if key != "record_id"})
        == record["record_id"],
        "forward record content hash mismatch",
    )
    persisted = _time(record["persisted_at"])
    for key, value in record.items():
        if key.endswith("_at") and value is not None:
            _require(_time(value) <= persisted, f"future forward clock: {key}")
    for key in ("slot_id", "attempt_id", "market_id", "event_id", "observation_id"):
        if key in record:
            _require(_text(record[key]), f"invalid {key}")
    if "error" in record:
        _require(record["error"] is None or _text(record["error"]), "invalid error")
    if kind == "registration":
        _require(
            record["protocol"] == protocol
            and record["model_source_sha256"] == protocol["model_source_sha256"],
            "registration protocol mismatch",
        )
        _require(
            record["registered_at"] == record["persisted_at"]
            and persisted < _time(protocol["study_start"]),
            "registration is not prospective",
        )
        return
    registration = _one(prior_records, "registration", protocol_id=record["protocol_id"])
    _require(_time(registration["persisted_at"]) <= persisted, "record precedes registration")
    if kind == "resolution":
        captures = [
            row
            for row in prior_records
            if row["record_type"] == "market_capture"
            and row["market_id"] == record["market_id"]
            and row["event_id"] == record["event_id"]
            and row["observation"] is not None
        ]
        _require(bool(captures), "resolution has no original captured market")
        _require(record["result"] in {"yes", "no"}, "invalid resolution result")
        _require(
            _time(record["observed_at"]) <= _time(record["received_at"]),
            "outcome availability follows receipt",
        )
        for capture in captures:
            _require(
                _time(capture["decision_at"]) < _time(record["observed_at"]),
                "resolution predates decision",
            )
            end = capture["observation"]["inputs"]["market"]["observation_end_at"]
            _require(_time(end) <= _time(record["observed_at"]), "resolution predates event end")
        payload = record["raw_payload"]
        _require(isinstance(payload, dict), "missing raw resolution response")
        metadata = payload.get("market", payload)
        _require(
            metadata.get("ticker") == record["market_id"]
            and metadata.get("event_ticker") == record["event_id"]
            and metadata.get("result") == record["result"],
            "resolution metadata mismatch",
        )
        return
    if kind == "slot_attempt":
        scheduled = _time(record["scheduled_at"])
        _require(
            _time(protocol["study_start"]) <= scheduled < _time(protocol["study_end"])
            and scheduled.minute == 45
            and scheduled.second == scheduled.microsecond == 0,
            "attempt is not a scheduled study slot",
        )
        _require(record["scheduled_at"] == scheduled.isoformat(), "slot timestamp not canonical")
        _require(
            record["slot_id"]
            == content_id(
                {"protocol_id": record["protocol_id"], "scheduled_at": scheduled.isoformat()}
            ),
            "slot identity mismatch",
        )
        _require(scheduled <= _time(record["started_at"]), "attempt started before slot")
        _require(
            not any(row.get("attempt_id") == record["attempt_id"] for row in prior_records),
            "attempt identity reused",
        )
        return
    attempt = _one(prior_records, "slot_attempt", attempt_id=record["attempt_id"])
    _require(record["slot_id"] == attempt["slot_id"], "wrong slot identity")
    _require(_time(attempt["persisted_at"]) <= persisted, "record precedes attempt")
    if "scheduled_at" in record:
        _require(record["scheduled_at"] == attempt["scheduled_at"], "wrong scheduled slot")
    if kind == "selection":
        _validate_selection(record, attempt)
        return
    if kind == "slot_result" and not any(
        row["record_type"] == "selection" and row["attempt_id"] == record["attempt_id"]
        for row in prior_records
    ):
        late = _time(attempt["started_at"]) > _time(attempt["scheduled_at"]) + timedelta(seconds=60)
        _require(
            record["status"] == ("late" if late else "failed")
            and record["selected_market_ids"] == []
            and record["captured_market_ids"] == []
            and _text(record["error"])
            and _time(record["completed_at"]) >= _time(attempt["persisted_at"]),
            "result without selection must retain an explicit empty failure",
        )
        return
    selection = _one(prior_records, "selection", attempt_id=record["attempt_id"])
    _require(_time(selection["persisted_at"]) <= persisted, "record precedes selection")
    if kind == "slot_result":
        _require(
            record["status"] in {"valid", "partial", "failed", "no-market", "late"},
            "invalid slot status",
        )
        captures = [
            row
            for row in prior_records
            if row["record_type"] == "market_capture"
            and row["attempt_id"] == record["attempt_id"]
            and row["status"] == "valid"
        ]
        captured = {row["market_id"] for row in captures}
        _require(
            _strings(record["captured_market_ids"])
            and set(record["captured_market_ids"]) == captured,
            "slot captures mismatch",
        )
        _require(
            record["selected_market_ids"] == selection["selected_market_ids"],
            "slot selection mismatch",
        )
        valid = bool(captured) and captured == set(selection["selected_market_ids"])
        late = _time(attempt["started_at"]) > _time(attempt["scheduled_at"]) + timedelta(seconds=60)
        expected = (
            "late"
            if late
            else "valid"
            if valid
            else "partial"
            if captured
            else "no-market"
            if selection["discovery_complete"]
            and not selection["selected_market_ids"]
            and selection["error"] is None
            else "failed"
        )
        _require(record["status"] == expected, "forged slot coverage status")
        _require(
            _time(record["completed_at"])
            >= max(
                [_time(selection["persisted_at"])]
                + [_time(row["persisted_at"]) for row in captures]
            ),
            "slot completed before evidence",
        )
        return
    market = _selected_market(selection, record["market_id"])
    _require(record["event_id"] == market.event_ticker, "wrong captured event")
    _require(record["status"] in {"valid", "failed"}, "invalid capture status")
    if record["raw_book"] is not None:
        _require(
            isinstance(record["raw_book"], dict)
            and content_id(record["raw_book"]) == record["raw_book_id"],
            "book hash mismatch",
        )
    else:
        _require(record["raw_book_id"] is None, "book hash without book")
    valid = record["status"] == "valid"
    _require(record["error"] is None if valid else _text(record["error"]), "capture error mismatch")
    if kind == "market_capture":
        _require(
            record["selection_inputs_id"] == selection["selection_inputs_id"],
            "wrong selection input identity",
        )
        requested = _time(record["book_requested_at"])
        _require(_time(selection["persisted_at"]) <= requested, "book requested before selection")
        if record["book_received_at"] is not None:
            _require(
                requested <= _time(record["book_received_at"]), "book receipt precedes request"
            )
        if valid:
            decision = _time(record["decision_at"])
            _require(
                _time(attempt["started_at"])
                <= _time(attempt["scheduled_at"]) + timedelta(seconds=60),
                "late attempt cannot produce eligible captures",
            )
            _require(
                _time(record["book_received_at"])
                <= decision
                <= persisted
                <= decision + timedelta(seconds=5),
                "forecast persistence timing invalid",
            )
            _require(
                decision <= _time(record["scheduled_at"]) + timedelta(seconds=120),
                "decision missed slot deadline",
            )
            _require(
                record["forecast_persisted_at"] == record["persisted_at"],
                "forecast persistence clock mismatch",
            )
            _require(
                selection["discovery_complete"] and record["raw_book"] is not None,
                "successful capture lacks complete evidence",
            )
            _validate_observation(record, selection, protocol)
        else:
            _require(record["forecast_persisted_at"] is None, "failed capture claims persistence")
        return
    capture = _one(
        prior_records,
        "market_capture",
        attempt_id=record["attempt_id"],
        market_id=record["market_id"],
    )
    observation = capture["observation"]
    _require(
        isinstance(observation, dict) and observation["observation_id"] == record["observation_id"],
        "execution has no original observation",
    )
    scenarios = {row["name"]: row["latency_seconds"] for row in protocol["cost_scenarios"]}
    _require(record["scenario"] in scenarios, "unknown execution scenario")
    target = _time(capture["decision_at"]) + timedelta(seconds=scenarios[record["scenario"]])
    _require(_time(record["target_at"]) == target, "wrong execution target")
    requested = _time(record["requested_at"])
    _require(target <= requested, "execution request precedes latency target")
    if record["received_at"] is not None:
        _require(
            requested <= _time(record["received_at"]) <= persisted,
            "execution receipt precedes request or follows persistence",
        )
    if valid:
        received = _time(record["received_at"])
        _require(
            received <= target + timedelta(seconds=120),
            "execution receipt missed target deadline",
        )
        _require(
            received <= min(market.close_time, market.observation_end_at) - timedelta(seconds=300),
            "execution inside event horizon",
        )
        _require(
            capture["status"] == "valid" and record["raw_book"] is not None,
            "execution lacks original capture or book",
        )
        expected_snapshot = to_market_snapshot(
            market, KalshiOrderBook.model_validate(record["raw_book"]), observed_at=received
        )
        _require(
            record["snapshot"] == expected_snapshot.model_dump(mode="json"),
            "execution snapshot differs from original book",
        )


class ForwardStudyStore:
    """Small isolated tables in the existing database; each operation owns its connection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _connect(self, *, writable: bool) -> sqlite3.Connection:
        if writable:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=120.0)
        else:
            connection = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=120.0
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 120000")
        connection.execute("PRAGMA foreign_keys = ON")
        if writable:
            connection.execute("PRAGMA journal_mode = WAL")
        else:
            connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        for statement in _DDL:
            connection.execute(statement)

    @staticmethod
    def _object(connection: sqlite3.Connection, object_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT payload_gzip FROM forward_objects WHERE object_id = ?", (object_id,)
        ).fetchone()
        if row is None:
            raise ValueError("missing forward content object")
        try:
            value: dict[str, Any] = json.loads(gzip.decompress(row[0]))
            _require(
                isinstance(value, dict) and content_id(value) == object_id,
                "forward object content hash mismatch",
            )
        except (
            OSError,
            EOFError,
            UnicodeError,
            TypeError,
            json.JSONDecodeError,
            zlib.error,
        ) as exc:
            raise ValueError("corrupt forward compressed payload") from exc
        return value

    @classmethod
    def _save_object(cls, connection: sqlite3.Connection, value: dict[str, Any]) -> str:
        object_id = content_id(value)
        prior = connection.execute(
            "SELECT 1 FROM forward_objects WHERE object_id = ?", (object_id,)
        ).fetchone()
        if prior is not None:
            _require(cls._object(connection, object_id) == value, "forward object collision")
        else:
            connection.execute(
                "INSERT INTO forward_objects VALUES (?, ?)",
                (
                    object_id,
                    gzip.compress(canonical_json(value).encode(), mtime=0),
                ),
            )
        return object_id

    @classmethod
    def _pack(cls, connection: sqlite3.Connection, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._pack(connection, item) for item in value]
        if not isinstance(value, dict):
            return value
        return {
            key: {"$forward_object": cls._save_object(connection, item)}
            if key in _SHARED and isinstance(item, dict)
            else cls._pack(connection, item)
            for key, item in value.items()
        }

    @classmethod
    def _unpack(cls, connection: sqlite3.Connection, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._unpack(connection, item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"$forward_object"}:
            return cls._object(connection, value["$forward_object"])
        return {key: cls._unpack(connection, item) for key, item in value.items()}

    @classmethod
    def _load(cls, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        record: dict[str, Any] = cls._unpack(connection, cls._object(connection, row["payload_id"]))
        _require(
            record.get("record_id") == row["record_id"]
            and record.get("protocol_id") == row["protocol_id"]
            and record.get("record_type") == row["record_type"]
            and record.get("attempt_id") == row["attempt_id"]
            and record.get("market_id") == row["market_id"]
            and _identity(record) == row["natural_key"],
            "forward record index mismatch",
        )
        _require(
            content_id({key: value for key, value in record.items() if key != "record_id"})
            == record["record_id"],
            "forward record content hash mismatch",
        )
        observation = record.get("observation") or {}
        _require(
            observation.get("observation_id") == row["observation_id"]
            and observation.get("run_id") == row["run_id"],
            "forecast identity index mismatch",
        )
        if record["record_type"] == "selection" and record["spot"] is not None:
            spot = record["spot"]
            proof = connection.execute(
                "SELECT 1 FROM forward_source_proofs WHERE record_id = ? AND object_id = ?",
                (record["record_id"], spot["input_id"]),
            ).fetchone()
            _require(proof is not None, "missing selection source proof")
            cls._validate_spot(spot, cls._object(connection, spot["input_id"]))
        return record

    @staticmethod
    def _validate_spot(spot: dict[str, Any], payload: dict[str, Any]) -> None:
        if "spot" in payload:
            context = ResearchContext.model_validate(payload)
            _require(
                context.content_id() == content_id(payload),
                "selection context changes during validation",
            )
            candle = context.spot
        else:
            candle = SpotCandle.model_validate(payload)
        _require(
            content_id(payload) == spot["input_id"]
            and float(candle.close) == spot["price"]
            and candle.end_at == _time(spot["observed_at"])
            and candle.retrieved_at == _time(spot["received_at"])
            and candle.provider == spot["source"]
            and candle.product_id == "BTC-USD"
            and candle.interval_seconds == 60,
            "selection spot source proof mismatch",
        )

    @classmethod
    def _dependencies(
        cls, connection: sqlite3.Connection, record: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT * FROM forward_records WHERE protocol_id = ? AND record_type = 'registration' "
            "UNION SELECT * FROM forward_records WHERE protocol_id = ? AND attempt_id = ? "
            "UNION SELECT * FROM forward_records WHERE protocol_id = ? "
            "AND record_type = 'market_capture' AND market_id = ? ORDER BY sequence",
            (
                record["protocol_id"],
                record["protocol_id"],
                record.get("attempt_id"),
                record["protocol_id"],
                record.get("market_id"),
            ),
        )
        return [cls._load(connection, row) for row in rows]

    @classmethod
    def _insert(cls, connection: sqlite3.Connection, record: dict[str, Any]) -> None:
        proof_id: str | None = None
        if record["record_type"] == "selection" and record["spot"] is not None:
            try:
                payload = load_input_object(connection, record["spot"]["input_id"])
            except sqlite3.OperationalError as exc:
                raise ValueError("missing selection source proof table") from exc
            _require(payload is not None, "missing selection source proof")
            assert payload is not None
            cls._validate_spot(record["spot"], payload)
            proof_id = cls._save_object(connection, payload)
        payload_id = cls._save_object(connection, cls._pack(connection, record))
        observation = record.get("observation") or {}
        connection.execute(
            "INSERT INTO forward_records (protocol_id, record_type, natural_key, record_id, "
            "payload_id, attempt_id, market_id, observation_id, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["protocol_id"],
                record["record_type"],
                _identity(record),
                record["record_id"],
                payload_id,
                record.get("attempt_id"),
                record.get("market_id"),
                observation.get("observation_id"),
                observation.get("run_id"),
            ),
        )
        if proof_id is not None:
            connection.execute(
                "INSERT INTO forward_source_proofs VALUES (?, ?)", (record["record_id"], proof_id)
            )

    def register(self, protocol: dict[str, Any]) -> dict[str, Any]:
        protocol = validate_forward_protocol(protocol)
        with closing(self._connect(writable=True)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize(connection)
            row = connection.execute(
                "SELECT * FROM forward_records "
                "WHERE protocol_id = ? AND record_type = 'registration'",
                (protocol["protocol_id"],),
            ).fetchone()
            if row is not None:
                record = self._load(connection, row)
                validate_forward_record(record, protocol, [])
                return record
            now = _now().isoformat()
            record = {
                "schema": _SCHEMA,
                "record_type": "registration",
                "protocol_id": protocol["protocol_id"],
                "persisted_at": now,
                "registered_at": now,
                "protocol": protocol,
                "model_source_sha256": protocol["model_source_sha256"],
            }
            record["record_id"] = content_id(record)
            validate_forward_record(record, protocol, [])
            self._insert(connection, record)
            return record

    def append(self, record: dict[str, Any]) -> bool:
        record = json.loads(canonical_json(record))
        kind = record.get("record_type")
        if not isinstance(kind, str) or kind not in _FIELDS:
            raise ValueError("unknown forward record type")
        _require(kind != "registration", "use register for prospective registration")
        _keys(
            record,
            (_COMMON - {"record_id", "persisted_at"}) | set(_FIELDS[kind].split()),
            "append input",
        )
        if kind == "market_capture":
            _require(record["forecast_persisted_at"] is None, "persistence clock is storage-owned")
        # Normalize identity clocks before looking up natural uniqueness.
        for key, value in record.items():
            if key.endswith("_at") and value is not None:
                record[key] = _time(value).isoformat()
        with closing(self._connect(writable=True)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize(connection)
            dependencies = self._dependencies(connection, record)
            registration = _one(dependencies, "registration", protocol_id=record["protocol_id"])
            protocol = validate_forward_protocol(registration["protocol"])
            validate_forward_record(registration, protocol, [])
            prior = connection.execute(
                "SELECT * FROM forward_records WHERE protocol_id = ? AND record_type = ? "
                "AND natural_key = ?",
                (record["protocol_id"], kind, _identity(record)),
            ).fetchone()
            if prior is not None:
                original = self._load(connection, prior)
                incoming = {
                    key: value
                    for key, value in original.items()
                    if key not in {"record_id", "persisted_at"}
                }
                if kind == "market_capture":
                    incoming["forecast_persisted_at"] = None
                _require(incoming == record, "conflicting immutable forward natural identity")
                return False
            record["persisted_at"] = _now().isoformat()
            if kind == "market_capture" and record["status"] == "valid":
                record["forecast_persisted_at"] = record["persisted_at"]
            record["record_id"] = content_id(record)
            validate_forward_record(record, protocol, dependencies)
            if kind == "market_capture" and record["observation"] is not None:
                observation = record["observation"]
                existing = connection.execute(
                    "SELECT 1 FROM forward_records WHERE observation_id = ? OR run_id = ?",
                    (observation["observation_id"], observation["run_id"]),
                ).fetchone()
                _require(existing is None, "forecast or run identity reused")
            # Validation can involve forecast reproduction. Stamp again immediately before
            # writing, so its CPU time cannot be hidden behind an earlier persistence clock.
            record["persisted_at"] = _now().isoformat()
            if kind == "market_capture" and record["status"] == "valid":
                record["forecast_persisted_at"] = record["persisted_at"]
                persisted = _time(record["persisted_at"])
                _require(
                    _time(record["decision_at"])
                    <= persisted
                    <= _time(record["decision_at"]) + timedelta(seconds=5),
                    "forecast persistence deadline elapsed during validation",
                )
                _require(
                    persisted
                    < _time(record["observation"]["inputs"]["market"]["observation_start_at"]),
                    "forecast persistence reached observation window",
                )
            record["record_id"] = content_id(
                {key: value for key, value in record.items() if key != "record_id"}
            )
            self._insert(connection, record)
            return True

    def records(self, protocol_id: str) -> list[dict[str, Any]]:
        """Audit the complete study, including every source proof and reproduced decision."""
        return self._read_records(protocol_id)

    def slot_attempt(self, protocol_id: str, scheduled_at: datetime) -> dict[str, Any] | None:
        """Read only the indexed first attempt and its prospective registration."""
        stamp = _time(scheduled_at.isoformat()).isoformat()
        rows = self._read_records(protocol_id, scheduled_at=stamp)
        return next((row for row in rows if row["record_type"] == "slot_attempt"), None)

    def attempt_records(self, protocol_id: str, attempt_id: str) -> list[dict[str, Any]]:
        """Validate one attempt's evidence without loading unrelated historical slots."""
        return [
            row
            for row in self._read_records(protocol_id, attempt_id=attempt_id)
            if row.get("attempt_id") == attempt_id
        ]

    def resolution_candidates(self, protocol_id: str) -> Iterator[tuple[dict[str, Any], set[str]]]:
        """Stream captured markets and known outcomes, retaining revision re-polling.

        Only one market's original capture dependencies are hydrated at a time. No
        database transaction remains open while the caller awaits a venue response.
        """
        after = ""
        while self.path.exists():
            with closing(self._connect(writable=False)) as connection:
                connection.execute("BEGIN")
                if not self._has_evidence(connection):
                    return
                row = connection.execute(
                    "SELECT market_id FROM forward_records WHERE protocol_id = ? "
                    "AND market_id > ? AND record_type = 'market_capture' "
                    "AND observation_id IS NOT NULL ORDER BY market_id LIMIT 1",
                    (protocol_id, after),
                ).fetchone()
            if row is None:
                return
            after = row["market_id"]
            records = self._read_records(protocol_id, market_id=after)
            captures = [
                record
                for record in records
                if record["record_type"] == "market_capture"
                and record["status"] == "valid"
                and record["observation"] is not None
            ]
            if captures:
                yield (
                    captures[-1],
                    {
                        record["result"]
                        for record in records
                        if record["record_type"] == "resolution"
                    },
                )

    @staticmethod
    def _has_evidence(connection: sqlite3.Connection) -> bool:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'forward_%'"
            )
        }
        if not tables:
            return False
        _require(
            {"forward_objects", "forward_records", "forward_source_proofs"} <= tables,
            "incomplete forward evidence schema",
        )
        return True

    def _read_records(
        self,
        protocol_id: str,
        *,
        attempt_id: str | None = None,
        scheduled_at: str | None = None,
        market_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with closing(self._connect(writable=False)) as connection:
            connection.execute("BEGIN")
            if not self._has_evidence(connection):
                return []
            records: list[dict[str, Any]] = []
            registration: dict[str, Any] | None = None
            by_attempt: dict[str, list[dict[str, Any]]] = {}
            by_market: dict[str, list[dict[str, Any]]] = {}
            identities: set[tuple[str, str]] = set()
            query = "SELECT * FROM forward_records WHERE protocol_id = ?"
            parameters: tuple[str, ...] = (protocol_id,)
            if attempt_id is not None or scheduled_at is not None:
                query += (
                    " AND record_type = 'registration' UNION ALL "
                    "SELECT * FROM forward_records WHERE protocol_id = ? AND "
                )
                if attempt_id is not None:
                    query += "attempt_id = ?"
                    parameters += (protocol_id, attempt_id)
                else:
                    assert scheduled_at is not None
                    query += "record_type = 'slot_attempt' AND natural_key = ?"
                    parameters += (protocol_id, canonical_json([scheduled_at]))
            elif market_id is not None:
                query += (
                    " AND record_type = 'registration' UNION ALL "
                    "SELECT * FROM forward_records WHERE protocol_id = ? "
                    "AND market_id = ? AND record_type IN ('market_capture', 'resolution') "
                    "UNION ALL SELECT * FROM forward_records WHERE protocol_id = ? "
                    "AND record_type IN ('slot_attempt', 'selection') AND attempt_id IN "
                    "(SELECT attempt_id FROM forward_records WHERE protocol_id = ? "
                    "AND record_type = 'market_capture' AND market_id = ?)"
                )
                parameters += (protocol_id, market_id, protocol_id, protocol_id, market_id)
            query += " ORDER BY sequence"
            for row in connection.execute(query, parameters):
                record = self._load(connection, row)
                if record["record_type"] == "registration":
                    _require(registration is None, "conflicting registrations")
                    registration = record
                    validate_forward_protocol(record["protocol"])
                    dependencies = []
                else:
                    _require(registration is not None, "missing prospective registration")
                    dependencies = [registration] if registration is not None else []
                    dependencies += by_attempt.get(record.get("attempt_id", ""), [])
                    if record["record_type"] == "resolution":
                        dependencies += by_market.get(record["market_id"], [])
                assert registration is not None
                validate_forward_record(record, registration["protocol"], dependencies)
                _require(
                    _time(record["persisted_at"]) <= _now(),
                    "forward evidence persistence is in the future",
                )
                identity = (record["record_type"], _identity(record))
                _require(identity not in identities, "duplicate natural identity")
                identities.add(identity)
                records.append(record)
                if "attempt_id" in record:
                    by_attempt.setdefault(record["attempt_id"], []).append(record)
                if record["record_type"] == "market_capture":
                    by_market.setdefault(record["market_id"], []).append(record)
            return records
