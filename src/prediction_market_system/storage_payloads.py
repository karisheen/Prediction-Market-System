"""Lossless physical encoding and references for immutable SQLite evidence.

The six large ``payload_json`` columns contain zlib-compressed canonical UTF-8
JSON, not SQL JSON text. References are an internal representation only; callers
must use ``load_payload`` before exposing logical evidence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from typing import Any

STORAGE_SCHEMA_VERSION = "compact_evidence_v1"
_TABLES = frozenset(
    {
        "input_objects",
        "forecast_ledger",
        "run_manifests",
        "forecasts",
        "opportunities",
        "paper_alert_market_checks",
        "code_identities",
    }
)
_REFERENCE_TABLES = frozenset(
    {
        "run_manifests",
        "forecasts",
        "opportunities",
        "paper_alert_market_checks",
    }
)


def _canonical(value: dict[str, Any]) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("storage payload must be a JSON object")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def encode_payload(value: dict[str, Any]) -> bytes:
    return zlib.compress(_canonical(value))


def decode_payload(value: str | bytes) -> dict[str, Any]:
    if isinstance(value, bytes):
        try:
            value = zlib.decompress(value).decode("utf-8")
        except (zlib.error, UnicodeDecodeError) as exc:
            raise ValueError("invalid compressed storage payload") from exc
    elif not isinstance(value, str):
        raise ValueError("storage payload must be legacy TEXT or compressed BLOB")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("storage payload must decode to a JSON object")
    return decoded


def _table(table: str) -> None:
    if table not in _TABLES:
        raise ValueError(f"unsupported storage payload table: {table}")


def _reference_key(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid storage reference key: {key}")
    return value


def payload_references_ledger(value: str | bytes) -> bool:
    """Whether a transient index payload can be discarded without losing evidence."""
    if isinstance(value, str):
        return False
    payload = decode_payload(value)
    return (
        payload.get("$storage") == STORAGE_SCHEMA_VERSION
        and payload.get("kind") == "ledger"
        and isinstance(payload.get("forecast_id"), str)
        and isinstance(payload.get("with_inputs"), bool)
    )


def _lookup(connection: sqlite3.Connection, table: str, column: str, key: str) -> dict[str, Any]:
    try:
        row = connection.execute(
            f"SELECT payload_json FROM {table} WHERE {column} = ?", (key,)
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table:" in str(exc):
            raise ValueError(f"dangling storage reference table: {table}") from exc
        raise
    if row is None:
        raise ValueError(f"dangling storage reference: {table}.{column}={key}")
    return load_payload(connection, table, row[0])


def _ledger_opportunity(
    connection: sqlite3.Connection, forecast_id: str, *, with_inputs: bool
) -> dict[str, Any]:
    try:
        row = connection.execute(
            "SELECT payload_json, input_id FROM forecast_ledger WHERE forecast_id = ?",
            (forecast_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table:" in str(exc):
            raise ValueError("dangling storage reference table: forecast_ledger") from exc
        raise
    if row is None:
        raise ValueError(f"dangling storage reference: forecast_ledger.forecast_id={forecast_id}")
    payload = load_payload(connection, "forecast_ledger", row[0])
    forecast = payload.get("forecast")
    if not isinstance(forecast, dict):
        raise ValueError("ledger reference has no forecast object")
    if with_inputs and "input_manifest" not in forecast:
        forecast["input_manifest"] = _lookup(connection, "input_objects", "input_id", row[1])
    return payload


def load_payload(connection: sqlite3.Connection, table: str, value: str | bytes) -> dict[str, Any]:
    _table(table)
    payload = decode_payload(value)
    # Legacy TEXT is always logical data, even if it resembles an internal envelope.
    if table not in _REFERENCE_TABLES or isinstance(value, str):
        return payload
    if payload.get("$storage") != STORAGE_SCHEMA_VERSION:
        return payload
    kind = payload.get("kind")
    if kind in ("inline", "code", "opportunity") and not isinstance(payload.get("value"), dict):
        raise ValueError("storage reference value must be a JSON object")
    if kind == "inline":
        logical = payload["value"]
    elif kind == "code" and table == "run_manifests":
        logical = payload["value"]
        code_id = _reference_key(payload, "code_id")
        code = _lookup(connection, "code_identities", "code_id", code_id)
        if hashlib.sha256(_canonical(code)).hexdigest() != code_id:
            raise ValueError("shared code identity does not match its recorded hash")
        logical["code"] = code
    elif kind == "ledger" and table in {"forecasts", "opportunities"}:
        if not isinstance(payload.get("with_inputs"), bool):
            raise ValueError("ledger reference requires a boolean with_inputs flag")
        logical = _ledger_opportunity(
            connection, _reference_key(payload, "forecast_id"), with_inputs=payload["with_inputs"]
        )
        if table == "forecasts":
            logical = logical["forecast"]
    elif kind == "opportunity" and table == "paper_alert_market_checks":
        logical = payload["value"]
        logical["opportunity"] = _lookup(
            connection, "opportunities", "opportunity_id", _reference_key(payload, "opportunity_id")
        )
    else:
        raise ValueError(f"invalid storage reference kind for {table}: {kind}")
    if not isinstance(logical, dict):
        raise ValueError("storage reference must hydrate a JSON object")
    return logical


def compact_payload(connection: sqlite3.Connection, table: str, value: dict[str, Any]) -> bytes:
    """Reference only byte-for-byte canonical logical matches; preserve all others."""
    _table(table)
    if not isinstance(value, dict):
        raise ValueError("storage payload must be a JSON object")
    if table not in _REFERENCE_TABLES:
        return encode_payload(value)
    envelope: dict[str, Any] = {
        "$storage": STORAGE_SCHEMA_VERSION,
        "kind": "inline",
        "value": value,
    }
    if table == "run_manifests" and isinstance(value.get("code"), dict):
        code = value["code"]
        code_bytes = _canonical(code)
        code_id = hashlib.sha256(code_bytes).hexdigest()
        prior = connection.execute(
            "SELECT payload_json FROM code_identities WHERE code_id = ?", (code_id,)
        ).fetchone()
        if prior is not None:
            if _canonical(decode_payload(prior[0])) != code_bytes:
                raise ValueError("immutable code identity collision")
        else:
            connection.execute(
                "INSERT INTO code_identities (code_id, payload_json) VALUES (?, ?)",
                (code_id, zlib.compress(code_bytes)),
            )
        envelope = {
            "$storage": STORAGE_SCHEMA_VERSION,
            "kind": "code",
            "code_id": code_id,
            "value": {key: item for key, item in value.items() if key != "code"},
        }
    elif table in {"forecasts", "opportunities"}:
        forecast = value if table == "forecasts" else value.get("forecast", {})
        forecast_id = forecast.get("forecast_id") if isinstance(forecast, dict) else None
        if forecast_id is not None:
            row = connection.execute(
                "SELECT payload_json, input_id FROM forecast_ledger WHERE forecast_id = ?",
                (forecast_id,),
            ).fetchone()
            if row is None:
                return encode_payload(envelope)
            candidate = load_payload(connection, "forecast_ledger", row[0])
            candidate_forecast = candidate.get("forecast")
            if not isinstance(candidate_forecast, dict):
                return encode_payload(envelope)
            with_inputs = "input_manifest" in forecast
            if with_inputs and "input_manifest" not in candidate_forecast:
                inputs = connection.execute(
                    "SELECT payload_json FROM input_objects WHERE input_id = ?", (row[1],)
                ).fetchone()
                # Preexisting broken historical FK evidence stays inline, not repaired.
                if inputs is None:
                    return encode_payload(envelope)
                candidate_forecast["input_manifest"] = load_payload(
                    connection, "input_objects", inputs[0]
                )
            if table == "forecasts":
                candidate = candidate_forecast
            if _canonical(candidate) == _canonical(value):
                envelope = {
                    "$storage": STORAGE_SCHEMA_VERSION,
                    "kind": "ledger",
                    "forecast_id": forecast_id,
                    "with_inputs": with_inputs,
                }
    elif table == "paper_alert_market_checks":
        opportunity = value.get("opportunity")
        opportunity_id = (
            opportunity.get("opportunity_id") if isinstance(opportunity, dict) else None
        )
        if isinstance(opportunity, dict) and opportunity_id is not None:
            row = connection.execute(
                "SELECT payload_json FROM opportunities WHERE opportunity_id = ?", (opportunity_id,)
            ).fetchone()
            if row is not None and _canonical(
                load_payload(connection, "opportunities", row[0])
            ) == _canonical(opportunity):
                envelope = {
                    "$storage": STORAGE_SCHEMA_VERSION,
                    "kind": "opportunity",
                    "opportunity_id": opportunity_id,
                    "value": {key: item for key, item in value.items() if key != "opportunity"},
                }
    return encode_payload(envelope)
