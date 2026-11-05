"""Immutable registration and deterministic schedule for the independent forward study.

This module grants no delivery authority and does not modify the Phase 1 protocols.
A different study requires a new preregistration, not an edited v1 document.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from prediction_market_system.engine import EngineConfig
from prediction_market_system.evidence import content_id

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORWARD_PROTOCOL_PATH = _REPOSITORY_ROOT / "experiments/forward/protocol-v1.json"
_FROZEN_PROTOCOL_ID = "sha256:e7f440c05ac51bdc8a13456bfa8185e15cc6420c9bffa60a6053cd1103e0056f"
_MODEL_PATHS = frozenset(
    f"src/prediction_market_system/{name}"
    for name in (
        "engine.py",
        "domain.py",
        "recipe.py",
        "research.py",
        "calibration.py",
        "venues/kalshi.py",
    )
)


def _utc(value: Any, name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{name} must be an offset-aware timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be an offset-aware timestamp")
    return value.astimezone(UTC)


def forward_protocol_id(document: dict[str, Any]) -> str:
    """Canonical identity, excluding only the identity field itself."""
    if not isinstance(document, dict):
        raise ValueError("forward protocol must be an object")
    try:
        return "sha256:" + content_id(
            {key: value for key, value in document.items() if key != "protocol_id"}
        )
    except (TypeError, ValueError) as error:
        raise ValueError("forward protocol must contain finite JSON values") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol key: {key}")
        result[key] = value
    return result


def load_forward_protocol(path: Path) -> dict[str, Any]:
    """Load an unambiguous JSON document and validate its frozen identity and code."""
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return validate_forward_protocol(document)


def validate_forward_protocol(document: dict[str, Any]) -> dict[str, Any]:
    """Reject tampering, unsupported registrations, invalid clocks and model drift.

    The supported digest pins *every* field, not merely selected numerical gates.
    Rehashing an edited document does not authorize a new experiment. Source pins
    cover the probability/semantics implementation, not evolving capture/report code.
    """
    identity = forward_protocol_id(document)
    if document.get("protocol_id") != identity:
        raise ValueError("forward protocol identity mismatch")
    if identity != _FROZEN_PROTOCOL_ID or document.get("protocol_version") != "pms-forward-v1":
        raise ValueError("unsupported immutable forward protocol")
    frozen = _utc(document["frozen_at"], "frozen_at")
    start = _utc(document["study_start"], "study_start")
    end = _utc(document["study_end"], "study_end")
    cutoff = _utc(document["report_as_of"], "report_as_of")
    if not frozen < start < end < cutoff or frozen > datetime.now(UTC):
        raise ValueError("invalid prospective freeze or study/report windows")
    if document["research_only"] is not True or document["delivery_authorization"] is not False:
        raise ValueError("forward study cannot grant delivery authorization")
    previous = start
    for fold in document["folds"]:
        fold_start = _utc(fold["start"], "fold start")
        fold_end = _utc(fold["end"], "fold end")
        if fold_start != previous or not fold_start < fold_end <= end:
            raise ValueError("forward folds must partition the study window")
        previous = fold_end
    if previous != end:
        raise ValueError("forward folds must cover the study window")
    config = EngineConfig.model_validate(document["engine_config"])
    if config.model_dump(mode="json") != document["engine_config"]:
        raise ValueError("forward engine configuration must be complete and exact")
    sources = document["model_source_sha256"]
    if set(sources) != _MODEL_PATHS:
        raise ValueError("forward model source pins are incomplete")
    for relative, expected in sources.items():
        try:
            actual = hashlib.sha256((_REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"forward model source unavailable: {relative}") from error
        if actual != expected:
            raise ValueError(f"forward model source drift: {relative}")
    try:
        lock_hash = hashlib.sha256((_REPOSITORY_ROOT / "uv.lock").read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("forward dependency lock unavailable") from error
    if lock_hash != document["dependency_lock_sha256"]:
        raise ValueError("forward dependency lock drift")
    return document


def scheduled_slots(
    document: dict[str, Any], start: datetime | None = None, end: datetime | None = None
) -> list[datetime]:
    """Return UTC hourly minute45 instants in the intersected half-open window.

    Bounds filter the frozen schedule; they never move its origin. Empty or disjoint
    intersections return [], while inverted bounds and naive datetimes are errors.
    """
    validate_forward_protocol(document)
    study_start = _utc(document["study_start"], "study_start")
    study_end = _utc(document["study_end"], "study_end")
    requested_start = study_start if start is None else _utc(start, "start")
    requested_end = study_end if end is None else _utc(end, "end")
    if start is not None and end is not None and requested_start > requested_end:
        raise ValueError("scheduled slot start must not follow end")
    lower = max(study_start, requested_start)
    upper = min(study_end, requested_end)
    if lower >= upper:
        return []
    schedule = document["schedule"]
    slot = lower.replace(minute=schedule["minute"], second=schedule["second"], microsecond=0)
    interval = timedelta(seconds=schedule["interval_seconds"])
    if slot < lower:
        slot += interval
    result: list[datetime] = []
    while slot < upper:
        result.append(slot)
        slot += interval
    return result
