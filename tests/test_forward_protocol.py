import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from prediction_market_system.evidence import content_id
from prediction_market_system.forward_protocol import (
    DEFAULT_FORWARD_PROTOCOL_PATH,
    forward_protocol_id,
    load_forward_protocol,
    scheduled_slots,
    validate_forward_protocol,
)


@pytest.fixture
def protocol() -> dict[str, Any]:
    return load_forward_protocol(DEFAULT_FORWARD_PROTOCOL_PATH)


def test_frozen_identity_and_research_boundary(protocol: dict[str, Any]) -> None:
    original = copy.deepcopy(protocol)
    assert validate_forward_protocol(protocol) == original
    assert protocol["protocol_id"] == forward_protocol_id(protocol)
    assert forward_protocol_id(protocol) == "sha256:" + content_id(
        {key: value for key, value in protocol.items() if key != "protocol_id"}
    )
    assert protocol["research_only"] is True
    assert protocol["delivery_authorization"] is False
    assert "supersedes" not in protocol
    assert datetime.fromisoformat(protocol["frozen_at"]) < datetime.fromisoformat(
        protocol["study_start"]
    )


def test_schedule_is_complete_and_fold_partitioned(protocol: dict[str, Any]) -> None:
    slots = scheduled_slots(protocol)
    assert len(slots) == 1440 == len(set(slots))
    assert slots[0] == datetime(2026, 11, 6, 0, 45, tzinfo=UTC)
    assert slots[-1] == datetime(2027, 1, 4, 23, 45, tzinfo=UTC)
    assert all(
        right - left == timedelta(hours=1) for left, right in zip(slots, slots[1:], strict=False)
    )
    fold_slots = [
        scheduled_slots(
            protocol, datetime.fromisoformat(f["start"]), datetime.fromisoformat(f["end"])
        )
        for f in protocol["folds"]
    ]
    assert [len(values) for values in fold_slots] == [720, 720]
    assert fold_slots[0] + fold_slots[1] == slots


def test_schedule_intersects_without_shifting_origin(protocol: dict[str, Any]) -> None:
    first = datetime(2026, 11, 6, 0, 45, tzinfo=UTC)
    assert scheduled_slots(protocol, first, first + timedelta(hours=1)) == [first]
    assert (
        scheduled_slots(protocol, first + timedelta(microseconds=1), first + timedelta(hours=1))
        == []
    )
    assert scheduled_slots(protocol, first, first) == []
    assert scheduled_slots(protocol, end=first) == []
    assert scheduled_slots(protocol, start=datetime(2028, 1, 1, tzinfo=UTC)) == []
    assert scheduled_slots(protocol, end=datetime(2025, 1, 1, tzinfo=UTC)) == []
    assert (
        scheduled_slots(
            protocol, start=datetime(2028, 1, 1, tzinfo=UTC), end=datetime(2028, 1, 2, tzinfo=UTC)
        )
        == []
    )
    offset_start = datetime.fromisoformat("2026-11-05T19:45:00-05:00")
    assert scheduled_slots(protocol, offset_start, first + timedelta(seconds=1)) == [first]


def test_schedule_rejects_naive_and_inverted_bounds(protocol: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="offset-aware"):
        scheduled_slots(protocol, datetime(2026, 11, 6))
    with pytest.raises(ValueError, match="offset-aware"):
        scheduled_slots(protocol, end=datetime(2026, 11, 6))
    with pytest.raises(ValueError, match="must not follow"):
        scheduled_slots(
            protocol, datetime(2027, 1, 2, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("study_start", "2026-11-06T00:00:00"),
        ("study_end", "2026-11-01T00:00:00+00:00"),
        ("report_as_of", "2026-11-06T00:00:00+00:00"),
        ("frozen_at", "2026-11-07T00:00:00+00:00"),
        ("engine_config", {}),
        ("bootstrap_replicates", 10),
        ("research_only", False),
        ("delivery_authorization", True),
        ("unknown_policy", "silently changed"),
    ],
)
def test_tampering_and_rehashed_replacements_rejected(
    protocol: dict[str, Any], key: str, value: Any
) -> None:
    protocol[key] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_forward_protocol(protocol)
    protocol["protocol_id"] = forward_protocol_id(protocol)
    with pytest.raises(ValueError, match="unsupported immutable"):
        validate_forward_protocol(protocol)


def test_nonfinite_and_nonobject_protocol_rejected(protocol: dict[str, Any]) -> None:
    protocol["confidence"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        forward_protocol_id(protocol)
    with pytest.raises(ValueError, match="must be an object"):
        validate_forward_protocol([])  # type: ignore[arg-type]


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"protocol_version":"a","protocol_version":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate protocol key"):
        load_forward_protocol(path)


@pytest.mark.parametrize("relative", ["src/prediction_market_system/engine.py", "uv.lock"])
def test_source_and_lock_drift_rejected(
    protocol: dict[str, Any], monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    original_read = Path.read_bytes
    target = DEFAULT_FORWARD_PROTOCOL_PATH.parents[2] / relative

    def changed_source(path: Path) -> bytes:
        if path == target:
            return b"changed implementation"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    with pytest.raises(ValueError, match="drift"):
        validate_forward_protocol(protocol)


def test_v2_semantics_costs_and_statistics_preserved(protocol: dict[str, Any]) -> None:
    reference = DEFAULT_FORWARD_PROTOCOL_PATH.parents[2] / "experiments/phase1/protocol-v2.json"
    v2 = json.loads(reference.read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "model_version",
        "structural_weight",
        "expected_annual_return",
        "production_features",
        "source_selection_branches",
        "decision_costs",
        "cost_scenarios",
        "minimum_resolved_events",
        "minimum_dates",
        "minimum_validation_folds",
        "minimum_fold_resolved_events",
        "bootstrap_replicates",
        "bootstrap_seed",
        "confidence",
        "log_loss_epsilon",
        "bootstrap",
        "metrics",
        "regimes",
    ):
        assert protocol[key] == v2[key], key
    assert protocol["coverage"]["minimum_valid_capture_fraction"] == 0.95
    assert protocol["coverage"]["require_each_fold"] is True
    assert protocol["coverage"]["execution_direction_requires_complete_slots"] is True
    assert protocol["coverage"]["execution_direction_requires_all_entry_snapshots"] is True
    assert protocol["coverage"]["execution_direction_requires_all_fill_resolutions"] is True
    assert protocol["capture"]["maximum_forecast_persistence_delay_seconds"] == 5
    assert protocol["engine_config"]["binary_fee_coefficient"] == 0.07
    assert protocol["engine_config"]["uncertainty_margin"] == 0.05
    assert protocol["minimum_ask_size"] == 1
    assert protocol["engine_config"]["minimum_ask_size"] == 10
