from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from prediction_market_system import forward_capture, forward_cli, forward_storage
from prediction_market_system.cli import app
from prediction_market_system.evidence import content_id
from prediction_market_system.forward_protocol import (
    DEFAULT_FORWARD_PROTOCOL_PATH,
    forward_protocol_id,
)
from prediction_market_system.forward_storage import ForwardStudyStore

runner = CliRunner()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, datetime]:
    clock = {"now": datetime(2026, 11, 5, 13, 41, tzinfo=UTC)}

    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            value = clock["now"]
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PMS_DATABASE_PATH", str(tmp_path / "default.db"))
    monkeypatch.setattr(forward_cli, "datetime", Clock)
    monkeypatch.setattr(forward_storage, "_now", lambda: clock["now"])
    monkeypatch.setattr(forward_capture, "_now", lambda: clock["now"])
    return clock


def test_registration_is_immutable_and_idempotent_after_start(
    tmp_path: Path, clock: dict[str, datetime]
) -> None:
    database = tmp_path / "study.db"
    first = runner.invoke(app, ["forward-register", "--db", str(database)])
    assert first.exit_code == 0, first.output
    original = json.loads(first.output)
    clock["now"] = datetime(2026, 11, 6, 0, 46, tzinfo=UTC)
    again = runner.invoke(app, ["forward-register", "--db", str(database)])
    assert again.exit_code == 0, again.output
    assert json.loads(again.output) == original
    records = ForwardStudyStore(database).records(original["protocol_id"])
    assert len(records) == 1
    assert records[0]["record_type"] == "registration"
    refused = runner.invoke(app, ["forward-register", "--db", str(tmp_path / "late.db")])
    assert refused.exit_code != 0


def test_registration_output_preserves_protocol_identity_and_denies_authorization(
    tmp_path: Path, clock: dict[str, datetime]
) -> None:
    database = tmp_path / "study.db"
    result = runner.invoke(app, ["forward-register", "--db", str(database)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "pms-forward-evidence-v1"
    assert payload["protocol"]["delivery_authorization"] is False
    assert payload["protocol"] == json.loads(DEFAULT_FORWARD_PROTOCOL_PATH.read_text())
    assert forward_protocol_id(payload["protocol"]) == payload["protocol_id"]
    assert (
        content_id({key: value for key, value in payload.items() if key != "record_id"})
        == payload["record_id"]
    )
    assert ForwardStudyStore(database).records(payload["protocol_id"]) == [payload]


def test_capture_before_first_slot_does_not_register_or_create_database(
    tmp_path: Path, clock: dict[str, datetime]
) -> None:
    for now in (clock["now"], datetime(2026, 11, 6, 0, 44, tzinfo=UTC)):
        clock["now"] = now
        result = runner.invoke(app, ["forward-capture"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["status"] == "not-started"
        assert not (tmp_path / "default.db").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["forward-capture", "--scheduled-at", "not-a-clock"],
        ["forward-capture", "--scheduled-at", "2026-11-06T00:45:00"],
        ["forward-capture", "--scheduled-at", "2026-11-06T00:44:00Z"],
        ["forward-capture", "--scheduled-at", "2026-11-06T00:45:00Z"],
        ["forward-capture", "--scheduled-at", "2027-01-05T00:45:00Z"],
        ["forward-capture", "--dry-run", "--scheduled-at", "2026-11-06T00:45:00Z"],
        ["forward-report", "--as-of", "2026-11-05T13:41:00"],
        ["forward-report", "--as-of", "2027-01-07T00:00:00Z"],
    ],
)
def test_invalid_or_future_clocks_fail_without_creating_evidence(
    arguments: list[str], tmp_path: Path, clock: dict[str, datetime]
) -> None:
    result = runner.invoke(app, arguments)
    assert result.exit_code != 0
    assert not (tmp_path / "default.db").exists()


def test_late_slot_retained_once_without_backfill_or_source_requests(
    tmp_path: Path, clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = runner.invoke(app, ["forward-register"])
    assert registered.exit_code == 0, registered.output
    protocol_id = json.loads(registered.output)["protocol_id"]
    clock["now"] = datetime(2026, 11, 6, 2, 10, tzinfo=UTC)

    async def reject_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("late slots must not request market data")

    monkeypatch.setattr(httpx.AsyncClient, "send", reject_network)
    first = runner.invoke(app, ["forward-capture"])
    assert first.exit_code != 0
    assert json.loads(first.output)["capture"]["status"] == "late"
    store = ForwardStudyStore(tmp_path / "default.db")
    before = store.records(protocol_id)
    attempts = [record for record in before if record["record_type"] == "slot_attempt"]
    assert len(attempts) == 1
    assert attempts[0]["scheduled_at"] == "2026-11-06T01:45:00+00:00"
    second = runner.invoke(app, ["forward-capture"])
    assert second.exit_code != 0
    assert store.records(protocol_id) == before


def test_live_diagnostic_source_failure_is_redacted_and_never_creates_study_database(
    tmp_path: Path, clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    async def unavailable(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> Any:
        requests.append(request.url)
        raise httpx.ConnectError("source unavailable token=cli-test-secret", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", unavailable)
    protocol_bytes = DEFAULT_FORWARD_PROTOCOL_PATH.read_bytes()
    result = runner.invoke(app, ["forward-capture", "--dry-run"])
    assert result.exit_code != 0
    assert requests, "diagnostic must exercise real source acquisition"
    payload = json.loads(result.output)
    assert payload["diagnostic"] is True
    assert payload["eligible"] is False
    assert payload["status"] == "failed"
    assert "[REDACTED]" in result.output
    assert payload["error"] or any(row["error"] for row in payload["capture_clocks"])
    assert "cli-test-secret" not in result.output
    assert not (tmp_path / "default.db").exists()
    assert DEFAULT_FORWARD_PROTOCOL_PATH.read_bytes() == protocol_bytes


def test_report_is_read_only_and_cannot_bypass_embargo(
    tmp_path: Path, clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market_system import forward_report

    monkeypatch.setattr(forward_report, "_utcnow", lambda: clock["now"])
    destination = tmp_path / "coverage.json"
    result = runner.invoke(app, ["forward-report", "--output", str(destination)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "pms-forward-report-v1"
    assert payload["delivery_authorization"] is False
    assert (
        payload["protocol_id"]
        == json.loads(DEFAULT_FORWARD_PROTOCOL_PATH.read_text())["protocol_id"]
    )
    assert payload["status"] == "coverage-only"
    assert payload["coverage"]["expected_slots"] == 1440
    assert payload["coverage"]["elapsed_slots"] == 0
    assert json.loads(destination.read_text()) == payload
    assert not (tmp_path / "default.db").exists()
    forbidden = {"brier", "log_loss", "pnl", "outcome_yes", "probability_scores"}

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for child in value.values() for key in keys(child)}
        if isinstance(value, list):
            return {key for child in value for key in keys(child)}
        return set()

    assert not keys(payload) & forbidden
    original = destination.read_bytes()
    again = runner.invoke(app, ["forward-report", "--output", str(destination)])
    assert again.exit_code != 0
    assert destination.read_bytes() == original
    forbidden_output = runner.invoke(
        app, ["forward-report", "--output", str(tmp_path / "default.db")]
    )
    assert forbidden_output.exit_code != 0
    assert not (tmp_path / "default.db").exists()


def test_invalid_protocol_errors_redact_secrets_without_database_creation(
    tmp_path: Path, clock: dict[str, datetime]
) -> None:
    missing = tmp_path / "missing-token=cli-test-secret"
    result = runner.invoke(app, ["forward-register", "--protocol", str(missing)])
    assert result.exit_code != 0
    assert "cli-test-secret" not in result.output
    assert not (tmp_path / "default.db").exists()


def test_structured_output_preserves_nonsecret_flags_but_redacts_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    forward_cli._emit(
        {
            "delivery_authorization": False,
            "authorization": True,
            "access_token": None,
            "nested": [
                {"client_secret": False, "password": None},
                {"authorization": "Bearer cli-test-secret", "api_key": "cli-test-key"},
            ],
            "error": "https://user:cli-test-password@example.test/?token=cli-test-token",
        }
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["delivery_authorization"] is False
    assert payload["authorization"] is True
    assert payload["access_token"] is None
    assert payload["nested"] == [
        {"client_secret": False, "password": None},
        {"authorization": "[REDACTED]", "api_key": "[REDACTED]"},
    ]
    assert "cli-test-" not in output
    assert "[REDACTED]" in payload["error"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_is_rejected_before_writing(
    value: float, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "report.json"
    with pytest.raises(ValueError, match="Out of range float"):
        forward_cli._emit({"nested": [{"value": value}]}, destination)
    assert not destination.exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"capture": {"status": "failed"}, "resolution": {"status": "valid"}},
        {"capture": {"status": "valid"}, "resolution": {"status": "partial"}},
        {"status": "valid", "capture_clocks": [{"status": "failed", "error": None}]},
        {"status": "valid", "execution_clocks": [{"status": "failed", "error": None}]},
        {"status": "valid", "capture_clocks": [{"status": "valid", "error": "unavailable"}]},
    ],
)
def test_nested_operational_failures_are_not_success(payload: dict[str, Any]) -> None:
    assert forward_cli._failed(payload)


def test_successful_diagnostic_with_absent_errors_is_not_failure() -> None:
    assert not forward_cli._failed(
        {
            "status": "valid",
            "eligible": False,
            "error": None,
            "capture_clocks": [{"status": "valid", "error": None}],
            "execution_clocks": [{"status": "valid", "error": None}],
        }
    )
