import gc
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from prediction_market_system.cli import app
from prediction_market_system.discord import DiscordWebhookClient
from prediction_market_system.heartbeat import HeartbeatCheck, collect_checks, run_heartbeat
from prediction_market_system.storage import SQLiteRepository

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

runner = CliRunner()


def database(tmp_path: Path) -> Path:
    path = tmp_path / "heartbeat.db"
    SQLiteRepository(path).initialize()
    return path


def execute(path: Path, sql: str, parameters: tuple[object, ...]) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(sql, parameters)


def record_scan(path: Path, observed_at: datetime, cycle_id: str = "cycle") -> None:
    execute(
        path,
        "INSERT INTO paper_alert_cycles VALUES (?, 'KXBTC', ?)",
        (cycle_id, observed_at.isoformat()),
    )


def record_research(path: Path, completed_at: datetime, status: str = "succeeded") -> None:
    execute(
        path,
        """
            INSERT INTO research_data_sync_runs (
                run_id, status, symbol, started_at, completed_at, request_json
            ) VALUES (?, ?, 'BTC', ?, ?, '{}')
            """,
        (
            f"run-{completed_at.isoformat()}-{status}",
            status,
            (completed_at - timedelta(minutes=1)).isoformat(),
            completed_at.isoformat(),
        ),
    )


def record_archive(path: Path, day: datetime, status: str, error: str | None = None) -> None:
    execute(
        path,
        """
            INSERT OR REPLACE INTO paper_validation_archive_runs (
                series_ticker, symbol, start_at, end_at, period_interval_minutes,
                status, started_at, completed_at, error
            ) VALUES ('KXBTC', 'BTC', ?, ?, 1, ?, ?, ?, ?)
            """,
        (
            day.isoformat(),
            (day + timedelta(days=1)).isoformat(),
            status,
            NOW.isoformat(),
            NOW.isoformat(),
            error,
        ),
    )


def checks_at(path: Path, now: datetime = NOW) -> list[HeartbeatCheck]:
    return collect_checks(
        path,
        series_ticker="KXBTC",
        symbol="BTC",
        now=now,
        max_scan_age=timedelta(minutes=30),
        max_research_age=timedelta(minutes=150),
        archive_lookback=timedelta(days=8),
    )


class Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def __call__(self, content: str) -> str:
        self.messages.append(content)
        return "message-id"


def healthy_database(tmp_path: Path) -> Path:
    path = database(tmp_path)
    record_scan(path, NOW - timedelta(minutes=4))
    record_research(path, NOW - timedelta(minutes=40))
    record_archive(path, datetime(2026, 10, 17, tzinfo=UTC), "succeeded")
    return path


@pytest.mark.asyncio
async def test_healthy_jobs_send_nothing(tmp_path: Path) -> None:
    path = healthy_database(tmp_path)
    state_path = tmp_path / "state.json"
    send = Recorder()

    result = await run_heartbeat(
        checks=collect_checks(
            path,
            series_ticker="KXBTC",
            symbol="BTC",
            now=NOW,
            max_scan_age=timedelta(minutes=30),
            max_research_age=timedelta(minutes=150),
            archive_lookback=timedelta(days=8),
        ),
        state_path=state_path,
        series_ticker="KXBTC",
        now=NOW,
        send=send,
    )

    assert all(check.healthy for check in result.checks)
    assert send.messages == []
    assert not state_path.exists()


@pytest.mark.asyncio
async def test_stalled_scans_alert_once_then_recover_once(tmp_path: Path) -> None:
    path = healthy_database(tmp_path)
    state_path = tmp_path / "state.json"
    send = Recorder()
    stalled_at = NOW + timedelta(hours=5)

    async def beat(now: datetime) -> None:
        checks = collect_checks(
            path,
            series_ticker="KXBTC",
            symbol="BTC",
            now=now,
            max_scan_age=timedelta(minutes=30),
            max_research_age=timedelta(hours=24),
            archive_lookback=timedelta(days=8),
        )
        await run_heartbeat(
            checks=checks,
            state_path=state_path,
            series_ticker="KXBTC",
            now=now,
            send=send,
        )

    await beat(stalled_at)
    await beat(stalled_at + timedelta(minutes=10))

    assert len(send.messages) == 1
    assert "Live scans problem" in send.messages[0]
    assert "5h 4m ago" in send.messages[0]
    assert "scans" in json.loads(state_path.read_text())["active"]

    record_scan(path, stalled_at + timedelta(minutes=15), cycle_id="resumed")
    await beat(stalled_at + timedelta(minutes=20))
    await beat(stalled_at + timedelta(minutes=30))

    assert len(send.messages) == 2
    assert "Live scans recovered after 5h 24m" in send.messages[1]
    assert json.loads(state_path.read_text())["active"] == {}


@pytest.mark.asyncio
async def test_failed_archive_day_is_reported_until_caught_up(tmp_path: Path) -> None:
    path = healthy_database(tmp_path)
    failed_day = datetime(2026, 10, 16, tzinfo=UTC)
    record_archive(path, failed_day, "failed", "Client error '429 Too Many Requests'")
    state_path = tmp_path / "state.json"
    send = Recorder()

    await run_heartbeat(
        checks=checks_at(path),
        state_path=state_path,
        series_ticker="KXBTC",
        now=NOW,
        send=send,
    )
    record_archive(path, failed_day, "succeeded")
    await run_heartbeat(
        checks=checks_at(path),
        state_path=state_path,
        series_ticker="KXBTC",
        now=NOW,
        send=send,
    )

    assert len(send.messages) == 2
    assert "Validation archive problem" in send.messages[0]
    assert "2026-10-16" in send.messages[0]
    assert "429 Too Many Requests" in send.messages[0]
    assert "Validation archive recovered" in send.messages[1]


@pytest.mark.asyncio
async def test_stale_research_and_missing_scans_are_problems(tmp_path: Path) -> None:
    path = database(tmp_path)
    record_research(path, NOW - timedelta(hours=1), status="failed")
    record_research(path, NOW - timedelta(hours=4))

    checks = {check.key: check for check in checks_at(path)}

    assert not checks["scans"].healthy
    assert checks["scans"].detail == "no scans recorded"
    assert not checks["research"].healthy
    assert "4h 0m ago" in checks["research"].detail
    assert checks["archive"].healthy


@pytest.mark.asyncio
async def test_failed_delivery_keeps_problem_pending(tmp_path: Path) -> None:
    path = database(tmp_path)
    record_research(path, NOW)
    state_path = tmp_path / "state.json"

    async def fail(content: str) -> str:
        raise httpx.ConnectError("offline")

    result = await run_heartbeat(
        checks=checks_at(path),
        state_path=state_path,
        series_ticker="KXBTC",
        now=NOW,
        send=fail,
    )

    assert [notice.key for notice, _ in result.failed] == ["scans"]
    assert not state_path.exists()


def test_dry_run_command_neither_sends_nor_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = database(tmp_path)
    record_research(path, datetime.now(UTC))
    gc.collect()
    before = path.read_bytes()
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(path))
    monkeypatch.delenv("PMS_DISCORD_WEBHOOK_URL", raising=False)

    result = runner.invoke(
        app,
        [
            "paper-alert-heartbeat",
            "--series",
            "KXBTC",
            "--symbol",
            "BTC",
            "--state-path",
            str(state_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Would send:" in result.output
    assert "Live scans problem" in result.output
    assert not state_path.exists()
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_operational_notice_disables_mentions() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "123"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DiscordWebhookClient(
        "https://discord.com/api/webhooks/1/token",
        client=http_client,
    )

    message_id = await client.send_operational_notice("@everyone scans stalled")

    assert message_id == "123"
    assert payloads == [{"content": "@everyone scans stalled", "allowed_mentions": {"parse": []}}]
    await http_client.aclose()
