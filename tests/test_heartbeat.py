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
from prediction_market_system.db_backup import LAST_ATTEMPT_NAME
from prediction_market_system.discord import DiscordWebhookClient
from prediction_market_system.heartbeat import (
    HeartbeatCheck,
    backup_check,
    collect_checks,
    run_heartbeat,
)
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


def record_backup(backup_dir: Path, created_at: datetime) -> None:
    backup_dir.mkdir(exist_ok=True)
    name = f"prediction_markets-{created_at.strftime('%Y%m%dT%H%M%SZ')}.db.gz"
    (backup_dir / name).write_bytes(b"backup")
    (backup_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "created_at": created_at.isoformat(),
                "backup_file": name,
                "backup_bytes": 3 << 29,
            }
        )
    )
    record_attempt(backup_dir, created_at, "succeeded")


def record_attempt(
    backup_dir: Path, started_at: datetime, status: str, error: str | None = None
) -> None:
    backup_dir.mkdir(exist_ok=True)
    (backup_dir / LAST_ATTEMPT_NAME).write_text(
        json.dumps(
            {
                "status": status,
                "started_at": started_at.isoformat(),
                "completed_at": (started_at + timedelta(minutes=5)).isoformat(),
                "error": error,
            }
        )
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
        backup_dir=path.parent / "backups",
        max_backup_age=timedelta(hours=36),
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
    record_backup(tmp_path / "backups", NOW - timedelta(hours=8))
    return path


@pytest.mark.asyncio
async def test_healthy_jobs_send_nothing(tmp_path: Path) -> None:
    path = healthy_database(tmp_path)
    state_path = tmp_path / "state.json"
    send = Recorder()

    result = await run_heartbeat(
        checks=checks_at(path),
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
            backup_dir=tmp_path / "backups",
            max_backup_age=timedelta(hours=36),
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
    record_backup(tmp_path / "backups", NOW)
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
            "--backup-dir",
            str(tmp_path / "backups"),
            "--max-backup-age-hours",
            "36",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Would send:" in result.output
    assert "Live scans problem" in result.output
    assert "Database backup problem" in result.output
    assert not state_path.exists()
    assert not (tmp_path / "backups").exists()
    assert path.read_bytes() == before


def test_missing_backup_directory_is_reported_without_creating_it(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"

    check = backup_check(backup_dir, now=NOW, max_age=timedelta(hours=36))

    assert not check.healthy
    assert check.detail == f"no successful backup in {backup_dir}"
    assert not backup_dir.exists()


def test_stale_backup_is_a_problem(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    record_backup(backup_dir, NOW - timedelta(hours=40))

    check = backup_check(backup_dir, now=NOW, max_age=timedelta(hours=36))

    assert not check.healthy
    assert check.since == NOW - timedelta(hours=40)
    assert "1d 16h 0m ago" in check.detail
    assert "1.50 GiB" in check.detail


def test_backup_manifest_without_archive_is_not_success(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    record_backup(backup_dir, NOW - timedelta(hours=2))
    for archive in backup_dir.glob("*.db.gz"):
        archive.unlink()

    check = backup_check(backup_dir, now=NOW, max_age=timedelta(hours=36))

    assert not check.healthy
    assert check.detail.startswith("no successful backup")


@pytest.mark.asyncio
async def test_failed_backup_attempt_alerts_until_next_success(tmp_path: Path) -> None:
    path = healthy_database(tmp_path)
    backup_dir = tmp_path / "backups"
    failed_at = NOW - timedelta(hours=1)
    record_attempt(backup_dir, failed_at, "failed", "refusing to back up: 3.0 GiB free")
    state_path = tmp_path / "state.json"
    send = Recorder()

    checks = {check.key: check for check in checks_at(path)}
    assert not checks["backup"].healthy
    assert "failed: refusing to back up: 3.0 GiB free" in checks["backup"].detail
    assert "last success 2026-10-18 04:00 UTC" in checks["backup"].detail

    await run_heartbeat(
        checks=checks_at(path),
        state_path=state_path,
        series_ticker="KXBTC",
        now=NOW,
        send=send,
    )
    record_backup(backup_dir, NOW - timedelta(minutes=10))
    await run_heartbeat(
        checks=checks_at(path),
        state_path=state_path,
        series_ticker="KXBTC",
        now=NOW,
        send=send,
    )

    assert len(send.messages) == 2
    assert "Database backup problem" in send.messages[0]
    assert "Database backup recovered after 55m" in send.messages[1]


def test_unreadable_backup_records_are_a_problem(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    (backup_dir / LAST_ATTEMPT_NAME).write_text("{not json")

    check = backup_check(backup_dir, now=NOW, max_age=timedelta(hours=36))

    assert not check.healthy
    assert check.detail.startswith("unreadable backup records")


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
