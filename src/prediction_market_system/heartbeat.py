"""Operational liveness checks for the scheduled paper-alert jobs."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SendNotice = Callable[[str], Awaitable[object]]


@dataclass(frozen=True)
class HeartbeatCheck:
    key: str
    label: str
    healthy: bool
    detail: str
    since: datetime | None = None


@dataclass(frozen=True)
class HeartbeatNotice:
    key: str
    recovered: bool
    content: str


@dataclass(frozen=True)
class HeartbeatResult:
    checks: list[HeartbeatCheck]
    notices: list[HeartbeatNotice]
    delivered: list[HeartbeatNotice]
    failed: list[tuple[HeartbeatNotice, str]]


def _parse(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_age(delta: timedelta) -> str:
    minutes = max(int(delta.total_seconds() // 60), 0)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    if not database_path.exists():
        raise FileNotFoundError(f"database not found: {database_path}")
    connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def collect_checks(
    database_path: Path,
    *,
    series_ticker: str,
    symbol: str,
    now: datetime,
    max_scan_age: timedelta,
    max_research_age: timedelta,
    archive_lookback: timedelta,
) -> list[HeartbeatCheck]:
    series = series_ticker.upper()
    normalized_symbol = symbol.upper()
    with closing(_connect_read_only(database_path)) as connection:
        scan_row = connection.execute(
            "SELECT MAX(observed_at) AS latest FROM paper_alert_cycles WHERE series_ticker = ?",
            (series,),
        ).fetchone()
        research_row = connection.execute(
            """
            SELECT MAX(COALESCE(completed_at, started_at)) AS latest
            FROM research_data_sync_runs
            WHERE symbol = ? AND status = 'succeeded'
            """,
            (normalized_symbol,),
        ).fetchone()
        archive_rows = connection.execute(
            """
            SELECT start_at, error
            FROM paper_validation_archive_runs
            WHERE series_ticker = ? AND symbol = ? AND status = 'failed' AND start_at >= ?
            ORDER BY start_at
            """,
            (series, normalized_symbol, (now - archive_lookback).isoformat()),
        ).fetchall()

    checks: list[HeartbeatCheck] = []

    if scan_row["latest"] is None:
        checks.append(HeartbeatCheck("scans", "Live scans", False, "no scans recorded"))
    else:
        latest_scan = _parse(scan_row["latest"])
        age = now - latest_scan
        checks.append(
            HeartbeatCheck(
                "scans",
                "Live scans",
                age <= max_scan_age,
                f"last scan {_format_time(latest_scan)} ({_format_age(age)} ago)",
                since=latest_scan,
            )
        )

    if research_row["latest"] is None:
        checks.append(
            HeartbeatCheck("research", "Research refresh", False, "no successful refresh recorded")
        )
    else:
        latest_research = _parse(research_row["latest"])
        age = now - latest_research
        checks.append(
            HeartbeatCheck(
                "research",
                "Research refresh",
                age <= max_research_age,
                f"last success {_format_time(latest_research)} ({_format_age(age)} ago)",
                since=latest_research,
            )
        )

    if archive_rows:
        days = ", ".join(_parse(row["start_at"]).date().isoformat() for row in archive_rows)
        error = str(archive_rows[-1]["error"] or "unknown error").split(" for url ")[0]
        checks.append(
            HeartbeatCheck(
                "archive",
                "Validation archive",
                False,
                f"failed days awaiting catch-up: {days} (latest error: {error[:200]})",
            )
        )
    else:
        checks.append(
            HeartbeatCheck("archive", "Validation archive", True, "no unrecovered failed days")
        )
    return checks


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    active = payload.get("active", {})
    return active if isinstance(active, dict) else {}


def save_state(path: Path, active: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.new")
    temporary.write_text(json.dumps({"active": active}, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def plan_notices(
    checks: list[HeartbeatCheck],
    active: dict[str, dict[str, Any]],
    *,
    series_ticker: str,
    now: datetime,
) -> list[HeartbeatNotice]:
    prefix = f"Prediction Market System health ({series_ticker.upper()})"
    notices: list[HeartbeatNotice] = []
    for check in checks:
        if not check.healthy and check.key not in active:
            notices.append(
                HeartbeatNotice(
                    check.key,
                    recovered=False,
                    content=f"⚠️ {prefix}: {check.label} problem — {check.detail}.",
                )
            )
        elif check.healthy and check.key in active:
            started = active[check.key].get("since") or active[check.key].get("alerted_at")
            duration = f" after {_format_age(now - _parse(started))}" if started else ""
            notices.append(
                HeartbeatNotice(
                    check.key,
                    recovered=True,
                    content=f"✅ {prefix}: {check.label} recovered{duration} — {check.detail}.",
                )
            )
    return notices


async def run_heartbeat(
    *,
    checks: list[HeartbeatCheck],
    state_path: Path,
    series_ticker: str,
    now: datetime,
    send: SendNotice | None,
) -> HeartbeatResult:
    active = load_state(state_path)
    notices = plan_notices(checks, active, series_ticker=series_ticker, now=now)
    if send is None:
        return HeartbeatResult(checks, notices, delivered=[], failed=[])

    by_key = {check.key: check for check in checks}
    delivered: list[HeartbeatNotice] = []
    failed: list[tuple[HeartbeatNotice, str]] = []
    for notice in notices:
        try:
            await send(notice.content)
        except Exception as exc:
            failed.append((notice, str(exc)))
            continue
        delivered.append(notice)
        if notice.recovered:
            active.pop(notice.key, None)
        else:
            since = by_key[notice.key].since
            active[notice.key] = {
                "alerted_at": now.isoformat(),
                "since": (since or now).isoformat(),
            }
    if delivered:
        save_state(state_path, active)
    return HeartbeatResult(checks, notices, delivered, failed)
