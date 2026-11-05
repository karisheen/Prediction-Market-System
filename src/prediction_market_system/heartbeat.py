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

from prediction_market_system.archive import ARCHIVE_COVERAGE_VERSION
from prediction_market_system.db_backup import last_attempt, latest_manifest

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


def _campaign_archive_check(
    connection: sqlite3.Connection,
    configuration_json: str,
    *,
    series: str,
    symbol: str,
    now: datetime,
) -> HeartbeatCheck:
    configuration = json.loads(configuration_json)
    if not isinstance(configuration, dict):
        raise ValueError("active campaign configuration is not an object")
    start = datetime.fromisoformat(configuration["campaign_start"].replace("Z", "+00:00"))
    if start.tzinfo is None or start.utcoffset() != timedelta(0):
        raise ValueError("active campaign start must be timezone-aware UTC")
    if start != start.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("active campaign start must be midnight UTC")
    period = configuration["period_minutes"]
    if type(period) is not int or period not in (1, 60, 1440):
        raise ValueError("active campaign has an invalid archive period")
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    expected_days = max((today - start).days, 0)
    rows = connection.execute(
        """
        SELECT start_at, end_at, status, counts_json, error
        FROM paper_validation_archive_runs
        WHERE series_ticker = ? AND symbol = ? AND period_interval_minutes = ?
          AND start_at >= ? AND start_at < ?
        ORDER BY start_at
        """,
        (series, symbol, period, start.isoformat(), today.isoformat()),
    ).fetchall()
    windows: dict[datetime, str | None] = {}
    failed_windows: set[datetime] = set()
    for row in rows:
        window_start = datetime.fromisoformat(str(row["start_at"]).replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(str(row["end_at"]).replace("Z", "+00:00"))
        if (
            window_start.utcoffset() != timedelta(0)
            or window_end.utcoffset() != timedelta(0)
            or window_start != window_start.replace(hour=0, minute=0, second=0, microsecond=0)
            or window_end - window_start != timedelta(days=1)
            or window_start in windows
        ):
            raise ValueError(f"invalid or ambiguous archive window at {row['start_at']}")
        if row["status"] not in ("succeeded", "failed", "running"):
            raise ValueError(f"invalid archive status at {row['start_at']}")
        if row["status"] == "failed":
            failed_windows.add(window_start)
        counts = json.loads(row["counts_json"]) if row["counts_json"] is not None else {}
        if not isinstance(counts, dict):
            raise ValueError(f"archive counts are not an object at {row['start_at']}")
        if (
            row["status"] == "succeeded"
            and counts.get("coverage_complete") == 1
            and counts.get("coverage_version") == ARCHIVE_COVERAGE_VERSION
        ):
            windows[window_start] = None
        else:
            reason = (
                "incomplete or unverified coverage"
                if row["status"] == "succeeded"
                else str(row["status"])
            )
            if row["error"]:
                error = str(row["error"]).split(" for url ")[0]
                reason += f": {error[:200]}"
            windows[window_start] = reason

    missing = incomplete = pending = 0
    first_gap: datetime | None = None
    examples: list[str] = []
    for offset in range(expected_days):
        day = start + timedelta(days=offset)
        # The daily collector runs after UTC midnight. Give the newest completed
        # day one scheduling cycle, but never defer a recorded failure.
        if (
            windows.get(day, "missing") is not None
            and day not in failed_windows
            and now < day + timedelta(days=2)
        ):
            pending += 1
            continue
        if day not in windows:
            missing += 1
            reason = "missing"
        elif windows[day] is not None:
            incomplete += 1
            reason = str(windows[day])
        else:
            continue
        if first_gap is None:
            first_gap = day
        if len(examples) < 5:
            examples.append(f"{day.date().isoformat()} ({reason})")
    covered = expected_days - missing - incomplete - pending
    pending_detail = (
        f"{pending} completed UTC day pending within 24h daily scheduling grace; "
        if pending
        else ""
    )
    if first_gap is not None:
        return HeartbeatCheck(
            "archive",
            "Validation archive",
            False,
            f"active campaign coverage backlog: {covered}/{expected_days} completed UTC days "
            f"certified at {period}m; {missing} missing, {incomplete} incomplete; "
            f"{pending_detail}awaiting catch-up: {', '.join(examples)}",
            since=first_gap + timedelta(days=1),
        )
    return HeartbeatCheck(
        "archive",
        "Validation archive",
        True,
        f"active campaign collection current: {covered}/{expected_days} completed UTC days "
        f"certified at {period}m; {pending_detail}accumulating evidence is not model approval",
    )


def _archive_check(
    connection: sqlite3.Connection,
    *,
    series: str,
    symbol: str,
    now: datetime,
    lookback: timedelta,
) -> HeartbeatCheck:
    try:
        campaigns = connection.execute(
            """
            SELECT configuration_json FROM validation_campaign_registrations
            WHERE series_ticker = ? AND symbol = ? AND superseded_at IS NULL
            LIMIT 2
            """,
            (series, symbol),
        ).fetchall()
        if len(campaigns) > 1:
            raise ValueError("multiple active campaign registrations")
        if campaigns:
            return _campaign_archive_check(
                connection,
                campaigns[0]["configuration_json"],
                series=series,
                symbol=symbol,
                now=now,
            )
        rows = connection.execute(
            """
            SELECT start_at, error FROM paper_validation_archive_runs
            WHERE series_ticker = ? AND symbol = ? AND status = 'failed' AND start_at >= ?
            ORDER BY start_at
            """,
            (series, symbol, (now - lookback).isoformat()),
        ).fetchall()
        if rows:
            days = ", ".join(_parse(row["start_at"]).date().isoformat() for row in rows)
            error = str(rows[-1]["error"] or "unknown error").split(" for url ")[0]
            return HeartbeatCheck(
                "archive",
                "Validation archive",
                False,
                f"failed days awaiting catch-up: {days} (latest error: {error[:200]})",
            )
    except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError) as exc:
        return HeartbeatCheck(
            "archive",
            "Validation archive",
            False,
            f"cannot establish archive coverage from metadata: {exc}",
        )
    return HeartbeatCheck("archive", "Validation archive", True, "no unrecovered failed days")


def collect_checks(
    database_path: Path,
    *,
    series_ticker: str,
    symbol: str,
    now: datetime,
    max_scan_age: timedelta,
    max_research_age: timedelta,
    archive_lookback: timedelta,
    backup_dir: Path,
    max_backup_age: timedelta,
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
        archive = _archive_check(
            connection,
            series=series,
            symbol=normalized_symbol,
            now=now,
            lookback=archive_lookback,
        )

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

    checks.append(archive)
    checks.append(backup_check(backup_dir, now=now, max_age=max_backup_age))
    return checks


def backup_check(backup_dir: Path, *, now: datetime, max_age: timedelta) -> HeartbeatCheck:
    try:
        manifest = latest_manifest(backup_dir)
        attempt = last_attempt(backup_dir)
    except (OSError, ValueError, KeyError) as exc:
        return HeartbeatCheck(
            "backup", "Database backup", False, f"unreadable backup records: {exc}"
        )

    if attempt is not None and attempt.get("status") == "failed":
        failed_at = _parse(attempt.get("completed_at") or attempt["started_at"])
        error = str(attempt.get("error") or "unknown error")
        previous = (
            f"; last success {_format_time(_parse(manifest['created_at']))}"
            if manifest is not None
            else "; no successful backup"
        )
        return HeartbeatCheck(
            "backup",
            "Database backup",
            False,
            f"last attempt {_format_time(failed_at)} failed: {error[:200]}{previous}",
            since=failed_at,
        )
    if manifest is None:
        return HeartbeatCheck(
            "backup", "Database backup", False, f"no successful backup in {backup_dir}"
        )

    created_at = _parse(manifest["created_at"])
    age = now - created_at
    size = int(manifest["backup_bytes"]) / (1 << 30)
    return HeartbeatCheck(
        "backup",
        "Database backup",
        age <= max_age,
        f"last backup {_format_time(created_at)} ({_format_age(age)} ago, {size:.2f} GiB)",
        since=created_at,
    )


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
