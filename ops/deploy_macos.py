from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

APP_LABEL_PREFIX = "com.karisheen.prediction-market"
SOURCE_IGNORES = {
    ".env",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
}


@dataclass(frozen=True)
class AgentSpec:
    suffix: str
    arguments: tuple[str, ...]
    schedule: dict[str, Any]
    log_name: str

    @property
    def label(self) -> str:
        return f"{APP_LABEL_PREFIX}-{self.suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, verify, and atomically activate the macOS paper system."
    )
    parser.add_argument(
        "--app-root",
        type=Path,
        default=Path.home() / "Library/Application Support/PredictionMarketSystem",
    )
    parser.add_argument("--series", default="KXBTC")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--campaign-start", default="2026-08-07T00:00:00Z")
    parser.add_argument("--watch-retention-days", type=int, default=14)
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip release tests and static checks. Intended only for emergency rollback.",
    )
    return parser.parse_args()


def _run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=check,
        env=env,
        text=True,
    )


def _git_revision(source: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=source,
        text=True,
    )
    if status.strip():
        raise RuntimeError("refusing to deploy a dirty working tree; commit the release first")
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=source,
        text=True,
    ).strip()


def _ignore_source(_path: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in SOURCE_IGNORES}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _agent_specs(
    *,
    series: str,
    symbol: str,
    campaign_start: str,
    watch_retention_days: int,
) -> tuple[AgentSpec, ...]:
    return (
        AgentSpec(
            suffix="alerts",
            arguments=(
                "paper-alerts",
                "--series",
                series,
                "--symbol",
                symbol,
                "--interval",
                "60",
                "--realized-window-days",
                "30",
            ),
            schedule={"RunAtLoad": True, "StartInterval": 300},
            log_name="paper-alerts.log",
        ),
        AgentSpec(
            suffix="research",
            arguments=(
                "paper-alert-research",
                "--series",
                series,
                "--symbol",
                symbol,
                "--interval",
                "60",
                "--realized-window-days",
                "30",
            ),
            schedule={"RunAtLoad": True, "StartInterval": 3600},
            log_name="paper-alert-research.log",
        ),
        AgentSpec(
            suffix="archive",
            arguments=(
                "paper-alert-archive",
                "--series",
                series,
                "--symbol",
                symbol,
                "--campaign-start",
                campaign_start,
                "--period",
                "1",
                "--catch-up-days",
                "7",
                "--max-events",
                "100",
                "--range-contracts-per-event",
                "3",
                "--history-hours",
                "24",
            ),
            schedule={"StartCalendarInterval": {"Hour": 1, "Minute": 15}},
            log_name="paper-alert-archive.log",
        ),
        AgentSpec(
            suffix="validation",
            arguments=(
                "paper-alert-validate",
                "--series",
                series,
                "--symbol",
                symbol,
                "--campaign-start",
                campaign_start,
                "--period",
                "1",
                "--train-days",
                "90",
                "--test-days",
                "30",
                "--step-days",
                "30",
                "--realized-window-days",
                "30",
                "--minimum-calibration-samples",
                "30",
                "--minimum-validation-events",
                "20",
                "--minimum-validation-folds",
                "2",
                "--minimum-return-on-cost",
                "0",
                "--maximum-brier-score",
                "0.25",
                "--max-events",
                "5000",
                "--send-discord",
            ),
            schedule={
                "StartCalendarInterval": {"Weekday": 1, "Hour": 2, "Minute": 15}
            },
            log_name="paper-alert-validation.log",
        ),
        AgentSpec(
            suffix="maintenance",
            arguments=(
                "paper-alert-maintain",
                "--series",
                series,
                "--watch-retention-days",
                str(watch_retention_days),
                "--apply",
            ),
            schedule={"StartCalendarInterval": {"Hour": 3, "Minute": 15}},
            log_name="paper-alert-maintenance.log",
        ),
    )


def _plist_payload(
    spec: AgentSpec,
    *,
    app_path: Path,
    data_path: Path,
    uv_path: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Label": spec.label,
        "ProgramArguments": [uv_path, "run", "pms", *spec.arguments],
        "WorkingDirectory": str(app_path),
        "ProcessType": "Background",
        "StandardOutPath": str(data_path / spec.log_name),
        "StandardErrorPath": str(data_path / spec.log_name),
    }
    payload.update(spec.schedule)
    return payload


def _write_plist(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".plist.new")
    with temporary.open("wb") as file:
        plistlib.dump(payload, file, sort_keys=False)
    os.replace(temporary, path)


def _stop_agents(uid: int, specs: tuple[AgentSpec, ...]) -> None:
    for spec in specs:
        _run(
            ["launchctl", "bootout", f"gui/{uid}/{spec.label}"],
            check=False,
        )


def _start_agents(uid: int, launch_agents: Path, specs: tuple[AgentSpec, ...]) -> None:
    started: list[AgentSpec] = []
    try:
        for spec in specs:
            _run(
                ["launchctl", "bootstrap", f"gui/{uid}", str(launch_agents / f"{spec.label}.plist")]
            )
            started.append(spec)
    except subprocess.CalledProcessError:
        _stop_agents(uid, tuple(started))
        raise


def _prepare_shared_state(app_root: Path, app_path: Path) -> tuple[Path, Path]:
    shared_env = app_root / ".env"
    shared_data = app_root / "data"
    existing_app = app_path.resolve() if app_path.exists() else app_path

    if not shared_env.exists():
        existing_env = existing_app / ".env"
        if not existing_env.exists():
            raise RuntimeError(f"runtime configuration is missing: {existing_env}")
        shutil.copy2(existing_env, shared_env)
        shared_env.chmod(0o600)

    if not shared_data.exists():
        existing_data = existing_app / "data"
        if existing_data.exists() and not existing_data.is_symlink():
            existing_data.rename(shared_data)
            existing_data.symlink_to(shared_data, target_is_directory=True)
        else:
            shared_data.mkdir(parents=True)
    return shared_env, shared_data


def _build_release(
    *,
    source: Path,
    release: Path,
    shared_env: Path,
    shared_data: Path,
    revision: str,
    skip_checks: bool,
) -> None:
    shutil.copytree(source, release, ignore=_ignore_source)
    (release / ".env").symlink_to(shared_env)
    (release / "data").symlink_to(shared_data, target_is_directory=True)
    (release / "DEPLOYED_REVISION").write_text(f"{revision}\n", encoding="utf-8")

    _run(["uv", "sync", "--frozen"], cwd=release)
    if skip_checks:
        return
    check_environment = os.environ.copy()
    check_environment["PMS_DATABASE_PATH"] = str(release / ".deployment-check.db")
    for command in (
        ["uv", "run", "pytest"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "mypy"],
    ):
        _run(command, cwd=release, env=check_environment)
    for suffix in ("", "-shm", "-wal"):
        (release / f".deployment-check.db{suffix}").unlink(missing_ok=True)


def _activate_release(app_path: Path, release: Path, releases: Path, release_id: str) -> Path:
    temporary_link = app_path.with_name("app.new")
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(release, target_is_directory=True)

    if app_path.is_symlink():
        previous = app_path.resolve()
        os.replace(temporary_link, app_path)
        return previous
    if app_path.exists():
        previous = releases / f"pre-atomic-{release_id}"
        app_path.rename(previous)
        os.replace(temporary_link, app_path)
        return previous

    os.replace(temporary_link, app_path)
    return release


def _point_app_at(app_path: Path, release: Path) -> None:
    temporary_link = app_path.with_name("app.rollback")
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(release, target_is_directory=True)
    os.replace(temporary_link, app_path)


def main() -> int:
    args = _parse_args()
    if args.watch_retention_days < 1:
        raise ValueError("--watch-retention-days must be positive")

    source = Path(__file__).resolve().parents[1]
    revision = _git_revision(source)
    now = datetime.now(UTC)
    release_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{revision}"
    app_root = args.app_root.expanduser().resolve()
    releases = app_root / "releases"
    release = releases / release_id
    app_path = app_root / "app"
    launch_agents = Path.home() / "Library/LaunchAgents"
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise RuntimeError("uv is not available on PATH")

    specs = _agent_specs(
        series=args.series.upper(),
        symbol=args.symbol.upper(),
        campaign_start=args.campaign_start,
        watch_retention_days=args.watch_retention_days,
    )
    uid = os.getuid()
    app_root.mkdir(parents=True, exist_ok=True)
    releases.mkdir(parents=True, exist_ok=True)
    launch_agents.mkdir(parents=True, exist_ok=True)

    _stop_agents(uid, specs)
    previous: Path | None = None
    try:
        shared_env, shared_data = _prepare_shared_state(app_root, app_path)
        _build_release(
            source=source,
            release=release,
            shared_env=shared_env,
            shared_data=shared_data,
            revision=revision,
            skip_checks=args.skip_checks,
        )
        previous = _activate_release(app_path, release, releases, release_id)
        _run([uv_path, "run", "pms", "init-db"], cwd=app_path)
        for spec in specs:
            _write_plist(
                launch_agents / f"{spec.label}.plist",
                _plist_payload(
                    spec,
                    app_path=app_path,
                    data_path=shared_data,
                    uv_path=uv_path,
                ),
            )
        _start_agents(uid, launch_agents, specs)
    except Exception:
        pre_atomic = releases / f"pre-atomic-{release_id}"
        if previous is not None and app_path.is_symlink():
            _point_app_at(app_path, previous)
        elif previous is None and pre_atomic.exists() and not app_path.exists():
            pre_atomic.rename(app_path)
        active_release = app_path.is_symlink() and app_path.resolve() == release
        if release.exists() and release != previous and not active_release:
            shutil.rmtree(release)
        restorable_specs = tuple(
            spec for spec in specs if (launch_agents / f"{spec.label}.plist").exists()
        )
        _start_agents(uid, launch_agents, restorable_specs)
        raise

    print(f"Activated {release_id}")
    print(f"Application pointer: {app_path} -> {release}")
    print("Paper-alert schedule is shadow-only; Discord delivery is disabled.")
    print(f"Detailed WATCH retention: {args.watch_retention_days} days")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
