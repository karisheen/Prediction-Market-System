from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
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
    "backups",
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


@dataclass(frozen=True)
class AgentState:
    plist: bytes | None
    loaded: bool
    running: bool


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
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=check,
        env=env,
        text=True,
        capture_output=capture_output,
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
    specs = (
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
                "--spot-interval",
                "1",
                "--research-interval",
                "60",
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
            schedule={"StartCalendarInterval": {"Weekday": 1, "Hour": 2, "Minute": 15}},
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
                "--batch-size",
                "5000",
                "--apply",
            ),
            schedule={"StartCalendarInterval": {"Hour": 3, "Minute": 15}},
            log_name="paper-alert-maintenance.log",
        ),
        AgentSpec(
            suffix="backup",
            arguments=("db-backup", "--keep", "3"),
            schedule={"StartCalendarInterval": {"Hour": 4, "Minute": 15}},
            log_name="db-backup.log",
        ),
        AgentSpec(
            suffix="heartbeat",
            arguments=(
                "paper-alert-heartbeat",
                "--series",
                series,
                "--symbol",
                symbol,
            ),
            schedule={"RunAtLoad": True, "StartInterval": 600},
            log_name="paper-alert-heartbeat.log",
        ),
    )
    if series == "KXBTC" and symbol == "BTC":
        specs += (
            AgentSpec(
                suffix="forward",
                arguments=("forward-capture",),
                schedule={"StartCalendarInterval": {"Minute": 45}},
                log_name="forward-capture.log",
            ),
        )
    return specs


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


def _write_plist_bytes(path: Path, contents: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(contents)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plist(path: Path, payload: dict[str, Any]) -> None:
    _write_plist_bytes(path, plistlib.dumps(payload, sort_keys=False))


def _agent_status(uid: int, label: str) -> tuple[bool, bool]:
    result = _run(
        ["launchctl", "print", f"gui/{uid}/{label}"],
        check=False,
        capture_output=True,
    )
    if result.returncode == 113:
        return False, False
    result.check_returncode()
    lines = (result.stdout or "").splitlines()
    running = any(
        line.strip() == "state = running" or line.strip().startswith("pid = ") for line in lines
    )
    return True, running


def _snapshot_agents(
    uid: int, launch_agents: Path, labels: tuple[str, ...]
) -> dict[str, AgentState]:
    states = {}
    for label in labels:
        path = launch_agents / f"{label}.plist"
        loaded, running = _agent_status(uid, label)
        contents = path.read_bytes() if path.exists() else None
        if loaded and contents is None:
            raise RuntimeError(f"cannot safely restore loaded agent without its plist: {label}")
        states[label] = AgentState(contents, loaded, running)
    return states


def _restore_agents(
    *,
    uid: int,
    launch_agents: Path,
    states: dict[str, AgentState],
    changed: set[str],
    app_path: Path,
    previous: Path | None,
    pre_atomic: Path,
    had_app: bool,
) -> list[str]:
    """Restore exact predeployment files and loaded membership, not desired specs."""
    errors: list[str] = []
    stopped: set[str] = set()
    restored: set[str] = set()
    for label in states:
        if label not in changed:
            continue
        try:
            loaded, _ = _agent_status(uid, label)
            if loaded:
                _run(["launchctl", "bootout", f"gui/{uid}/{label}"])
            stopped.add(label)
        except Exception as exc:
            errors.append(f"stopping {label}: {exc}")
    pointer_restored = False
    try:
        if previous is not None and app_path.is_symlink():
            if had_app:
                _point_app_at(app_path, previous)
            else:
                app_path.unlink()
        elif previous is None and pre_atomic.exists() and not app_path.exists():
            pre_atomic.rename(app_path)
        pointer_restored = True
    except Exception as exc:
        errors.append(f"application pointer: {exc}")
    for label, state in states.items():
        if label not in changed:
            continue
        path = launch_agents / f"{label}.plist"
        try:
            if state.plist is None:
                path.unlink(missing_ok=True)
            else:
                _write_plist_bytes(path, state.plist)
            restored.add(label)
        except Exception as exc:
            errors.append(f"plist {label}: {exc}")
    if pointer_restored:
        for label, state in states.items():
            if state.loaded and label in stopped and label in restored:
                try:
                    _run(
                        [
                            "launchctl",
                            "bootstrap",
                            f"gui/{uid}",
                            str(launch_agents / f"{label}.plist"),
                        ]
                    )
                except Exception as exc:
                    errors.append(f"loading {label}: {exc}")
    return errors


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
        ["uv", "run", "ruff", "format", "--check", "."],
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
    if release.exists():
        raise RuntimeError(f"release already exists: {release}")

    forward_label = f"{APP_LABEL_PREFIX}-forward"
    labels = tuple(dict.fromkeys([*(spec.label for spec in specs), forward_label]))
    states = _snapshot_agents(uid, launch_agents, labels)
    desired = {
        spec.label: _plist_payload(
            spec, app_path=app_path, data_path=app_root / "data", uv_path=uv_path
        )
        for spec in specs
    }
    preserved: set[str] = set()
    forward = states[forward_label]
    if forward.loaded and forward_label in desired:
        try:
            matches = (
                forward.plist is not None
                and plistlib.loads(forward.plist) == desired[forward_label]
            )
        except (plistlib.InvalidFileException, ValueError):
            matches = False
        if not matches:
            # Even an idle job can start while gates run. Never reload a loaded
            # changed definition: an operator must first drain it safely.
            raise RuntimeError("loaded forward agent definition changed; refusing unsafe reload")
        preserved.add(forward_label)
    if forward.loaded and forward_label not in desired and forward.running:
        raise RuntimeError("forward capture is active; refusing to remove its agent")

    previous: Path | None = None
    had_app = app_path.exists() or app_path.is_symlink()
    changed: set[str] = set()
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
        # Keep the existing forward job loaded through all gates and the atomic
        # handoff. Its running process keeps the old cwd; its next run uses app.
        if forward.loaded and forward_label not in desired:
            _, running = _agent_status(uid, forward_label)
            if running:
                raise RuntimeError("forward capture is active; refusing to remove its agent")
        for label, state in states.items():
            if label in preserved:
                continue
            changed.add(label)
            if state.loaded:
                _run(["launchctl", "bootout", f"gui/{uid}/{label}"])
        previous = _activate_release(app_path, release, releases, release_id)
        _run([uv_path, "run", "pms", "init-db"], cwd=app_path)
        if args.series.upper() == "KXBTC" and args.symbol.upper() == "BTC":
            # The release CLI resolves its own immutable protocol constant. A late
            # deployment succeeds only if this study was already preregistered.
            _run([uv_path, "run", "pms", "forward-register"], cwd=app_path)
        for label, payload in desired.items():
            if label not in preserved:
                _write_plist(launch_agents / f"{label}.plist", payload)
        if forward_label not in desired:
            (launch_agents / f"{forward_label}.plist").unlink(missing_ok=True)
        for spec in specs:
            if spec.label not in preserved:
                _run(
                    [
                        "launchctl",
                        "bootstrap",
                        f"gui/{uid}",
                        str(launch_agents / f"{spec.label}.plist"),
                    ]
                )
    except Exception as deployment_error:
        errors = _restore_agents(
            uid=uid,
            launch_agents=launch_agents,
            states=states,
            changed=changed,
            app_path=app_path,
            previous=previous,
            pre_atomic=releases / f"pre-atomic-{release_id}",
            had_app=had_app,
        )
        active_release = app_path.is_symlink() and app_path.resolve() == release
        if release.exists() and release != previous and not active_release and not errors:
            try:
                shutil.rmtree(release)
            except Exception as exc:
                errors.append(f"removing new release: {exc}")
        if errors:
            raise RuntimeError(
                "deployment rollback failed: " + "; ".join(errors)
            ) from deployment_error
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
