import argparse
import importlib.util
import plistlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_deploy_module() -> ModuleType:
    path = Path(__file__).parents[1] / "ops" / "deploy_macos.py"
    spec = importlib.util.spec_from_file_location("deploy_macos", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deploy = _load_deploy_module()
_agent_specs = deploy._agent_specs
_ignore_source = deploy._ignore_source
_plist_payload = deploy._plist_payload


def test_launch_agents_use_shadow_mode_and_schedule_maintenance() -> None:
    specs = _agent_specs(
        series="KXBTC",
        symbol="BTC",
        campaign_start="2026-08-07T00:00:00Z",
        watch_retention_days=14,
    )
    by_suffix = {spec.suffix: spec for spec in specs}

    assert "--send-discord" not in by_suffix["alerts"].arguments
    assert "--allow-unapproved-discord" not in by_suffix["alerts"].arguments
    validation_args = by_suffix["validation"].arguments
    assert validation_args[validation_args.index("--period") + 1] == "1"
    assert validation_args[validation_args.index("--spot-interval") + 1] == "1"
    assert validation_args[validation_args.index("--research-interval") + 1] == "60"
    assert by_suffix["maintenance"].arguments == (
        "paper-alert-maintain",
        "--series",
        "KXBTC",
        "--watch-retention-days",
        "14",
        "--batch-size",
        "5000",
        "--apply",
    )
    assert by_suffix["maintenance"].schedule == {"StartCalendarInterval": {"Hour": 3, "Minute": 15}}
    assert by_suffix["heartbeat"].arguments == (
        "paper-alert-heartbeat",
        "--series",
        "KXBTC",
        "--symbol",
        "BTC",
    )
    assert by_suffix["heartbeat"].schedule == {"RunAtLoad": True, "StartInterval": 600}
    assert by_suffix["backup"].arguments == ("db-backup", "--keep", "3")
    assert by_suffix["backup"].schedule == {"StartCalendarInterval": {"Hour": 4, "Minute": 15}}
    assert by_suffix["backup"].log_name == "db-backup.log"
    assert len(specs) == 8
    assert by_suffix["forward"].arguments == ("forward-capture",)
    assert by_suffix["forward"].schedule == {"StartCalendarInterval": {"Minute": 45}}
    assert "RunAtLoad" not in by_suffix["forward"].schedule
    assert by_suffix["forward"].log_name == "forward-capture.log"
    assert by_suffix["alerts"].schedule == {"RunAtLoad": True, "StartInterval": 300}
    assert by_suffix["research"].schedule == {"RunAtLoad": True, "StartInterval": 3600}
    assert by_suffix["archive"].schedule == {"StartCalendarInterval": {"Hour": 1, "Minute": 15}}
    assert by_suffix["validation"].schedule == {
        "StartCalendarInterval": {"Weekday": 1, "Hour": 2, "Minute": 15}
    }


def test_plist_uses_stable_application_pointer_and_shared_logs() -> None:
    spec = _agent_specs(
        series="KXBTC",
        symbol="BTC",
        campaign_start="2026-08-07T00:00:00Z",
        watch_retention_days=14,
    )[0]
    payload = _plist_payload(
        spec,
        app_path=Path("/runtime/app"),
        data_path=Path("/runtime/data"),
        uv_path="/usr/local/bin/uv",
    )

    assert payload["WorkingDirectory"] == "/runtime/app"
    assert payload["StandardOutPath"] == "/runtime/data/paper-alerts.log"
    assert payload["ProgramArguments"][:3] == ["/usr/local/bin/uv", "run", "pms"]


def test_release_copy_excludes_runtime_state_and_secrets() -> None:
    ignored = _ignore_source(
        "/source",
        [".env", ".git", ".venv", "backups", "data", "module.py", "module.pyc"],
    )

    assert ignored == {".env", ".git", ".venv", "backups", "data", "module.pyc"}


@pytest.mark.parametrize(("series", "symbol"), [("KXETH", "ETH"), ("KXBTC", "ETH")])
def test_forward_job_is_only_enabled_for_the_frozen_population(series: str, symbol: str) -> None:
    specs = _agent_specs(
        series=series,
        symbol=symbol,
        campaign_start="2026-08-07T00:00:00Z",
        watch_retention_days=14,
    )
    assert len(specs) == 7
    assert all(spec.suffix != "forward" for spec in specs)


@pytest.mark.parametrize("registration_fails", [False, True])
def test_deployment_gates_register_before_jobs_and_restore_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registration_fails: bool
) -> None:
    source = tmp_path / "source"
    (source / "ops").mkdir(parents=True)
    (source / "application.py").write_text("release source\n")
    (source / ".env").write_text("do not copy source credentials\n")
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "runtime"
    root.mkdir()
    previous = root / "previous"
    previous.mkdir()
    app = root / "app"
    app.symlink_to(previous, target_is_directory=True)
    (root / ".env").write_text("shared configuration\n")
    (root / "data").mkdir()
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    specs = _agent_specs(
        series="KXBTC",
        symbol="BTC",
        campaign_start="2026-08-07T00:00:00Z",
        watch_retention_days=14,
    )
    for spec in specs[:-1]:
        deploy._write_plist(
            agents / f"{spec.label}.plist",
            _plist_payload(spec, app_path=app, data_path=root / "data", uv_path="/uv"),
        )
    commands: list[tuple[list[str], Path | None, dict[str, str] | None]] = []
    loaded = {spec.label for spec in specs[:-1]}

    def run(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        env: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        commands.append((arguments, cwd, env))
        if arguments[:2] == ["launchctl", "print"]:
            label = arguments[-1].split("/")[-1]
            return subprocess.CompletedProcess(
                arguments, 0 if label in loaded else 113, "state = waiting\n", ""
            )
        if arguments[:2] == ["launchctl", "bootout"]:
            loaded.discard(arguments[-1].split("/")[-1])
        if arguments[:2] == ["launchctl", "bootstrap"]:
            loaded.add(Path(arguments[-1]).stem)
        if arguments[-1] == "forward-register" and registration_fails:
            raise subprocess.CalledProcessError(2, arguments)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(deploy, "__file__", str(source / "ops/deploy_macos.py"))
    monkeypatch.setattr(deploy, "_git_revision", lambda source: "release123")
    monkeypatch.setattr(deploy, "_run", run)
    monkeypatch.setattr(deploy.shutil, "which", lambda command: "/uv")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        deploy,
        "_parse_args",
        lambda: argparse.Namespace(
            app_root=root,
            series="KXBTC",
            symbol="BTC",
            campaign_start="2026-08-07T00:00:00Z",
            watch_retention_days=14,
            skip_checks=False,
        ),
    )
    if registration_fails:
        with pytest.raises(subprocess.CalledProcessError):
            deploy.main()
        assert app.resolve() == previous
    else:
        assert deploy.main() == 0
        release = app.resolve()
        assert release != previous
        assert (release / ".env").is_symlink()
        assert (release / ".env").resolve() == root / ".env"
        assert (release / "application.py").read_text() == "release source\n"
        with (agents / f"{specs[-1].label}.plist").open("rb") as file:
            payload: dict[str, Any] = plistlib.load(file)
        assert payload["StartCalendarInterval"] == {"Minute": 45}
        assert "RunAtLoad" not in payload
        assert payload["ProgramArguments"] == ["/uv", "run", "pms", "forward-capture"]
    argv = [command for command, _, _ in commands]
    init = argv.index(["/uv", "run", "pms", "init-db"])
    registration = argv.index(["/uv", "run", "pms", "forward-register"])
    assert init < registration
    gates = [
        ["uv", "run", "pytest"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "ruff", "format", "--check", "."],
        ["uv", "run", "mypy"],
    ]
    for gate in gates:
        index = argv.index(gate)
        assert index < init
        assert commands[index][2] is not None
        assert commands[index][2]["PMS_DATABASE_PATH"].endswith(".deployment-check.db")
    bootstraps = [
        index for index, args in enumerate(argv) if args[:2] == ["launchctl", "bootstrap"]
    ]
    assert bootstraps and min(bootstraps) > registration
    assert len(bootstraps) == (7 if registration_fails else 8)


@pytest.fixture
def deployment_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = tmp_path / "source"
    (source / "ops").mkdir(parents=True)
    (source / "application.py").write_text("new release\n")
    home = tmp_path / "home"
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    root = tmp_path / "runtime"
    previous = root / "previous"
    previous.mkdir(parents=True)
    app = root / "app"
    app.symlink_to(previous, target_is_directory=True)
    (root / ".env").write_text("configuration\n")
    (root / "data").mkdir()
    specs = _agent_specs(
        series="KXBTC",
        symbol="BTC",
        campaign_start="2026-08-07T00:00:00Z",
        watch_retention_days=14,
    )
    for spec in specs:
        deploy._write_plist(
            agents / f"{spec.label}.plist",
            _plist_payload(spec, app_path=app, data_path=root / "data", uv_path="/uv"),
        )
    args = argparse.Namespace(
        app_root=root,
        series="KXBTC",
        symbol="BTC",
        campaign_start="2026-08-07T00:00:00Z",
        watch_retention_days=14,
        skip_checks=False,
    )
    state: dict[str, Any] = {
        "agents": agents,
        "app": app,
        "previous": previous,
        "specs": specs,
        "args": args,
        "loaded": {spec.label for spec in specs},
        "active": set(),
        "events": [],
        "fail_bootstrap": None,
        "fail_restore": None,
        "bootstrap_failed": False,
        "fail_gate": False,
    }

    def run(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        env: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        state["events"].append((arguments, set(state["loaded"]), app.resolve()))
        if arguments[:2] == ["launchctl", "print"]:
            label = arguments[-1].split("/")[-1]
            if label not in state["loaded"]:
                return subprocess.CompletedProcess(arguments, 113, "", "Could not find service")
            status = (
                "state = running\npid = 123\n" if label in state["active"] else "state = waiting\n"
            )
            return subprocess.CompletedProcess(arguments, 0, status, "")
        if arguments[:2] == ["launchctl", "bootout"]:
            state["loaded"].discard(arguments[-1].split("/")[-1])
        if arguments[:2] == ["launchctl", "bootstrap"]:
            label = Path(arguments[-1]).stem
            if label == state["fail_bootstrap"] and not state["bootstrap_failed"]:
                state["bootstrap_failed"] = True
                raise subprocess.CalledProcessError(5, arguments)
            if label == state["fail_restore"] and state["bootstrap_failed"]:
                raise subprocess.CalledProcessError(6, arguments)
            assert label not in state["loaded"], "bootstrapping an already loaded job"
            assert Path(arguments[-1]).exists()
            state["loaded"].add(label)
        if arguments == ["uv", "run", "mypy"] and state["fail_gate"]:
            raise subprocess.CalledProcessError(1, arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(deploy, "__file__", str(source / "ops/deploy_macos.py"))
    monkeypatch.setattr(deploy, "_git_revision", lambda source: "release123")
    monkeypatch.setattr(deploy, "_run", run)
    monkeypatch.setattr(deploy.shutil, "which", lambda command: "/uv")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(deploy, "_parse_args", lambda: args)
    return state


@pytest.mark.parametrize("gate_fails", [False, True])
def test_forward_agent_remains_loaded_through_gates_and_atomic_handoff(
    deployment_state: dict[str, Any], gate_fails: bool
) -> None:
    state = deployment_state
    forward = state["specs"][-1].label
    state["active"].add(forward)
    state["fail_gate"] = gate_fails
    original = (state["agents"] / f"{forward}.plist").read_bytes()
    if gate_fails:
        with pytest.raises(subprocess.CalledProcessError):
            deploy.main()
        assert state["app"].resolve() == state["previous"]
    else:
        assert deploy.main() == 0
        assert state["app"].resolve() != state["previous"]
    for command, loaded, _ in state["events"]:
        assert forward in loaded, f"forward capture unavailable during {command}"
        assert not (command[:2] == ["launchctl", "bootout"] and command[-1].endswith(f"/{forward}"))
        assert not (command[:2] == ["launchctl", "bootstrap"] and Path(command[-1]).stem == forward)
    assert forward in state["loaded"]
    assert (state["agents"] / f"{forward}.plist").read_bytes() == original
    assert any(command == ["uv", "run", "mypy"] for command, _, _ in state["events"])


def test_nonmatching_population_removes_obsolete_forward_agent(
    deployment_state: dict[str, Any],
) -> None:
    state = deployment_state
    state["args"].series = "KXETH"
    state["args"].symbol = "ETH"
    forward = state["specs"][-1].label
    assert deploy.main() == 0
    assert forward not in state["loaded"]
    assert not (state["agents"] / f"{forward}.plist").exists()
    assert state["loaded"] == {spec.label for spec in state["specs"][:-1]}


@pytest.mark.parametrize("obsolete", [False, True])
def test_bootstrap_failure_restores_exact_plists_and_prior_loaded_set(
    deployment_state: dict[str, Any], obsolete: bool
) -> None:
    state = deployment_state
    specs = state["specs"]
    forward_path = state["agents"] / f"{specs[-1].label}.plist"
    if obsolete:
        state["args"].series = "KXETH"
        state["args"].symbol = "ETH"
    else:
        forward_path.unlink()
        state["loaded"].remove(specs[-1].label)
    # A preexisting but unloaded agent must not be loaded by rollback.
    state["loaded"].remove(specs[1].label)
    old_plist = state["agents"] / f"{specs[0].label}.plist"
    payload = plistlib.loads(old_plist.read_bytes())
    payload["EnvironmentVariables"] = {"OLD_RELEASE_ONLY": "preserve"}
    old_plist.write_bytes(plistlib.dumps(payload, fmt=plistlib.FMT_BINARY))
    original = {path.name: path.read_bytes() for path in state["agents"].glob("*.plist")}
    loaded = set(state["loaded"])
    state["fail_bootstrap"] = specs[-2 if obsolete else -1].label
    with pytest.raises(subprocess.CalledProcessError) as error:
        deploy.main()
    assert error.value.returncode == 5
    assert state["app"].resolve() == state["previous"]
    assert {path.name: path.read_bytes() for path in state["agents"].glob("*.plist")} == original
    assert state["loaded"] == loaded


def test_changed_loaded_forward_definition_fails_before_activation(
    deployment_state: dict[str, Any],
) -> None:
    state = deployment_state
    forward = state["specs"][-1].label
    state["active"].add(forward)
    path = state["agents"] / f"{forward}.plist"
    payload = plistlib.loads(path.read_bytes())
    payload["StartCalendarInterval"] = {"Minute": 44}
    path.write_bytes(plistlib.dumps(payload))
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="forward"):
        deploy.main()
    assert state["app"].resolve() == state["previous"]
    assert forward in state["loaded"]
    assert path.read_bytes() == original
    assert not any(command[:2] == ["launchctl", "bootout"] for command, _, _ in state["events"])


def test_rollback_error_preserves_deployment_failure_context(
    deployment_state: dict[str, Any],
) -> None:
    state = deployment_state
    forward = state["specs"][-1].label
    state["loaded"].remove(forward)
    (state["agents"] / f"{forward}.plist").unlink()
    state["fail_bootstrap"] = forward
    state["fail_restore"] = state["specs"][0].label
    with pytest.raises(RuntimeError, match="rollback") as error:
        deploy.main()
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert error.value.__cause__.returncode == 5
    assert "restored" not in str(error.value).lower()
