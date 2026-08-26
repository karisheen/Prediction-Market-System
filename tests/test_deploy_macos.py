import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
    assert by_suffix["maintenance"].schedule == {
        "StartCalendarInterval": {"Hour": 3, "Minute": 15}
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
        [".env", ".git", ".venv", "data", "module.py", "module.pyc"],
    )

    assert ignored == {".env", ".git", ".venv", "data", "module.pyc"}
