import csv
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from prediction_market_system.cli import app
from prediction_market_system.evidence import code_identity, content_id
from prediction_market_system.experiment import compare_observations, load_protocol
from prediction_market_system.experiment_cli import (
    MANIFEST_NAME,
    OBSERVATION_FIELDS,
    write_comparison_artifacts,
)
from prediction_market_system.experiment_decisions import assess_decisions
from prediction_market_system.experiment_inputs import load_fixture

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY / "experiments/phase1/protocol-v2.json"
SUPERSEDED = REPOSITORY / "experiments/phase1/protocol-v1.json"
FIXTURE = REPOSITORY / "experiments/phase1/fixture-v1.json"
ARTIFACTS = {"comparison.json", "comparison.csv", "calibration.svg", MANIFEST_NAME}
DECISION_ARTIFACTS = ARTIFACTS | {"decisions.json", "decisions.csv"}
runner = CliRunner()


def fixture_report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(PROTOCOL)
    observations, source = load_fixture(FIXTURE, protocol)
    report = compare_observations(
        observations,
        protocol,
        source_kind=source["source_kind"],
        as_of=datetime.fromisoformat(protocol["report_as_of"]),
    )
    return report, protocol, source


def snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


@pytest.mark.parametrize("sources", [[], ["--fixture", "--database"]])
@pytest.mark.parametrize("force_color", [False, True], ids=["plain", "color"])
def test_phase1_report_requires_exactly_one_source(
    tmp_path: Path, sources: list[str], force_color: bool
) -> None:
    database = tmp_path / "never-created.db"
    arguments = {"--fixture": str(FIXTURE), "--database": str(database)}
    output = tmp_path / "evidence"

    result = runner.invoke(
        app,
        [
            "phase1-report",
            *[item for option in sources for item in (option, arguments[option])],
            "--protocol",
            str(PROTOCOL),
            "--output",
            str(output),
        ],
        color=force_color,
        env={
            "FORCE_COLOR": "1" if force_color else None,
            "NO_COLOR": None if force_color else "1",
            "COLUMNS": "80",
        },
    )

    assert result.exit_code == 2
    assert not output.exists()
    assert not database.exists()


def test_phase1_report_never_reuses_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "comparison.json").write_text("prior evidence")

    result = runner.invoke(
        app,
        [
            "phase1-report",
            "--fixture",
            str(FIXTURE),
            "--protocol",
            str(PROTOCOL),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "already exists" in result.output
    assert snapshot(output) == {"comparison.json": b"prior evidence"}


def test_writer_refuses_collision_and_keeps_first_evidence(tmp_path: Path) -> None:
    report, protocol, source = fixture_report()
    output = tmp_path / "evidence"
    write_comparison_artifacts(report, protocol, source, output)
    first = snapshot(output)

    with pytest.raises(FileExistsError):
        write_comparison_artifacts(report, protocol, source, output)

    assert set(first) == ARTIFACTS
    assert snapshot(output) == first


@pytest.mark.parametrize(
    ("report_changes", "source_changes"),
    [
        ({"source_kind": "forward-evaluation"}, {"source_kind": "forward-evaluation"}),
        ({"source_kind": "retrospective-local"}, {"source_kind": "retrospective-local"}),
        ({"conclusion": {"status": "supported", "claim": "edge", "basis": "fixture"}}, {}),
        ({"protocol_id": "sha256:" + "0" * 64}, {}),
        ({}, {"protocol_id": "sha256:" + "0" * 64}),
        ({}, {"synthetic": False}),
        ({}, {"source": {"path": "fixture-v1.json"}}),
    ],
)
def test_writer_publishes_only_inconclusive_synthetic_or_descriptive_evidence(
    tmp_path: Path, report_changes: dict[str, Any], source_changes: dict[str, Any]
) -> None:
    report, protocol, source = fixture_report()
    output = tmp_path / "evidence"

    with pytest.raises(ValueError):
        write_comparison_artifacts(
            {**report, **report_changes}, protocol, {**source, **source_changes}, output
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "tamper", ["protocol-edited", "v1-missing", "v1-reformatted", "v1-document-edited"]
)
def test_writer_refuses_protocol_whose_frozen_identity_or_lineage_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    report, protocol, source = fixture_report()
    lineage = protocol["supersedes"]
    preserved = tmp_path / "repository"
    for key in ("path", "documentation_path"):
        copy = preserved / lineage[key]
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((REPOSITORY / lineage[key]).read_bytes())
    monkeypatch.setattr("prediction_market_system.experiment_cli._REPOSITORY", preserved)
    write_comparison_artifacts(report, protocol, source, tmp_path / "untampered")
    superseded = preserved / lineage["path"]
    if tamper == "protocol-edited":
        protocol = {**protocol, "structural_weight": 0.6}
    elif tamper == "v1-missing":
        superseded.unlink()
    elif tamper == "v1-reformatted":
        superseded.write_text(json.dumps(json.loads(superseded.read_text()), indent=4))
    else:
        (preserved / lineage["documentation_path"]).write_text("rewritten after freeze\n")
    output = tmp_path / "evidence"

    with pytest.raises(ValueError):
        write_comparison_artifacts(report, protocol, source, output)

    assert not output.exists()


def test_failed_publication_leaves_no_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, protocol, source = fixture_report()
    output = tmp_path / "evidence"
    real_fsync = os.fsync
    calls = 0

    def failing_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="disk full"):
        write_comparison_artifacts(report, protocol, source, output)

    assert not output.exists()


def test_phase1_report_manifest_content_addresses_evidence_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "https://discord.com/api/webhooks/1/phase1-secret-token"
    monkeypatch.setenv("PMS_DISCORD_WEBHOOK_URL", secret)
    monkeypatch.chdir(REPOSITORY)
    output = tmp_path / "evidence"
    arguments = ["phase1-report", "--fixture", str(FIXTURE), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", ["/opt/venv/bin/pms", *arguments])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    files = snapshot(output)
    assert set(files) == ARTIFACTS
    manifest = json.loads(files[MANIFEST_NAME])
    content = manifest["content"]
    assert manifest["evidence_id"] == "sha256:" + content_id(content)
    assert json.loads(result.output)["evidence_id"] == manifest["evidence_id"]
    for name in ARTIFACTS - {MANIFEST_NAME}:
        assert content["artifacts"][name] == {
            "sha256": hashlib.sha256(files[name]).hexdigest(),
            "bytes": len(files[name]),
        }
    comparison = json.loads(files["comparison.json"])
    protocol = load_protocol(PROTOCOL)
    assert comparison["protocol_id"] == protocol["protocol_id"]
    assert content["protocol"] == {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": "pms-phase1-v2",
        "frozen_at": protocol["frozen_at"],
        "protocol_sha256": content_id(protocol),
        "file_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "supersedes": protocol["supersedes"],
        "trial_record": protocol["trial_record"],
    }
    assert (
        content["protocol"]["supersedes"]["protocol_id"]
        == (json.loads(SUPERSEDED.read_text())["protocol_id"])
    )
    assert content["configuration"]["report_as_of"] == protocol["report_as_of"]
    assert content["source_sha256"] == content_id(content["source"])
    assert "path" not in content["source"]["source"]
    assert str(REPOSITORY) not in json.dumps(content)
    identity = code_identity()
    stable_code = ("git_revision", "source_sha256", "lock_sha256")
    assert content["configuration_sha256"] == content_id(content["configuration"])
    assert content["analysis_id"] == "sha256:" + content_id(
        {
            "protocol_sha256": content_id(protocol),
            "configuration_sha256": content["configuration_sha256"],
            "source": content["source"]["source"],
            "code": {key: identity[key] for key in stable_code},
        }
    )
    assert json.loads(result.output)["analysis_id"] == content["analysis_id"]
    assert content["source"]["source_kind"] == "synthetic-fixture"
    assert content["evidence_label"].startswith("SYNTHETIC FIXTURE")
    assert (
        content["dataset"]["eligible_observations_sha256"]
        == (comparison["identities"]["eligible_observations_sha256"])
    )
    assert content["dataset"]["recipe_ids"] == comparison["identities"]["recipe_ids"]
    branches = content["dataset"]["source_selection_branches"]
    assert branches and [branch["recipe_id"] for branch in branches] == [
        branch["recipe_id"] for branch in comparison["source_selection_branches"]
    ]
    assert all(branch["branch"] == "manual-synthetic" for branch in branches)
    assert content["code"]["git_revision"] == identity["git_revision"]
    assert content["code"]["source_files"] == identity["source_files"]
    assert content["code"]["lock_sha256"] == identity["lock_sha256"]
    assert manifest["invocation"]["command"] == ["pms", *arguments]
    assert all(secret not in payload.decode() for payload in files.values())
    rows = list(csv.reader(files["comparison.csv"].decode().splitlines()))
    assert rows[0] == list(OBSERVATION_FIELDS)
    assert [row[0] for row in rows[1:]] == [
        row["observation_id"] for row in comparison["observations"]
    ]
    assert b"SYNTHETIC FIXTURE" in files["calibration.svg"]


@pytest.mark.parametrize("include_decisions", [False, True])
def test_identical_inputs_at_different_paths_produce_identical_evidence(
    tmp_path: Path, include_decisions: bool
) -> None:
    relocated = tmp_path / "copy" / FIXTURE.name
    relocated.parent.mkdir()
    relocated.write_bytes(FIXTURE.read_bytes())
    manifests = []
    for name, fixture in (("first", FIXTURE), ("second", relocated)):
        result = runner.invoke(
            app,
            [
                "phase1-report",
                "--fixture",
                str(fixture),
                *(["--include-decisions"] if include_decisions else []),
                "--protocol",
                str(PROTOCOL),
                "--output",
                str(tmp_path / name),
            ],
        )
        assert result.exit_code == 0, result.output
        manifests.append(json.loads((tmp_path / name / MANIFEST_NAME).read_text()))

    first, second = snapshot(tmp_path / "first"), snapshot(tmp_path / "second")
    artifacts = DECISION_ARTIFACTS if include_decisions else ARTIFACTS
    for name in artifacts - {MANIFEST_NAME}:
        assert first[name] == second[name]
    assert manifests[0]["content"] == manifests[1]["content"]
    assert manifests[0]["evidence_id"] == manifests[1]["evidence_id"]


@pytest.mark.parametrize("include_decisions", [False, True])
def test_database_without_evidence_schema_is_descriptive_and_untouched(
    tmp_path: Path, include_decisions: bool
) -> None:
    database = tmp_path / "unrelated.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.close()
    before = database.read_bytes()
    output = tmp_path / "evidence"

    result = runner.invoke(
        app,
        [
            "phase1-report",
            "--database",
            str(database),
            *(["--include-decisions"] if include_decisions else []),
            "--protocol",
            str(PROTOCOL),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert database.read_bytes() == before
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["content"]["source"]["source_kind"] == "retrospective-local"
    assert manifest["content"]["evidence_label"].startswith("DESCRIPTIVE RETROSPECTIVE")
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["observations"] == []
    assert comparison["conclusion"]["status"] == "inconclusive"
    assert (output / "comparison.csv").read_text().splitlines() == [",".join(OBSERVATION_FIELDS)]
    assert b"no resolved forecasts to calibrate" in (output / "calibration.svg").read_bytes()
    if include_decisions:
        decisions = json.loads((output / "decisions.json").read_text())
        assert decisions["source_kind"] == "retrospective-local"
        assert decisions["conclusion"]["status"] == "inconclusive"
        for scenario in decisions["scenarios"]:
            assert scenario["decisions"] == []
            summary = scenario["summary"]
            assert summary["eligible_forecasts"] == 0
            assert summary["filled_orders"] == 0
            assert summary["total_cost_dollars"] == 0.0
            assert summary["no_trade_frequency"] is None
            assert summary["recorded_watch_frequency"] is None
            assert summary["raw_pnl_dollars"] is None
            assert summary["haircut_adjusted_pnl_dollars"] is None
            assert summary["return_on_cost"] is None
        assert list(csv.DictReader((output / "decisions.csv").read_text().splitlines())) == []


def test_decision_flag_preserves_comparison_and_publishes_bound_fixture_evidence(
    tmp_path: Path,
) -> None:
    outputs = {}
    responses = {}
    for name, flags in (("probability", []), ("decisions", ["--include-decisions"])):
        output = tmp_path / name
        result = runner.invoke(
            app,
            [
                "phase1-report",
                "--fixture",
                str(FIXTURE),
                "--protocol",
                str(PROTOCOL),
                "--output",
                str(output),
                *flags,
            ],
        )
        assert result.exit_code == 0, result.output
        outputs[name] = snapshot(output)
        responses[name] = json.loads(result.output)

    probability = outputs["probability"]
    files = outputs["decisions"]
    assert set(probability) == ARTIFACTS
    assert set(files) == DECISION_ARTIFACTS
    for name in ARTIFACTS - {MANIFEST_NAME}:
        assert files[name] == probability[name]
    baseline_manifest = json.loads(probability[MANIFEST_NAME])
    assert "include_decisions" not in baseline_manifest["content"]["configuration"]
    assert "decision_scenarios" not in baseline_manifest["content"]["configuration"]
    assert "decisions" not in baseline_manifest["content"]
    assert "include_decisions" not in baseline_manifest["invocation"]

    comparison = json.loads(files["comparison.json"])
    decisions = json.loads(files["decisions.json"])
    manifest = json.loads(files[MANIFEST_NAME])
    content = manifest["content"]
    assert manifest["evidence_id"] == "sha256:" + content_id(content)
    assert content["configuration"]["include_decisions"] is True
    assert manifest["invocation"]["include_decisions"] is True
    assert content["configuration_sha256"] != baseline_manifest["content"]["configuration_sha256"]
    assert content["decisions"] == {
        "report_sha256": content_id(decisions),
        "comparison_sha256": content_id(comparison),
        "population_sha256": comparison["identities"]["eligible_observations_sha256"],
        "protocol_id": comparison["protocol_id"],
        "source_kind": "synthetic-fixture",
        "source_sha256": content["source_sha256"],
    }
    assert decisions["comparison_sha256"] == content_id(comparison)
    assert decisions["conclusion"]["status"] == "inconclusive"
    assert responses["decisions"]["output"] == str(tmp_path / "decisions")
    assert responses["decisions"]["artifacts"] == content["artifacts"]
    for name in DECISION_ARTIFACTS - {MANIFEST_NAME}:
        assert content["artifacts"][name] == {
            "sha256": hashlib.sha256(files[name]).hexdigest(),
            "bytes": len(files[name]),
        }
    assert content["configuration"]["decision_scenarios"] == [
        {
            "name": scenario["name"],
            "configuration": scenario["configuration"],
            "configuration_id": scenario["configuration_id"],
        }
        for scenario in decisions["scenarios"]
    ]
    scenarios = {scenario["name"]: scenario for scenario in decisions["scenarios"]}
    assert list(scenarios) == ["base", "adverse", "severe"]
    eligible_ids = {row["observation_id"] for row in comparison["observations"]}
    for scenario in scenarios.values():
        assert {row["observation_id"] for row in scenario["decisions"]} == eligible_ids
        assert scenario["summary"]["eligible_forecasts"] == 13
        assert scenario["configuration_id"] == content_id(scenario["configuration"])
        partial = next(
            row
            for row in scenario["decisions"]
            if row["market_id"] == "SYNTH-KXBTC-26AUG1117-B111750"
        )
        assert (
            partial["filled_contracts"] == {"base": 2, "adverse": 1, "severe": 0}[scenario["name"]]
        )
        withdrawn = next(
            row
            for row in scenario["decisions"]
            if row["market_id"] == "SYNTH-KXBTC-26SEP0817-B121250"
        )
        assert withdrawn["status"] == "nonfill"
        assert withdrawn["reason"] == "side_ask_unavailable"
        assert withdrawn["filled_contracts"] == 0
        assert withdrawn["execution_snapshot"]["yes_ask"] is None
    csv_rows = list(csv.DictReader(files["decisions.csv"].decode().splitlines()))
    assert {(row["scenario"], row["observation_id"]) for row in csv_rows} == {
        (name, observation_id) for name in scenarios for observation_id in eligible_ids
    }
    for row in csv_rows:
        assert "decision_quotes" not in row
        assert "execution_snapshot" not in row
        assert "execution_snapshots" not in row
        decision = next(
            value
            for value in scenarios[row["scenario"]]["decisions"]
            if value["observation_id"] == row["observation_id"]
        )
        assert row["status"] == decision["status"]
        assert int(row["filled_contracts"]) == decision["filled_contracts"]
        assert float(row["cost_dollars"]) == decision["cost_dollars"]


@pytest.mark.parametrize("first_has_decisions", [False, True])
def test_decision_publication_cannot_overwrite_either_report_mode(
    tmp_path: Path, first_has_decisions: bool
) -> None:
    report, protocol, source = fixture_report()
    observations, _ = load_fixture(FIXTURE, protocol)
    decisions = assess_decisions(observations, report, protocol)
    output = tmp_path / "immutable"
    write_comparison_artifacts(
        report, protocol, source, output, decisions=decisions if first_has_decisions else None
    )
    first = snapshot(output)
    with pytest.raises(FileExistsError):
        write_comparison_artifacts(
            report, protocol, source, output, decisions=None if first_has_decisions else decisions
        )
    assert snapshot(output) == first


def test_writer_rejects_decisions_from_another_comparison_dataset(tmp_path: Path) -> None:
    report, protocol, source = fixture_report()
    observations, _ = load_fixture(FIXTURE, protocol)
    decisions = assess_decisions(observations, report, protocol)
    excluded_id = report["observations"][0]["observation_id"]
    other_report = compare_observations(
        [row for row in observations if row["observation_id"] != excluded_id],
        protocol,
        source_kind=source["source_kind"],
        as_of=datetime.fromisoformat(protocol["report_as_of"]),
    )
    output = tmp_path / "wrong-dataset"
    with pytest.raises(ValueError, match="does not match the supplied comparison"):
        write_comparison_artifacts(other_report, protocol, source, output, decisions=decisions)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_sha256", "another-population"),
        ("protocol_id", "another-protocol"),
        ("source_kind", "retrospective-local"),
    ],
)
def test_writer_rejects_decision_identity_mismatches_before_publication(
    tmp_path: Path, field: str, value: str
) -> None:
    report, protocol, source = fixture_report()
    observations, _ = load_fixture(FIXTURE, protocol)
    decisions = assess_decisions(observations, report, protocol)
    decisions[field] = value
    output = tmp_path / "mismatched"
    with pytest.raises(ValueError):
        write_comparison_artifacts(report, protocol, source, output, decisions=decisions)
    assert not output.exists()


def test_decision_write_failure_removes_all_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, protocol, source = fixture_report()
    observations, _ = load_fixture(FIXTURE, protocol)
    decisions = assess_decisions(observations, report, protocol)
    real_fsync = os.fsync
    calls = 0

    def failing_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("disk full writing decision CSV")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", failing_fsync)
    output = tmp_path / "partial"
    with pytest.raises(OSError, match="disk full"):
        write_comparison_artifacts(report, protocol, source, output, decisions=decisions)
    assert not output.exists()
