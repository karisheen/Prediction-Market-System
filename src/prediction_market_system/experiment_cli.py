"""Phase 1 research-only comparison and optional recorded-order cost sensitivity."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import html
import io
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from prediction_market_system.evidence import code_identity, content_id
from prediction_market_system.experiment import (
    ARMS,
    SOURCE_RETROSPECTIVE,
    SOURCE_SYNTHETIC,
    compare_observations,
    load_protocol,
    protocol_identity,
)
from prediction_market_system.experiment_decisions import assess_decisions
from prediction_market_system.experiment_inputs import load_fixture, load_local_database
from prediction_market_system.redaction import redact_payload, redact_secrets

DEFAULT_PROTOCOL = Path("experiments/phase1/protocol-v2.json")
# Paths recorded in a protocol's supersedes lineage are relative to the repository root.
_REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "pms-phase1-evidence-manifest-v1"
MANIFEST_NAME = "evidence-manifest.json"
# Neither source can support an unseen-event claim; any other source kind fails closed.
EVIDENCE_LABELS = {
    SOURCE_SYNTHETIC: "SYNTHETIC FIXTURE: implementation check, not empirical evidence",
    SOURCE_RETROSPECTIVE: (
        "DESCRIPTIVE RETROSPECTIVE: local archive summary, not unseen-event evidence"
    ),
}
OBSERVATION_FIELDS = (
    "observation_id",
    "event_id",
    "market_id",
    "series_id",
    "observed_at",
    "event_date",
    "observation_end_at",
    "expires_at",
    "regime",
    "state",
    "run_id",
    "input_id",
    "run_manifest_sha256",
    "recipe_id",
    "research_context_id",
    "feature_source",
    "uncertainty_source",
    "calibration_profile_id",
    "structural_probability_yes",
    "blend_probability_yes",
    "market_yes_ask",
    "market_yes_ask_size",
    "recorded_midpoint_anchor_probability_yes",
    "resolution_status",
    "unresolved_reason",
    "outcome_yes",
    "resolution_observed_at",
    "settlement_ts",
    "event_weight",
    "structural_brier",
    "structural_log_loss",
    "blend_brier",
    "blend_log_loss",
    "market_yes_ask_brier",
    "market_yes_ask_log_loss",
    "recorded_midpoint_anchor_brier",
    "recorded_midpoint_anchor_log_loss",
)
# Legend text, stroke colour and dash pattern per report arm.
_ARM_STYLES = {
    "structural": ("structural", "#1f77b4", ""),
    "blend": ("blend", "#2ca02c", ""),
    "market_yes_ask": ("YES ask (primary baseline)", "#d62728", ""),
    "recorded_midpoint_anchor": ("recorded midpoint (not executable)", "#7f7f7f", "4 3"),
}
_BRANCH_IDENTITY_KEYS = (
    "branch",
    "feature_source",
    "selected_volatility_kind",
    "selected_volatility_provider",
    "selected_volatility_window_seconds",
    "recipe_id",
)
_CODE_KEYS = ("git_revision", "source_sha256", "source_files", "lock_sha256")
_RUNTIME_KEYS = ("python", "platform")
_PLOT_LEFT, _PLOT_TOP, _PLOT_SIZE = 70.0, 90.0, 400.0


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _evidence_label(
    report: dict[str, Any], protocol: dict[str, Any], source: dict[str, Any]
) -> str:
    source_kind = source.get("source_kind")
    if not isinstance(source_kind, str) or source_kind not in EVIDENCE_LABELS:
        raise ValueError(f"unsupported phase1 source kind: {source_kind!r}")
    if source.get("synthetic") is not (source_kind == SOURCE_SYNTHETIC):
        raise ValueError("source synthetic flag contradicts its source kind")
    if report.get("source_kind") != source_kind:
        raise ValueError("report and source disagree on source kind")
    if report.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("report was not produced under the supplied protocol")
    if source.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("source was not loaded under the supplied protocol")
    if (report.get("conclusion") or {}).get("status") != "inconclusive":
        raise ValueError("synthetic and retrospective phase1 reports must remain inconclusive")
    return EVIDENCE_LABELS[source_kind]


def _preserved_bytes(relative: Any, expected_sha256: Any) -> bytes:
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise ValueError("protocol lineage must record each preserved path and its sha256")
    root = _REPOSITORY.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"superseded protocol file is not preserved: {relative}")
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"superseded protocol file changed after freeze: {relative}")
    return payload


def _protocol_record(protocol: dict[str, Any], file_sha256: str | None) -> dict[str, Any]:
    protocol_id = protocol.get("protocol_id")
    if protocol_identity(protocol) != protocol_id:
        raise ValueError("protocol content does not match its frozen protocol_id")
    lineage = protocol.get("supersedes")
    # A replacement is explicit only while the protocol it replaces stays exactly as frozen.
    if lineage is not None:
        if not isinstance(lineage, dict) or lineage.get("protocol_id") == protocol_id:
            raise ValueError("protocol supersedes record is malformed")
        claimed = lineage.get("protocol_id")
        preserved = json.loads(_preserved_bytes(lineage.get("path"), lineage.get("file_sha256")))
        if (
            not isinstance(preserved, dict)
            or preserved.get("protocol_id") != claimed
            or protocol_identity(preserved) != claimed
        ):
            raise ValueError("superseded protocol identity does not match its preserved file")
        if "documentation_path" in lineage:
            _preserved_bytes(lineage["documentation_path"], lineage.get("documentation_sha256"))
    return {
        "protocol_id": protocol_id,
        "protocol_version": protocol["protocol_version"],
        "frozen_at": protocol["frozen_at"],
        "protocol_sha256": content_id(protocol),
        "file_sha256": file_sha256,
        "supersedes": lineage,
        "trial_record": protocol.get("trial_record"),
    }


def _source_record(source: dict[str, Any]) -> dict[str, Any]:
    origin = source.get("source")
    if not isinstance(origin, dict) or not isinstance(origin.get("sha256"), str):
        raise ValueError("source metadata must identify its content by sha256")
    # A local path identifies the host, not the evidence; the content hash is the identity.
    return {**source, "source": {key: value for key, value in origin.items() if key != "path"}}


def _comparison_csv(report: dict[str, Any]) -> bytes:
    expected = {*OBSERVATION_FIELDS, "execution"}
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(OBSERVATION_FIELDS)
    for row in report["observations"]:
        if set(row) != expected:
            raise ValueError("observation row fields do not match the phase1 CSV columns")
        values = [row[field] for field in OBSERVATION_FIELDS]
        if not all(value is None or isinstance(value, str | int | float) for value in values):
            raise ValueError("observation CSV values must be scalar")
        writer.writerow(values)
    return buffer.getvalue().encode()


def _decision_record(
    decisions: dict[str, Any],
    report: dict[str, Any],
    protocol: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    _evidence_label(decisions, protocol, source)
    if decisions.get("comparison_sha256") != content_id(report):
        raise ValueError("decision report does not match the supplied comparison")
    population = report["identities"]["eligible_observations_sha256"]
    if decisions.get("population_sha256") != population:
        raise ValueError("decision report does not match the comparison population")
    for scenario in decisions["scenarios"]:
        if scenario["configuration_id"] != content_id(scenario["configuration"]):
            raise ValueError("decision scenario configuration identity does not match")
    return {
        "report_sha256": content_id(decisions),
        "comparison_sha256": content_id(report),
        "population_sha256": population,
        "protocol_id": protocol["protocol_id"],
        "source_kind": source["source_kind"],
        "source_sha256": content_id(source),
    }


def _decisions_csv(decisions: dict[str, Any]) -> bytes:
    rows = [
        {
            "scenario": scenario["name"],
            "scenario_configuration_id": scenario["configuration_id"],
            **{
                key: value
                for key, value in decision.items()
                if key not in {"decision_quotes", "execution_snapshot"}
                and (value is None or isinstance(value, str | int | float))
            },
        }
        for scenario in decisions["scenarios"]
        for decision in scenario["decisions"]
    ]
    leading = ("scenario", "scenario_configuration_id", "observation_id")
    fields = [*leading, *sorted({key for row in rows for key in row} - set(leading))]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _plot_x(probability: float) -> str:
    return f"{_PLOT_LEFT + probability * _PLOT_SIZE:.2f}"


def _plot_y(frequency: float) -> str:
    return f"{_PLOT_TOP + (1.0 - frequency) * _PLOT_SIZE:.2f}"


def _calibration_svg(report: dict[str, Any], label: str) -> bytes:
    calibration = report["calibration"]
    arms = calibration["arms"]
    if set(arms) != set(ARMS):
        raise ValueError("calibration arms do not match the phase1 arms")
    right, bottom = _PLOT_LEFT + _PLOT_SIZE, _PLOT_TOP + _PLOT_SIZE
    text = html.escape
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="560" '
        'viewBox="0 0 720 560" font-family="sans-serif">',
        '<rect width="720" height="560" fill="#ffffff"/>',
        f'<text x="20" y="26" font-size="14" font-weight="bold">{text(label)}</text>',
        f'<text x="20" y="46" font-size="10">{text(report["protocol_version"])} '
        f"{text(report['protocol_id'])}</text>",
        f'<text x="20" y="62" font-size="10">as of {text(report["as_of"])}; '
        f"{calibration['resolved_forecasts']} resolved forecasts in "
        f"{calibration['resolved_events']} events; "
        f"conclusion {text(report['conclusion']['status'])}</text>",
        f'<text x="20" y="78" font-size="10">weighting: '
        f"{text(calibration['weighting'])}; "
        f"pools {len(report['source_selection_branches'])} source-selection branches</text>",
    ]
    for step in range(6):
        tick = step / 5
        parts.append(
            f'<line x1="{_plot_x(tick)}" y1="{_PLOT_TOP:.2f}" x2="{_plot_x(tick)}" '
            f'y2="{bottom:.2f}" stroke="#e5e5e5"/>'
            f'<line x1="{_PLOT_LEFT:.2f}" y1="{_plot_y(tick)}" x2="{right:.2f}" '
            f'y2="{_plot_y(tick)}" stroke="#e5e5e5"/>'
            f'<text x="{_plot_x(tick)}" y="{bottom + 16:.2f}" font-size="10" '
            f'text-anchor="middle">{tick:.1f}</text>'
            f'<text x="{_PLOT_LEFT - 8:.2f}" y="{_plot_y(tick)}" font-size="10" '
            f'text-anchor="end" dominant-baseline="middle">{tick:.1f}</text>'
        )
    parts += [
        f'<rect x="{_PLOT_LEFT:.2f}" y="{_PLOT_TOP:.2f}" width="{_PLOT_SIZE:.2f}" '
        f'height="{_PLOT_SIZE:.2f}" fill="none" stroke="#000000"/>',
        f'<line x1="{_plot_x(0.0)}" y1="{_plot_y(0.0)}" x2="{_plot_x(1.0)}" '
        f'y2="{_plot_y(1.0)}" stroke="#000000" stroke-dasharray="2 4"/>',
        f'<text x="{_plot_x(0.5)}" y="{bottom + 36:.2f}" font-size="11" '
        'text-anchor="middle">mean forecast probability (event-weighted)</text>',
        f'<text x="18" y="{_plot_y(0.5)}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 18 {_plot_y(0.5)})">observed YES frequency (event-weighted)'
        "</text>",
    ]
    plotted = 0
    for index, arm in enumerate(ARMS):
        legend, colour, dash = _ARM_STYLES[arm]
        points = [
            bin_
            for bin_ in arms[arm]
            if bin_["mean_probability"] is not None and bin_["mean_outcome"] is not None
        ]
        plotted += len(points)
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        if len(points) > 1:
            coordinates = " ".join(
                f"{_plot_x(point['mean_probability'])},{_plot_y(point['mean_outcome'])}"
                for point in points
            )
            parts.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{colour}"{dash_attribute}/>'
            )
        for point in points:
            parts.append(
                f'<circle cx="{_plot_x(point["mean_probability"])}" '
                f'cy="{_plot_y(point["mean_outcome"])}" r="4" fill="{colour}">'
                f"<title>{text(arm)} bin {point['bin']} {point['lower']:.2f}-"
                f"{point['upper']:.2f}: mean p {point['mean_probability']:.4f}, "
                f"observed {point['mean_outcome']:.4f}, {point['forecasts']} forecasts, "
                f"{point['events']} events, weight {point['weight']:.4f}</title></circle>"
            )
        legend_y = _PLOT_TOP + 10 + index * 22
        parts.append(
            f'<line x1="{right + 20:.2f}" y1="{legend_y:.2f}" x2="{right + 44:.2f}" '
            f'y2="{legend_y:.2f}" stroke="{colour}" stroke-width="2"{dash_attribute}/>'
            f'<text x="{right + 50:.2f}" y="{legend_y:.2f}" font-size="10" '
            f'dominant-baseline="middle">{text(legend)}</text>'
        )
    if plotted == 0:
        parts.append(
            f'<text x="{_plot_x(0.5)}" y="{_plot_y(0.5)}" font-size="12" '
            'text-anchor="middle">no resolved forecasts to calibrate</text>'
        )
    parts.append("</svg>")
    return ("\n".join(parts) + "\n").encode()


def _manifest(
    report: dict[str, Any],
    protocol: dict[str, Any],
    protocol_record: dict[str, Any],
    source: dict[str, Any],
    label: str,
    artifacts: dict[str, bytes],
    decisions: dict[str, Any] | None = None,
    decision_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = code_identity()
    configuration: dict[str, Any] = {
        "command": "phase1-report",
        "protocol_id": protocol["protocol_id"],
        "report_as_of": protocol["report_as_of"],
        "source_kind": source["source_kind"],
    }
    if decisions is not None:
        configuration["include_decisions"] = True
        configuration["decision_scenarios"] = [
            {
                "name": scenario["name"],
                "configuration": scenario["configuration"],
                "configuration_id": scenario["configuration_id"],
            }
            for scenario in decisions["scenarios"]
        ]
    code = {key: identity[key] for key in _CODE_KEYS}
    # Engine floats can differ in the last ulp across platforms, changing dataset and artifact
    # hashes; analysis_id covers only the platform-independent inputs.
    analysis = {
        "protocol_sha256": protocol_record["protocol_sha256"],
        "configuration_sha256": content_id(configuration),
        "source": source["source"],
        "code": {key: code[key] for key in ("git_revision", "source_sha256", "lock_sha256")},
    }
    content = redact_payload(
        {
            "evidence_label": label,
            "analysis_id": "sha256:" + content_id(analysis),
            "protocol": protocol_record,
            "configuration": configuration,
            "configuration_sha256": content_id(configuration),
            "source": source,
            "source_sha256": content_id(source),
            "dataset": {
                **{
                    key: report["identities"][key]
                    for key in (
                        "eligible_observations_sha256",
                        "excluded_observations_sha256",
                        "recipe_ids",
                    )
                },
                "source_selection_branches": [
                    {key: branch[key] for key in _BRANCH_IDENTITY_KEYS}
                    for branch in report["source_selection_branches"]
                ],
            },
            "code": code,
            **({"decisions": decision_record} if decisions is not None else {}),
            "artifacts": {
                name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                for name, payload in sorted(artifacts.items())
            },
        }
    )
    # Invocation and host details differ between otherwise identical analyses, so they stay
    # outside the content address.
    return {
        "schema": MANIFEST_SCHEMA,
        "evidence_id": "sha256:" + content_id(content),
        "content": content,
        "invocation": redact_payload(
            {
                "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
                **({"include_decisions": True} if decisions is not None else {}),
                "generated_at": datetime.now(UTC).isoformat(),
                **{key: identity[key] for key in _RUNTIME_KEYS},
            }
        ),
    }


def _publish(output: Path, files: dict[str, bytes]) -> None:
    # Exclusive creation of the directory and of every file: an existing path, even an empty
    # directory, is never reused, and a concurrent writer to the same path loses cleanly.
    output.mkdir(parents=True)
    written: list[Path] = []
    try:
        for name, payload in files.items():
            path = output / name
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
            written.append(path)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            output.rmdir()
        raise


def write_comparison_artifacts(
    report: dict[str, Any],
    protocol: dict[str, Any],
    source: dict[str, Any],
    output: Path,
    *,
    protocol_file_sha256: str | None = None,
    decisions: dict[str, Any] | None = None,
) -> None:
    """Publish immutable evidence into a new directory; the manifest is written last.

    Without ``protocol_file_sha256`` the manifest records the protocol file hash as null.
    """
    label = _evidence_label(report, protocol, source)
    protocol_record = _protocol_record(protocol, protocol_file_sha256)
    record = _source_record(redact_payload(source))
    decision_record = (
        _decision_record(decisions, report, protocol, record) if decisions is not None else None
    )
    safe_report = redact_payload(report)
    artifacts = {
        "comparison.json": _json_bytes(safe_report),
        "comparison.csv": _comparison_csv(safe_report),
        "calibration.svg": _calibration_svg(safe_report, label),
    }
    safe_decisions = redact_payload(decisions) if decisions is not None else None
    if safe_decisions is not None:
        artifacts["decisions.json"] = _json_bytes(safe_decisions)
        artifacts["decisions.csv"] = _decisions_csv(safe_decisions)
    manifest = _manifest(
        safe_report,
        protocol,
        protocol_record,
        record,
        label,
        artifacts,
        safe_decisions,
        decision_record,
    )
    _publish(output, {**artifacts, MANIFEST_NAME: _json_bytes(manifest)})


def register_experiment_commands(app: typer.Typer) -> None:
    @app.command("phase1-report")
    def phase1_report(
        output: Annotated[
            Path, typer.Option(help="New evidence directory; an existing path is never reused.")
        ],
        fixture: Annotated[
            Path | None, typer.Option(help="Declarative synthetic fixture JSON.")
        ] = None,
        database: Annotated[
            Path | None, typer.Option(help="Existing local SQLite database, opened read-only.")
        ] = None,
        protocol: Annotated[
            Path, typer.Option(help="Frozen phase1 protocol JSON; v1 is a superseded trial.")
        ] = DEFAULT_PROTOCOL,
        include_decisions: Annotated[
            bool, typer.Option(help="Include research-only recorded-order cost sensitivity.")
        ] = False,
    ) -> None:
        """Synthetic or descriptive comparison, optionally with paper cost sensitivity."""
        if (fixture is None) == (database is None):
            raise typer.BadParameter("provide exactly one of --fixture or --database")
        if os.path.lexists(output):
            raise typer.BadParameter("output already exists; evidence is never overwritten")
        try:
            frozen_bytes = protocol.read_bytes()
            frozen = load_protocol(protocol)
            if json.loads(frozen_bytes) != frozen:
                raise ValueError("protocol file changed while it was being loaded")
            if fixture is not None:
                observations, source = load_fixture(fixture, frozen)
            else:
                assert database is not None
                observations, source = load_local_database(database, frozen)
            report = compare_observations(
                observations,
                frozen,
                source_kind=source["source_kind"],
                as_of=datetime.fromisoformat(frozen["report_as_of"]),
            )
            decisions = (
                assess_decisions(observations, report, frozen) if include_decisions else None
            )
            write_comparison_artifacts(
                report,
                frozen,
                source,
                output,
                protocol_file_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
                decisions=decisions,
            )
            manifest = json.loads((output / MANIFEST_NAME).read_text())
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        typer.echo(
            json.dumps(
                redact_payload(
                    {
                        "output": str(output),
                        "evidence_id": manifest["evidence_id"],
                        "analysis_id": manifest["content"]["analysis_id"],
                        "evidence_label": manifest["content"]["evidence_label"],
                        "artifacts": manifest["content"]["artifacts"],
                    }
                ),
                sort_keys=True,
                indent=2,
            )
        )
