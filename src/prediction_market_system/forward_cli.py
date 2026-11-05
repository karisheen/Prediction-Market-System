"""Independent prospective research commands; never delivery or trading authority."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from prediction_market_system.config import Settings
from prediction_market_system.forward_capture import (
    capture_forward_slot,
    probe_forward_capture,
    resolve_forward_outcomes,
)
from prediction_market_system.forward_protocol import (
    DEFAULT_FORWARD_PROTOCOL_PATH,
    load_forward_protocol,
    scheduled_slots,
)
from prediction_market_system.forward_report import build_forward_report
from prediction_market_system.forward_storage import ForwardStudyStore
from prediction_market_system.redaction import redact_payload, redact_secrets


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("use an ISO-8601 timestamp with a timezone") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _emit(payload: dict[str, Any], output: Path | None = None) -> None:
    encoded = json.dumps(redact_payload(payload), sort_keys=True, indent=2, allow_nan=False)
    if output is not None:
        # Reports must not overwrite evidence, protocols or an earlier report.
        with output.open("x", encoding="utf-8") as file:
            file.write(encoded + "\n")
    typer.echo(encoded)


def _failed(payload: Any) -> bool:
    """Propagate operational failures, including nested diagnostic capture clocks."""
    if isinstance(payload, dict):
        if (
            payload.get("error")
            or payload.get("errors")
            or payload.get("status")
            in {
                "failed",
                "partial",
                "late",
                "already-attempted",
            }
        ):
            return True
        return any(_failed(item) for item in payload.values())
    if isinstance(payload, list):
        return any(_failed(item) for item in payload)
    return False


async def _capture_then_resolve(
    *,
    store: ForwardStudyStore,
    protocol: dict[str, Any],
    scheduled_at: datetime | None,
    settings: Settings,
) -> dict[str, Any]:
    capture = (
        await capture_forward_slot(
            store=store, protocol=protocol, scheduled_at=scheduled_at, settings=settings
        )
        if scheduled_at is not None
        else {"status": "study-ended", "protocol_id": protocol["protocol_id"]}
    )
    # Discovery and decision clocks take priority over unrelated settlement requests.
    resolution = await resolve_forward_outcomes(store=store, protocol=protocol)
    return {"capture": capture, "resolution": resolution}


def register_forward_commands(app: typer.Typer) -> None:
    @app.command("forward-register")
    def forward_register(
        protocol: Annotated[Path, typer.Option(help="Immutable study protocol.")] = (
            DEFAULT_FORWARD_PROTOCOL_PATH
        ),
        db: Annotated[Path | None, typer.Option(help="Defaults to Settings.database_path.")] = None,
    ) -> None:
        """Preregister the independent study before its start; identical retries are safe."""
        try:
            document = load_forward_protocol(protocol)
            store = ForwardStudyStore(db if db is not None else Settings().database_path)
            _emit(store.register(document))
        except Exception as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None

    @app.command("forward-capture")
    def forward_capture(
        protocol: Annotated[Path, typer.Option(help="Immutable study protocol.")] = (
            DEFAULT_FORWARD_PROTOCOL_PATH
        ),
        db: Annotated[Path | None, typer.Option(help="Defaults to Settings.database_path.")] = None,
        scheduled_at: Annotated[
            str | None,
            typer.Option(help="Frozen hourly slot identity, not an observation-clock override."),
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option(
                help="Actual live diagnostic only; never registers or writes study evidence."
            ),
        ] = False,
    ) -> None:
        """Capture the latest minute-45 slot first, then collect pending public resolutions."""
        try:
            document = load_forward_protocol(protocol)
            settings = Settings()
            if db is not None:
                settings = settings.model_copy(update={"database_path": db})
            if dry_run:
                if scheduled_at is not None:
                    raise ValueError("--scheduled-at cannot be combined with --dry-run")
                result = asyncio.run(probe_forward_capture(protocol=document, settings=settings))
                result = {**result, "diagnostic": True, "eligible": False}
                failed = _failed(result)
            else:
                now = datetime.now(UTC)
                slots = scheduled_slots(document)
                if scheduled_at is not None:
                    slot = _timestamp(scheduled_at)
                    if slot not in slots:
                        raise ValueError("--scheduled-at must identify an exact frozen hourly slot")
                    if slot > now:
                        raise ValueError("--scheduled-at cannot be in the future")
                else:
                    slot = now.replace(minute=45, second=0, microsecond=0)
                    if slot > now:
                        slot -= timedelta(hours=1)
                    if slot < slots[0]:
                        _emit({"status": "not-started", "protocol_id": document["protocol_id"]})
                        return
                store = ForwardStudyStore(settings.database_path)
                result = asyncio.run(
                    _capture_then_resolve(
                        store=store,
                        protocol=document,
                        scheduled_at=slot if slot in slots else None,
                        settings=settings,
                    )
                )
                failed = _failed(result)
            _emit(result)
        except Exception as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        if failed:
            raise typer.Exit(code=1)

    @app.command("forward-resolve")
    def forward_resolve(
        protocol: Annotated[Path, typer.Option(help="Immutable study protocol.")] = (
            DEFAULT_FORWARD_PROTOCOL_PATH
        ),
        db: Annotated[Path | None, typer.Option(help="Defaults to Settings.database_path.")] = None,
    ) -> None:
        """Collect captured-market outcomes; print operational status, not scores."""
        try:
            document = load_forward_protocol(protocol)
            store = ForwardStudyStore(db if db is not None else Settings().database_path)
            result = asyncio.run(resolve_forward_outcomes(store=store, protocol=document))
            _emit(result)
        except Exception as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
        if _failed(result):
            raise typer.Exit(code=1)

    @app.command("forward-report")
    def forward_report(
        protocol: Annotated[Path, typer.Option(help="Immutable study protocol.")] = (
            DEFAULT_FORWARD_PROTOCOL_PATH
        ),
        db: Annotated[Path | None, typer.Option(help="Defaults to Settings.database_path.")] = None,
        as_of: Annotated[
            str | None, typer.Option(help="Nonfuture UTC cutoff; cannot relax the final embargo.")
        ] = None,
        output: Annotated[
            Path | None, typer.Option(help="Optional NEW JSON file; never overwrites evidence.")
        ] = None,
    ) -> None:
        """Read-only coverage until the frozen cutoff, then the frozen final study report."""
        try:
            now = datetime.now(UTC)
            cutoff = _timestamp(as_of) if as_of is not None else now
            if cutoff > now:
                raise ValueError("--as-of cannot be in the future")
            document = load_forward_protocol(protocol)
            database = db if db is not None else Settings().database_path
            if output is not None and output.resolve() == database.resolve():
                raise ValueError("report output cannot be the study database")
            report = build_forward_report(
                store=ForwardStudyStore(database), protocol=document, as_of=cutoff
            )
            _emit(report, output)
        except Exception as exc:
            raise typer.BadParameter(redact_secrets(str(exc))) from None
