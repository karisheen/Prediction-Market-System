"""Read-only observation adapters for the Phase 1 market-baseline comparison.

``load_fixture`` evaluates a small declarative synthetic fixture through the real engine.
``load_local_database`` reads recorded forward-evaluation evidence from an existing SQLite
database without writing, migrating, or reclassifying it. A private SQLite online-backup
snapshot binds its source hash to all extracted evidence, including committed WAL data.
Adapters establish identities, source integrity, the archive window, and outcome availability;
protocol eligibility (model, contract structure, timing, freshness, quotes) belongs to
``experiment.compare_observations``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid5

from pydantic import ValidationError

from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketRegimeSnapshot,
    MarketSnapshot,
    Opportunity,
    TerminalRangeContract,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import (
    AMBIGUOUS_EVIDENCE_CLASS,
    EVIDENCE_FORWARD_SHADOW,
    KNOWN_EVIDENCE_CLASSES,
    content_id,
)
from prediction_market_system.research import ResearchDataUnavailable
from prediction_market_system.storage import SQLiteRepository

SOURCE_SYNTHETIC_FIXTURE = "synthetic-fixture"
SOURCE_RETROSPECTIVE_LOCAL = "retrospective-local"
FIXTURE_VERSION = "pms-phase1-fixture-v1"
SYNTHETIC_RUN_MANIFEST_SCHEMA = "pms-phase1-synthetic-run-manifest-v1"
KALSHI_SNAPSHOT_SOURCE = "kalshi_market_snapshots"

_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "forecast_ledger": (
        "forecast_id",
        "run_id",
        "input_id",
        "market_id",
        "event_id",
        "series_id",
        "observed_at",
        "recipe_id",
        "probability_yes",
        "market_probability_yes",
        "state",
        "payload_json",
    ),
    "run_manifests": ("run_id", "kind", "recorded_at", "manifest_sha256", "payload_json"),
    "input_objects": ("input_id", "payload_json"),
}
_RESEARCH_TABLES = (
    "crypto_spot_candles",
    "research_provider_revisions",
    "research_revision_observations",
)
_OPTIONAL_SCHEMA: dict[str, tuple[tuple[str, ...], str]] = {
    "kalshi_resolutions": (
        ("ticker", "observed_at", "result", "settlement_ts"),
        "no resolution evidence: every observation remains unresolved",
    ),
    KALSHI_SNAPSHOT_SOURCE: (
        ("ticker", "observed_at", "status", "payload_json"),
        "no later execution snapshots are attached",
    ),
    "forecasts": (("forecast_id",), "legacy forecast rows cannot be inventoried"),
    **{
        table: ((), "rows referencing research contexts are excluded as irreproducible")
        for table in _RESEARCH_TABLES
    },
}
_QUOTE_FIELDS = ("yes_bid", "yes_ask", "no_bid", "no_ask", "yes_ask_size", "no_ask_size")


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be an ISO string: {value!r}")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return parsed.astimezone(UTC)


def _reason(error: Exception) -> str:
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return f"{location}: {first['msg']}" if location else str(first["msg"])
    return str(error)


def _finite_or_none(value: Any) -> float | None:
    """Native quote value as a float; missing stays missing, never synthesized."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("quote values must be numeric")
    try:
        number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"quote value is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"quote value is not finite: {value!r}")
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Boundaries:
    report_as_of: datetime
    archive_start: datetime
    archive_end: datetime
    execution_horizon_seconds: float

    @classmethod
    def from_protocol(cls, protocol: dict[str, Any]) -> _Boundaries:
        try:
            latencies = [
                protocol["decision_costs"]["latency_seconds"],
                *(scenario["latency_seconds"] for scenario in protocol["cost_scenarios"]),
            ]
            # A fill snapshot may arrive up to one freshness bound after the slowest latency.
            boundaries = cls(
                report_as_of=_utc(protocol["report_as_of"]),
                archive_start=_utc(protocol["archive_start"]),
                archive_end=_utc(protocol["archive_end"]),
                execution_horizon_seconds=max(latencies) + protocol["maximum_input_age_seconds"],
            )
        except KeyError as error:
            raise ValueError(f"protocol is missing {error.args[0]}") from error
        if boundaries.archive_start >= boundaries.archive_end:
            raise ValueError("protocol archive window is empty")
        if boundaries.execution_horizon_seconds < 0:
            raise ValueError("protocol latency must be non-negative")
        return boundaries

    def outside_window(self, observed_at: datetime) -> str | None:
        if observed_at > self.report_as_of:
            return "observed after report_as_of"
        if observed_at < self.archive_start:
            return "observed before archive_start"
        if observed_at >= self.archive_end:
            return "observed at or after archive_end"
        return None

    def execution_end(self, observed_at: datetime, expires_at: datetime) -> datetime:
        return min(
            observed_at + timedelta(seconds=self.execution_horizon_seconds),
            expires_at,
            self.report_as_of,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "report_as_of": self.report_as_of.isoformat(),
            "archive_start": self.archive_start.isoformat(),
            "archive_end": self.archive_end.isoformat(),
            "execution_horizon_seconds": self.execution_horizon_seconds,
            "observation_rule": (
                "archive_start <= observed_at < archive_end and observed_at <= report_as_of"
            ),
            "resolution_rule": (
                "latest record with observed_at <= report_as_of and settlement_ts null or "
                "<= report_as_of; conflicting, non-binary, or pre-forecast records exclude"
            ),
            "execution_rule": (
                "observed_at < snapshot.observed_at <= min(observed_at + "
                "execution_horizon_seconds, market.expires_at, report_as_of); horizon = "
                "slowest protocol latency + maximum_input_age_seconds"
            ),
        }


def _engine_config(protocol: dict[str, Any]) -> EngineConfig:
    try:
        costs = protocol["decision_costs"]
        return EngineConfig(
            structural_weight=protocol["structural_weight"],
            maximum_input_age_seconds=protocol["maximum_input_age_seconds"],
            binary_fee_type=costs["fee_type"],
            binary_fee_coefficient=costs["fee_coefficient"],
            slippage_bps=costs["slippage_bps"],
            resolution_haircut=costs["resolution_haircut"],
            uncertainty_margin=costs["uncertainty_margin"],
            minimum_ask_size=costs["minimum_ask_size"],
            fractional_kelly=costs["fractional_kelly"],
            paper_bankroll=costs["paper_bankroll"],
            max_bankroll_fraction=costs["max_bankroll_fraction"],
            max_event_bankroll_fraction=costs["max_event_bankroll_fraction"],
            min_conservative_edge=costs["min_conservative_edge"],
            minimum_seconds_to_expiry=costs["minimum_seconds_to_expiry"],
        )
    except KeyError as error:
        raise ValueError(f"protocol is missing {error.args[0]}") from error


def _excluded(
    observation_id: str,
    event_id: str | None,
    market_id: str,
    observed_at: datetime | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "event_id": event_id,
        "market_id": market_id,
        "observed_at": observed_at.isoformat() if observed_at is not None else None,
        "exclusion_reason": reason,
    }


def _observation(
    *,
    observation_id: str,
    opportunity: dict[str, Any],
    inputs: dict[str, Any],
    run_id: str,
    run_manifest: dict[str, Any],
    research_context: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
    resolution_withheld: bool,
    execution: list[dict[str, Any]],
) -> dict[str, Any]:
    market = opportunity["market"]
    return {
        "observation_id": observation_id,
        "event_id": market.get("event_id"),
        "market_id": market["market_id"],
        "observed_at": _utc(opportunity["forecast"]["generated_at"]).isoformat(),
        "opportunity": opportunity,
        "inputs": inputs,
        "run_id": run_id,
        "run_manifest": run_manifest,
        "run_manifest_sha256": content_id(run_manifest),
        "input_id": content_id(inputs),
        "research_context": research_context,
        "resolution": resolution,
        "resolution_unavailable_reason": (
            "outcome-after-cutoff" if resolution is None and resolution_withheld else None
        ),
        "execution": execution,
    }


def _select_resolution(
    records: list[dict[str, Any]],
    *,
    forecast_at: datetime,
    report_as_of: datetime,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Outcome known at ``report_as_of``: (resolution, exclusion reason, later record withheld)."""
    available: list[tuple[datetime, datetime | None, str]] = []
    withheld = False
    for record in records:
        observed = _utc(record["observed_at"])
        settlement = record.get("settlement_ts")
        settled = _utc(settlement) if settlement is not None else None
        if observed > report_as_of or (settled is not None and settled > report_as_of):
            withheld = True
            continue
        available.append((observed, settled, str(record["result"])))
    if not available:
        return None, None, withheld
    if len({result for _, _, result in available}) > 1:
        return None, "conflicting resolution records available at report_as_of", withheld
    if min(observed for observed, _, _ in available) <= forecast_at:
        return None, "resolution was recorded at or before the forecast", withheld
    observed, settled, result = max(available, key=lambda item: item[0])
    if result not in {"yes", "no"}:
        return None, f"unsupported resolution result: {result!r}", withheld
    resolution = {
        "result": result,
        "observed_at": observed.isoformat(),
        "settlement_ts": settled.isoformat() if settled is not None else None,
    }
    return resolution, None, withheld


def _order(row: dict[str, Any]) -> tuple[datetime, str]:
    observed = row["observed_at"]
    return (
        _utc(observed) if observed is not None else datetime.min.replace(tzinfo=UTC),
        str(row["observation_id"]),
    )


def _execution_slice(
    times: list[datetime],
    snapshots: list[dict[str, Any]],
    *,
    observed_at: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Snapshots strictly after the observation and no later than ``end``; ``times`` sorted."""
    if end <= observed_at:
        return []
    return snapshots[bisect_right(times, observed_at) : bisect_right(times, end)]


def _consumed(observations: list[dict[str, Any]], tables: list[str]) -> dict[str, Any]:
    rows = [row for row in observations if "exclusion_reason" not in row]
    contexts = {
        row["inputs"]["crypto"]["input_provenance"]["research_context_id"]
        for row in rows
        if row["research_context"] is not None
    }
    return {
        "run_ids": sorted({row["run_id"] for row in rows}),
        "input_ids": sorted({row["input_id"] for row in rows}),
        "run_manifest_sha256s": sorted({row["run_manifest_sha256"] for row in rows}),
        "research_context_ids": sorted(contexts),
        "tables": tables,
    }


def _summarize(
    observations: list[dict[str, Any]],
    inventory: Counter[str],
    exclusions: Counter[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [row for row in observations if "exclusion_reason" not in row]
    for row in observations:
        if "exclusion_reason" in row:
            exclusions["malformed", row["exclusion_reason"]] += 1
    inventory["observations_emitted"] = len(rows)
    inventory["observations_excluded_by_adapter"] = len(observations) - len(rows)
    inventory["resolved_at_report_as_of"] = sum(row["resolution"] is not None for row in rows)
    inventory["unresolved_at_report_as_of"] = sum(row["resolution"] is None for row in rows)
    inventory["explicit_events_emitted"] = len(
        {row["event_id"] for row in rows if row["event_id"] is not None}
    )
    inventory["execution_snapshots_attached"] = sum(len(row["execution"]) for row in rows)
    exclusion_list = [
        {"category": category, "reason": reason, "count": count}
        for (category, reason), count in sorted(exclusions.items())
    ]
    return dict(sorted(inventory.items())), exclusion_list


def load_fixture(
    path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the declarative synthetic fixture through the real engine; nothing is persisted.

    Fixture documents are author-controlled: structural errors raise. Deliberately malformed
    or engine-rejected cases become explicit excluded rows, never silent drops.
    """
    raw = path.read_bytes()
    fixture = json.loads(raw)
    if (
        not isinstance(fixture, dict)
        or fixture.get("fixture_version") != FIXTURE_VERSION
        or fixture.get("synthetic") is not True
    ):
        raise ValueError(f"fixture must be a synthetic {FIXTURE_VERSION} document")
    boundaries = _Boundaries.from_protocol(protocol)
    symbol = str(protocol["symbol"]).upper()
    if (
        str(fixture["series_id"]).upper() != str(protocol["series"]).upper()
        or str(fixture["symbol"]).upper() != symbol
    ):
        raise ValueError("fixture series or symbol does not match the protocol")
    engine = CryptoThresholdEngine(_engine_config(protocol))
    namespace = UUID(str(fixture["uuid_namespace"]))

    markets: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for event in fixture["events"]:
        for market in event["markets"]:
            if market["market_id"] in markets:
                raise ValueError(f"duplicate fixture market: {market['market_id']}")
            markets[market["market_id"]] = (event, market)

    observations: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    inventory: Counter[str] = Counter()
    exclusions: Counter[tuple[str, str]] = Counter()
    for case in fixture["observations"]:
        case_id = str(case["case_id"])
        if any(existing["case_id"] == case_id for existing in cases):
            raise ValueError(f"duplicate fixture case: {case_id}")
        if case["market_id"] not in markets:
            raise ValueError(f"fixture case {case_id} references an unknown market")
        event, market = markets[case["market_id"]]
        cases.append(
            {
                "case_id": case_id,
                "observation_id": str(uuid5(namespace, f"forecast:{case_id}")),
                "event_id": event["event_id"],
                "market_id": market["market_id"],
                "purpose": case["purpose"],
                "labels": list(case["labels"]),
                "deliberate_failure": bool(case["deliberate_failure"]),
                "expected": case["expected"],
                "core_exclusion": case.get("core_exclusion"),
            }
        )
        outside = boundaries.outside_window(_utc(case["observed_at"]))
        if outside is not None:
            inventory["cases_outside_window"] += 1
            exclusions["window", outside] += 1
            continue
        observations.append(
            _fixture_observation(
                fixture,
                event,
                market,
                case,
                engine=engine,
                namespace=namespace,
                boundaries=boundaries,
                symbol=symbol,
                inventory=inventory,
            )
        )

    observations.sort(key=_order)
    inventory["cases_declared"] = len(cases)
    inventory_summary, exclusion_list = _summarize(observations, inventory, exclusions)
    metadata = {
        "source_kind": SOURCE_SYNTHETIC_FIXTURE,
        "synthetic": True,
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "wal_sha256": None,
        },
        "dataset_sha256": content_id(observations),
        "protocol_id": protocol.get("protocol_id"),
        "boundaries": boundaries.metadata(),
        "schema": {
            "status": "not-applicable",
            "tables_present": [],
            "missing_required": [],
            "missing_optional": [],
        },
        "inventory": inventory_summary,
        "exclusions": exclusion_list,
        "consumed": _consumed(observations, []),
        "fixture": {
            "fixture_id": fixture["fixture_id"],
            "fixture_version": fixture["fixture_version"],
            "label": fixture["label"],
            "cases": cases,
            "failure_examples": [
                case["observation_id"] for case in cases if case["deliberate_failure"]
            ],
        },
    }
    return observations, metadata


def _fixture_observation(
    fixture: dict[str, Any],
    event: dict[str, Any],
    market_spec: dict[str, Any],
    case: dict[str, Any],
    *,
    engine: CryptoThresholdEngine,
    namespace: UUID,
    boundaries: _Boundaries,
    symbol: str,
    inventory: Counter[str],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    forecast_id = uuid5(namespace, f"forecast:{case_id}")
    observation_id = str(forecast_id)
    observed_at = _utc(case["observed_at"])
    market_id = str(market_spec["market_id"])
    event_id = str(event["event_id"]) if market_spec.get("explicit_event_id", True) else None
    fixture_reference = {"fixture_id": fixture["fixture_id"], "case_id": case_id}
    observation_end = _utc(event["observation_end_at"])
    window = int(market_spec["settlement_window_seconds"])
    lower, upper = market_spec["lower_bound"], market_spec["upper_bound"]
    quotes, crypto_spec = case["quotes"], case["crypto"]
    ending = observation_end.isoformat()
    rule = (
        f"[SYNTHETIC FIXTURE] Resolves YES if the simple arithmetic average of the sixty "
        f"seconds of {symbol} index prices before {ending} is at least {lower} and below {upper}."
        if window
        else f"[SYNTHETIC FIXTURE] Resolves YES if the {symbol} index price at {ending} is at "
        f"least {lower} and below {upper}; no settlement averaging."
    )
    try:
        contract = TerminalRangeContract(
            lower_bound=lower, upper_bound=upper, settlement_window_seconds=window
        )
        market = MarketSnapshot(
            market_id=market_id,
            question=f"[SYNTHETIC FIXTURE] {symbol} terminal range {lower}-{upper} at {ending}",
            venue=fixture["venue"],
            observed_at=observed_at,
            expires_at=_utc(event["close_at"]),
            observation_end_at=observation_end,
            observation_start_at=observation_end - timedelta(seconds=window) if window else None,
            expected_settlement_at=_utc(event["expected_settlement_at"]),
            yes_bid=quotes["yes_bid"],
            yes_ask=quotes["yes_ask"],
            no_bid=quotes["no_bid"],
            no_ask=quotes["no_ask"],
            yes_ask_size=quotes["yes_ask_size"],
            no_ask_size=quotes["no_ask_size"],
            resolution_rule=rule,
            series_id=fixture["series_id"],
            event_id=event_id,
            contract_label=f"{lower} to below {upper}",
            source_metadata={
                "synthetic": True,
                **fixture_reference,
                "purpose": case["purpose"],
                "labels": list(case["labels"]),
                "deliberate_failure": bool(case["deliberate_failure"]),
            },
        )
        crypto = CryptoSnapshot(
            symbol=symbol,
            observed_at=_utc(crypto_spec["observed_at"]),
            spot_price=crypto_spec["spot_price"],
            strike_price=engine.reference_price(contract),
            annualized_volatility=crypto_spec["annualized_volatility"],
            feature_recipe={
                "source_selection": SOURCE_SYNTHETIC_FIXTURE,
                "volatility_estimator": "synthetic-fixture-annualized-volatility",
                "drift_estimator": "synthetic-fixture-zero-drift",
            },
            input_provenance={"source": SOURCE_SYNTHETIC_FIXTURE, **fixture_reference},
        )
        regime_spec = event.get("regime")
        thresholds = fixture["regime_thresholds"]
        regime = (
            MarketRegimeSnapshot(
                symbol=symbol,
                observed_at=crypto.observed_at,
                source_start_at=crypto.observed_at
                - timedelta(seconds=thresholds["lookback_seconds"]),
                trailing_return=regime_spec["trailing_return"],
                realized_volatility=regime_spec["realized_volatility"],
                price_trend=regime_spec["price_trend"],
                volatility=regime_spec["volatility"],
                trend_threshold=thresholds["trend_threshold"],
                low_volatility_threshold=thresholds["low_volatility_threshold"],
                high_volatility_threshold=thresholds["high_volatility_threshold"],
            )
            if regime_spec is not None
            else None
        )
    except ValueError as error:
        return _excluded(
            observation_id,
            event_id,
            market_id,
            observed_at,
            f"malformed synthetic input: {_reason(error)}",
        )
    try:
        forecast, opportunity = engine.evaluate(market, crypto, contract)
    except ValueError as error:
        return _excluded(
            observation_id,
            event_id,
            market_id,
            observed_at,
            f"engine rejected synthetic input: {_reason(error)}",
        )
    forecast = forecast.model_copy(update={"forecast_id": forecast_id})
    opportunity = opportunity.model_copy(
        update={
            "opportunity_id": uuid5(namespace, f"opportunity:{case_id}"),
            "forecast": forecast,
            "market_regime": regime,
        }
    )
    payload = opportunity.model_dump(mode="json")
    inputs: dict[str, Any] = payload["forecast"].pop("input_manifest")
    run_manifest = {
        "schema": SYNTHETIC_RUN_MANIFEST_SCHEMA,
        "kind": SOURCE_SYNTHETIC_FIXTURE,
        "run_id": observation_id,
        "recorded_at": forecast.generated_at.isoformat(),
        "configuration": {"recipe_id": forecast.recipe_id, "engine": inputs["engine_config"]},
        "inputs": {"input_id": content_id(inputs)},
        "fixture": fixture_reference,
    }

    resolution_spec = market_spec.get("resolution")
    resolution, reason, withheld = _select_resolution(
        [resolution_spec] if resolution_spec is not None else [],
        forecast_at=observed_at,
        report_as_of=boundaries.report_as_of,
    )
    if reason is not None:
        return _excluded(observation_id, event_id, market_id, observed_at, reason)
    if resolution is None and withheld:
        inventory["resolutions_withheld_after_report_as_of"] += 1

    declared = sorted(
        (
            (
                _utc(snapshot["observed_at"]),
                {
                    "snapshot_id": str(uuid5(namespace, f"execution:{case_id}:{index}")),
                    "market_id": market_id,
                    "observed_at": _utc(snapshot["observed_at"]).isoformat(),
                    "source": SOURCE_SYNTHETIC_FIXTURE,
                    "status": snapshot.get("status", "active"),
                    **{field: _finite_or_none(snapshot.get(field)) for field in _QUOTE_FIELDS},
                    "label": snapshot["label"],
                },
            )
            for index, snapshot in enumerate(case.get("execution", []))
        ),
        key=lambda item: item[0],
    )
    execution = _execution_slice(
        [observed for observed, _ in declared],
        [snapshot for _, snapshot in declared],
        observed_at=observed_at,
        end=boundaries.execution_end(observed_at, market.expires_at),
    )
    inventory["execution_snapshots_outside_horizon"] += len(declared) - len(execution)
    return _observation(
        observation_id=observation_id,
        opportunity=payload,
        inputs=inputs,
        run_id=observation_id,
        run_manifest=run_manifest,
        research_context=None,
        resolution=resolution,
        resolution_withheld=withheld,
        execution=execution,
    )


class _ReadOnlyRepository(SQLiteRepository):
    """Existing repository reads over an externally owned ``mode=ro`` connection."""

    def __init__(self, database_path: Path, connection: sqlite3.Connection) -> None:
        super().__init__(database_path)
        self._read_only_connection = connection

    def _connect(self) -> sqlite3.Connection:
        return self._read_only_connection


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


@contextmanager
def _database_snapshot(path: Path) -> Iterator[Path]:
    """Freeze committed source data, including WAL pages, without modifying the source.

    The source connection closes before the snapshot is hashed or queried. All extraction
    and research reconstruction share this private backup, which is removed on every exit.
    """
    with TemporaryDirectory(prefix="pms-phase1-snapshot-") as directory:
        snapshot = Path(directory) / "snapshot.db"
        with (
            closing(_open_read_only(path)) as source,
            closing(sqlite3.connect(snapshot)) as destination,
        ):
            source.backup(destination)
            # Keep the backup self-contained even when the source uses WAL journaling.
            destination.execute("PRAGMA journal_mode = DELETE")
        yield snapshot


def _inspect_schema(connection: sqlite3.Connection) -> dict[str, set[str]]:
    names = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    ]
    return {
        name: {
            str(column[1])
            for column in connection.execute(
                'PRAGMA table_info("{}")'.format(name.replace('"', '""'))
            )
        }
        for name in names
    }


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    """Only internal constant table names are accepted by callers."""
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def load_local_database(
    path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read recorded forward-evaluation evidence without writing, migrating, or reclassifying.

    Only ledger rows whose immutable run manifest is ``forward-evaluation`` for the protocol
    series are emitted. Historical, manual, ambiguous-legacy, and legacy ``forecasts`` rows are
    inventoried, never retrofitted into forward records. A missing required schema yields an
    explicit empty population plus inventory. ``source.sha256`` identifies the private
    online-backup snapshot queried, not the live database's main file or a separate WAL.
    """
    boundaries = _Boundaries.from_protocol(protocol)
    series = str(protocol["series"]).upper()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    with _database_snapshot(path) as snapshot:
        source = {
            "path": str(path),
            "sha256": _sha256_file(snapshot),
            "bytes": snapshot.stat().st_size,
            "wal_sha256": None,
            "snapshot_method": "sqlite-online-backup",
            "hash_scope": "queried-backup-snapshot",
            "snapshot_semantics": (
                "consistent committed SQLite state including WAL; read-only source closed "
                "before hashing and extraction; private backup removed after extraction"
            ),
        }
        with closing(_open_read_only(snapshot)) as connection:
            reader = _DatabaseReader(snapshot, connection, boundaries, series)
            observations = reader.observations()
    inventory_summary, exclusion_list = _summarize(
        observations, reader.inventory, reader.exclusions
    )
    metadata = {
        "source_kind": SOURCE_RETROSPECTIVE_LOCAL,
        "synthetic": False,
        "source": source,
        "dataset_sha256": content_id(observations),
        "protocol_id": protocol.get("protocol_id"),
        "boundaries": boundaries.metadata(),
        "schema": reader.schema_report,
        "inventory": {**inventory_summary, **reader.ledger_classes},
        "exclusions": exclusion_list,
        "consumed": _consumed(observations, sorted(reader.tables_read)),
        "fixture": None,
    }
    return observations, metadata


class _DatabaseReader:
    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        boundaries: _Boundaries,
        series: str,
    ) -> None:
        self.connection = connection
        self.boundaries = boundaries
        self.series = series
        self.repository = _ReadOnlyRepository(path, connection)
        self.inventory: Counter[str] = Counter()
        self.exclusions: Counter[tuple[str, str]] = Counter()
        self.tables_read: set[str] = set()
        self.ledger_classes: dict[str, dict[str, int]] = {
            "ledger_rows_by_evidence_class": {},
            "series_ledger_rows_by_evidence_class": {},
        }
        self.schema = _inspect_schema(connection)
        self.missing_required: list[dict[str, Any]] = [
            {"table": table, "columns": sorted(set(columns) - self.schema.get(table, set()))}
            for table, columns in _REQUIRED_SCHEMA.items()
            if table not in self.schema or not set(columns) <= self.schema[table]
        ]
        missing_optional: list[dict[str, Any]] = [
            {
                "table": table,
                "columns": sorted(set(columns) - self.schema.get(table, set())),
                "effect": effect,
            }
            for table, (columns, effect) in _OPTIONAL_SCHEMA.items()
            if table not in self.schema or not set(columns) <= self.schema[table]
        ]
        self.unavailable = {str(missing["table"]) for missing in missing_optional}
        self.schema_report = {
            "status": "missing-required" if self.missing_required else "complete",
            "tables_present": sorted(self.schema),
            "missing_required": self.missing_required,
            "missing_optional": missing_optional,
        }
        self._resolution_cache: dict[str, list[dict[str, Any]]] = {}
        self._snapshot_cache: dict[str, tuple[list[datetime], list[dict[str, Any]]]] = {}
        self._context_cache: dict[str, tuple[dict[str, Any] | None, str | None]] = {}

    def observations(self) -> list[dict[str, Any]]:
        self._inventory_legacy_forecasts()
        self._inventory_ledger()
        if self.missing_required:
            unavailable = "; ".join(
                f"{missing['table']} (missing columns: {', '.join(missing['columns']) or 'none'})"
                for missing in self.missing_required
            )
            self.exclusions[
                "schema",
                f"required schema unavailable: {unavailable}; no qualified population",
            ] += self.inventory["ledger_rows_total"]
            return []
        self.tables_read.update(_REQUIRED_SCHEMA)
        candidates = self.connection.execute(
            """SELECT f.forecast_id, f.market_id, f.observed_at
               FROM forecast_ledger f JOIN run_manifests m ON m.run_id = f.run_id
               WHERE m.kind = ? AND UPPER(f.series_id) = ?
               ORDER BY f.observed_at, f.forecast_id""",
            (EVIDENCE_FORWARD_SHADOW, self.series),
        ).fetchall()
        observations: list[dict[str, Any]] = []
        for candidate in candidates:
            forecast_id, market_id = str(candidate["forecast_id"]), str(candidate["market_id"])
            try:
                observed_at = _utc(candidate["observed_at"])
            except ValueError:
                observations.append(
                    _excluded(forecast_id, None, market_id, None, "malformed ledger observed_at")
                )
                continue
            outside = self.boundaries.outside_window(observed_at)
            if outside is not None:
                self.exclusions["window", outside] += 1
                continue
            row = self.connection.execute(
                """SELECT f.forecast_id, f.run_id, f.input_id, f.market_id, f.recipe_id,
                          f.probability_yes, f.market_probability_yes, f.state, f.payload_json,
                          m.manifest_sha256, m.payload_json AS manifest_json,
                          m.recorded_at AS manifest_recorded_at,
                          i.payload_json AS inputs_json
                   FROM forecast_ledger f JOIN run_manifests m ON m.run_id = f.run_id
                   LEFT JOIN input_objects i ON i.input_id = f.input_id
                   WHERE f.forecast_id = ?""",
                (forecast_id,),
            ).fetchone()
            observations.append(self._ledger_observation(row, observed_at))
        observations.sort(key=_order)
        return observations

    def _inventory_legacy_forecasts(self) -> None:
        if "forecasts" in self.unavailable:
            return
        self.tables_read.add("forecasts")
        if "forecast_id" in self.schema.get("forecast_ledger", set()):
            legacy = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM forecasts WHERE forecast_id NOT IN "
                    "(SELECT forecast_id FROM forecast_ledger)"
                ).fetchone()[0]
            )
        else:
            legacy = _table_count(self.connection, "forecasts")
        self.inventory["legacy_forecasts_without_ledger"] = legacy
        if legacy:
            self.exclusions[
                "legacy",
                "legacy forecast without a v2 ledger record (legacy-unreconstructible)",
            ] += legacy

    def _inventory_ledger(self) -> None:
        if self.missing_required:
            # Classification joins may themselves be unavailable. Count each existing
            # ledger row once; the combined schema exclusion owns this population.
            self.inventory["ledger_rows_total"] = 0
            if "forecast_ledger" in self.schema:
                self.tables_read.add("forecast_ledger")
                self.inventory["ledger_rows_total"] = _table_count(
                    self.connection, "forecast_ledger"
                )
            return
        by_class: Counter[str] = Counter()
        in_series: Counter[str] = Counter()
        for row in self.connection.execute(
            """SELECT UPPER(COALESCE(f.series_id, '')) AS series, m.kind AS kind,
                      COUNT(*) AS count
               FROM forecast_ledger f LEFT JOIN run_manifests m ON m.run_id = f.run_id
               GROUP BY UPPER(COALESCE(f.series_id, '')), m.kind"""
        ):
            kind = row["kind"]
            evidence_class = (
                str(kind) if kind in KNOWN_EVIDENCE_CLASSES else AMBIGUOUS_EVIDENCE_CLASS
            )
            count = int(row["count"])
            by_class[evidence_class] += count
            if row["series"] == self.series:
                in_series[evidence_class] += count
            else:
                self.exclusions["population", "ledger row outside the protocol series"] += count
        for evidence_class, count in in_series.items():
            if evidence_class == AMBIGUOUS_EVIDENCE_CLASS:
                self.exclusions[
                    "legacy",
                    "ledger row lacks a classifiable run manifest (ambiguous-legacy)",
                ] += count
            elif evidence_class != EVIDENCE_FORWARD_SHADOW:
                self.exclusions[
                    "population",
                    f"{evidence_class} ledger evidence is not forward-evaluation; "
                    "never reclassified",
                ] += count
        self.inventory["ledger_rows_total"] = sum(by_class.values())
        self.inventory["series_forward_rows"] = in_series[EVIDENCE_FORWARD_SHADOW]
        self.ledger_classes = {
            "ledger_rows_by_evidence_class": dict(sorted(by_class.items())),
            "series_ledger_rows_by_evidence_class": dict(sorted(in_series.items())),
        }

    def _ledger_observation(self, row: sqlite3.Row, observed_at: datetime) -> dict[str, Any]:
        forecast_id, market_id = str(row["forecast_id"]), str(row["market_id"])
        event_id: str | None = None

        def excluded(reason: str) -> dict[str, Any]:
            return _excluded(forecast_id, event_id, market_id, observed_at, reason)

        try:
            payload = json.loads(str(row["payload_json"]))
            opportunity = Opportunity.model_validate(payload)
        except ValueError as error:
            return excluded(f"malformed ledger payload: {_reason(error)}")
        event_id = opportunity.market.event_id
        forecast = opportunity.forecast
        if str(forecast.forecast_id) != forecast_id or forecast.input_manifest:
            return excluded("ledger payload does not match the ledger forecast record")
        if (
            forecast.market_id != market_id
            or opportunity.market.market_id != market_id
            or forecast.generated_at != observed_at
            or forecast.recipe_id != row["recipe_id"]
            or forecast.probability_yes != row["probability_yes"]
            or forecast.market_probability_yes != row["market_probability_yes"]
            or opportunity.state.value != row["state"]
        ):
            return excluded("ledger columns disagree with the recorded payload")
        if forecast.recipe_id is None:
            return excluded("legacy ledger row without recipe identity")

        try:
            manifest = json.loads(str(row["manifest_json"]))
            if not isinstance(manifest, dict) or content_id(manifest) != row["manifest_sha256"]:
                return excluded("run manifest does not match its recorded hash")
            recorded_at = _utc(row["manifest_recorded_at"])
            if recorded_at != _utc(manifest.get("recorded_at")):
                return excluded("run manifest recording time disagrees with its recorded payload")
            if recorded_at != observed_at:
                return excluded("run manifest was not recorded at the forecast observation time")
            manifest_inputs = manifest.get("inputs")
            if (
                manifest.get("run_id") != row["run_id"]
                or not isinstance(manifest_inputs, dict)
                or manifest_inputs.get("input_id") != row["input_id"]
            ):
                return excluded("run manifest does not reference the ledger run and input")
            if row["inputs_json"] is None:
                return excluded("recorded input object is missing")
            inputs = json.loads(str(row["inputs_json"]))
            if not isinstance(inputs, dict) or content_id(inputs) != row["input_id"]:
                return excluded("recorded input object does not match its identity")
        except ValueError as error:
            return excluded(f"malformed run manifest or input object: {_reason(error)}")

        crypto = inputs.get("crypto")
        provenance = crypto.get("input_provenance") if isinstance(crypto, dict) else None
        context_id = provenance.get("research_context_id") if isinstance(provenance, dict) else None
        research_context: dict[str, Any] | None = None
        if context_id is not None:
            if not isinstance(context_id, str):
                return excluded("malformed research context reference")
            research_context, reason = self._research_context(context_id)
            if reason is not None:
                return excluded(reason)

        try:
            resolution, reason, withheld = _select_resolution(
                self._resolutions(market_id),
                forecast_at=observed_at,
                report_as_of=self.boundaries.report_as_of,
            )
        except ValueError as error:
            return excluded(f"malformed resolution record: {_reason(error)}")
        if reason is not None:
            return excluded(reason)
        if resolution is None and withheld:
            self.inventory["resolutions_withheld_after_report_as_of"] += 1

        times, snapshots = self._snapshots(market_id)
        execution = _execution_slice(
            times,
            snapshots,
            observed_at=observed_at,
            end=self.boundaries.execution_end(observed_at, opportunity.market.expires_at),
        )
        return _observation(
            observation_id=forecast_id,
            opportunity=payload,
            inputs=inputs,
            run_id=str(row["run_id"]),
            run_manifest=manifest,
            research_context=research_context,
            resolution=resolution,
            resolution_withheld=withheld,
            execution=execution,
        )

    def _research_context(self, context_id: str) -> tuple[dict[str, Any] | None, str | None]:
        if context_id not in self._context_cache:
            self.inventory["research_contexts_referenced"] += 1
            result: tuple[dict[str, Any] | None, str | None]
            if self.unavailable.intersection(_RESEARCH_TABLES):
                result = (None, "research context not reproducible: research tables missing")
            else:
                self.tables_read.update(_RESEARCH_TABLES)
                try:
                    context = self.repository.research_context_by_id(context_id)
                except (
                    ResearchDataUnavailable,
                    ValueError,
                    KeyError,
                    TypeError,
                    sqlite3.Error,
                ) as error:
                    result = (
                        None,
                        "research context not reproducible from stored point-in-time "
                        f"sources: {_reason(error)}",
                    )
                else:
                    result = (
                        (None, "referenced research context is missing")
                        if context is None
                        else (context.model_dump(mode="json"), None)
                    )
            self.inventory[
                "research_contexts_reproduced"
                if result[1] is None
                else "research_contexts_unavailable"
            ] += 1
            self._context_cache[context_id] = result
        return self._context_cache[context_id]

    def _resolutions(self, ticker: str) -> list[dict[str, Any]]:
        if "kalshi_resolutions" in self.unavailable:
            return []
        if ticker not in self._resolution_cache:
            self.tables_read.add("kalshi_resolutions")
            self._resolution_cache[ticker] = [
                {
                    "result": row["result"],
                    "observed_at": row["observed_at"],
                    "settlement_ts": row["settlement_ts"],
                }
                for row in self.connection.execute(
                    "SELECT observed_at, result, settlement_ts FROM kalshi_resolutions "
                    "WHERE ticker = ?",
                    (ticker,),
                )
            ]
        return self._resolution_cache[ticker]

    def _snapshots(self, ticker: str) -> tuple[list[datetime], list[dict[str, Any]]]:
        if KALSHI_SNAPSHOT_SOURCE in self.unavailable:
            return [], []
        if ticker not in self._snapshot_cache:
            self.tables_read.add(KALSHI_SNAPSHOT_SOURCE)
            parsed: list[tuple[datetime, dict[str, Any]]] = []
            for row in self.connection.execute(
                "SELECT observed_at, status, payload_json FROM kalshi_market_snapshots "
                "WHERE ticker = ?",
                (ticker,),
            ):
                try:
                    observed = _utc(row["observed_at"])
                    payload = json.loads(str(row["payload_json"]))
                    if not isinstance(payload, dict):
                        raise ValueError("snapshot payload is not an object")
                    # Kalshi's complementary book: a NO ask fills against YES bids, the same
                    # mapping as venues.kalshi.normalize_order_book.
                    quotes = {
                        "yes_bid": _finite_or_none(payload.get("yes_bid_dollars")),
                        "yes_ask": _finite_or_none(payload.get("yes_ask_dollars")),
                        "no_bid": _finite_or_none(payload.get("no_bid_dollars")),
                        "no_ask": _finite_or_none(payload.get("no_ask_dollars")),
                        "yes_ask_size": _finite_or_none(payload.get("yes_ask_size_fp")),
                        "no_ask_size": _finite_or_none(payload.get("yes_bid_size_fp")),
                    }
                except ValueError:
                    self.inventory["execution_snapshots_malformed"] += 1
                    continue
                parsed.append(
                    (
                        observed,
                        {
                            "snapshot_id": (
                                f"{KALSHI_SNAPSHOT_SOURCE}:{ticker}:{row['observed_at']}"
                            ),
                            "market_id": ticker,
                            "observed_at": observed.isoformat(),
                            "source": KALSHI_SNAPSHOT_SOURCE,
                            "status": row["status"],
                            **quotes,
                            "label": None,
                        },
                    )
                )
            parsed.sort(key=lambda item: item[0])
            self._snapshot_cache[ticker] = (
                [observed for observed, _ in parsed],
                [snapshot for _, snapshot in parsed],
            )
        return self._snapshot_cache[ticker]
