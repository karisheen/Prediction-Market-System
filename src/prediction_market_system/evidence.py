from __future__ import annotations

import hashlib
import json
import math
import platform
import sqlite3
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prediction_market_system.domain import Opportunity, ProbabilityForecast
    from prediction_market_system.research import ResearchContext

# Immutable ledger kinds. Forward-shadow reporting may count only EVIDENCE_FORWARD_SHADOW.
EVIDENCE_FORWARD_SHADOW = "forward-evaluation"
EVIDENCE_HISTORICAL = "historical-evaluation"
EVIDENCE_MANUAL_RESEARCH = "manual-evaluation"
KNOWN_EVIDENCE_CLASSES = frozenset(
    {EVIDENCE_FORWARD_SHADOW, EVIDENCE_HISTORICAL, EVIDENCE_MANUAL_RESEARCH}
)
AMBIGUOUS_EVIDENCE_CLASS = "ambiguous-legacy"

EVIDENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_manifests (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS input_objects (
    input_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_ledger (
    forecast_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES run_manifests(run_id),
    input_id TEXT NOT NULL REFERENCES input_objects(input_id),
    market_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    series_id TEXT,
    observed_at TEXT NOT NULL,
    recipe_id TEXT,
    probability_yes REAL NOT NULL,
    market_probability_yes REAL NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_recipe_time ON forecast_ledger(recipe_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_ledger_market_time ON forecast_ledger(market_id, observed_at);
CREATE TABLE IF NOT EXISTS venue_source_revisions (
    revision_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source_key TEXT NOT NULL,
    series_ticker TEXT NOT NULL,
    available_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_venue_revision_key_time
    ON venue_source_revisions(kind, source_key, available_at);
CREATE TABLE IF NOT EXISTS venue_revision_observations (
    kind TEXT NOT NULL,
    source_key TEXT NOT NULL,
    available_at TEXT NOT NULL,
    revision_id TEXT NOT NULL REFERENCES venue_source_revisions(revision_id),
    PRIMARY KEY(kind, source_key, available_at, revision_id)
);
CREATE VIEW IF NOT EXISTS venue_revision_history AS
    SELECT r.kind, r.source_key, r.series_ticker, o.available_at, r.payload_json
    FROM venue_source_revisions r JOIN venue_revision_observations o USING(revision_id);
CREATE TABLE IF NOT EXISTS holdout_usage (
    run_id TEXT NOT NULL REFERENCES run_manifests(run_id),
    series_ticker TEXT NOT NULL,
    event_id TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY(run_id, event_id, recipe_id)
);
CREATE INDEX IF NOT EXISTS idx_holdout_event ON holdout_usage(series_ticker,event_id);
CREATE TABLE IF NOT EXISTS campaign_history (
    report_id TEXT PRIMARY KEY,
    series_ticker TEXT NOT NULL,
    symbol TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_campaign_registrations (
    campaign_id TEXT PRIMARY KEY,
    series_ticker TEXT NOT NULL,
    symbol TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    superseded_at TEXT,
    configuration_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_campaign_registration_active
    ON validation_campaign_registrations(series_ticker, symbol, superseded_at);
CREATE TABLE IF NOT EXISTS paper_model_validations_v2 (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    calibration_profile_id TEXT,
    accepted INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, model_name, recipe_id)
);
CREATE INDEX IF NOT EXISTS idx_v2_validations_profile
    ON paper_model_validations_v2(calibration_profile_id, generated_at);
"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@lru_cache(maxsize=1)
def code_identity() -> dict[str, Any]:
    """Hash actual source and dependency lock, including an uncommitted working tree."""
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    lock = root / "uv.lock"
    revision = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        revision = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_revision": revision,
        "source_sha256": content_id(sources),
        "source_files": sources,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def insert_immutable(
    connection: sqlite3.Connection, table: str, key_column: str, key: str, payload: str
) -> bool:
    """Only internal constant table names are accepted by callers."""
    allowed = {("input_objects", "input_id"), ("campaign_history", "report_id")}
    if (table, key_column) not in allowed:
        raise ValueError("unsupported immutable table")
    row = connection.execute(
        f"SELECT payload_json FROM {table} WHERE {key_column} = ?", (key,)
    ).fetchone()
    if row is not None and str(row[0]) != payload:
        raise ValueError(f"immutable {table} identity collision")
    return row is None


def save_manifest(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    kind: str,
    recorded_at: datetime,
    configuration: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    manifest = {
        "schema": "pms-run-manifest-v2",
        "kind": kind,
        "run_id": run_id,
        "recorded_at": recorded_at.isoformat(),
        "code": code_identity(),
        "configuration": configuration,
        "inputs": inputs,
    }
    payload = canonical_json(manifest)
    digest = content_id(manifest)
    prior = connection.execute(
        "SELECT manifest_sha256 FROM run_manifests WHERE run_id = ?", (run_id,)
    ).fetchone()
    if prior is not None:
        if str(prior[0]) != digest:
            raise ValueError("run manifest is immutable; run ID already has different evidence")
        return
    connection.execute(
        "INSERT INTO run_manifests VALUES (?, ?, ?, ?, ?)",
        (run_id, kind, recorded_at.isoformat(), digest, payload),
    )


def save_input_object(connection: sqlite3.Connection, payload: Any) -> str:
    """Store one content-addressed immutable input; identical content is written once."""
    input_id = content_id(payload)
    serialized = canonical_json(payload)
    if insert_immutable(connection, "input_objects", "input_id", input_id, serialized):
        connection.execute("INSERT INTO input_objects VALUES (?, ?)", (input_id, serialized))
    return input_id


def save_research_context(connection: sqlite3.Connection, context: ResearchContext) -> str:
    """Persist the exact research context once per content identity.

    Forecasts reference ``research_context_id`` from ``CryptoSnapshot.input_provenance``
    instead of embedding hundreds of kilobytes of source candles per evaluation.
    """
    input_id = save_input_object(connection, context.model_dump(mode="json"))
    if input_id != context.content_id():
        raise ValueError("research context identity does not match its canonical payload")
    return input_id


def load_input_object(connection: sqlite3.Connection, input_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT payload_json FROM input_objects WHERE input_id = ?", (input_id,)
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(str(row[0]))
    return payload if isinstance(payload, dict) else None


def resolve_evidence_class(ledger_kind: str | None, *, venue: str) -> str:
    """Map a ledger write onto a known evidence class.

    Kalshi live scans default to forward-shadow only when the caller does not
    pass an explicit kind. Historical replay must pass ``EVIDENCE_HISTORICAL``.
    Unknown kinds fail closed.
    """
    if ledger_kind is None:
        if venue.casefold() == "kalshi":
            return EVIDENCE_FORWARD_SHADOW
        return EVIDENCE_MANUAL_RESEARCH
    if ledger_kind not in KNOWN_EVIDENCE_CLASSES:
        raise ValueError(f"unknown evidence class: {ledger_kind}")
    return ledger_kind


def save_ledger_evaluation(
    connection: sqlite3.Connection,
    forecast: ProbabilityForecast,
    opportunity: Opportunity,
    *,
    ledger_kind: str | None = None,
) -> None:
    inputs = forecast.input_manifest
    input_id = save_input_object(connection, inputs)
    run_id = str(forecast.forecast_id)
    evidence_class = resolve_evidence_class(ledger_kind, venue=opportunity.market.venue)
    save_manifest(
        connection,
        run_id=run_id,
        kind=evidence_class,
        recorded_at=forecast.generated_at,
        configuration={"recipe_id": forecast.recipe_id, "engine": inputs.get("engine_config")},
        inputs={"input_id": input_id},
    )
    payload = opportunity.model_dump(mode="json")
    payload["forecast"].pop("input_manifest", None)
    serialized = canonical_json(payload)
    previous = connection.execute(
        "SELECT payload_json FROM forecast_ledger WHERE forecast_id = ?", (run_id,)
    ).fetchone()
    if previous is not None:
        if str(previous[0]) != serialized:
            raise ValueError("forecast ledger is immutable; forecast already recorded differently")
        return
    connection.execute(
        "INSERT INTO forecast_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            run_id,
            input_id,
            forecast.market_id,
            opportunity.market.event_id or forecast.market_id,
            opportunity.market.series_id,
            forecast.generated_at.isoformat(),
            forecast.recipe_id,
            forecast.probability_yes,
            forecast.market_probability_yes,
            opportunity.state.value,
            serialized,
        ),
    )


def save_venue_revision(
    connection: sqlite3.Connection,
    *,
    kind: str,
    source_key: str,
    series_ticker: str,
    available_at: datetime,
    payload: dict[str, Any],
) -> None:
    if available_at.tzinfo is None:
        raise ValueError("source availability must include a timezone")
    revision_id = content_id({"kind": kind, "key": source_key, "payload": payload})
    connection.execute(
        "INSERT OR IGNORE INTO venue_source_revisions VALUES (?, ?, ?, ?, ?, ?)",
        (
            revision_id,
            kind,
            source_key,
            series_ticker.upper(),
            available_at.astimezone(UTC).isoformat(),
            canonical_json(payload),
        ),
    )
    observed = available_at.astimezone(UTC).isoformat()
    prior = connection.execute(
        "SELECT revision_id FROM venue_revision_observations "
        "WHERE kind=? AND source_key=? AND available_at<=? "
        "ORDER BY available_at DESC LIMIT 1",
        (kind, source_key, observed),
    ).fetchone()
    if prior is None or str(prior[0]) != revision_id:
        connection.execute(
            "INSERT OR IGNORE INTO venue_revision_observations VALUES (?, ?, ?, ?)",
            (kind, source_key, observed, revision_id),
        )


class EvidenceRepositoryMixin:
    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def recorded_opportunity(self, forecast_id: str) -> Opportunity | None:
        from prediction_market_system.domain import Opportunity

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM opportunities WHERE forecast_id = ?",
                (forecast_id,),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("forecast has multiple persisted allocations")
        return Opportunity.model_validate_json(str(rows[0][0]))

    def forward_event_entry_exposure(self, *, event_id: str, exclude_forecast_id: str) -> float:
        """Sum persisted forward-shadow entry allocations for one event, excluding one forecast."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT f.forecast_id, f.payload_json, m.kind AS evidence_class
                   FROM forecast_ledger f
                   LEFT JOIN run_manifests m ON m.run_id = f.run_id
                   WHERE f.event_id = ? AND f.state IN ('ENTER YES', 'ENTER NO')""",
                (event_id,),
            ).fetchall()
        total = 0.0
        for row in rows:
            kind = row["evidence_class"]
            if kind is None or str(kind) not in KNOWN_EVIDENCE_CLASSES:
                raise ValueError("event exposure cannot use unclassified ledger evidence")
            if str(kind) != EVIDENCE_FORWARD_SHADOW:
                continue
            if str(row["forecast_id"]) == exclude_forecast_id:
                continue
            payload = json.loads(str(row["payload_json"]))
            total += float(payload["suggested_max_exposure"])
        return total

    def explain_forecast(self, forecast_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT f.payload_json, i.payload_json AS inputs, r.payload_json AS manifest
                   FROM forecast_ledger f JOIN input_objects i USING(input_id)
                   JOIN run_manifests r USING(run_id) WHERE f.forecast_id = ?""",
                (forecast_id,),
            ).fetchone()
            if row is None:
                legacy = connection.execute(
                    "SELECT payload_json FROM forecasts WHERE forecast_id = ?", (forecast_id,)
                ).fetchone()
                if legacy is None:
                    raise ValueError("forecast not found")
                return {
                    "classification": "legacy-unreconstructible",
                    "evidence_class": AMBIGUOUS_EVIDENCE_CLASS,
                    "forecast": json.loads(legacy[0]),
                }
            inputs = json.loads(row[1])
            provenance = inputs.get("crypto", {}).get("input_provenance", {})
            context_id = provenance.get("research_context_id")
            research_context = (
                load_input_object(connection, context_id) if isinstance(context_id, str) else None
            )
        opportunity = json.loads(row[0])
        manifest = json.loads(row[2])
        kind = manifest.get("kind") if isinstance(manifest, dict) else None
        evidence_class = str(kind) if kind in KNOWN_EVIDENCE_CLASSES else AMBIGUOUS_EVIDENCE_CLASS
        return {
            "classification": (
                "v2-recorded-inputs" if opportunity["forecast"].get("recipe_id") else "legacy"
            ),
            "evidence_class": evidence_class,
            "opportunity": opportunity,
            "inputs": inputs,
            "research_context": research_context,
            "research_context_persisted": research_context is not None,
            "manifest": manifest,
        }

    def shadow_report(self, *, series_ticker: str, as_of: datetime) -> dict[str, Any]:
        if as_of.tzinfo is None:
            raise ValueError("report timestamp must include a timezone")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT f.*, m.kind AS evidence_class, (
                       SELECT result FROM kalshi_resolutions r
                       WHERE r.ticker=f.market_id AND r.observed_at<=?
                         AND (r.settlement_ts IS NULL OR r.settlement_ts<=?)
                       ORDER BY r.observed_at DESC LIMIT 1
                   ) AS outcome FROM forecast_ledger f
                   LEFT JOIN run_manifests m ON m.run_id = f.run_id
                   WHERE f.series_id=? AND f.observed_at<=? ORDER BY f.observed_at""",
                (as_of.isoformat(), as_of.isoformat(), series_ticker.upper(), as_of.isoformat()),
            ).fetchall()
        forward_rows: list[sqlite3.Row] = []
        excluded = 0
        for row in rows:
            kind = row["evidence_class"]
            if kind is None or str(kind) not in KNOWN_EVIDENCE_CLASSES:
                raise ValueError("shadow report cannot classify ledger evidence")
            if str(kind) != EVIDENCE_FORWARD_SHADOW:
                excluded += 1
                continue
            forward_rows.append(row)
        groups: dict[str, list[sqlite3.Row]] = {}
        for row in forward_rows:
            groups.setdefault(str(row["recipe_id"] or "legacy"), []).append(row)
        reports: list[dict[str, Any]] = []
        for recipe, group in sorted(groups.items()):
            event_scores: dict[str, list[tuple[float, float, float, float]]] = {}
            resolved = 0
            for row in group:
                if row["outcome"] not in {"yes", "no"}:
                    continue
                resolved += 1
                outcome = float(row["outcome"] == "yes")
                probability = float(row["probability_yes"])
                baseline = float(row["market_probability_yes"])
                scores = (
                    (probability - outcome) ** 2,
                    (baseline - outcome) ** 2,
                    _log_loss(probability, outcome),
                    _log_loss(baseline, outcome),
                )
                event_scores.setdefault(str(row["event_id"]), []).append(scores)
            means = [
                sum(
                    sum(score[i] for score in scores) / len(scores)
                    for scores in event_scores.values()
                )
                / len(event_scores)
                if event_scores
                else None
                for i in range(4)
            ]
            reports.append(
                {
                    "recipe_id": recipe,
                    "forecasts": len(group),
                    "resolved_forecasts": resolved,
                    "unresolved_forecasts": len(group) - resolved,
                    "independent_resolved_events": len(event_scores),
                    "watch_forecasts": sum(row["state"] == "WATCH" for row in group),
                    "event_weighted_brier": means[0],
                    "market_brier_same_population": means[1],
                    "event_weighted_log_loss": means[2],
                    "market_log_loss_same_population": means[3],
                }
            )
        return {
            "series": series_ticker.upper(),
            "as_of": as_of.isoformat(),
            "recipes": reports,
            "excluded_non_forward_forecasts": excluded,
            "evidence": "forward recorded forecasts; no execution or approval implied",
        }

    def register_validation_campaign(
        self,
        *,
        series_ticker: str,
        symbol: str,
        configuration: dict[str, Any],
        replace: bool = False,
    ) -> str:
        """Preregister (freeze) a validation campaign's configuration.

        The campaign identity is the content hash of every gate, fold geometry,
        engine and model setting that could change which evidence is examined or
        what counts as approval. Re-running with different settings is refused so
        thresholds cannot be tuned after seeing results; ``replace`` explicitly
        supersedes the active registration while preserving the old one.
        """
        campaign_id = content_id(
            {
                "series": series_ticker.upper(),
                "symbol": symbol.upper(),
                "configuration": configuration,
            }
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT campaign_id FROM validation_campaign_registrations "
                "WHERE series_ticker=? AND symbol=? AND superseded_at IS NULL",
                (series_ticker.upper(), symbol.upper()),
            ).fetchall()
            active_ids = {str(row[0]) for row in active}
            if active_ids == {campaign_id}:
                return campaign_id
            if active_ids and not replace:
                raise ValueError(
                    "validation campaign configuration is frozen "
                    f"(active campaign {next(iter(active_ids))[:12]}); "
                    "rerun with the registered settings or explicitly replace the campaign"
                )
            if active_ids:
                connection.execute(
                    "UPDATE validation_campaign_registrations SET superseded_at=? "
                    "WHERE series_ticker=? AND symbol=? AND superseded_at IS NULL",
                    (now, series_ticker.upper(), symbol.upper()),
                )
            connection.execute(
                "INSERT INTO validation_campaign_registrations VALUES (?, ?, ?, ?, NULL, ?) "
                "ON CONFLICT(campaign_id) DO UPDATE SET superseded_at=NULL",
                (
                    campaign_id,
                    series_ticker.upper(),
                    symbol.upper(),
                    now,
                    canonical_json(configuration),
                ),
            )
        return campaign_id

    def campaign_registration(self, campaign_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT campaign_id, series_ticker, symbol, registered_at,
                          superseded_at, configuration_json
                   FROM validation_campaign_registrations WHERE campaign_id=?""",
                (campaign_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "campaign_id": str(row["campaign_id"]),
            "series_ticker": str(row["series_ticker"]),
            "symbol": str(row["symbol"]),
            "registered_at": str(row["registered_at"]),
            "superseded_at": row["superseded_at"],
            "configuration": json.loads(str(row["configuration_json"])),
        }

    def campaign_is_active(self, campaign_id: str) -> bool:
        registration = self.campaign_registration(campaign_id)
        return registration is not None and registration["superseded_at"] is None

    def campaign_report(self, *, series_ticker: str, symbol: str) -> dict[str, Any]:
        with self._connect() as connection:
            registrations = connection.execute(
                """SELECT campaign_id,registered_at,superseded_at,configuration_json
                   FROM validation_campaign_registrations
                   WHERE series_ticker=? AND symbol=? ORDER BY registered_at""",
                (series_ticker.upper(), symbol.upper()),
            ).fetchall()
            rows = connection.execute(
                """SELECT start_at,end_at,status,counts_json,error
                   FROM paper_validation_archive_runs
                   WHERE series_ticker=? AND symbol=? ORDER BY start_at""",
                (series_ticker.upper(), symbol.upper()),
            ).fetchall()
            usages = connection.execute(
                """SELECT event_id,COUNT(DISTINCT run_id) AS uses FROM holdout_usage
                   WHERE series_ticker=? GROUP BY event_id HAVING uses>1 ORDER BY event_id""",
                (series_ticker.upper(),),
            ).fetchall()
        return {
            "series": series_ticker.upper(),
            "symbol": symbol.upper(),
            "campaign_registrations": [
                {
                    "campaign_id": str(row["campaign_id"]),
                    "registered_at": str(row["registered_at"]),
                    "superseded_at": row["superseded_at"],
                    "configuration": json.loads(str(row["configuration_json"])),
                }
                for row in registrations
            ],
            "archive_windows": [dict(row) for row in rows],
            "reused_holdout_events": [dict(row) for row in usages],
            "note": "archive success is not point-in-time forecast eligibility or model approval",
        }

    def compare_runs(self, first: str, second: str) -> dict[str, Any]:
        with self._connect() as connection:
            results = []
            for run_id in (first, second):
                row = connection.execute(
                    "SELECT result_json FROM backtest_runs WHERE run_id=?", (run_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"backtest run not found: {run_id}")
                results.append(json.loads(row[0]))
        populations: list[dict[tuple[str, str, str], dict[str, Any]]] = []
        for result in results:
            population: dict[tuple[str, str, str], dict[str, Any]] = {}
            for fold in result["folds"]:
                for forecast in fold.get("forecasts", []):
                    key = (forecast["event_ticker"], forecast["ticker"], forecast["observed_at"])
                    if key in population:
                        raise ValueError("comparison requires one forecast per exact observation")
                    population[key] = forecast
            populations.append(population)
        shared = populations[0].keys() & populations[1].keys()
        paired: list[dict[str, Any]] = []
        for key in sorted(shared):
            left, right = populations[0][key], populations[1][key]
            if left.get("outcome_yes") is None or left.get("outcome_yes") != right.get(
                "outcome_yes"
            ):
                continue
            if left.get("input_population_id") != right.get("input_population_id"):
                continue
            outcome = float(left["outcome_yes"])
            paired.append(
                {
                    "event_id": key[0],
                    "ticker": key[1],
                    "observed_at": key[2],
                    "first_brier": (left["probability_yes"] - outcome) ** 2,
                    "second_brier": (right["probability_yes"] - outcome) ** 2,
                }
            )
        deltas: dict[str, list[float]] = {}
        for pair in paired:
            deltas.setdefault(pair["event_id"], []).append(
                pair["second_brier"] - pair["first_brier"]
            )
        return {
            "first_run": first,
            "second_run": second,
            "first_population": len(populations[0]),
            "second_population": len(populations[1]),
            "identical_observations": len(paired),
            "independent_events": len(deltas),
            "event_weighted_brier_delta_second_minus_first": (
                sum(sum(values) / len(values) for values in deltas.values()) / len(deltas)
                if deltas
                else None
            ),
            "comparable_full_population": bool(paired)
            and len(paired) == len(populations[0])
            and len(paired) == len(populations[1]),
            "evidence": "descriptive paired comparison, not independent promotion evidence",
        }


def _log_loss(probability: float, outcome: float) -> float:
    bounded = min(max(probability, 1e-12), 1 - 1e-12)
    return -(outcome * math.log(bounded) + (1 - outcome) * math.log1p(-bounded))
