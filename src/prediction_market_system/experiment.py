"""Phase 1 read-only comparison of structural, blended and executable-ask probabilities.

Nothing here fits, tunes, approves, or delivers. Synthetic fixtures and retrospective
local evidence can check implementation and describe recorded data; they never
establish forecast superiority on unseen events.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from prediction_market_system.calibration import UncertaintyCalibrationProfile
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSnapshot,
    Opportunity,
    TerminalRangeContract,
)
from prediction_market_system.engine import (
    CryptoThresholdEngine,
    EngineConfig,
    remaining_seconds_to_entry_horizon,
)
from prediction_market_system.evidence import EVIDENCE_FORWARD_SHADOW, canonical_json, content_id
from prediction_market_system.recipe import MODEL_VERSION
from prediction_market_system.research import ResearchContext, ResearchDataUnavailable
from prediction_market_system.venues.kalshi import KalshiMarket

REPORT_SCHEMA = "pms-phase1-comparison-v1"
# v1 was superseded before use because it did not pin the production feature recipe.
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"pms-phase1-v2"})
SOURCE_SYNTHETIC = "synthetic-fixture"
SOURCE_RETROSPECTIVE = "retrospective-local"
# Synthetic manifests must never carry the forward-evaluation class.
MANIFEST_KIND_BY_SOURCE = {
    SOURCE_SYNTHETIC: "synthetic-fixture",
    SOURCE_RETROSPECTIVE: EVIDENCE_FORWARD_SHADOW,
}
# Manual features are exempt from production pinning only with this explicit label, which
# is part of the feature recipe and therefore of the recipe identity.
SYNTHETIC_FEATURE_SELECTION = "synthetic-fixture"
# Values ResearchContext.to_crypto_snapshot writes; any other protocol value could never match.
PRODUCTION_FEATURE_CONSTANTS = {
    "semantics": "point-in-time-complete-windows-v2",
    "volatility_selection": "matching-interval-dvol-else-realized",
}
PRODUCTION_FEATURE_INTERVALS = (
    "spot_interval_seconds",
    "realized_interval_seconds",
    "realized_window_seconds",
)
ARMS = ("structural", "blend", "market_yes_ask", "recorded_midpoint_anchor")
METRICS = ("brier", "log_loss")
COMPARISONS = (
    ("structural", "market_yes_ask", "primary"),
    ("blend", "market_yes_ask", "primary"),
    ("blend", "structural", "primary"),
    ("structural", "recorded_midpoint_anchor", "secondary"),
    ("blend", "recorded_midpoint_anchor", "secondary"),
)
PROTOCOL_KEYS = (
    "protocol_version",
    "protocol_id",
    "frozen_at",
    "series",
    "symbol",
    "contract_family",
    "model_name",
    "model_version",
    "structural_weight",
    "expected_annual_return",
    "production_features",
    "market_baseline",
    "secondary_market_baseline",
    "maximum_input_age_seconds",
    "minimum_ask_size",
    "settlement_window_seconds",
    "minimum_seconds_to_entry_horizon",
    "report_as_of",
    "archive_start",
    "archive_end",
    "holdout_start",
    "holdout_end",
    "training_start",
    "training_end",
    "validation_start",
    "validation_end",
    "validation_folds",
    "holdout_folds",
    "minimum_calibration_events",
    "minimum_resolved_events",
    "minimum_dates",
    "minimum_validation_folds",
    "minimum_fold_resolved_events",
    "log_loss_epsilon",
    "calibration_bins",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence",
    "decision_costs",
    "cost_scenarios",
)
_TIMESTAMP_KEYS = (
    "frozen_at",
    "report_as_of",
    "archive_start",
    "archive_end",
    "holdout_start",
    "holdout_end",
    "training_start",
    "training_end",
    "validation_start",
    "validation_end",
)
_OBSERVATION_KEYS = frozenset(
    {
        "observation_id",
        "event_id",
        "market_id",
        "observed_at",
        "opportunity",
        "inputs",
        "run_id",
        "run_manifest",
        "run_manifest_sha256",
        "input_id",
        "research_context",
        "resolution",
    }
)
_INPUT_KEYS = frozenset({"market", "crypto", "contract", "engine_config", "recipe"})
# Recorded forecasts are compared with reproduced ones within a tolerance so that
# last-ulp libm differences across platforms cannot silently change eligibility.
_REPRODUCTION_TOLERANCE = 1e-9
_REPRODUCED_PROBABILITIES = (
    "structural_probability_yes",
    "probability_yes",
    "market_probability_yes",
    "lower_probability_yes",
    "upper_probability_yes",
)


def protocol_identity(protocol: dict[str, Any]) -> str:
    return "sha256:" + content_id({k: v for k, v in protocol.items() if k != "protocol_id"})


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    _settings(protocol)
    return protocol


@dataclass(frozen=True)
class _Fold:
    name: str
    kind: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class _Settings:
    protocol_id: str
    protocol_version: str
    series: str
    symbol: str
    model_name: str
    model_version: str
    structural_weight: float
    expected_annual_return: float
    production_features: dict[str, Any]
    settlement_window_seconds: int
    maximum_input_age_seconds: int
    minimum_ask_size: float
    minimum_seconds_to_entry_horizon: int
    report_as_of: datetime
    archive_start: datetime
    archive_end: datetime
    training_start: datetime
    training_end: datetime
    folds: tuple[_Fold, ...]
    log_loss_epsilon: float
    calibration_bins: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence: float
    minimum_resolved_events: int
    minimum_dates: int
    minimum_calibration_events: int
    minimum_validation_folds: int
    minimum_fold_resolved_events: int


def _protocol_int(value: Any, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"protocol {name} must be an integer >= {minimum}")
    return value


def _protocol_real(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"protocol {name} must be a finite number")
    return float(value)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _production_features(value: Any) -> dict[str, Any]:
    expected_keys = {*PRODUCTION_FEATURE_CONSTANTS, *PRODUCTION_FEATURE_INTERVALS}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(
            "protocol production_features must define exactly: " + ", ".join(sorted(expected_keys))
        )
    for key, expected in PRODUCTION_FEATURE_CONSTANTS.items():
        if value[key] != expected:
            raise ValueError(f"protocol production_features.{key} must be {expected!r}")
    for key in PRODUCTION_FEATURE_INTERVALS:
        _protocol_int(value[key], f"production_features.{key}", 1)
    if value["realized_window_seconds"] % value["realized_interval_seconds"]:
        raise ValueError("protocol realized window must contain whole realized intervals")
    return dict(value)


def _folds(
    protocol: dict[str, Any], key: str, kind: str, start: datetime, end: datetime
) -> list[_Fold]:
    value = protocol[key]
    if not isinstance(value, list):
        raise ValueError(f"protocol {key} must be a list")
    folds: list[_Fold] = []
    for fold in value:
        name = fold.get("name") if isinstance(fold, dict) else None
        fold_start = _parse_timestamp(fold.get("start")) if isinstance(fold, dict) else None
        fold_end = _parse_timestamp(fold.get("end")) if isinstance(fold, dict) else None
        if not isinstance(name, str) or not name or fold_start is None or fold_end is None:
            raise ValueError(f"protocol {key} entries need a name, start and end")
        if not start <= fold_start < fold_end <= end:
            raise ValueError(f"protocol {key} entries must lie within their window")
        folds.append(_Fold(name=name, kind=kind, start=fold_start, end=fold_end))
    return folds


def _settings(protocol: Any) -> _Settings:
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    missing = sorted(set(PROTOCOL_KEYS).difference(protocol))
    if missing:
        raise ValueError("protocol is missing required keys: " + ", ".join(missing))
    if protocol["protocol_version"] not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(f"unsupported protocol version: {protocol['protocol_version']!r}")
    if protocol["protocol_id"] != protocol_identity(protocol):
        raise ValueError("protocol_id does not match protocol content")
    times: dict[str, datetime] = {}
    for key in _TIMESTAMP_KEYS:
        parsed = _parse_timestamp(protocol[key])
        if parsed is None:
            raise ValueError(f"protocol {key} must be an ISO timestamp with a timezone")
        times[key] = parsed
    if not times["archive_start"] < times["archive_end"] <= times["report_as_of"]:
        raise ValueError("protocol archive window must end no later than the report cutoff")
    if not times["archive_end"] <= times["holdout_start"] < times["holdout_end"]:
        raise ValueError("protocol holdout window must follow the archive window")
    if times["frozen_at"] > times["holdout_start"]:
        raise ValueError("protocol must be frozen before the holdout window begins")
    if not (
        times["training_start"]
        < times["training_end"]
        <= times["validation_start"]
        < times["validation_end"]
    ):
        raise ValueError("protocol training window must precede its validation window")
    for key, expected in (
        ("contract_family", "terminal-range"),
        ("market_baseline", "yes_ask"),
        ("secondary_market_baseline", "recorded_midpoint_anchor"),
    ):
        if protocol[key] != expected:
            raise ValueError(f"protocol {key} must be {expected!r}")
    for key in ("series", "symbol"):
        if not isinstance(protocol[key], str) or not protocol[key]:
            raise ValueError(f"protocol {key} must be a non-empty string")
    window = _protocol_int(protocol["settlement_window_seconds"], "settlement_window_seconds", 1)
    engine_model = CryptoThresholdEngine.model_name(
        TerminalRangeContract(lower_bound=1.0, upper_bound=2.0, settlement_window_seconds=window)
    )
    if protocol["model_name"] != engine_model:
        raise ValueError(f"protocol model_name must be the engine model {engine_model!r}")
    if protocol["model_version"] != MODEL_VERSION:
        raise ValueError(f"protocol model_version must be {MODEL_VERSION!r}")
    weight = _protocol_real(protocol["structural_weight"], "structural_weight")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("protocol structural_weight must be within [0, 1]")
    epsilon = _protocol_real(protocol["log_loss_epsilon"], "log_loss_epsilon")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("protocol log_loss_epsilon must be within (0, 0.5)")
    confidence = _protocol_real(protocol["confidence"], "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("protocol confidence must be within (0, 1)")
    minimum_ask_size = _protocol_real(protocol["minimum_ask_size"], "minimum_ask_size")
    if minimum_ask_size < 0.0:
        raise ValueError("protocol minimum_ask_size must be non-negative")
    if not isinstance(protocol["decision_costs"], dict):
        raise ValueError("protocol decision_costs must be an object")
    if not isinstance(protocol["cost_scenarios"], list):
        raise ValueError("protocol cost_scenarios must be a list")
    seed = protocol["bootstrap_seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("protocol bootstrap_seed must be an integer")
    folds = _folds(
        protocol,
        "validation_folds",
        "validation",
        times["validation_start"],
        times["validation_end"],
    ) + _folds(protocol, "holdout_folds", "holdout", times["holdout_start"], times["holdout_end"])
    return _Settings(
        protocol_id=protocol["protocol_id"],
        protocol_version=protocol["protocol_version"],
        series=protocol["series"],
        symbol=protocol["symbol"],
        model_name=engine_model,
        model_version=MODEL_VERSION,
        structural_weight=weight,
        expected_annual_return=_protocol_real(
            protocol["expected_annual_return"], "expected_annual_return"
        ),
        production_features=_production_features(protocol["production_features"]),
        settlement_window_seconds=window,
        maximum_input_age_seconds=_protocol_int(
            protocol["maximum_input_age_seconds"], "maximum_input_age_seconds", 0
        ),
        minimum_ask_size=minimum_ask_size,
        minimum_seconds_to_entry_horizon=_protocol_int(
            protocol["minimum_seconds_to_entry_horizon"], "minimum_seconds_to_entry_horizon", 0
        ),
        report_as_of=times["report_as_of"],
        archive_start=times["archive_start"],
        archive_end=times["archive_end"],
        training_start=times["training_start"],
        training_end=times["training_end"],
        folds=tuple(folds),
        log_loss_epsilon=epsilon,
        calibration_bins=_protocol_int(protocol["calibration_bins"], "calibration_bins", 1),
        bootstrap_replicates=_protocol_int(
            protocol["bootstrap_replicates"], "bootstrap_replicates", 1
        ),
        bootstrap_seed=seed,
        confidence=confidence,
        minimum_resolved_events=_protocol_int(
            protocol["minimum_resolved_events"], "minimum_resolved_events", 0
        ),
        minimum_dates=_protocol_int(protocol["minimum_dates"], "minimum_dates", 0),
        minimum_calibration_events=_protocol_int(
            protocol["minimum_calibration_events"], "minimum_calibration_events", 0
        ),
        minimum_validation_folds=_protocol_int(
            protocol["minimum_validation_folds"], "minimum_validation_folds", 0
        ),
        minimum_fold_resolved_events=_protocol_int(
            protocol["minimum_fold_resolved_events"], "minimum_fold_resolved_events", 1
        ),
    )


class _Excluded(Exception):
    """``code`` is the frozen protocol's reason name; ``reason`` is the readable detail."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass(frozen=True)
class _Row:
    observation_id: str
    event_id: str
    market_id: str
    series_id: str
    observed_at: datetime
    observation_end_at: datetime
    expires_at: datetime
    regime: str
    state: str
    run_id: str
    input_id: str
    run_manifest_sha256: str
    recipe_id: str
    research_context_id: str | None
    feature_source: str
    selected_volatility_kind: str | None
    selected_volatility_provider: str | None
    selected_volatility_window_seconds: int | None
    uncertainty_source: str
    calibration_profile_id: str | None
    probabilities: dict[str, float]
    yes_ask_size: float
    outcome: int | None
    unresolved_reason: str | None
    resolution_observed_at: str | None
    settlement_ts: str | None
    execution: list[Any]

    @property
    def event_date(self) -> str:
        return self.observation_end_at.date().isoformat()

    @property
    def branch(self) -> tuple[str, str, str, int, str]:
        return (
            self.selected_volatility_kind or self.feature_source,
            self.selected_volatility_provider or "",
            self.feature_source,
            self.selected_volatility_window_seconds or 0,
            self.recipe_id,
        )


def _text(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    return value if isinstance(value, str) else None


def _excluded_entry(observation: Any, code: str, reason: str) -> dict[str, Any]:
    source = observation if isinstance(observation, dict) else {}
    return {
        "observation_id": _text(source, "observation_id"),
        "event_id": _text(source, "event_id"),
        "market_id": _text(source, "market_id"),
        "observed_at": _text(source, "observed_at"),
        "reason_code": code,
        "reason": reason,
    }


def _validated(model: Any, payload: Any, code: str, reason: str) -> Any:
    try:
        return model.model_validate(payload)
    except (ValidationError, ResearchDataUnavailable) as error:
        raise _Excluded(code, reason) from error


def _check_source_rules(
    market: MarketSnapshot,
    contract: TerminalRangeContract,
    observed_at: datetime,
    source_kind: str,
) -> None:
    """Re-derive contract semantics from the recorded venue market, never from rule keywords."""
    raw = market.source_metadata
    if source_kind == SOURCE_SYNTHETIC:
        if raw.get("synthetic") is not True:
            raise _Excluded(
                "unsupported_contract", "synthetic row lacks an explicit synthetic declaration"
            )
        return
    if market.venue.casefold() != "kalshi" or not raw:
        raise _Excluded("unsupported_contract", "missing raw Kalshi market semantics")
    try:
        venue_market = KalshiMarket.model_validate(raw)
        venue_contract = venue_market.evaluation_contract(observed_at)
        venue_observation_end = venue_market.observation_end_at
    except (ValidationError, ValueError) as error:
        raise _Excluded(
            "unsupported_contract", f"recorded Kalshi market is unsupported: {error}"
        ) from error
    if (
        venue_market.ticker != market.market_id
        or venue_market.event_ticker != market.event_id
        or venue_market.normalized_series_ticker != market.series_id
        or venue_market.close_time != market.expires_at
    ):
        raise _Excluded("identity_mismatch", "recorded Kalshi market identity does not match")
    if venue_contract != contract:
        raise _Excluded("identity_mismatch", "recorded Kalshi contract does not match inputs")
    if venue_observation_end != market.observation_end_at:
        raise _Excluded("identity_mismatch", "recorded Kalshi benchmark end does not match")
    if venue_market.resolution_rule != market.resolution_rule:
        raise _Excluded("identity_mismatch", "recorded Kalshi rule text does not match")


def _check_production_features(
    inputs: dict[str, Any], crypto: CryptoSnapshot, context: ResearchContext, settings: _Settings
) -> None:
    code = "feature_recipe_mismatch"
    recipe = inputs["recipe"]
    expected_return = settings.expected_annual_return
    if (
        not isinstance(recipe, dict)
        or recipe.get("expected_annual_return") != expected_return
        or crypto.expected_annual_return != expected_return
    ):
        raise _Excluded(code, "expected annual return is not the protocol value")
    features = crypto.feature_recipe
    if recipe.get("features") != features:
        raise _Excluded(code, "recorded recipe features do not match the crypto input")
    for key, value in sorted(settings.production_features.items()):
        if features.get(key) != value:
            raise _Excluded(code, f"feature recipe {key} is not the protocol value")
    pinned = settings.production_features
    realized = context.realized_volatility
    if (
        context.spot.interval_seconds != pinned["spot_interval_seconds"]
        or realized.raw_payload.get("interval_seconds") != pinned["realized_interval_seconds"]
        or realized.window_seconds != pinned["realized_window_seconds"]
    ):
        raise _Excluded(code, "research context windows are not the protocol windows")
    selected = context.implied_volatility or realized
    if (
        features.get("selected_volatility_kind") != selected.kind
        or features.get("selected_volatility_provider") != selected.provider
        or features.get("selected_volatility_window_seconds") != selected.window_seconds
    ):
        raise _Excluded(code, "selected volatility does not match the research context")


def _check_calibration_profile(
    profile: UncertaintyCalibrationProfile,
    crypto: CryptoSnapshot,
    recipe_id: str,
    observed_at: datetime,
    settings: _Settings,
) -> None:
    code = "calibration_profile_invalid"
    if (
        profile.symbol != crypto.symbol.upper()
        or profile.model_name != settings.model_name
        or profile.model_version != settings.model_version
        or profile.recipe_id != recipe_id
    ):
        raise _Excluded(code, "calibration profile does not match the forecast recipe")
    if profile.generated_at > observed_at:
        raise _Excluded(code, "calibration profile was generated after the forecast")
    if profile.training_start < settings.training_start or profile.cutoff_at > min(
        settings.training_end, observed_at
    ):
        raise _Excluded(code, "calibration profile is outside the protocol training window")
    if profile.independent_event_count < settings.minimum_calibration_events:
        raise _Excluded(code, "calibration profile has too few independent events")


def _evaluate_row(observation: Any, settings: _Settings, source_kind: str, as_of: datetime) -> _Row:
    if not isinstance(observation, dict):
        raise _Excluded("malformed", "observation is not an object")
    if observation.get("exclusion_reason") is not None:
        raise _Excluded("source_exclusion", str(observation["exclusion_reason"]))
    missing = sorted(_OBSERVATION_KEYS.difference(observation))
    if missing:
        raise _Excluded("malformed", "missing observation fields: " + ", ".join(missing))
    for key in ("observation_id", "event_id", "market_id", "run_id", "input_id"):
        if not isinstance(observation[key], str) or not observation[key]:
            code = "missing_event_id" if key == "event_id" else "malformed"
            raise _Excluded(code, f"invalid {key}")
    observed_at = _parse_timestamp(observation["observed_at"])
    if observed_at is None:
        raise _Excluded("invalid_timestamp", "invalid observed_at")
    if observed_at > as_of:
        raise _Excluded("outside_window", "observed after report cutoff")
    if not settings.archive_start <= observed_at < settings.archive_end:
        raise _Excluded("outside_window", "observed outside archive window")

    inputs = observation["inputs"]
    if not isinstance(inputs, dict) or not _INPUT_KEYS.issubset(inputs):
        raise _Excluded("malformed", "missing forecast inputs")
    if content_id(inputs) != observation["input_id"]:
        raise _Excluded("identity_mismatch", "input hash mismatch")
    manifest = observation["run_manifest"]
    if not isinstance(manifest, dict) or content_id(manifest) != observation["run_manifest_sha256"]:
        raise _Excluded("identity_mismatch", "run manifest hash mismatch")
    if manifest.get("run_id") != observation["run_id"]:
        raise _Excluded("identity_mismatch", "run manifest does not identify this run")
    recorded_at = _parse_timestamp(manifest.get("recorded_at"))
    if recorded_at is None:
        raise _Excluded("invalid_timestamp", "invalid run manifest recorded_at")
    if recorded_at != observed_at:
        raise _Excluded("forecast_time_mismatch", "run manifest was not recorded at forecast time")
    manifest_inputs = manifest.get("inputs")
    if (
        not isinstance(manifest_inputs, dict)
        or manifest_inputs.get("input_id") != observation["input_id"]
    ):
        raise _Excluded("identity_mismatch", "run manifest does not reference these inputs")
    expected_kind = MANIFEST_KIND_BY_SOURCE[source_kind]
    if manifest.get("kind") != expected_kind:
        raise _Excluded("evidence_class", f"run manifest kind is not {expected_kind}")

    opportunity: Opportunity = _validated(
        Opportunity, observation["opportunity"], "malformed", "invalid opportunity payload"
    )
    market: MarketSnapshot = _validated(
        MarketSnapshot, inputs["market"], "invalid_quote", "invalid market input"
    )
    crypto: CryptoSnapshot = _validated(
        CryptoSnapshot, inputs["crypto"], "malformed", "invalid crypto input"
    )
    config: EngineConfig = _validated(
        EngineConfig, inputs["engine_config"], "malformed", "invalid engine configuration input"
    )
    contract_payload = inputs["contract"]
    if not isinstance(contract_payload, dict) or "lower_bound" not in contract_payload:
        raise _Excluded(
            "unsupported_contract", "unsupported contract structure: not a terminal range"
        )
    contract: TerminalRangeContract = _validated(
        TerminalRangeContract, contract_payload, "unsupported_contract", "invalid contract input"
    )
    if contract.settlement_window_seconds != settings.settlement_window_seconds:
        raise _Excluded(
            "unsupported_contract", "unsupported contract structure: settlement averaging window"
        )
    profile_payload = inputs.get("calibration_profile")
    profile: UncertaintyCalibrationProfile | None = (
        None
        if profile_payload is None
        else _validated(
            UncertaintyCalibrationProfile,
            profile_payload,
            "calibration_profile_invalid",
            "invalid calibration profile input",
        )
    )
    forecast = opportunity.forecast

    if opportunity.market.model_dump(mode="json") != inputs["market"]:
        raise _Excluded("identity_mismatch", "opportunity market does not match forecast inputs")
    if market.market_id != observation["market_id"] or forecast.market_id != market.market_id:
        raise _Excluded("identity_mismatch", "market id does not match forecast")
    if market.observed_at != observed_at or forecast.generated_at != observed_at:
        raise _Excluded("forecast_time_mismatch", "observation time does not match forecast")
    event_id = market.event_id
    if not event_id:
        raise _Excluded("missing_event_id", "missing explicit event id")
    if event_id == market.market_id:
        raise _Excluded("missing_event_id", "event id is a contract-id fallback")
    if event_id != observation["event_id"]:
        raise _Excluded("missing_event_id", "event id does not match market")
    series_id = market.series_id
    if series_id is None or series_id.upper() != settings.series.upper():
        raise _Excluded("out_of_scope", "series is not the protocol series")
    if crypto.symbol.upper() != settings.symbol.upper():
        raise _Excluded("out_of_scope", "symbol is not the protocol symbol")
    _check_source_rules(market, contract, observed_at, source_kind)

    if (
        forecast.model_name != settings.model_name
        or forecast.model_version != settings.model_version
    ):
        raise _Excluded("model_mismatch", "model identity is not the protocol model")
    recorded_recipe = inputs["recipe"]
    if (
        config.structural_weight != settings.structural_weight
        or not isinstance(recorded_recipe, dict)
        or recorded_recipe.get("structural_weight") != settings.structural_weight
    ):
        raise _Excluded("model_mismatch", "structural weight is not the protocol weight")
    engine = CryptoThresholdEngine(config)
    recipe_id = forecast.recipe_id
    if recipe_id is None or recipe_id != engine.recipe_id(crypto):
        raise _Excluded("model_mismatch", "recipe does not match inputs")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("recipe_id") != recipe_id:
        raise _Excluded("identity_mismatch", "run manifest recipe does not match forecast")

    observation_end_at = market.observation_end_at
    if observation_end_at is None:
        raise _Excluded("missing_observation_provenance", "missing explicit observation end")
    window_start = observation_end_at - timedelta(seconds=contract.settlement_window_seconds)
    if market.observation_start_at not in (None, window_start):
        raise _Excluded(
            "missing_observation_provenance", "observation start does not match averaging window"
        )
    if observed_at > window_start:
        raise _Excluded("timing_ineligible", "averaging window already started")
    if remaining_seconds_to_entry_horizon(market, observed_at) < (
        settings.minimum_seconds_to_entry_horizon
    ):
        raise _Excluded("timing_ineligible", "inside minimum time to expiry")
    input_age = (observed_at - crypto.observed_at).total_seconds()
    if input_age < 0:
        raise _Excluded("future_input", "crypto input is from the future")
    if input_age > settings.maximum_input_age_seconds:
        raise _Excluded("stale_input", "crypto input is stale")

    context_id = crypto.input_provenance.get("research_context_id")
    context_payload = observation["research_context"]
    selected_kind: str | None = None
    selected_provider: str | None = None
    selected_window: int | None = None
    if context_id is None:
        if source_kind != SOURCE_SYNTHETIC:
            raise _Excluded(
                "research_context_unavailable",
                "manual features are allowed only for synthetic sources",
            )
        if crypto.feature_recipe.get("source_selection") != SYNTHETIC_FEATURE_SELECTION:
            raise _Excluded(
                "feature_recipe_mismatch", "manual features are not explicitly labelled synthetic"
            )
        if context_payload is not None:
            raise _Excluded(
                "identity_mismatch", "research context is not referenced by forecast inputs"
            )
        feature_source = "manual-synthetic"
    else:
        if not isinstance(context_payload, dict):
            raise _Excluded("research_context_unavailable", "research context is not available")
        try:
            stored_context_id = content_id(context_payload)
        except (TypeError, ValueError) as error:
            raise _Excluded("research_context_invalid", "invalid research context") from error
        if stored_context_id != context_id:
            raise _Excluded("research_context_invalid", "research context identity mismatch")
        context: ResearchContext = _validated(
            ResearchContext, context_payload, "research_context_invalid", "invalid research context"
        )
        # Validation may omit an inadmissible optional input; that must not pass silently.
        if context.content_id() != context_id:
            raise _Excluded(
                "research_context_invalid", "research context changed during validation"
            )
        try:
            context.validate_at(
                observed_at, maximum_spot_age_seconds=settings.maximum_input_age_seconds
            )
            snapshot = context.to_crypto_snapshot(
                strike_price=crypto.strike_price,
                expected_annual_return=crypto.expected_annual_return,
            )
        except (ResearchDataUnavailable, ValueError) as error:
            raise _Excluded(
                "research_context_invalid", f"research context inadmissible: {error}"
            ) from error
        if snapshot != crypto:
            raise _Excluded(
                "research_context_invalid", "crypto input does not match research context"
            )
        _check_production_features(inputs, crypto, context, settings)
        selected = context.implied_volatility or context.realized_volatility
        selected_kind = selected.kind
        selected_provider = selected.provider
        selected_window = selected.window_seconds
        feature_source = "research-context"

    if profile is not None:
        _check_calibration_profile(profile, crypto, recipe_id, observed_at, settings)
    try:
        reproduced, _ = engine.evaluate(market, crypto, contract, profile)
    except (ResearchDataUnavailable, ValueError) as error:
        raise _Excluded(
            "forecast_not_reproducible", f"forecast does not reproduce: {error}"
        ) from error
    if content_id(reproduced.input_manifest) != observation["input_id"]:
        raise _Excluded("forecast_not_reproducible", "forecast inputs do not reproduce")
    if (
        reproduced.model_name != forecast.model_name
        or reproduced.recipe_id != recipe_id
        or reproduced.uncertainty_source != forecast.uncertainty_source
        or reproduced.calibration_profile_id != forecast.calibration_profile_id
        or any(
            not math.isclose(
                getattr(reproduced, name),
                getattr(forecast, name),
                rel_tol=0.0,
                abs_tol=_REPRODUCTION_TOLERANCE,
            )
            for name in _REPRODUCED_PROBABILITIES
        )
    ):
        raise _Excluded("forecast_not_reproducible", "forecast does not reproduce from inputs")

    if market.yes_ask is None:
        raise _Excluded("no_displayed_ask", "missing displayed YES ask")
    if market.yes_ask_size is None or market.yes_ask_size < settings.minimum_ask_size:
        raise _Excluded("no_displayed_ask", "displayed YES ask size below protocol minimum")
    probabilities = {
        "structural": forecast.structural_probability_yes,
        "blend": forecast.probability_yes,
        "market_yes_ask": market.yes_ask,
        "recorded_midpoint_anchor": forecast.market_probability_yes,
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise _Excluded("invalid_arm", "invalid probability arm")

    execution = observation.get("execution", [])
    if not isinstance(execution, list):
        raise _Excluded("malformed", "invalid execution snapshots")
    outcome, unresolved_reason, resolved_at, settlement_ts = _resolution(
        observation["resolution"],
        observed_at,
        as_of,
        observation.get("resolution_unavailable_reason"),
    )
    return _Row(
        observation_id=observation["observation_id"],
        event_id=event_id,
        market_id=market.market_id,
        series_id=series_id,
        observed_at=observed_at,
        observation_end_at=observation_end_at,
        expires_at=market.expires_at,
        regime=opportunity.market_regime.label if opportunity.market_regime else "unknown",
        state=opportunity.state.value,
        run_id=observation["run_id"],
        input_id=observation["input_id"],
        run_manifest_sha256=observation["run_manifest_sha256"],
        recipe_id=recipe_id,
        research_context_id=str(context_id) if context_id is not None else None,
        feature_source=feature_source,
        selected_volatility_kind=selected_kind,
        selected_volatility_provider=selected_provider,
        selected_volatility_window_seconds=selected_window,
        uncertainty_source=forecast.uncertainty_source,
        calibration_profile_id=(
            str(forecast.calibration_profile_id) if forecast.calibration_profile_id else None
        ),
        probabilities=probabilities,
        yes_ask_size=market.yes_ask_size,
        outcome=outcome,
        unresolved_reason=unresolved_reason,
        resolution_observed_at=resolved_at,
        settlement_ts=settlement_ts,
        execution=execution,
    )


def _resolution(
    resolution: Any, observed_at: datetime, as_of: datetime, unavailable_reason: Any
) -> tuple[int | None, str | None, str | None, str | None]:
    """Return only as-of outcomes, preserving an adapter's withheld-label reason."""
    if unavailable_reason not in (None, "outcome-after-cutoff"):
        raise _Excluded("malformed", "invalid resolution_unavailable_reason")
    if unavailable_reason is not None and resolution is not None:
        raise _Excluded("malformed", "resolution_unavailable_reason contradicts resolution")
    if resolution is None:
        return None, unavailable_reason or "missing-resolution", None, None
    if not isinstance(resolution, dict) or resolution.get("result") not in ("yes", "no"):
        raise _Excluded("malformed", "invalid resolution")
    resolved_at = _parse_timestamp(resolution.get("observed_at"))
    raw_settlement = resolution.get("settlement_ts")
    settlement = None if raw_settlement is None else _parse_timestamp(raw_settlement)
    if resolved_at is None or (raw_settlement is not None and settlement is None):
        raise _Excluded("invalid_timestamp", "invalid resolution timestamp")
    if resolved_at <= observed_at or (settlement is not None and settlement <= observed_at):
        raise _Excluded("outcome_leak", "resolution known before forecast")
    if resolved_at > as_of or (settlement is not None and settlement > as_of):
        return None, "outcome-after-cutoff", None, None
    return (
        int(resolution["result"] == "yes"),
        None,
        resolved_at.isoformat(),
        settlement.isoformat() if settlement is not None else None,
    )


def _row_scores(row: _Row, epsilon: float) -> dict[tuple[str, str], float]:
    outcome = row.outcome
    assert outcome is not None
    scores: dict[tuple[str, str], float] = {}
    for arm in ARMS:
        probability = row.probabilities[arm]
        bounded = min(max(probability, epsilon), 1.0 - epsilon)
        scores[(arm, "brier")] = (probability - outcome) ** 2
        scores[(arm, "log_loss")] = -math.log(bounded) if outcome else -math.log1p(-bounded)
    return scores


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cluster_bootstrap(
    sums: list[list[float]], sizes: list[int], settings: _Settings, method: str
) -> list[dict[str, Any] | None]:
    """Resample clusters; each statistic is summed event deltas over sampled event count.

    One draw sequence is shared by every statistic, so paired arms see identical resamples.
    """
    clusters = len(sizes)
    if clusters < 2:
        return [None for _ in sums]
    rng = random.Random(settings.bootstrap_seed)
    replicates: list[list[float]] = [[] for _ in sums]
    for _ in range(settings.bootstrap_replicates):
        counts = [0] * clusters
        for _ in range(clusters):
            counts[rng.randrange(clusters)] += 1
        drawn = [(index, count) for index, count in enumerate(counts) if count]
        events = sum(sizes[index] * count for index, count in drawn)
        for values, statistic in zip(replicates, sums, strict=True):
            values.append(math.fsum(statistic[index] * count for index, count in drawn) / events)
    alpha = (1.0 - settings.confidence) / 2.0
    intervals: list[dict[str, Any] | None] = []
    for values in replicates:
        ordered = sorted(values)
        intervals.append(
            {
                "method": method,
                "clusters": clusters,
                "replicates": len(values),
                "seed": settings.bootstrap_seed,
                "confidence": settings.confidence,
                "lower": _quantile(ordered, alpha),
                "upper": _quantile(ordered, 1.0 - alpha),
                "standard_error": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return intervals


def _event_means(
    rows: list[_Row], scores: dict[str, dict[tuple[str, str], float]]
) -> dict[str, dict[tuple[str, str], float]]:
    grouped: dict[str, list[_Row]] = {}
    for row in rows:
        grouped.setdefault(row.event_id, []).append(row)
    return {
        event_id: {
            (arm, metric): math.fsum(scores[row.observation_id][(arm, metric)] for row in members)
            / len(members)
            for arm in ARMS
            for metric in METRICS
        }
        for event_id, members in sorted(grouped.items())
    }


def _arm_metrics(
    event_means: dict[str, dict[tuple[str, str], float]],
) -> dict[str, dict[str, float | None]]:
    return {
        arm: {
            metric: _mean([means[(arm, metric)] for means in event_means.values()])
            for metric in METRICS
        }
        for arm in ARMS
    }


def _observation_payload(
    row: _Row, weight: float | None, scores: dict[tuple[str, str], float] | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation_id": row.observation_id,
        "event_id": row.event_id,
        "market_id": row.market_id,
        "series_id": row.series_id,
        "observed_at": row.observed_at.isoformat(),
        "event_date": row.event_date,
        "observation_end_at": row.observation_end_at.isoformat(),
        "expires_at": row.expires_at.isoformat(),
        "regime": row.regime,
        "state": row.state,
        "run_id": row.run_id,
        "input_id": row.input_id,
        "run_manifest_sha256": row.run_manifest_sha256,
        "recipe_id": row.recipe_id,
        "research_context_id": row.research_context_id,
        "feature_source": row.feature_source,
        "uncertainty_source": row.uncertainty_source,
        "calibration_profile_id": row.calibration_profile_id,
        "structural_probability_yes": row.probabilities["structural"],
        "blend_probability_yes": row.probabilities["blend"],
        "market_yes_ask": row.probabilities["market_yes_ask"],
        "market_yes_ask_size": row.yes_ask_size,
        "recorded_midpoint_anchor_probability_yes": row.probabilities["recorded_midpoint_anchor"],
        "resolution_status": "resolved" if row.outcome is not None else "unresolved",
        "unresolved_reason": row.unresolved_reason,
        "outcome_yes": row.outcome,
        "resolution_observed_at": row.resolution_observed_at,
        "settlement_ts": row.settlement_ts,
        "event_weight": weight,
    }
    for arm in ARMS:
        for metric in METRICS:
            payload[f"{arm}_{metric}"] = scores[(arm, metric)] if scores is not None else None
    payload["execution"] = row.execution
    return payload


def _calibration(
    resolved: list[_Row], weights: dict[str, float], bins: int
) -> dict[str, list[dict[str, Any]]]:
    table: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        members: list[list[_Row]] = [[] for _ in range(bins)]
        for row in resolved:
            members[min(int(row.probabilities[arm] * bins), bins - 1)].append(row)
        entries: list[dict[str, Any]] = []
        for index, group in enumerate(members):
            weight = math.fsum(weights[row.observation_id] for row in group)
            mean_probability: float | None = None
            mean_outcome: float | None = None
            if group:
                mean_probability = (
                    math.fsum(weights[row.observation_id] * row.probabilities[arm] for row in group)
                    / weight
                )
                mean_outcome = (
                    math.fsum(weights[row.observation_id] * (row.outcome or 0) for row in group)
                    / weight
                )
            entries.append(
                {
                    "bin": index,
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "forecasts": len(group),
                    "events": len({row.event_id for row in group}),
                    "weight": weight,
                    "mean_probability": mean_probability,
                    "mean_outcome": mean_outcome,
                }
            )
        table[arm] = entries
    return table


def _source_selection_branches(
    eligible: list[_Row], scores: dict[str, dict[tuple[str, str], float]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[_Row]] = {}
    for row in eligible:
        grouped.setdefault(row.branch, []).append(row)
    branches: list[dict[str, Any]] = []
    for (label, _, feature_source, _, recipe_id), members in sorted(grouped.items()):
        first = members[0]
        resolved = [row for row in members if row.outcome is not None]
        means = _event_means(resolved, scores)
        branches.append(
            {
                "branch": label,
                "feature_source": feature_source,
                "selected_volatility_kind": first.selected_volatility_kind,
                "selected_volatility_provider": first.selected_volatility_provider,
                "selected_volatility_window_seconds": first.selected_volatility_window_seconds,
                "recipe_id": recipe_id,
                "eligible_forecasts": len(members),
                "resolved_forecasts": len(resolved),
                "eligible_events": len({row.event_id for row in members}),
                "resolved_events": len(means),
                "resolved_dates": len({row.event_date for row in resolved}),
                "arms": _arm_metrics(means),
                "note": (
                    "predeclared source-selection branch; pooled only under the fixed "
                    "selection rule; descriptive only, no branch-level uncertainty"
                ),
            }
        )
    return branches


def _fold_coverage(
    eligible: list[_Row], resolved_events: list[str], settings: _Settings
) -> list[dict[str, Any]]:
    first_decision: dict[str, datetime] = {}
    for row in eligible:
        current = first_decision.get(row.event_id)
        if current is None or row.observed_at < current:
            first_decision[row.event_id] = row.observed_at
    coverage: list[dict[str, Any]] = []
    for fold in settings.folds:
        count = sum(
            fold.start <= first_decision[event_id] < fold.end for event_id in resolved_events
        )
        coverage.append(
            {
                "name": fold.name,
                "kind": fold.kind,
                "start": fold.start.isoformat(),
                "end": fold.end.isoformat(),
                "resolved_events": count,
                "qualifies": count >= settings.minimum_fold_resolved_events,
            }
        )
    return coverage


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sort_key(observation: Any) -> tuple[str, str]:
    if not isinstance(observation, dict):
        return "", canonical_json(observation)
    identifier = observation.get("observation_id")
    return (identifier if isinstance(identifier, str) else ""), ""


def _observation_identity(observation: dict[str, Any]) -> str:
    try:
        return content_id(observation)
    except (TypeError, ValueError):
        return f"unserializable:{id(observation)}"


def _duplicate_exclusions(ordered: list[Any]) -> dict[int, tuple[str, str]]:
    """Map positions in ``ordered`` to duplicate exclusions; independent of input order."""
    positions: dict[str, list[int]] = {}
    for index, observation in enumerate(ordered):
        if isinstance(observation, dict) and isinstance(observation.get("observation_id"), str):
            positions.setdefault(observation["observation_id"], []).append(index)
    excluded: dict[int, tuple[str, str]] = {}
    for indexes in positions.values():
        if len(indexes) < 2:
            continue
        identities = {_observation_identity(ordered[index]) for index in indexes}
        if len(identities) == 1:
            for index in indexes[1:]:
                excluded[index] = ("duplicate_observation", "identical duplicate observation_id")
        else:
            for index in indexes:
                excluded[index] = ("duplicate_conflict", "conflicting duplicate observation_id")
    return excluded


def compare_observations(
    observations: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    source_kind: str,
    as_of: datetime,
) -> dict[str, Any]:
    if source_kind not in MANIFEST_KIND_BY_SOURCE:
        raise ValueError(f"unsupported source kind: {source_kind!r}")
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("report timestamp must include a timezone")
    as_of = as_of.astimezone(UTC)
    settings = _settings(protocol)
    if as_of != settings.report_as_of:
        raise ValueError("report timestamp must match protocol report_as_of")

    ordered = sorted(observations, key=_sort_key)
    duplicates = _duplicate_exclusions(ordered)
    excluded: list[dict[str, Any]] = []
    candidates: list[_Row] = []
    for index, observation in enumerate(ordered):
        if index in duplicates:
            excluded.append(_excluded_entry(observation, *duplicates[index]))
            continue
        try:
            candidates.append(_evaluate_row(observation, settings, source_kind, as_of))
        except _Excluded as exclusion:
            excluded.append(_excluded_entry(observation, exclusion.code, exclusion.reason))

    event_dates: dict[str, set[str]] = {}
    for row in candidates:
        event_dates.setdefault(row.event_id, set()).add(row.event_date)
    eligible: list[_Row] = []
    for row in candidates:
        if len(event_dates[row.event_id]) > 1:
            excluded.append(
                {
                    "observation_id": row.observation_id,
                    "event_id": row.event_id,
                    "market_id": row.market_id,
                    "observed_at": row.observed_at.isoformat(),
                    "reason_code": "event_date_conflict",
                    "reason": "event observation-end date conflict",
                }
            )
        else:
            eligible.append(row)
    excluded.sort(
        key=lambda entry: (entry["observation_id"] or "", entry["reason"], canonical_json(entry))
    )

    resolved = [row for row in eligible if row.outcome is not None]
    scores = {row.observation_id: _row_scores(row, settings.log_loss_epsilon) for row in resolved}
    resolved_per_event = Counter(row.event_id for row in resolved)
    weights = {row.observation_id: 1.0 / resolved_per_event[row.event_id] for row in resolved}
    event_means = _event_means(resolved, scores)
    event_date = {row.event_id: row.event_date for row in eligible}
    resolved_events = list(event_means)
    resolved_dates = sorted({event_date[event_id] for event_id in resolved_events})

    statistic_keys = [
        (minuend, subtrahend, role, metric)
        for minuend, subtrahend, role in COMPARISONS
        for metric in METRICS
    ]
    deltas = [
        [
            event_means[event_id][(minuend, metric)] - event_means[event_id][(subtrahend, metric)]
            for event_id in resolved_events
        ]
        for minuend, subtrahend, _, metric in statistic_keys
    ]
    event_intervals = _cluster_bootstrap(
        deltas, [1] * len(resolved_events), settings, "event-cluster-percentile"
    )
    date_index = {value: index for index, value in enumerate(resolved_dates)}
    date_sizes = [0] * len(resolved_dates)
    for event_id in resolved_events:
        date_sizes[date_index[event_date[event_id]]] += 1
    date_sums: list[list[float]] = []
    for values in deltas:
        per_date: list[list[float]] = [[] for _ in resolved_dates]
        for event_id, value in zip(resolved_events, values, strict=True):
            per_date[date_index[event_date[event_id]]].append(value)
        date_sums.append([math.fsum(group) for group in per_date])
    date_intervals = _cluster_bootstrap(
        date_sums, date_sizes, settings, "utc-observation-end-date-block-percentile"
    )
    paired_differences = [
        {
            "comparison": f"{minuend}_minus_{subtrahend}",
            "minuend": minuend,
            "subtrahend": subtrahend,
            "baseline_role": role,
            "metric": metric,
            "point_estimate": _mean(values),
            "resolved_forecasts": len(resolved),
            "events": len(resolved_events),
            "dates": len(resolved_dates),
            "event_bootstrap": event_interval,
            "date_block_bootstrap": date_interval,
        }
        for (minuend, subtrahend, role, metric), values, event_interval, date_interval in zip(
            statistic_keys, deltas, event_intervals, date_intervals, strict=True
        )
    ]

    rows_by_event: dict[str, list[_Row]] = {}
    for row in eligible:
        rows_by_event.setdefault(row.event_id, []).append(row)
    events: list[dict[str, Any]] = []
    for event_id, members in sorted(rows_by_event.items()):
        entry: dict[str, Any] = {
            "event_id": event_id,
            "event_date": event_date[event_id],
            "eligible_forecasts": len(members),
            "resolved_forecasts": resolved_per_event[event_id],
            "regimes": sorted({row.regime for row in members}),
        }
        means = event_means.get(event_id)
        for arm in ARMS:
            for metric in METRICS:
                entry[f"{arm}_{metric}"] = means[(arm, metric)] if means else None
        events.append(entry)

    regimes: list[dict[str, Any]] = []
    for regime in sorted({row.regime for row in eligible}):
        members = [row for row in eligible if row.regime == regime]
        regime_means = _event_means([row for row in members if row.outcome is not None], scores)
        regimes.append(
            {
                "regime": regime,
                "eligible_forecasts": len(members),
                "resolved_forecasts": sum(row.outcome is not None for row in members),
                "resolved_events": len(regime_means),
                "arms": _arm_metrics(regime_means),
                "note": "descriptive only; no regime-level uncertainty is reported",
            }
        )

    branches = _source_selection_branches(eligible, scores)
    folds = _fold_coverage(eligible, resolved_events, settings)
    observation_rows = [
        _observation_payload(row, weights.get(row.observation_id), scores.get(row.observation_id))
        for row in eligible
    ]
    unresolved_reasons = Counter(row.unresolved_reason for row in eligible if row.outcome is None)
    counts: dict[str, Any] = {
        "input_observations": len(observations),
        "eligible": len(eligible),
        "excluded": len(excluded),
        "resolved": len(resolved),
        "unresolved": len(eligible) - len(resolved),
        "unresolved_missing_resolution": unresolved_reasons["missing-resolution"],
        "unresolved_outcome_after_cutoff": unresolved_reasons["outcome-after-cutoff"],
        "watch": sum(row.state == "WATCH" for row in eligible),
        "entry": sum(row.state in {"ENTER YES", "ENTER NO"} for row in eligible),
        "eligible_events": len(rows_by_event),
        "resolved_events": len(resolved_events),
        "eligible_dates": len(set(event_date.values())),
        "resolved_dates": len(resolved_dates),
        "source_selection_branches": len(branches),
        "qualifying_validation_folds": sum(
            fold["qualifies"] for fold in folds if fold["kind"] == "validation"
        ),
        "qualifying_holdout_folds": sum(
            fold["qualifies"] for fold in folds if fold["kind"] == "holdout"
        ),
        "excluded_by_reason": dict(sorted(Counter(entry["reason"] for entry in excluded).items())),
        "excluded_by_reason_code": dict(
            sorted(Counter(entry["reason_code"] for entry in excluded).items())
        ),
    }
    observed_times = [row.observed_at for row in eligible]
    return {
        "report_schema": REPORT_SCHEMA,
        "protocol_id": settings.protocol_id,
        "protocol_version": settings.protocol_version,
        "source_kind": source_kind,
        "as_of": as_of.isoformat(),
        "observations": observation_rows,
        "excluded": excluded,
        "counts": counts,
        "metrics": {
            "population": (
                "common resolved population: every eligible forecast with valid structural, "
                "blend and displayed YES ask arms, including WATCH recommendations"
            ),
            "weighting": "mean within event, then equal weight per independent event",
            "resolved_forecasts": len(resolved),
            "resolved_events": len(resolved_events),
            "resolved_dates": len(resolved_dates),
            "arms": _arm_metrics(event_means),
        },
        "events": events,
        "calibration": {
            "bins": settings.calibration_bins,
            "weighting": "each resolved forecast weighted 1 / resolved forecasts in its event",
            "resolved_forecasts": len(resolved),
            "resolved_events": len(resolved_events),
            "arms": _calibration(resolved, weights, settings.calibration_bins),
        },
        "paired_differences": paired_differences,
        "regimes": regimes,
        "source_selection_branches": branches,
        "coverage": {
            "eligible_fraction": _fraction(len(eligible), len(observations)),
            "resolved_fraction": _fraction(len(resolved), len(eligible)),
            "resolved_event_fraction": _fraction(len(resolved_events), len(rows_by_event)),
            "first_observed_at": min(observed_times).isoformat() if observed_times else None,
            "last_observed_at": max(observed_times).isoformat() if observed_times else None,
            "archive_start": settings.archive_start.isoformat(),
            "archive_end": settings.archive_end.isoformat(),
            "folds": folds,
        },
        "identities": {
            "protocol_id": settings.protocol_id,
            "protocol_version": settings.protocol_version,
            "source_kind": source_kind,
            "as_of": as_of.isoformat(),
            "protocol_report_as_of": settings.report_as_of.isoformat(),
            "model_name": settings.model_name,
            "model_version": settings.model_version,
            "structural_weight": settings.structural_weight,
            "expected_annual_return": settings.expected_annual_return,
            "production_features": settings.production_features,
            "recipe_ids": sorted({row.recipe_id for row in eligible}),
            "run_ids": sorted({row.run_id for row in eligible}),
            "input_ids": sorted({row.input_id for row in eligible}),
            "run_manifest_sha256": sorted({row.run_manifest_sha256 for row in eligible}),
            "research_context_ids": sorted(
                {row.research_context_id for row in eligible if row.research_context_id}
            ),
            "eligible_observations_sha256": content_id(observation_rows),
            "excluded_observations_sha256": content_id(excluded),
        },
        "conclusion": {
            "status": "inconclusive",
            "claim": (
                "structural or blended forecasts outperform the executable YES ask on unseen events"
            ),
            "basis": (
                f"{source_kind} evidence cannot test unseen-event performance; this holds "
                "regardless of point estimates or interval signs"
            ),
        },
        "limitations": _limitations(settings, source_kind, counts),
    }


def _limitations(settings: _Settings, source_kind: str, counts: dict[str, Any]) -> list[str]:
    limitations = [
        (
            "Synthetic fixture values are deliberate constructions, not empirical market "
            "evidence; manual synthetic features are exempt from production feature pinning."
            if source_kind == SOURCE_SYNTHETIC
            else "Retrospective local evidence was available during development and is not "
            "untouched held-out evidence."
        ),
        "No holdout-window forecasts are scored; the population is limited to the archive "
        f"window {settings.archive_start.isoformat()} to {settings.archive_end.isoformat()}.",
        f"{counts['qualifying_validation_folds']} qualifying validation folds and "
        f"{counts['qualifying_holdout_folds']} qualifying holdout folds; the protocol requires "
        f"at least {settings.minimum_validation_folds}, each with "
        f"{settings.minimum_fold_resolved_events} resolved events.",
        "The recorded midpoint anchor is a secondary descriptive baseline, not an executable "
        "price.",
        "Regime results are descriptive; no regime-level uncertainty is reported.",
    ]
    resolved_events = counts["resolved_events"]
    resolved_dates = counts["resolved_dates"]
    if resolved_events < settings.minimum_resolved_events:
        limitations.append(
            f"Only {resolved_events} resolved independent events; the protocol minimum is "
            f"{settings.minimum_resolved_events}."
        )
    if resolved_dates < settings.minimum_dates:
        limitations.append(
            f"Only {resolved_dates} resolved event dates; the protocol minimum is "
            f"{settings.minimum_dates}."
        )
    if resolved_events < settings.minimum_calibration_events:
        limitations.append(
            f"Calibration uses {resolved_events} resolved events, below the protocol minimum "
            f"of {settings.minimum_calibration_events}; bins are descriptive only."
        )
    if resolved_events < 2:
        limitations.append("Fewer than two resolved events: no event bootstrap interval.")
    if resolved_dates < 2:
        limitations.append("Fewer than two resolved event dates: no date-block bootstrap interval.")
    if counts["source_selection_branches"] > 1:
        limitations.append(
            f"{counts['source_selection_branches']} predeclared source-selection branches are "
            "pooled under the fixed selection rule; see source_selection_branches."
        )
    if counts["unresolved"]:
        limitations.append(
            f"{counts['unresolved']} eligible forecasts are unresolved at the report cutoff and "
            "are not scored."
        )
    if counts["excluded"]:
        limitations.append(
            f"{counts['excluded']} observations were excluded; see the excluded reasons."
        )
    return limitations
