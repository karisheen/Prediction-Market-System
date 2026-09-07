from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from prediction_market_system.backtest import BacktestConfig, BacktestModelValidation
from prediction_market_system.engine import EngineConfig, deployment_policy_id
from prediction_market_system.recipe import MODEL_VERSION


class ValidationCampaignState(StrEnum):
    COLLECTING = "COLLECTING EVIDENCE"
    REJECTED = "NOT YET APPROVED"
    PARTIALLY_APPROVED = "PARTIALLY APPROVED"
    APPROVED = "APPROVED"


@dataclass(frozen=True)
class ValidationCampaignReport:
    series_ticker: str
    symbol: str
    generated_at: datetime
    state: ValidationCampaignState
    coverage_start: datetime
    coverage_end: datetime
    coverage_days: int
    required_days: int
    validations: tuple[BacktestModelValidation, ...] = ()
    run_id: str | None = None
    campaign_id: str | None = None

    @property
    def progress(self) -> float:
        if self.required_days <= 0:
            return 1.0
        return min(self.coverage_days / self.required_days, 1.0)


def deployment_campaign_settings(config: BacktestConfig, engine: EngineConfig) -> dict[str, Any]:
    """Frozen campaign fields that can be reconstructed from a persisted backtest run."""
    return {
        "period_minutes": config.period_minutes,
        "spot_interval_seconds": config.spot_interval_seconds,
        "realized_interval_seconds": config.realized_interval_seconds,
        "train_days": config.train_days,
        "test_days": config.test_days,
        "step_days": config.step_days,
        "realized_window_days": config.realized_window_days,
        "latency_seconds": config.latency_seconds,
        "max_volume_participation": config.max_volume_participation,
        "expected_annual_return": config.expected_annual_return,
        "require_calibration": config.require_calibration,
        "minimum_calibration_samples": config.minimum_calibration_samples,
        "maximum_calibration_bins": config.maximum_calibration_bins,
        "calibration_confidence": config.calibration_confidence,
        "calibration_lead_seconds": config.calibration_lead_seconds,
        "minimum_validation_events": config.minimum_validation_events,
        "minimum_validation_folds": config.minimum_validation_folds,
        "minimum_return_on_cost": config.minimum_return_on_cost,
        "maximum_brier_score": config.maximum_brier_score,
        "engine": engine.model_dump(mode="json"),
        "model_version": MODEL_VERSION,
        "deployment_policy_id": deployment_policy_id(engine),
    }


def frozen_campaign_configuration(
    *,
    campaign_start: datetime,
    max_events: int,
    config: BacktestConfig,
    engine: EngineConfig,
) -> dict[str, Any]:
    """Preregistered campaign identity: fold geometry, gates, and deployment policy."""
    return {
        "campaign_start": campaign_start.isoformat(),
        "max_events": max_events,
        **deployment_campaign_settings(config, engine),
    }


def campaign_matches_backtest(
    registered: dict[str, Any],
    *,
    config: BacktestConfig,
    engine: EngineConfig,
) -> bool:
    if registered.get("campaign_start") != config.start.isoformat():
        return False
    expected = deployment_campaign_settings(config, engine)
    return all(registered.get(key) == value for key, value in expected.items())
