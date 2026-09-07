"""Deployability, policy compatibility, and delivery-authorization boundaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from test_backtest import START, config
from test_discord import evaluation
from test_engine import TERMINAL_ABOVE, crypto_snapshot, market_snapshot

from prediction_market_system.authorization import authorize_delivery
from prediction_market_system.backtest import (
    BacktestFoldResult,
    BacktestModelValidation,
    BacktestResult,
    WalkForwardFold,
)
from prediction_market_system.calibration import CalibrationBin, UncertaintyCalibrationProfile
from prediction_market_system.engine import (
    CryptoThresholdEngine,
    EngineConfig,
    deployment_policy_id,
)
from prediction_market_system.recipe import MODEL_VERSION
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.validation import frozen_campaign_configuration


def _profile(*, recipe_id: str, research_only: bool = False) -> UncertaintyCalibrationProfile:
    generated = datetime.now(UTC) - timedelta(days=1)
    return UncertaintyCalibrationProfile(
        generated_at=generated,
        symbol="BTC",
        model_name="crypto-terminal-above-threshold-market-anchor",
        model_version=MODEL_VERSION,
        recipe_id=recipe_id,
        research_only=research_only,
        training_start=START - timedelta(days=2),
        cutoff_at=START,
        confidence_level=0.95,
        sample_count=100,
        independent_event_count=100,
        brier_score=0.01,
        bins=(
            CalibrationBin(
                lower_probability=0,
                upper_probability=1,
                mean_probability=0.9,
                observed_frequency=0.9,
                outcome_interval_lower=0.85,
                outcome_interval_upper=0.95,
                uncertainty_margin=0.05,
                sample_count=100,
                minimum_horizon_seconds=1,
                maximum_horizon_seconds=31_557_600,
            ),
        ),
    )


def _accepted_result(
    engine: EngineConfig,
    profile: UncertaintyCalibrationProfile,
    *,
    accepted: bool = True,
    require_calibration: bool = True,
) -> BacktestResult:
    backtest = config().model_copy(
        update={
            "require_calibration": require_calibration,
            "minimum_validation_events": 1,
            "minimum_validation_folds": 1,
            "minimum_calibration_samples": 1,
        }
    )
    fold = BacktestFoldResult(
        fold=WalkForwardFold(
            index=0,
            train_start=backtest.start,
            train_end=backtest.start + timedelta(days=1),
            test_start=backtest.start + timedelta(days=1),
            test_end=backtest.end,
        ),
        markets_considered=1,
        evaluated_signals=1,
        missing_context_signals=0,
        missing_fee_signals=0,
        missing_calibration_signals=0,
        calibration_profiles=(),
        trades=(),
        total_cost_dollars=Decimal("1.00"),
        total_pnl_dollars=Decimal("0.10"),
        return_on_cost=0.10,
        brier_score=0.05,
    )
    validation = BacktestModelValidation(
        model_name=profile.model_name,
        model_version=profile.model_version,
        recipe_id=profile.recipe_id,
        calibration_profile_id=profile.profile_id,
        independent_calibration_events=profile.independent_event_count,
        held_out_events=20,
        held_out_folds=2,
        held_out_trades=5,
        total_cost_dollars=Decimal("1.00"),
        total_pnl_dollars=Decimal("0.10"),
        return_on_cost=0.10,
        brier_score=0.05,
        accepted_for_paper_alerts=accepted,
        rejection_reasons=() if accepted else ("synthetic rejection",),
    )
    return BacktestResult(
        config=backtest,
        folds=(fold,),
        unsupported_markets=(),
        unresolved_markets=(),
        markets_without_candles=(),
        total_trades=5,
        partial_fills=0,
        total_cost_dollars=Decimal("1.00"),
        total_pnl_dollars=Decimal("0.10"),
        return_on_cost=0.10,
        brier_score=0.05,
        deployment_profiles=(profile,),
        model_validations=(validation,),
        engine_config=engine,
    )


def _register_campaign(repository: SQLiteRepository, engine: EngineConfig, backtest_config) -> str:
    return repository.register_validation_campaign(
        series_ticker=backtest_config.series_ticker,
        symbol=backtest_config.symbol,
        configuration=frozen_campaign_configuration(
            campaign_start=backtest_config.start,
            max_events=500,
            config=backtest_config,
            engine=engine,
        ),
    )


def test_ordinary_backtest_cannot_create_deployable_approval(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    engine = EngineConfig(fee_rate=0.02)
    recipe_id = CryptoThresholdEngine(engine).recipe_id(crypto_snapshot())
    profile = _profile(recipe_id=recipe_id)
    result = _accepted_result(engine, profile)
    saved = repository.save_backtest_result(result)
    assert saved.model_validations[0].accepted_for_paper_alerts is True
    assert saved.model_validations[0].campaign_id is None
    assert not repository.is_calibration_approved(profile.profile_id)
    assert not repository.is_calibration_approved(
        profile.profile_id, deployment_policy_id=deployment_policy_id(engine)
    )


def test_frozen_campaign_can_create_deployable_approval(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    engine = EngineConfig(fee_rate=0.02, slippage_bps=25)
    recipe_id = CryptoThresholdEngine(engine).recipe_id(crypto_snapshot())
    profile = _profile(recipe_id=recipe_id)
    result = _accepted_result(engine, profile)
    campaign_id = _register_campaign(repository, engine, result.config)
    saved = repository.save_backtest_result(result, campaign_id=campaign_id)
    assert saved.model_validations[0].campaign_id == campaign_id
    assert saved.model_validations[0].deployment_policy_id == deployment_policy_id(engine)
    assert repository.is_calibration_approved(
        profile.profile_id, deployment_policy_id=deployment_policy_id(engine)
    )


def test_replaced_campaign_cannot_reuse_prior_approval(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    engine = EngineConfig(fee_rate=0.02)
    recipe_id = CryptoThresholdEngine(engine).recipe_id(crypto_snapshot())
    profile = _profile(recipe_id=recipe_id)
    result = _accepted_result(engine, profile)
    campaign_id = _register_campaign(repository, engine, result.config)
    repository.save_backtest_result(result, campaign_id=campaign_id)
    policy = deployment_policy_id(engine)
    assert repository.is_calibration_approved(profile.profile_id, deployment_policy_id=policy)

    loosened = result.config.model_copy(update={"maximum_brier_score": 0.5})
    replacement = repository.register_validation_campaign(
        series_ticker=result.config.series_ticker,
        symbol=result.config.symbol,
        configuration=frozen_campaign_configuration(
            campaign_start=loosened.start,
            max_events=500,
            config=loosened,
            engine=engine,
        ),
        replace=True,
    )
    assert replacement != campaign_id
    assert not repository.campaign_is_active(campaign_id)
    assert not repository.is_calibration_approved(profile.profile_id, deployment_policy_id=policy)
    report = repository.campaign_report(
        series_ticker=result.config.series_ticker, symbol=result.config.symbol
    )
    registrations = report["campaign_registrations"]
    assert [item["campaign_id"] for item in registrations] == [campaign_id, replacement]
    assert registrations[0]["superseded_at"] is not None
    assert registrations[1]["superseded_at"] is None
    with pytest.raises(ValueError, match="no longer active"):
        repository.save_backtest_result(
            result.model_copy(update={"run_id": uuid4()}),
            campaign_id=campaign_id,
        )


def test_deployment_policy_must_match_for_approval_reuse(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    engine = EngineConfig(fee_rate=0.02, slippage_bps=25, fractional_kelly=0.25)
    recipe_id = CryptoThresholdEngine(engine).recipe_id(crypto_snapshot())
    profile = _profile(recipe_id=recipe_id)
    result = _accepted_result(engine, profile)
    campaign_id = _register_campaign(repository, engine, result.config)
    repository.save_backtest_result(result, campaign_id=campaign_id)
    policy = deployment_policy_id(engine)
    assert repository.is_calibration_approved(profile.profile_id, deployment_policy_id=policy)
    assert repository.is_calibration_approved(
        profile.profile_id, deployment_policy_id=deployment_policy_id(engine.model_copy())
    )
    assert not repository.is_calibration_approved(
        profile.profile_id,
        deployment_policy_id=deployment_policy_id(engine.model_copy(update={"fee_rate": 0.0})),
    )
    assert not repository.is_calibration_approved(
        profile.profile_id,
        deployment_policy_id=deployment_policy_id(engine.model_copy(update={"slippage_bps": 100})),
    )
    assert not repository.is_calibration_approved(
        profile.profile_id,
        deployment_policy_id=deployment_policy_id(
            engine.model_copy(update={"fractional_kelly": 1.0})
        ),
    )


def test_research_only_profile_is_not_deployable_under_a_campaign(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    engine = EngineConfig(fee_rate=0.02)
    recipe_id = CryptoThresholdEngine(engine).recipe_id(crypto_snapshot())
    profile = _profile(recipe_id=recipe_id, research_only=True)
    result = _accepted_result(engine, profile)
    campaign_id = _register_campaign(repository, engine, result.config)
    saved = repository.save_backtest_result(result, campaign_id=campaign_id)
    assert saved.model_validations[0].accepted_for_paper_alerts is True
    assert not repository.is_calibration_approved(
        profile.profile_id, deployment_policy_id=deployment_policy_id(engine)
    )


def test_legacy_validation_payload_without_policy_fails_closed(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    engine = EngineConfig(fee_rate=0.02)
    recipe_id = CryptoThresholdEngine(engine).recipe_id(crypto_snapshot())
    profile = _profile(recipe_id=recipe_id)
    result = _accepted_result(engine, profile)
    campaign_id = _register_campaign(repository, engine, result.config)
    saved = repository.save_backtest_result(result, campaign_id=campaign_id)
    with sqlite3.connect(repository.database_path) as connection:
        (payload,) = connection.execute(
            "SELECT payload_json FROM paper_model_validations_v2 WHERE calibration_profile_id=?",
            (str(profile.profile_id),),
        ).fetchone()
        stripped = json.loads(payload)
        stripped.pop("campaign_id")
        stripped.pop("deployment_policy_id")
        connection.execute(
            "UPDATE paper_model_validations_v2 SET payload_json=? WHERE calibration_profile_id=?",
            (json.dumps(stripped), str(profile.profile_id)),
        )
        connection.commit()
    assert not repository.is_calibration_approved(
        profile.profile_id, deployment_policy_id=deployment_policy_id(engine)
    )
    assert saved.model_validations[0].campaign_id == campaign_id


def test_mutated_opportunity_economics_cannot_authorize_delivery(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    repository.save_evaluation(forecast, opportunity)
    as_of = forecast.generated_at
    authorize_delivery(repository, opportunity, allow_unapproved=True, as_of=as_of)

    mutated_price = opportunity.model_copy(
        update={"executable_price": 0.99, "conservative_net_edge": 0.90}
    )
    assert mutated_price.executable_price == 0.99
    assert opportunity.executable_price == pytest.approx(0.42)
    with pytest.raises(ValueError, match="reproducible executable decision"):
        authorize_delivery(repository, mutated_price, allow_unapproved=True, as_of=as_of)

    mutated_edge = opportunity.model_copy(update={"conservative_net_edge": 0.90})
    with pytest.raises(ValueError, match="reproducible executable decision"):
        authorize_delivery(repository, mutated_edge, allow_unapproved=True, as_of=as_of)

    mutated_probability = opportunity.model_copy(update={"conservative_probability": 0.99})
    with pytest.raises(ValueError, match="reproducible executable decision"):
        authorize_delivery(repository, mutated_probability, allow_unapproved=True, as_of=as_of)

    annotated = opportunity.model_copy(
        update={"warnings": (*opportunity.warnings, "regime annotation")}
    )
    authorize_delivery(repository, annotated, allow_unapproved=True, as_of=as_of)


def test_compact_manifest_keeps_recipe_identity_without_live_provider_state() -> None:
    from prediction_market_system.backtest import _compact_manifest

    forecast, _ = CryptoThresholdEngine().evaluate(
        market_snapshot(), crypto_snapshot(), TERMINAL_ABOVE
    )
    compact = _compact_manifest(forecast)
    assert compact["recipe"] == forecast.input_manifest["recipe"]
    assert compact["engine_config"] == forecast.input_manifest["engine_config"]
    assert compact["contract"] == forecast.input_manifest["contract"]
    assert "market" not in compact
    assert "crypto" not in compact
    replay = CryptoThresholdEngine(EngineConfig.model_validate(compact["engine_config"])).recipe_id(
        crypto_snapshot()
    )
    assert replay == forecast.recipe_id
