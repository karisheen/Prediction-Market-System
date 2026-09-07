"""Regression boundaries missing from the inherited v2 implementation."""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from test_engine import NOW, TERMINAL_ABOVE, crypto_snapshot, market_snapshot
from test_research import research_context

from prediction_market_system.calibration import CalibrationSample, fit_uncertainty_profiles
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.research import ResearchContext, ResearchDataUnavailable


def test_touch_without_contractual_history_fails_closed() -> None:
    from prediction_market_system.domain import ThresholdModelKind

    contract = TERMINAL_ABOVE.model_copy(update={"model_kind": ThresholdModelKind.BARRIER})
    market = market_snapshot().model_copy(update={"observation_start_at": NOW - timedelta(days=1)})
    # Current spot below the barrier cannot establish that yesterday's barrier was never hit.
    with pytest.raises(ValueError, match="history"):
        CryptoThresholdEngine().evaluate(market, crypto_snapshot(spot=90), contract)


def test_realized_provenance_must_reproduce_from_stored_history(tmp_path: Path) -> None:
    from test_research import research_candles

    from prediction_market_system.storage import SQLiteRepository

    as_of = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    context = research_context(as_of)
    candles = research_candles(as_of)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    context_id = repository.save_research_context(context)
    assert context_id == context.content_id()
    assert "source_candles" not in context.realized_volatility.raw_payload

    # The recorded number is not trusted: the immutable spot store must reproduce it.
    with pytest.raises(ResearchDataUnavailable):
        repository.research_context_by_id(context_id)
    repository.save_research_data(spot_candles=candles[:12] + candles[13:])
    with pytest.raises(ResearchDataUnavailable):
        repository.research_context_by_id(context_id)
    repository.save_research_data(spot_candles=candles)
    assert repository.research_context_by_id(context_id) == context

    payload = context.model_dump(mode="json")
    payload["realized_volatility"]["annualized_volatility"] *= 1.5
    tampered = ResearchContext.model_validate(payload)
    assert tampered.content_id() != context_id


def test_sparse_probability_support_is_not_entire_occupied_bin() -> None:
    samples = tuple(
        CalibrationSample(
            market_id=str(index),
            event_id=str(index),
            symbol="BTC",
            model_name="test",
            model_version="2.0.0",
            probability_yes=0.21,
            outcome_yes=index % 5 == 0,
            observed_at=NOW - timedelta(days=2),
            resolved_at=NOW - timedelta(days=1),
        )
        for index in range(100)
    )
    (profile,) = fit_uncertainty_profiles(
        samples, training_start=NOW - timedelta(days=3), cutoff_at=NOW
    )
    assert profile.margin_for(0.39) == 1.0


def test_horizon_regions_are_not_extrapolated_or_pooled() -> None:
    from prediction_market_system.recipe import MODEL_VERSION

    samples = tuple(
        CalibrationSample(
            market_id=f"{horizon}-{index}",
            event_id=f"{horizon}-{index}",
            symbol="BTC",
            model_name="test",
            model_version=MODEL_VERSION,
            recipe_id="recipe",
            horizon_seconds=horizon,
            probability_yes=0.2,
            outcome_yes=index % 5 == 0,
            observed_at=NOW - timedelta(days=2),
            resolved_at=NOW - timedelta(days=1),
        )
        for horizon in (60, 3600)
        for index in range(100)
    )
    (profile,) = fit_uncertainty_profiles(
        samples,
        training_start=NOW - timedelta(days=3),
        cutoff_at=NOW,
    )
    assert profile.margin_for(0.2, horizon_seconds=60) < 1
    assert profile.margin_for(0.2, horizon_seconds=300) == 1
    assert profile.margin_for(0.2, horizon_seconds=7200) == 1
    assert not profile.research_only
    (legacy,) = fit_uncertainty_profiles(
        tuple(item.model_copy(update={"horizon_seconds": None}) for item in samples),
        training_start=NOW - timedelta(days=3),
        cutoff_at=NOW,
    )
    assert legacy.research_only
    assert legacy.margin_for(0.2, horizon_seconds=60) == 1


def test_coarse_calibration_cannot_hide_opposite_tail_errors() -> None:
    samples = tuple(
        CalibrationSample(
            market_id=f"{index}-{probability}",
            event_id=str(index),
            symbol="BTC",
            model_name="test",
            model_version="2.0.0",
            probability_yes=probability,
            outcome_yes=probability < 0.5,
            observed_at=NOW - timedelta(days=2),
            resolved_at=NOW - timedelta(days=1),
        )
        for index in range(1000)
        for probability in (0.1, 0.9)
    )
    (profile,) = fit_uncertainty_profiles(
        samples,
        training_start=NOW - timedelta(days=3),
        cutoff_at=NOW,
        maximum_bins=1,
    )
    assert profile.margin_for(0.1) >= 0.9
    assert profile.margin_for(0.9) >= 0.9


@pytest.mark.parametrize(
    "changes",
    [
        {"market_type": "scalar"},
        {"notional_value_dollars": 2},
        {"status": "paused"},
        {"updated_time": NOW + timedelta(days=20)},
        {"open_time": NOW + timedelta(days=20)},
    ],
)
def test_live_and_historical_contract_eligibility_agree(changes: dict[str, object]) -> None:
    from test_backtest import START, FixedResearchSource, config, historical_market

    from prediction_market_system.backtest import HistoricalBacktester
    from prediction_market_system.venues.kalshi import (
        KalshiOrderBook,
        UnsupportedMarketError,
        to_market_snapshot,
    )

    history = historical_market()
    # Use the replay's actual test boundary for future-dated metadata checks.
    changes = {
        key: START + timedelta(days=20) if hasattr(value, "tzinfo") else value
        for key, value in changes.items()
    }
    metadata = history.metadata_snapshots[0][1].model_copy(update=changes)
    with pytest.raises(UnsupportedMarketError):
        to_market_snapshot(
            metadata,
            KalshiOrderBook(no_dollars=[(".5", "100")]),
            observed_at=START + timedelta(days=1, hours=1),
        )
    replay = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(
        config(),
        (history.model_copy(update={"metadata_snapshots": ((START, metadata),)}),),
    )
    assert replay.forecasts == () and replay.total_trades == 0


def test_changed_execution_revision_cannot_improve_past_fill() -> None:
    from test_backtest import FixedResearchSource, config, historical_market

    from prediction_market_system.backtest import HistoricalBacktester

    history = historical_market()
    available, candle = history.candlestick_revisions[1]
    revised = candle.model_copy(update={"volume": 100000})
    changed = history.model_copy(
        update={
            "candlesticks": (history.candlesticks[0], revised),
            "candlestick_revisions": (
                *history.candlestick_revisions,
                (available + timedelta(days=1), revised),
            ),
        }
    )
    runner = HistoricalBacktester(FixedResearchSource(), EngineConfig())
    original = runner.run(config(), (history,))
    replay = runner.run(config(), (changed,))
    assert replay.total_cost_dollars == original.total_cost_dollars
    assert replay.total_pnl_dollars == original.total_pnl_dollars


@pytest.mark.asyncio
async def test_forecasts_reference_persisted_research_context_instead_of_embedding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_paper_alerts as alerts

    from prediction_market_system.authorization import authorize_delivery
    from prediction_market_system.paper_alerts import PaperAlertRunner, classify_market_regime
    from prediction_market_system.storage import SQLiteRepository

    context, candles = alerts.research_context(end_price="130", realized_volatility=0.60)
    regime = classify_market_regime(context, candles)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    repository.save_research_data(spot_candles=candles)
    profile = alerts.calibration_profile()
    monkeypatch.setattr(repository, "latest_uncertainty_calibration", lambda **kwargs: profile)
    monkeypatch.setattr(repository, "uncertainty_calibration", lambda profile_id: profile)
    monkeypatch.setattr(repository, "is_calibration_approved", lambda profile_id, **kwargs: True)
    publisher = alerts.FakePublisher()
    runner = PaperAlertRunner(
        repository=repository,
        engine=CryptoThresholdEngine(
            EngineConfig(
                min_conservative_edge=0.0,
                binary_fee_coefficient=0.0,
                slippage_bps=0.0,
                resolution_haircut=0.0,
            )
        ),
        market_reader=alerts.FakeMarketReader(),
        alert_service=publisher,
        clock=lambda: alerts.AS_OF,
    )
    result = await runner.run(
        markets=[alerts.kalshi_market()],
        context=context,
        regime=regime,
        cycle_id="cycle",
        deliver_entries=True,
    )
    assert result.delivered == 1
    opportunity = publisher.published[0]
    provenance = opportunity.forecast.input_manifest["crypto"]["input_provenance"]
    assert provenance["research_context_id"] == context.content_id()

    with sqlite3.connect(tmp_path / "audit.db") as connection:
        (forecast_json,) = connection.execute(
            "SELECT payload_json FROM forecasts WHERE forecast_id=?",
            (str(opportunity.forecast.forecast_id),),
        ).fetchone()
        (context_count,) = connection.execute(
            "SELECT COUNT(*) FROM input_objects WHERE input_id=?", (context.content_id(),)
        ).fetchone()
    # Hundreds of kilobytes of source candles are stored once, not per forecast.
    assert "source_candles" not in forecast_json and len(forecast_json) < 20_000
    assert context_count == 1
    assert repository.research_context_by_id(context.content_id()) == context
    explained = repository.explain_forecast(str(opportunity.forecast.forecast_id))
    assert explained["research_context_persisted"] is True
    assert explained["research_context"] == context.model_dump(mode="json")

    # Delivery authorization requires the exact persisted context, not a claim of one.
    fresh = SQLiteRepository(tmp_path / "fresh.db")
    fresh.initialize()
    monkeypatch.setattr(fresh, "uncertainty_calibration", lambda profile_id: profile)
    with pytest.raises(ValueError, match="research context is not persisted"):
        authorize_delivery(fresh, opportunity, allow_unapproved=True, as_of=alerts.AS_OF)
