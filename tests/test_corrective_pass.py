from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from test_backtest import (
    START,
    FixedResearchSource,
    candle,
    config,
    historical_market,
    market,
)
from test_engine import TERMINAL_ABOVE, crypto_snapshot, market_snapshot

from prediction_market_system.authorization import authorize_delivery
from prediction_market_system.backtest import BacktestConfig, HistoricalBacktester
from prediction_market_system.calibration import CalibrationBin, UncertaintyCalibrationProfile
from prediction_market_system.cli import (
    _fetch_prospective_candlesticks,
    availability_after_receipt,
    save_prospective_kalshi_history,
)
from prediction_market_system.domain import RecommendationState
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import (
    AMBIGUOUS_EVIDENCE_CLASS,
    EVIDENCE_FORWARD_SHADOW,
    EVIDENCE_HISTORICAL,
    EVIDENCE_MANUAL_RESEARCH,
)
from prediction_market_system.recipe import (
    DECISION_SPOT_INTERVAL_SECONDS,
    DEFAULT_RESEARCH_INTERVAL_SECONDS,
    MANAGED_KALSHI_PERIOD_MINUTES,
)
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.validation import (
    campaign_matches_backtest,
    frozen_campaign_configuration,
)


def _held_out_profile(
    engine: CryptoThresholdEngine,
    crypto,
    contract,
    *,
    observed_at: datetime,
) -> UncertaintyCalibrationProfile:
    return UncertaintyCalibrationProfile(
        generated_at=observed_at - timedelta(days=1),
        symbol=crypto.symbol,
        model_name=engine.model_name(contract),
        model_version=engine.model_version,
        recipe_id=engine.recipe_id(crypto),
        research_only=False,
        training_start=observed_at - timedelta(days=60),
        cutoff_at=observed_at - timedelta(days=1),
        confidence_level=0.95,
        sample_count=30,
        independent_event_count=30,
        brier_score=0.20,
        bins=(
            CalibrationBin(
                lower_probability=0,
                upper_probability=1,
                mean_probability=0.5,
                observed_frequency=0.5,
                outcome_interval_lower=0.4,
                outcome_interval_upper=0.6,
                uncertainty_margin=0.03,
                sample_count=30,
                minimum_horizon_seconds=1,
                maximum_horizon_seconds=31557600,
            ),
        ),
    )


def _persist_profile(repository: SQLiteRepository, profile: UncertaintyCalibrationProfile) -> None:
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "INSERT INTO uncertainty_calibrations "
            "(profile_id, symbol, model_name, model_version, cutoff_at, "
            "generated_at, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(profile.profile_id),
                profile.symbol,
                profile.model_name,
                profile.model_version,
                profile.cutoff_at.isoformat(),
                profile.generated_at.isoformat(),
                profile.model_dump_json(),
            ),
        )


def _entry_engine(**overrides: object) -> CryptoThresholdEngine:
    payload = {
        "uncertainty_margin": 0.03,
        "structural_weight": 0.70,
        "fee_rate": 0.0,
        "binary_fee_coefficient": 0.0,
        "slippage_bps": 25,
        "resolution_haircut": 0.01,
        "minimum_seconds_to_expiry": 300,
        "maximum_input_age_seconds": 120,
    }
    payload.update(overrides)
    return CryptoThresholdEngine(EngineConfig(**payload))


def test_authorize_delivery_fails_after_crossing_minimum_time_to_expiry(
    tmp_path: Path,
) -> None:
    observed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    engine = _entry_engine()
    market = market_snapshot(expires_in=timedelta(seconds=310)).model_copy(
        update={"observed_at": observed_at, "expires_at": observed_at + timedelta(seconds=310)}
    )
    crypto = crypto_snapshot().model_copy(update={"observed_at": observed_at})
    profile = _held_out_profile(engine, crypto, TERMINAL_ABOVE, observed_at=observed_at)
    repository = SQLiteRepository(tmp_path / "h1.db")
    repository.initialize()
    _persist_profile(repository, profile)
    _, opportunity = engine.evaluate(market, crypto, TERMINAL_ABOVE, profile)
    assert opportunity.state is RecommendationState.ENTER_YES
    repository.save_evaluation(opportunity.forecast, opportunity)

    authorize_delivery(repository, opportunity, allow_unapproved=True, as_of=observed_at)
    # Five seconds later 305s remain — still above the 300s control.
    authorize_delivery(
        repository,
        opportunity,
        allow_unapproved=True,
        as_of=observed_at + timedelta(seconds=5),
    )

    with pytest.raises(ValueError, match="minimum-time-to-expiry exclusion window"):
        authorize_delivery(
            repository,
            opportunity,
            allow_unapproved=True,
            as_of=observed_at + timedelta(seconds=20),
        )

    _, later = engine.evaluate(
        market.model_copy(update={"observed_at": observed_at + timedelta(seconds=20)}),
        crypto,
        TERMINAL_ABOVE,
        profile,
    )
    assert later.state is RecommendationState.WATCH


def test_authorize_delivery_rejects_ended_observation_window(tmp_path: Path) -> None:
    observed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    engine = _entry_engine(minimum_seconds_to_expiry=0)
    market = market_snapshot(expires_in=timedelta(seconds=600)).model_copy(
        update={
            "observed_at": observed_at,
            "expires_at": observed_at + timedelta(seconds=600),
            "observation_end_at": observed_at + timedelta(seconds=50),
        }
    )
    crypto = crypto_snapshot().model_copy(update={"observed_at": observed_at})
    profile = _held_out_profile(engine, crypto, TERMINAL_ABOVE, observed_at=observed_at)
    repository = SQLiteRepository(tmp_path / "h1-obs.db")
    repository.initialize()
    _persist_profile(repository, profile)
    _, opportunity = engine.evaluate(market, crypto, TERMINAL_ABOVE, profile)
    assert opportunity.state is RecommendationState.ENTER_YES
    repository.save_evaluation(opportunity.forecast, opportunity)
    authorize_delivery(repository, opportunity, allow_unapproved=True, as_of=observed_at)
    with pytest.raises(ValueError, match="benchmark observation has already ended"):
        authorize_delivery(
            repository,
            opportunity,
            allow_unapproved=True,
            as_of=observed_at + timedelta(seconds=50),
        )


def test_default_and_managed_recipes_match_live_and_interval_changes_are_incompatible() -> None:
    as_of = START + timedelta(days=1, hours=1)
    source = FixedResearchSource()
    engine = CryptoThresholdEngine()
    live = source.research_context_as_of(
        symbol="BTC",
        as_of=as_of,
        interval_seconds=DEFAULT_RESEARCH_INTERVAL_SECONDS,
        spot_interval_seconds=DECISION_SPOT_INTERVAL_SECONDS,
        realized_window_seconds=24 * 60 * 60,
    )
    historical = BacktestConfig(
        series_ticker="KXBTC",
        symbol="BTC",
        start=START,
        end=START + timedelta(days=180),
    )
    historical_context = source.research_context_as_of(
        symbol="BTC",
        as_of=as_of,
        interval_seconds=historical.realized_interval_seconds,
        spot_interval_seconds=historical.spot_interval_seconds,
        realized_window_seconds=24 * 60 * 60,
    )
    managed = BacktestConfig(
        series_ticker="KXBTC",
        symbol="BTC",
        start=START,
        end=START + timedelta(days=180),
        period_minutes=MANAGED_KALSHI_PERIOD_MINUTES,
    )
    managed_context = source.research_context_as_of(
        symbol="BTC",
        as_of=as_of,
        interval_seconds=managed.realized_interval_seconds,
        spot_interval_seconds=managed.spot_interval_seconds,
        realized_window_seconds=24 * 60 * 60,
    )
    live_recipe = engine.recipe_id(live.to_crypto_snapshot(strike_price=100.0))
    assert historical.spot_interval_seconds == DECISION_SPOT_INTERVAL_SECONDS
    assert historical.realized_interval_seconds == DEFAULT_RESEARCH_INTERVAL_SECONDS
    assert managed.spot_interval_seconds == DECISION_SPOT_INTERVAL_SECONDS
    assert managed.realized_interval_seconds == DEFAULT_RESEARCH_INTERVAL_SECONDS
    assert live.spot.interval_seconds == DECISION_SPOT_INTERVAL_SECONDS
    assert live.realized_volatility.raw_payload["interval_seconds"] == (
        DEFAULT_RESEARCH_INTERVAL_SECONDS
    )
    historical_recipe = engine.recipe_id(historical_context.to_crypto_snapshot(strike_price=100.0))
    managed_recipe = engine.recipe_id(managed_context.to_crypto_snapshot(strike_price=100.0))
    assert historical_recipe == live_recipe
    assert managed_recipe == live_recipe

    different_spot = source.research_context_as_of(
        symbol="BTC",
        as_of=as_of,
        interval_seconds=DEFAULT_RESEARCH_INTERVAL_SECONDS,
        spot_interval_seconds=DEFAULT_RESEARCH_INTERVAL_SECONDS,
        realized_window_seconds=24 * 60 * 60,
    )
    different_research = source.research_context_as_of(
        symbol="BTC",
        as_of=as_of,
        interval_seconds=DECISION_SPOT_INTERVAL_SECONDS,
        spot_interval_seconds=DECISION_SPOT_INTERVAL_SECONDS,
        realized_window_seconds=24 * 60 * 60,
    )
    assert engine.recipe_id(different_spot.to_crypto_snapshot(strike_price=100.0)) != live_recipe
    different_research_id = engine.recipe_id(
        different_research.to_crypto_snapshot(strike_price=100.0)
    )
    assert different_research_id != live_recipe


def test_campaign_missing_interval_keys_fails_closed() -> None:
    engine = EngineConfig()
    campaign = BacktestConfig(
        series_ticker="KXBTC",
        symbol="BTC",
        start=START,
        end=START + timedelta(days=180),
        period_minutes=MANAGED_KALSHI_PERIOD_MINUTES,
    )
    registered = frozen_campaign_configuration(
        campaign_start=campaign.start,
        max_events=5000,
        config=campaign,
        engine=engine,
    )
    assert campaign_matches_backtest(registered, config=campaign, engine=engine)
    stale = dict(registered)
    stale.pop("spot_interval_seconds")
    stale.pop("realized_interval_seconds")
    assert not campaign_matches_backtest(stale, config=campaign, engine=engine)


def test_prospective_candle_receipt_is_replay_admissible_and_late_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    decision_at = START + timedelta(days=1, hours=1)
    settlement_at = START + timedelta(days=2, minutes=5)
    identity = market().model_copy(
        update={
            "status": "active",
            "result": "",
            "settlement_ts": None,
            "settlement_value_dollars": None,
            "expiration_value": "",
        }
    )
    first = candle(
        decision_at,
        bid=("0.18", "0.18", "0.20", "0.20"),
        ask=("0.20", "0.20", "0.22", "0.22"),
        volume="500",
    )
    second = candle(
        decision_at + timedelta(hours=1),
        bid=("0.20", "0.19", "0.22", "0.21"),
        ask=("0.22", "0.22", "0.25", "0.24"),
        volume="100",
    )
    resolved = market()

    def persist(path: Path, candle_receipt: datetime) -> SQLiteRepository:
        repository = SQLiteRepository(path)
        repository.initialize()
        repository.save_kalshi_history(
            series_ticker="KXBTC",
            observed_at=START,
            markets=[identity],
            candlesticks={},
            period_interval=60,
            series_fee_changes=[],
            event_fee_changes=[],
        )
        repository.save_kalshi_history(
            series_ticker="KXBTC",
            observed_at=candle_receipt,
            markets=[identity],
            candlesticks={identity.ticker: [first]},
            period_interval=60,
            series_fee_changes=[],
            event_fee_changes=[],
        )
        repository.save_kalshi_history(
            series_ticker="KXBTC",
            observed_at=candle_receipt + timedelta(hours=1),
            markets=[identity],
            candlesticks={identity.ticker: [second]},
            period_interval=60,
            series_fee_changes=[],
            event_fee_changes=[],
        )
        repository.save_kalshi_history(
            series_ticker="KXBTC",
            observed_at=settlement_at,
            markets=[resolved],
            candlesticks={},
            period_interval=60,
            series_fee_changes=[],
            event_fee_changes=[],
        )
        return repository

    timely = persist(tmp_path / "timely.db", decision_at + timedelta(seconds=10))
    late = persist(tmp_path / "late.db", settlement_at)
    timely_history = timely.load_kalshi_backtest_data(
        series_ticker="KXBTC",
        start=config().start,
        end=config().end,
        period_interval=60,
        max_events=10,
    )
    late_history = late.load_kalshi_backtest_data(
        series_ticker="KXBTC",
        start=config().start,
        end=config().end,
        period_interval=60,
        max_events=10,
    )
    assert timely_history[0].candlestick_revisions[0][0] == decision_at + timedelta(seconds=10)
    assert late_history[0].candlestick_revisions[0][0] == settlement_at
    backtester = HistoricalBacktester(FixedResearchSource(), EngineConfig())
    timely_result = backtester.run(config(), timely_history)
    late_result = backtester.run(config(), late_history)
    assert len(timely_result.forecasts) == 2
    assert timely_result.forecasts[0].observed_at == decision_at + timedelta(seconds=10)
    assert late_result.forecasts == ()
    assert late_result.incomplete_forecast_signals == 2


@pytest.mark.asyncio
async def test_prospective_candle_fetch_does_not_open_a_client_for_empty_markets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prediction_market_system.cli.KalshiClient",
        lambda: pytest.fail("empty scans must not open a Kalshi client"),
    )
    assert (await _fetch_prospective_candlesticks("KXBTC", [], as_of=datetime.now(UTC))) == {}


class _PersistingResearchSource:
    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository
        self._inner = FixedResearchSource()

    def research_context_as_of(self, **kwargs: object):
        return self._inner.research_context_as_of(**kwargs)

    def save_research_context(self, context) -> str:
        return self._repository.save_research_context(context)

    def save_evaluation(self, forecast, opportunity, *, ledger_kind: str | None = None) -> None:
        self._repository.save_evaluation(forecast, opportunity, ledger_kind=ledger_kind)


def test_persisted_historical_forecasts_are_explainable_without_duplicating_context(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "m1.db")
    repository.initialize()
    source = _PersistingResearchSource(repository)
    result = HistoricalBacktester(source, EngineConfig()).run(config(), (historical_market(),))
    assert len(result.forecasts) == 2
    forecast_id = str(result.forecasts[0].forecast.forecast_id)
    explained = repository.explain_forecast(forecast_id)
    assert explained["classification"] == "v2-recorded-inputs"
    assert explained["research_context_persisted"] is True
    assert explained["research_context"] is not None
    assert explained["manifest"]["kind"] == "historical-evaluation"
    provenance = explained["inputs"]["crypto"]["input_provenance"]
    context_id = provenance["research_context_id"]
    assert isinstance(context_id, str) and len(context_id) == 64
    assert explained["research_context"]["as_of"]
    second = repository.explain_forecast(str(result.forecasts[1].forecast.forecast_id))
    assert second["research_context_persisted"] is True

    context = source.research_context_as_of(
        symbol="BTC",
        as_of=START + timedelta(days=1, hours=1),
        interval_seconds=DEFAULT_RESEARCH_INTERVAL_SECONDS,
        spot_interval_seconds=DECISION_SPOT_INTERVAL_SECONDS,
        realized_window_seconds=24 * 60 * 60,
    )
    source.save_research_context(context)
    source.save_research_context(context)
    with sqlite3.connect(repository.database_path) as connection:
        (context_rows,) = connection.execute(
            "SELECT COUNT(*) FROM input_objects WHERE input_id=?", (context.content_id(),)
        ).fetchone()
        ledger_payloads = [
            str(row[0])
            for row in connection.execute("SELECT payload_json FROM forecast_ledger").fetchall()
        ]
        (forecast_inputs,) = connection.execute("SELECT COUNT(*) FROM input_objects").fetchone()
    assert context_rows == 1
    assert all("source_candles" not in payload for payload in ledger_payloads)
    # Two compact forecast manifests plus one or two research contexts, not one giant
    # context copy per forecast.
    assert forecast_inputs <= 6
    with pytest.raises(ValueError, match="forecast not found"):
        repository.explain_forecast("missing-forecast-id")


def test_pycdc_is_absent_from_project_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    assert "pycdc" not in (root / "pyproject.toml").read_text()
    assert "pycdc" not in (root / "uv.lock").read_text()


def test_prospective_availability_is_receipt_time_not_request_start(tmp_path: Path) -> None:
    request_started_at = START + timedelta(days=1, hours=1)
    received_at = request_started_at + timedelta(seconds=10)
    decision_at = request_started_at
    identity = market().model_copy(
        update={
            "status": "active",
            "result": "",
            "settlement_ts": None,
            "settlement_value_dollars": None,
            "expiration_value": "",
        }
    )
    first = candle(
        decision_at,
        bid=("0.18", "0.18", "0.20", "0.20"),
        ask=("0.20", "0.20", "0.22", "0.22"),
        volume="500",
    )
    assert (
        availability_after_receipt(
            request_started_at=request_started_at,
            received_at=received_at,
        )
        == received_at
    )
    with pytest.raises(ValueError, match="cannot precede request start"):
        availability_after_receipt(
            request_started_at=received_at,
            received_at=request_started_at,
        )

    repository = SQLiteRepository(tmp_path / "receipt.db")
    repository.initialize()
    repository.save_kalshi_history(
        series_ticker="KXBTC",
        observed_at=START,
        markets=[identity],
        candlesticks={},
        period_interval=60,
        series_fee_changes=[],
        event_fee_changes=[],
    )
    availability = save_prospective_kalshi_history(
        repository,
        series_ticker="KXBTC",
        markets=[identity],
        candlesticks={identity.ticker: [first]},
        request_started_at=request_started_at,
        received_at=received_at,
        period_interval=60,
    )
    assert availability == received_at
    history = repository.load_kalshi_backtest_data(
        series_ticker="KXBTC",
        start=config().start,
        end=config().end,
        period_interval=60,
        max_events=10,
    )
    assert history[0].candlestick_revisions[0][0] == received_at
    assert history[0].candlestick_revisions[0][0] != request_started_at
    backtester = HistoricalBacktester(FixedResearchSource(), EngineConfig())
    late_for_request_start = backtester.run(
        config().model_copy(update={"end": request_started_at + timedelta(seconds=1)}),
        history,
    )
    assert late_for_request_start.forecasts == ()
    timely = backtester.run(config(), history)
    assert timely.forecasts
    assert all(record.observed_at >= received_at for record in timely.forecasts)


def test_request_start_cannot_masquerade_as_candle_receipt(tmp_path: Path) -> None:
    request_started_at = datetime(2025, 1, 2, 1, 0, 0, tzinfo=UTC)
    received_at = datetime(2025, 1, 2, 1, 0, 10, tzinfo=UTC)
    identity = market().model_copy(
        update={
            "status": "active",
            "result": "",
            "settlement_ts": None,
            "settlement_value_dollars": None,
            "expiration_value": "",
        }
    )
    first = candle(
        request_started_at,
        bid=("0.18", "0.18", "0.20", "0.20"),
        ask=("0.20", "0.20", "0.22", "0.22"),
        volume="500",
    )
    repository = SQLiteRepository(tmp_path / "lookahead.db")
    repository.initialize()
    save_prospective_kalshi_history(
        repository,
        series_ticker="KXBTC",
        markets=[identity],
        candlesticks={identity.ticker: [first]},
        request_started_at=request_started_at,
        received_at=received_at,
        period_interval=60,
    )
    history = repository.load_kalshi_backtest_data(
        series_ticker="KXBTC",
        start=request_started_at - timedelta(days=1),
        end=received_at + timedelta(days=1),
        period_interval=60,
        max_events=10,
    )
    available_at = history[0].candlestick_revisions[0][0]
    assert available_at == received_at
    # A forecast at the request-start boundary would be look-ahead.
    assert available_at > request_started_at


def test_forward_shadow_report_excludes_historical_forecasts(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "h2.db")
    repository.initialize()
    source = _PersistingResearchSource(repository)
    historical = HistoricalBacktester(source, EngineConfig()).run(config(), (historical_market(),))
    assert historical.forecasts
    historical_id = str(historical.forecasts[0].forecast.forecast_id)

    engine = _entry_engine()
    observed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    crypto = crypto_snapshot().model_copy(update={"observed_at": observed_at})
    profile = _held_out_profile(engine, crypto, TERMINAL_ABOVE, observed_at=observed_at)
    market = market_snapshot().model_copy(
        update={
            "market_id": "forward-shadow-one",
            "venue": "kalshi",
            "series_id": "KXBTC",
            "event_id": "FORWARD-EVENT",
            "observed_at": observed_at,
            "expires_at": observed_at + timedelta(days=30),
            "observation_end_at": observed_at + timedelta(days=30),
        }
    )
    forecast, opportunity = engine.evaluate(market, crypto, TERMINAL_ABOVE, profile)
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)

    report = repository.shadow_report(series_ticker="KXBTC", as_of=observed_at)
    assert report["excluded_non_forward_forecasts"] == len(historical.forecasts)
    assert sum(item["forecasts"] for item in report["recipes"]) == 1
    assert "forward recorded forecasts" in report["evidence"]
    assert "untouched forward" not in report["evidence"]

    explained = repository.explain_forecast(historical_id)
    assert explained["evidence_class"] == EVIDENCE_HISTORICAL
    assert explained["manifest"]["kind"] == EVIDENCE_HISTORICAL
    assert explained["classification"] == "v2-recorded-inputs"
    reloaded = repository.explain_forecast(str(forecast.forecast_id))
    assert reloaded["evidence_class"] == EVIDENCE_FORWARD_SHADOW
    assert reloaded["manifest"]["kind"] == EVIDENCE_FORWARD_SHADOW

    manual_forecast = forecast.model_copy(update={"forecast_id": uuid4()})
    manual = opportunity.model_copy(update={"opportunity_id": uuid4(), "forecast": manual_forecast})
    repository.save_evaluation(manual_forecast, manual, ledger_kind=EVIDENCE_MANUAL_RESEARCH)
    after_manual = repository.shadow_report(series_ticker="KXBTC", as_of=observed_at)
    assert sum(item["forecasts"] for item in after_manual["recipes"]) == 1
    assert after_manual["excluded_non_forward_forecasts"] == len(historical.forecasts) + 1


def test_shadow_report_fails_closed_on_unclassified_ledger_rows(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "h2-legacy.db")
    repository.initialize()
    engine = _entry_engine()
    observed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    crypto = crypto_snapshot().model_copy(update={"observed_at": observed_at})
    profile = _held_out_profile(engine, crypto, TERMINAL_ABOVE, observed_at=observed_at)
    market = market_snapshot().model_copy(
        update={
            "series_id": "KXBTC",
            "event_id": "LEGACY-EVENT",
            "observed_at": observed_at,
        }
    )
    forecast, opportunity = engine.evaluate(market, crypto, TERMINAL_ABOVE, profile)
    repository.save_evaluation(forecast, opportunity, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE run_manifests SET kind = 'unexpected-kind', "
            "payload_json = json_set(payload_json, '$.kind', 'unexpected-kind') "
            "WHERE run_id = ?",
            (str(forecast.forecast_id),),
        )
        connection.commit()
    with pytest.raises(ValueError, match="cannot classify ledger evidence"):
        repository.shadow_report(series_ticker="KXBTC", as_of=observed_at)
    explained = repository.explain_forecast(str(forecast.forecast_id))
    assert explained["evidence_class"] == AMBIGUOUS_EVIDENCE_CLASS


def _event_entry_pair(repository: SQLiteRepository, observed_at: datetime):
    engine = CryptoThresholdEngine()
    crypto = crypto_snapshot().model_copy(update={"observed_at": observed_at})
    expires = observed_at + timedelta(days=30)
    profile = _held_out_profile(engine, crypto, TERMINAL_ABOVE, observed_at=observed_at)
    _persist_profile(repository, profile)
    market_a = market_snapshot().model_copy(
        update={
            "market_id": "event-cap-a",
            "venue": "kalshi",
            "event_id": "SHARED-EVENT",
            "series_id": "KXBTC",
            "observed_at": observed_at,
            "expires_at": expires,
            "observation_end_at": expires,
            "yes_ask_size": 1_000,
        }
    )
    market_b = market_a.model_copy(update={"market_id": "event-cap-b", "yes_ask_size": 100})
    _, opportunity_a = engine.evaluate(market_a, crypto, TERMINAL_ABOVE, profile)
    _, opportunity_b = engine.evaluate(market_b, crypto, TERMINAL_ABOVE, profile)
    assert opportunity_a.state is RecommendationState.ENTER_YES
    assert opportunity_b.state is RecommendationState.ENTER_YES
    assert opportunity_a.suggested_max_exposure == pytest.approx(199.99875)
    assert opportunity_b.suggested_max_exposure == pytest.approx(42.105)
    capped_a = opportunity_a.model_copy(update={"suggested_max_exposure": 157.89375})
    return engine, opportunity_a, capped_a, opportunity_b


def test_event_capped_allocation_cannot_increase_before_delivery(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "h3.db")
    repository.initialize()
    observed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    engine, uncapped_a, capped_a, opportunity_b = _event_entry_pair(repository, observed_at)
    repository.save_evaluation(capped_a.forecast, capped_a, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    repository.save_evaluation(
        opportunity_b.forecast, opportunity_b, ledger_kind=EVIDENCE_FORWARD_SHADOW
    )

    authorize_delivery(repository, capped_a, allow_unapproved=True, as_of=observed_at)
    reduced = capped_a.model_copy(update={"suggested_max_exposure": 100.0})
    authorize_delivery(repository, reduced, allow_unapproved=True, as_of=observed_at)

    mutated = capped_a.model_copy(update={"suggested_max_exposure": 199.99875})
    with pytest.raises(ValueError, match="persisted event-capped allocation"):
        authorize_delivery(repository, mutated, allow_unapproved=True, as_of=observed_at)
    assert mutated.suggested_max_exposure == pytest.approx(199.99875)
    assert capped_a.suggested_max_exposure == pytest.approx(157.89375)
    combined = (
        repository.forward_event_entry_exposure(
            event_id="SHARED-EVENT",
            exclude_forecast_id=str(capped_a.forecast.forecast_id),
        )
        + mutated.suggested_max_exposure
    )
    assert combined == pytest.approx(242.10375)
    assert combined > engine.event_exposure_cap

    uncapped_repo = SQLiteRepository(tmp_path / "h3-uncapped.db")
    uncapped_repo.initialize()
    _, uncapped_again, _, sibling = _event_entry_pair(uncapped_repo, observed_at)
    uncapped_repo.save_evaluation(
        uncapped_again.forecast, uncapped_again, ledger_kind=EVIDENCE_FORWARD_SHADOW
    )
    uncapped_repo.save_evaluation(sibling.forecast, sibling, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    with pytest.raises(ValueError, match="aggregate event exposure cap"):
        authorize_delivery(uncapped_repo, uncapped_again, allow_unapproved=True, as_of=observed_at)
