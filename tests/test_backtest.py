import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from prediction_market_system.backtest import (
    BacktestConfig,
    EffectiveFee,
    HistoricalBacktester,
    HistoricalMarketData,
    effective_fee_at,
    kalshi_taker_fee,
    walk_forward_folds,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.research import (
    ResearchContext,
    SpotCandle,
    calculate_realized_volatility,
)
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import (
    KalshiBidAskDistribution,
    KalshiCandlestick,
    KalshiEventFeeChange,
    KalshiMarket,
    KalshiPriceDistribution,
    KalshiSeriesFeeChange,
)

START = datetime(2025, 1, 1, tzinfo=UTC)


def market() -> KalshiMarket:
    return KalshiMarket(
        ticker="KXBTC-SMOKE-T100",
        event_ticker="KXBTC-SMOKE",
        market_type="binary",
        title="Will BTC be above $100?",
        yes_sub_title="$100 or above",
        no_sub_title="Below $100",
        close_time=START + timedelta(days=2),
        expected_expiration_time=START + timedelta(days=2),
        latest_expiration_time=START + timedelta(days=2, hours=1),
        status="settled",
        notional_value_dollars=Decimal("1.00"),
        can_close_early=False,
        strike_type="greater_or_equal",
        floor_strike=100.0,
        rules_primary="Resolves YES if BTC is at least $100 at expiry.",
        yes_bid_dollars=Decimal("0.20"),
        yes_ask_dollars=Decimal("0.22"),
        no_bid_dollars=Decimal("0.78"),
        no_ask_dollars=Decimal("0.80"),
        yes_bid_size_fp=Decimal("500"),
        yes_ask_size_fp=Decimal("500"),
        open_time=START,
        result="yes",
        settlement_value_dollars=Decimal("1.00"),
        settlement_ts=START + timedelta(days=2, minutes=5),
        expiration_value="120.00",
    )


def quote_distribution(
    open_value: str,
    low: str,
    high: str,
    close: str,
) -> KalshiBidAskDistribution:
    return KalshiBidAskDistribution(
        open=Decimal(open_value),
        low=Decimal(low),
        high=Decimal(high),
        close=Decimal(close),
    )


def candle(
    at: datetime,
    *,
    bid: tuple[str, str, str, str],
    ask: tuple[str, str, str, str],
    volume: str,
) -> KalshiCandlestick:
    return KalshiCandlestick(
        end_period_ts=int(at.timestamp()),
        yes_bid=quote_distribution(*bid),
        yes_ask=quote_distribution(*ask),
        price=KalshiPriceDistribution(),
        volume=Decimal(volume),
        open_interest=Decimal("1000"),
    )


def historical_market() -> HistoricalMarketData:
    history = HistoricalMarketData(
        series_ticker="KXBTC",
        market=market(),
        candlesticks=(
            candle(
                START + timedelta(days=1, hours=1),
                bid=("0.18", "0.18", "0.20", "0.20"),
                ask=("0.20", "0.20", "0.22", "0.22"),
                volume="500",
            ),
            candle(
                START + timedelta(days=1, hours=2),
                bid=("0.20", "0.19", "0.22", "0.21"),
                ask=("0.22", "0.22", "0.25", "0.24"),
                volume="100",
            ),
        ),
        series_fee_changes=(
            KalshiSeriesFeeChange(
                id="series-fee",
                series_ticker="KXBTC",
                fee_type="quadratic",
                fee_multiplier=0.07,
                scheduled_ts=START,
            ),
        ),
        event_fee_changes=(),
    )
    return with_provenance(history)


def with_provenance(history: HistoricalMarketData) -> HistoricalMarketData:
    """Synthetic forward-collected evidence; no resolved labels in metadata."""
    metadata = history.market.model_copy(
        update={
            "status": "active",
            "result": "",
            "settlement_ts": None,
            "settlement_value_dollars": None,
            "expiration_value": None,
        }
    )
    return history.model_copy(
        update={
            "metadata_observed_at": None,
            "metadata_snapshots": ((START, metadata),),
            "outcome_available_at": history.market.settlement_ts,
            "candlestick_revisions": tuple(
                (datetime.fromtimestamp(item.end_period_ts, UTC), item)
                for item in history.candlesticks
            ),
            "series_fee_revisions": tuple((START, item) for item in history.series_fee_changes),
            "event_fee_revisions": tuple((START, item) for item in history.event_fee_changes),
        }
    )


class FixedResearchSource:
    def research_context_as_of(
        self,
        *,
        symbol: str,
        as_of: datetime,
        event_ticker: str | None = None,
        interval_seconds: int = 3600,
        realized_window_seconds: int = 30 * 24 * 60 * 60,
        spot_interval_seconds: int | None = None,
        spot_max_age_seconds: int | None = None,
        optional_max_age_seconds: int = 2 * 60 * 60,
        event_max_age_seconds: int = 6 * 60 * 60,
    ) -> ResearchContext:
        del spot_max_age_seconds, optional_max_age_seconds, event_max_age_seconds
        decision_interval = (
            interval_seconds if spot_interval_seconds is None else spot_interval_seconds
        )
        spot = SpotCandle(
            provider="coinbase",
            product_id=f"{symbol}-USD",
            interval_seconds=decision_interval,
            start_at=as_of - timedelta(seconds=decision_interval),
            end_at=as_of,
            open=Decimal("119"),
            high=Decimal("121"),
            low=Decimal("118"),
            close=Decimal("120"),
            volume=Decimal("10"),
            retrieved_at=as_of,
            raw_payload={},
        )
        boundary = datetime.fromtimestamp(
            int(as_of.timestamp()) // interval_seconds * interval_seconds, UTC
        )
        research_template = SpotCandle(
            provider="coinbase",
            product_id=f"{symbol}-USD",
            interval_seconds=interval_seconds,
            start_at=boundary - timedelta(seconds=interval_seconds),
            end_at=boundary,
            open=Decimal("119"),
            high=Decimal("121"),
            low=Decimal("118"),
            close=Decimal("120"),
            volume=Decimal("10"),
            retrieved_at=as_of,
            raw_payload={},
        )
        source_candles = tuple(
            research_template.model_copy(
                update={
                    "start_at": boundary - timedelta(seconds=(index + 1) * interval_seconds),
                    "end_at": boundary - timedelta(seconds=index * interval_seconds),
                    "close": Decimal("120") if index % 2 == 0 else Decimal("120.1"),
                }
            )
            for index in range(realized_window_seconds // interval_seconds + 1)
        )
        volatility = calculate_realized_volatility(
            source_candles,
            symbol=symbol,
            as_of=as_of,
            window_seconds=realized_window_seconds,
        )
        return ResearchContext(
            symbol=symbol,
            event_ticker=event_ticker,
            as_of=as_of,
            spot=spot,
            realized_volatility=volatility,
        )


def config() -> BacktestConfig:
    return BacktestConfig(
        series_ticker="KXBTC",
        symbol="BTC",
        start=START,
        end=START + timedelta(days=3),
        period_minutes=60,
        realized_window_days=1,
        train_days=1,
        test_days=1,
        step_days=1,
        latency_seconds=30,
        max_volume_participation=0.10,
        require_calibration=False,
    )


def test_walk_forward_folds_are_chronological_and_non_overlapping() -> None:
    folds = walk_forward_folds(config())

    assert len(folds) == 2
    assert folds[0].train_start == START
    assert folds[0].train_end == START + timedelta(days=1)
    assert folds[0].test_start == folds[0].train_end
    assert folds[0].test_end == folds[1].test_start

    with pytest.raises(ValueError, match="overlapping test sets"):
        BacktestConfig(**(config().model_dump() | {"step_days": 1, "test_days": 2}))


def test_effective_fee_honors_event_override_and_explicit_clear() -> None:
    base = historical_market()
    override = KalshiEventFeeChange(
        id="override",
        event_ticker=base.market.event_ticker,
        series_ticker=base.series_ticker,
        fee_type_override="flat",
        fee_multiplier_override=0.01,
        scheduled_ts=START + timedelta(hours=1),
    )
    clear = KalshiEventFeeChange(
        id="clear",
        event_ticker=base.market.event_ticker,
        series_ticker=base.series_ticker,
        fee_type_override=None,
        fee_multiplier_override=None,
        scheduled_ts=START + timedelta(hours=2),
    )
    with_overrides = with_provenance(
        base.model_copy(update={"event_fee_changes": (override, clear)})
    )

    assert effective_fee_at(with_overrides, START + timedelta(minutes=30)) == EffectiveFee(
        fee_type="quadratic",
        multiplier=0.07,
    )
    assert effective_fee_at(with_overrides, START + timedelta(hours=1, minutes=30)) == (
        EffectiveFee(fee_type="flat", multiplier=0.01)
    )
    assert effective_fee_at(with_overrides, START + timedelta(hours=2, minutes=30)) == (
        EffectiveFee(fee_type="quadratic", multiplier=0.07)
    )
    assert kalshi_taker_fee(10, 0.25, EffectiveFee(fee_type="quadratic", multiplier=0.07)) == (
        Decimal("0.14")
    )


def test_backtest_uses_delayed_adverse_quote_partial_fill_and_persists(
    tmp_path: Path,
) -> None:
    backtester = HistoricalBacktester(
        FixedResearchSource(),
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            minimum_ask_size=10,
        ),
    )

    result = backtester.run(config(), (historical_market(),))

    assert result.total_trades == 1
    assert result.partial_fills == 1
    trade = result.folds[0].trades[0]
    assert trade.signal_at == START + timedelta(days=1, hours=1)
    assert trade.executed_at == START + timedelta(days=1, hours=2)
    assert trade.execution_price == pytest.approx(0.25)
    assert trade.filled_contracts == 10
    assert trade.uncertainty_margin == pytest.approx(0.03)
    assert trade.uncertainty_source == "fixed"
    assert trade.calibration_profile_id is None
    assert trade.fee_dollars == Decimal("0.14")
    assert trade.cost_dollars == Decimal("2.64625")
    assert trade.pnl_dollars == Decimal("7.35375")

    database_path = tmp_path / "backtest.db"
    repository = SQLiteRepository(database_path)
    repository.initialize()
    repository.save_backtest_result(result)
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT result_json FROM backtest_runs").fetchone()

    assert row is not None
    persisted = json.loads(row[0])
    assert persisted["run_id"] == str(result.run_id)
    assert persisted["total_trades"] == 1


def test_backtest_uses_configured_fee_when_schedule_has_no_changes() -> None:
    historical = historical_market().model_copy(
        update={"series_fee_changes": (), "event_fee_changes": (), "series_fee_revisions": ()}
    )
    result = HistoricalBacktester(
        FixedResearchSource(),
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            minimum_ask_size=10,
            binary_fee_coefficient=0.07,
        ),
    ).run(config(), (historical,))

    assert result.folds[0].missing_fee_signals == 0
    assert result.total_trades == 1
    assert result.folds[0].trades[0].fee_dollars == Decimal("0.14")


def test_backtest_rejects_touch_without_contractual_path_history() -> None:
    historical = historical_market()
    touch_market = historical.market.model_copy(
        update={
            "can_close_early": True,
            "rules_primary": ("Resolves YES if BTC reaches $100 at any time before expiry."),
        }
    )
    touch_history = with_provenance(historical.model_copy(update={"market": touch_market}))
    backtester = HistoricalBacktester(
        FixedResearchSource(),
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            minimum_ask_size=10,
        ),
    )

    result = backtester.run(config(), (touch_history,))

    assert result.total_trades == 0
    assert result.forecasts == ()
    assert result.incomplete_forecast_signals == 2


def test_backtest_calibrates_and_replays_terminal_ranges_by_event() -> None:
    training = historical_market()
    training_expiry = START + timedelta(hours=12)
    training_market = training.market.model_copy(
        update={
            "ticker": "KXBTC-TRAIN-B110",
            "event_ticker": "KXBTC-TRAIN",
            "title": "BTC range",
            "yes_sub_title": "$110 to $129.99",
            "no_sub_title": "Outside range",
            "strike_type": "between",
            "floor_strike": 110.0,
            "cap_strike": 129.99,
            "can_close_early": False,
            "rules_primary": "Resolves YES if BTC at expiry is between 110 and 129.99.",
            "close_time": training_expiry,
            "expected_expiration_time": training_expiry,
            "latest_expiration_time": training_expiry + timedelta(hours=1),
            "settlement_ts": training_expiry + timedelta(minutes=5),
            "result": "yes",
        }
    )
    training = training.model_copy(
        update={
            "market": training_market,
            "candlesticks": (
                candle(
                    START + timedelta(hours=1),
                    bid=("0.18", "0.18", "0.20", "0.20"),
                    ask=("0.20", "0.20", "0.22", "0.22"),
                    volume="500",
                ),
                candle(
                    START + timedelta(hours=2),
                    bid=("0.20", "0.19", "0.22", "0.21"),
                    ask=("0.22", "0.22", "0.25", "0.24"),
                    volume="100",
                ),
            ),
        }
    )
    training = with_provenance(training)
    test_history = historical_market()
    test_market = test_history.market.model_copy(
        update={
            "ticker": "KXBTC-TEST-B110",
            "event_ticker": "KXBTC-TEST",
            "title": "BTC range",
            "yes_sub_title": "$110 to $129.99",
            "no_sub_title": "Outside range",
            "strike_type": "between",
            "floor_strike": 110.0,
            "cap_strike": 129.99,
            "can_close_early": False,
            "rules_primary": "Resolves YES if BTC at expiry is between 110 and 129.99.",
        }
    )
    test_history = with_provenance(test_history.model_copy(update={"market": test_market}))
    calibrated = BacktestConfig(
        **(
            config().model_dump()
            | {
                "require_calibration": True,
                "minimum_calibration_samples": 1,
                "maximum_calibration_bins": 1,
            }
        )
    )

    result = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(
        calibrated,
        (training, test_history),
    )

    first_fold = result.folds[0]
    range_profile = next(
        profile
        for profile in first_fold.calibration_profiles
        if profile.model_name == "crypto-terminal-range-market-anchor"
    )
    assert range_profile.independent_event_count == 1
    assert test_market.ticker not in result.unsupported_markets
    assert first_fold.evaluated_signals > 0


def test_walk_forward_calibration_uses_only_resolved_training_markets(
    tmp_path: Path,
) -> None:
    training_histories = []
    for index, result_value in enumerate(("no", "yes", "yes"), start=1):
        historical = historical_market()
        signal_at = START + timedelta(hours=index)
        expiry = START + timedelta(hours=12)
        training_market = historical.market.model_copy(
            update={
                "ticker": f"TRAIN-{index}",
                "event_ticker": f"TRAIN-{index}-EVENT",
                "close_time": expiry,
                "expected_expiration_time": expiry,
                "latest_expiration_time": expiry + timedelta(hours=1),
                "result": result_value,
                "settlement_value_dollars": (
                    Decimal("1.00") if result_value == "yes" else Decimal("0.00")
                ),
                "settlement_ts": START + timedelta(hours=13),
            }
        )
        training_histories.append(
            historical.model_copy(
                update={
                    "market": training_market,
                    "candlesticks": (
                        candle(
                            signal_at,
                            bid=("0.18", "0.18", "0.20", "0.20"),
                            ask=("0.20", "0.20", "0.22", "0.22"),
                            volume="500",
                        ),
                    ),
                }
            )
        )
    training_histories = [with_provenance(history) for history in training_histories]

    future_outcome = training_histories[0].model_copy(
        update={
            "outcome_available_at": START + timedelta(days=1, seconds=1),
            "market": training_histories[0].market.model_copy(
                update={
                    "ticker": "FUTURE-OUTCOME",
                    "settlement_ts": START + timedelta(days=1, seconds=1),
                }
            ),
        }
    )

    calibrated_config = BacktestConfig(
        **(
            config().model_dump()
            | {
                "require_calibration": True,
                "minimum_calibration_samples": 3,
                "maximum_calibration_bins": 1,
            }
        )
    )
    backtester = HistoricalBacktester(FixedResearchSource(), EngineConfig())

    result = backtester.run(
        calibrated_config,
        (*training_histories, future_outcome, historical_market()),
    )

    first_fold = result.folds[0]
    assert len(first_fold.calibration_profiles) == 1
    profile = first_fold.calibration_profiles[0]
    assert profile.sample_count == 3
    assert profile.cutoff_at == first_fold.fold.test_start
    assert first_fold.missing_calibration_signals == 0
    assert first_fold.evaluated_signals > 0
    assert all(trade.uncertainty_source == "held_out" for trade in first_fold.trades)

    repository = SQLiteRepository(tmp_path / "calibrated-backtest.db")
    repository.initialize()
    repository.save_backtest_result(result)
    loaded = repository.latest_uncertainty_calibration(
        symbol="BTC",
        model_name=profile.model_name,
        model_version=profile.model_version,
        as_of=first_fold.fold.test_start,
        recipe_id=profile.recipe_id,
    )
    assert loaded is None  # Fitted now, not actually generated at the historical boundary.
    assert repository.uncertainty_calibration(profile.profile_id) == profile


def test_forecast_population_includes_watch_and_post_fill_observations() -> None:
    history = historical_market()
    source = FixedResearchSource()
    traded = HistoricalBacktester(
        source, EngineConfig(structural_weight=0.7, uncertainty_margin=0.03)
    ).run(config(), (history,))
    watched = HistoricalBacktester(
        source, EngineConfig(structural_weight=0.7, uncertainty_margin=0.03, minimum_ask_size=10000)
    ).run(config(), (history,))
    assert traded.total_trades == 1
    assert watched.total_trades == 0
    assert len(traded.forecasts) == len(watched.forecasts) == 2
    assert [record.probability_yes for record in traded.forecasts] == [
        record.probability_yes for record in watched.forecasts
    ]
    assert traded.brier_score == watched.brier_score
    assert traded.log_loss == watched.log_loss
    assert traded.market_brier_score == pytest.approx((0.79**2 + 0.775**2) / 2)
    assert traded.incomplete_forecast_signals == 0
    assert all(record.input_population_id for record in traded.forecasts)


def test_unknown_or_late_metadata_never_reconstructs_historical_features() -> None:
    history = historical_market()
    missing = history.model_copy(update={"metadata_snapshots": ()})
    late = history.model_copy(update={"metadata_snapshots": ((config().end, history.market),)})
    replay = HistoricalBacktester(FixedResearchSource(), EngineConfig())
    for inadmissible in (missing, late):
        result = replay.run(config(), (inadmissible,))
        assert result.forecasts == ()
        assert result.missing_metadata_signals == 2
        assert result.incomplete_forecast_signals == 2
    revised = history.model_copy(
        update={
            "market": history.market.model_copy(
                update={"floor_strike": 10000, "rules_primary": "Ambiguous final metadata"}
            )
        }
    )
    original_result = replay.run(config(), (history,))
    revised_result = replay.run(config(), (revised,))
    assert [r.probability_yes for r in revised_result.forecasts] == [
        r.probability_yes for r in original_result.forecasts
    ]


def test_delayed_candle_availability_sets_real_signal_time() -> None:
    history = historical_market()
    available_at, first = history.candlestick_revisions[0]
    delayed = history.model_copy(
        update={
            "candlestick_revisions": (
                (available_at + timedelta(seconds=30), first),
                history.candlestick_revisions[1],
            )
        }
    )
    result = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(config(), (delayed,))
    assert result.forecasts[0].observed_at == available_at + timedelta(seconds=30)
    unknown = history.model_copy(update={"candlestick_revisions": ()})
    rejected = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(config(), (unknown,))
    assert rejected.forecasts == ()
    assert rejected.incomplete_forecast_signals == 2


def test_late_outcomes_remain_unscored_not_silently_dropped() -> None:
    history = historical_market().model_copy(
        update={"outcome_available_at": config().end + timedelta(seconds=1)}
    )
    replay = HistoricalBacktester(FixedResearchSource(), EngineConfig())
    result = replay.run(config(), (history,))
    assert len(result.forecasts) == 2
    assert result.missing_outcome_signals == 2
    assert result.brier_score is None
    assert result.market_brier_score is None
    assert all(not validation.accepted_for_paper_alerts for validation in result.model_validations)
    assert (
        replay.calibrate(config(), (history,), training_start=START, cutoff_at=config().end) == ()
    )


def test_fee_revision_must_be_known_not_merely_scheduled() -> None:
    history = historical_market()
    fee = history.series_fee_changes[0]
    future = history.model_copy(
        update={"series_fee_revisions": ((START + timedelta(days=2), fee),)}
    )
    assert effective_fee_at(future, START + timedelta(days=1)) is None
    assert effective_fee_at(future, START + timedelta(days=2)) == EffectiveFee(
        fee_type="quadratic", multiplier=0.07
    )
    result = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(config(), (future,))
    assert len(result.forecasts) == 2
    assert result.total_trades == 0
    assert result.folds[0].missing_fee_signals == 2


def test_fill_budget_includes_rounded_fees_and_configured_costs() -> None:
    from prediction_market_system.backtest import _fill_trade
    from prediction_market_system.domain import MarketSide

    history = historical_market()
    trade = _fill_trade(
        0,
        history.market,
        START,
        START + timedelta(hours=1),
        history.candlesticks[1],
        MarketSide.YES,
        0.22,
        1.00,
        0.9,
        "test",
        0.03,
        "fixed",
        None,
        0.5,
        EffectiveFee(fee_type="quadratic", multiplier=0.07),
        1.0,
        fee_rate=0.1,
        slippage_bps=100,
    )
    assert trade is not None
    assert trade.filled_contracts == 3
    assert trade.cost_dollars <= Decimal("1.00")
    assert trade.cost_dollars == Decimal("0.8725")


def test_event_cap_uses_actual_all_in_spend_across_contracts() -> None:
    first = historical_market()
    second = with_provenance(
        first.model_copy(update={"market": first.market.model_copy(update={"ticker": "SECOND"})})
    )
    settings = EngineConfig(
        structural_weight=0.7,
        uncertainty_margin=0.03,
        paper_bankroll=100,
        max_event_bankroll_fraction=0.05,
        max_bankroll_fraction=0.05,
        slippage_bps=100,
        fee_rate=0.1,
    )
    result = HistoricalBacktester(FixedResearchSource(), settings).run(config(), (first, second))
    assert result.total_trades >= 1
    assert (
        sum((trade.cost_dollars for fold in result.folds for trade in fold.trades), Decimal()) <= 5
    )


def test_profitable_override_and_fixed_evidence_are_never_approved() -> None:
    from prediction_market_system.backtest import _model_validations
    from prediction_market_system.calibration import CalibrationBin, UncertaintyCalibrationProfile

    result = HistoricalBacktester(
        FixedResearchSource(), EngineConfig(structural_weight=0.7, uncertainty_margin=0.03)
    ).run(config(), (historical_market(),))
    first = result.folds[0]
    record = first.forecasts[0]
    profile = UncertaintyCalibrationProfile(
        symbol="BTC",
        model_name=record.model_name,
        model_version=CryptoThresholdEngine.model_version,
        recipe_id=record.recipe_id,
        research_only=False,
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
                maximum_horizon_seconds=31557600,
            ),
        ),
    )
    held_out = first.model_copy(
        update={
            "calibration_profiles": (profile,),
            "forecasts": tuple(
                item.model_copy(
                    update={
                        "uncertainty_source": "held_out",
                        "calibration_profile_id": profile.profile_id,
                    }
                )
                for item in first.forecasts
            ),
            "trades": tuple(
                item.model_copy(
                    update={
                        "uncertainty_source": "held_out",
                        "calibration_profile_id": profile.profile_id,
                    }
                )
                for item in first.trades
            ),
        }
    )
    strict = config().model_copy(
        update={
            "require_calibration": True,
            "minimum_validation_events": 1,
            "minimum_validation_folds": 1,
            "minimum_calibration_samples": 1,
        }
    )
    (valid,) = _model_validations(strict, (held_out,), (profile,))
    assert valid.accepted_for_paper_alerts
    (override,) = _model_validations(
        strict.model_copy(update={"require_calibration": False}), (held_out,), (profile,)
    )
    assert not override.accepted_for_paper_alerts
    (fixed,) = _model_validations(strict, (first,), (profile,))
    assert not fixed.accepted_for_paper_alerts
    (missing,) = _model_validations(
        strict, (held_out.model_copy(update={"incomplete_forecast_signals": 1}),), (profile,)
    )
    assert not missing.accepted_for_paper_alerts
