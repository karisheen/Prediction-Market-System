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
from prediction_market_system.engine import EngineConfig
from prediction_market_system.research import ResearchContext, SpotCandle, VolatilityObservation
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
    return HistoricalMarketData(
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


class FixedResearchSource:
    def research_context_as_of(
        self,
        *,
        symbol: str,
        as_of: datetime,
        event_ticker: str | None = None,
        interval_seconds: int = 3600,
        realized_window_seconds: int = 30 * 24 * 60 * 60,
        spot_max_age_seconds: int | None = None,
        optional_max_age_seconds: int = 2 * 60 * 60,
        event_max_age_seconds: int = 6 * 60 * 60,
    ) -> ResearchContext:
        del spot_max_age_seconds, optional_max_age_seconds, event_max_age_seconds
        spot = SpotCandle(
            provider="coinbase",
            product_id=f"{symbol}-USD",
            interval_seconds=interval_seconds,
            start_at=as_of - timedelta(seconds=interval_seconds),
            end_at=as_of,
            open=Decimal("119"),
            high=Decimal("121"),
            low=Decimal("118"),
            close=Decimal("120"),
            volume=Decimal("10"),
            retrieved_at=as_of,
            raw_payload={},
        )
        volatility = VolatilityObservation(
            provider="coinbase:calculated",
            symbol=symbol,
            kind="realized",
            window_seconds=realized_window_seconds,
            source_start_at=as_of - timedelta(seconds=realized_window_seconds),
            observed_at=as_of,
            annualized_volatility=0.20,
            retrieved_at=as_of,
            raw_payload={},
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
    with_overrides = base.model_copy(update={"event_fee_changes": (override, clear)})

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
    assert trade.requested_contracts == 44
    assert trade.filled_contracts == 10
    assert trade.uncertainty_margin == pytest.approx(0.03)
    assert trade.uncertainty_source == "fixed"
    assert trade.calibration_profile_id is None
    assert trade.fee_dollars == Decimal("0.14")
    assert trade.cost_dollars == Decimal("2.64")
    assert trade.pnl_dollars == Decimal("7.36")

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


def test_backtest_routes_explicit_touch_contract_to_barrier_model() -> None:
    historical = historical_market()
    touch_market = historical.market.model_copy(
        update={
            "can_close_early": True,
            "rules_primary": ("Resolves YES if BTC reaches $100 at any time before expiry."),
        }
    )
    touch_history = historical.model_copy(update={"market": touch_market})
    backtester = HistoricalBacktester(
        FixedResearchSource(),
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            minimum_ask_size=10,
        ),
    )

    result = backtester.run(config(), (touch_history,))

    assert result.total_trades == 1
    assert result.folds[0].trades[0].model_name == ("crypto-barrier-above-threshold-market-anchor")


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

    future_outcome = training_histories[0].model_copy(
        update={
            "market": training_histories[0].market.model_copy(
                update={
                    "ticker": "FUTURE-OUTCOME",
                    "settlement_ts": START + timedelta(days=1, seconds=1),
                }
            )
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
    )
    assert loaded == profile
