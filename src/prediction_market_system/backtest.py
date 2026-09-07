from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from typing import Annotated, Any, Literal, Protocol, Self, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prediction_market_system.calibration import (
    CalibrationSample,
    UncertaintyCalibrationProfile,
    fit_uncertainty_profiles,
)
from prediction_market_system.domain import (
    CryptoPriceContract,
    MarketSide,
    MarketSnapshot,
    Opportunity,
    ProbabilityForecast,
    RecommendationState,
)
from prediction_market_system.engine import BinaryFeeType, CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import EVIDENCE_HISTORICAL
from prediction_market_system.recipe import (
    DECISION_SPOT_INTERVAL_SECONDS,
    DEFAULT_RESEARCH_INTERVAL_SECONDS,
)
from prediction_market_system.research import ResearchContext, ResearchDataUnavailable
from prediction_market_system.venues.kalshi import (
    CandlestickPeriod,
    KalshiCandlestick,
    KalshiEventFeeChange,
    KalshiMarket,
    KalshiSeriesFeeChange,
    UnsupportedMarketError,
)

ChangeT = TypeVar("ChangeT", KalshiSeriesFeeChange, KalshiEventFeeChange)


class BacktestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BacktestConfig(BacktestModel):
    series_ticker: Annotated[str, Field(min_length=1)]
    symbol: Annotated[str, Field(min_length=1)]
    start: datetime
    end: datetime
    period_minutes: CandlestickPeriod = 60
    spot_interval_seconds: Annotated[int, Field(gt=0)] = DECISION_SPOT_INTERVAL_SECONDS
    realized_interval_seconds: Annotated[int, Field(gt=0)] = DEFAULT_RESEARCH_INTERVAL_SECONDS
    realized_window_days: Annotated[int, Field(gt=0)] = 30
    train_days: Annotated[int, Field(gt=0)] = 90
    test_days: Annotated[int, Field(gt=0)] = 30
    step_days: Annotated[int, Field(gt=0)] = 30
    latency_seconds: Annotated[int, Field(ge=0)] = 30
    max_volume_participation: Annotated[float, Field(gt=0.0, le=1.0)] = 0.10
    expected_annual_return: float = 0.0
    require_calibration: bool = True
    minimum_calibration_samples: Annotated[int, Field(gt=0)] = 30
    maximum_calibration_bins: Annotated[int, Field(gt=0, le=20)] = 5
    calibration_confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    calibration_lead_seconds: Annotated[int, Field(ge=0)] = 5 * 60
    minimum_validation_events: Annotated[int, Field(gt=0)] = 20
    minimum_validation_folds: Annotated[int, Field(gt=0)] = 2
    minimum_return_on_cost: float = 0.0
    maximum_brier_score: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25

    @field_validator("start", "end")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("backtest timestamps must include a timezone")
        return value.astimezone(UTC)

    @field_validator("series_ticker", "symbol")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.end <= self.start:
            raise ValueError("backtest end must follow start")
        if self.step_days < self.test_days:
            raise ValueError(
                "step_days must be at least test_days to prevent overlapping test sets"
            )
        first_test = self.start + timedelta(days=self.train_days)
        if first_test >= self.end:
            raise ValueError("backtest range must include at least one complete training boundary")
        return self


class WalkForwardFold(BacktestModel):
    index: Annotated[int, Field(ge=0)]
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class EffectiveFee(BacktestModel):
    fee_type: BinaryFeeType
    multiplier: Annotated[float, Field(ge=0.0)]


class HistoricalMarketData(BacktestModel):
    series_ticker: str
    market: KalshiMarket
    candlesticks: tuple[KalshiCandlestick, ...]
    series_fee_changes: tuple[KalshiSeriesFeeChange, ...]
    event_fee_changes: tuple[KalshiEventFeeChange, ...]
    metadata_observed_at: datetime | None = None
    metadata_snapshots: tuple[tuple[datetime, KalshiMarket], ...] = ()
    outcome_available_at: datetime | None = None
    candlestick_revisions: tuple[tuple[datetime, KalshiCandlestick], ...] = ()
    series_fee_revisions: tuple[tuple[datetime, KalshiSeriesFeeChange], ...] = ()
    event_fee_revisions: tuple[tuple[datetime, KalshiEventFeeChange], ...] = ()


class BacktestTrade(BacktestModel):
    fold_index: int
    event_ticker: str
    ticker: str
    signal_at: datetime
    executed_at: datetime
    side: MarketSide
    signal_price: float
    execution_price: float
    requested_contracts: int
    filled_contracts: int
    partial_fill: bool
    probability_yes: float
    model_name: str
    uncertainty_margin: float
    uncertainty_source: Literal["fixed", "held_out"]
    calibration_profile_id: UUID | None
    conservative_net_edge: float
    fee_type: BinaryFeeType
    fee_multiplier: float
    fee_dollars: Decimal
    cost_dollars: Decimal
    payout_dollars: Decimal
    pnl_dollars: Decimal
    result: Literal["yes", "no"]
    recipe_id: str | None = None
    model_version: str | None = None


class BacktestForecast(BacktestModel):
    fold_index: int
    event_ticker: str
    ticker: str
    observed_at: datetime
    model_name: str
    model_version: str
    recipe_id: str | None = None
    probability_yes: float
    market_probability_yes: float
    outcome_yes: bool | None = None
    outcome_available_at: datetime | None = None
    uncertainty_source: Literal["fixed", "held_out"]
    calibration_profile_id: UUID | None = None
    input_population_id: str = ""
    forecast: ProbabilityForecast


class BacktestFoldResult(BacktestModel):
    fold: WalkForwardFold
    markets_considered: int
    evaluated_signals: int
    missing_context_signals: int
    missing_fee_signals: int
    missing_calibration_signals: int
    calibration_profiles: tuple[UncertaintyCalibrationProfile, ...]
    trades: tuple[BacktestTrade, ...]
    total_cost_dollars: Decimal
    total_pnl_dollars: Decimal
    return_on_cost: float | None
    brier_score: float | None
    forecasts: tuple[BacktestForecast, ...] = ()
    eligible_signals: int = 0
    missing_metadata_signals: int = 0
    missing_outcome_signals: int = 0
    incomplete_forecast_signals: int = 0
    log_loss: float | None = None
    market_brier_score: float | None = None
    market_log_loss: float | None = None


class BacktestModelValidation(BacktestModel):
    model_name: str
    model_version: str
    calibration_profile_id: UUID | None
    independent_calibration_events: int
    held_out_events: int
    held_out_folds: int
    held_out_trades: int
    total_cost_dollars: Decimal
    total_pnl_dollars: Decimal
    return_on_cost: float | None
    brier_score: float | None
    accepted_for_paper_alerts: bool
    rejection_reasons: tuple[str, ...]
    recipe_id: str | None = None
    held_out_forecasts: int = 0
    incomplete_forecast_signals: int = 0
    log_loss: float | None = None
    market_brier_score: float | None = None
    market_log_loss: float | None = None
    campaign_id: str | None = None
    deployment_policy_id: str | None = None


class BacktestResult(BacktestModel):
    run_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: BacktestConfig
    folds: tuple[BacktestFoldResult, ...]
    unsupported_markets: tuple[str, ...]
    unresolved_markets: tuple[str, ...]
    markets_without_candles: tuple[str, ...]
    total_trades: int
    partial_fills: int
    total_cost_dollars: Decimal
    total_pnl_dollars: Decimal
    return_on_cost: float | None
    brier_score: float | None
    deployment_profiles: tuple[UncertaintyCalibrationProfile, ...]
    model_validations: tuple[BacktestModelValidation, ...]
    forecasts: tuple[BacktestForecast, ...] = ()
    eligible_signals: int = 0
    missing_metadata_signals: int = 0
    missing_outcome_signals: int = 0
    incomplete_forecast_signals: int = 0
    log_loss: float | None = None
    market_brier_score: float | None = None
    market_log_loss: float | None = None
    engine_config: EngineConfig | None = None
    input_manifest: dict[str, Any] = Field(default_factory=dict)


class BacktestResearchSource(Protocol):
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
    ) -> ResearchContext: ...


def walk_forward_folds(config: BacktestConfig) -> tuple[WalkForwardFold, ...]:
    folds: list[WalkForwardFold] = []
    test_start = config.start + timedelta(days=config.train_days)
    step = timedelta(days=config.step_days)
    train_window = timedelta(days=config.train_days)
    test_window = timedelta(days=config.test_days)
    while test_start < config.end:
        test_end = min(test_start + test_window, config.end)
        folds.append(
            WalkForwardFold(
                index=len(folds),
                train_start=test_start - train_window,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start += step
    return tuple(folds)


def effective_fee_at(market: HistoricalMarketData, as_of: datetime) -> EffectiveFee | None:
    series_change = _latest_known_change(market.series_fee_revisions, as_of)
    event_change = _latest_known_change(market.event_fee_revisions, as_of)
    fee_type = series_change.fee_type if series_change is not None else None
    multiplier = series_change.fee_multiplier if series_change is not None else None
    if event_change is not None and event_change.fee_type_override is not None:
        fee_type = event_change.fee_type_override
        multiplier = event_change.fee_multiplier_override
    if fee_type is None or multiplier is None:
        return None
    return EffectiveFee(fee_type=fee_type, multiplier=multiplier)


def kalshi_taker_fee(
    contracts: int,
    price: float,
    fee: EffectiveFee,
) -> Decimal:
    if contracts <= 0:
        return Decimal("0.00")
    price_decimal = Decimal(str(price))
    multiplier = Decimal(str(fee.multiplier))
    if fee.fee_type == "flat":
        raw_fee = multiplier * contracts
    else:
        raw_fee = multiplier * contracts * price_decimal * (Decimal("1") - price_decimal)
    return raw_fee.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def _latest_known_change(
    revisions: tuple[tuple[datetime, ChangeT], ...],
    as_of: datetime,
) -> ChangeT | None:
    # A revised schedule supersedes that provider ID even if its new effective
    # time is later. Selecting schedule first would resurrect the old revision.
    latest_by_id: dict[str, tuple[datetime, ChangeT]] = {}
    for available_at, change in sorted(revisions, key=lambda item: item[0]):
        if available_at <= as_of:
            latest_by_id[change.id] = (available_at, change)
    eligible = (item for item in latest_by_id.values() if item[1].scheduled_ts <= as_of)
    latest = max(eligible, key=lambda item: (item[1].scheduled_ts, item[0]), default=None)
    return None if latest is None else latest[1]


def _metadata_at(history: HistoricalMarketData, as_of: datetime) -> KalshiMarket | None:
    snapshots = history.metadata_snapshots
    if not snapshots and history.metadata_observed_at is not None:
        snapshots += ((history.metadata_observed_at, history.market),)
    latest = max(
        (item for item in snapshots if item[0] <= as_of),
        key=lambda item: item[0],
        default=None,
    )
    if (
        latest is not None
        and len({item.model_dump_json() for available, item in snapshots if available == latest[0]})
        != 1
    ):
        return None
    return None if latest is None else latest[1]


def _available_signals(
    history: HistoricalMarketData,
) -> tuple[tuple[datetime, KalshiCandlestick], ...]:
    signals: dict[tuple[datetime, int], KalshiCandlestick] = {}
    for available_at, candle in history.candlestick_revisions:
        key = (
            max(available_at, datetime.fromtimestamp(candle.end_period_ts, UTC)),
            candle.end_period_ts,
        )
        if key in signals and signals[key] != candle:
            raise ValueError("conflicting candle revisions have ambiguous availability order")
        signals[key] = candle
    return tuple((key[0], candle) for key, candle in sorted(signals.items()))


@dataclass(frozen=True)
class _BacktestCandidate:
    historical_market: HistoricalMarketData
    signal_market: KalshiMarket
    signal_at: datetime
    execution_at: datetime
    execution_candle: KalshiCandlestick
    opportunity: Opportunity
    forecast: ProbabilityForecast
    execution_fee: EffectiveFee


class HistoricalBacktester:
    def __init__(
        self,
        research_source: BacktestResearchSource,
        engine_config: EngineConfig,
    ) -> None:
        self._research_source = research_source
        self._engine_config = engine_config

    def _research_context(
        self,
        config: BacktestConfig,
        *,
        as_of: datetime,
        event_ticker: str,
    ) -> ResearchContext:
        return self._research_source.research_context_as_of(
            symbol=config.symbol,
            as_of=as_of,
            event_ticker=event_ticker,
            interval_seconds=config.realized_interval_seconds,
            spot_interval_seconds=config.spot_interval_seconds,
            realized_window_seconds=config.realized_window_days * 24 * 60 * 60,
            spot_max_age_seconds=self._engine_config.maximum_input_age_seconds,
        )

    def _record_historical_evaluation(
        self,
        context: ResearchContext,
        forecast: ProbabilityForecast,
        opportunity: Opportunity,
    ) -> None:
        save_context = getattr(self._research_source, "save_research_context", None)
        save_eval = getattr(self._research_source, "save_evaluation", None)
        if save_context is not None:
            save_context(context)
        if save_eval is not None:
            save_eval(forecast, opportunity, ledger_kind=EVIDENCE_HISTORICAL)

    def run(
        self,
        config: BacktestConfig,
        markets: tuple[HistoricalMarketData, ...],
    ) -> BacktestResult:
        unsupported: set[str] = set()
        unresolved: set[str] = set()
        without_candles: set[str] = set()
        traded_markets: set[str] = set()
        remaining_by_event: dict[str, float] = {}
        fold_results: list[BacktestFoldResult] = []
        configured_fee = EffectiveFee(
            fee_type=("flat" if self._engine_config.binary_fee_type == "flat" else "quadratic"),
            multiplier=self._engine_config.binary_fee_coefficient,
        )

        for fold in walk_forward_folds(config):
            calibration_profiles = self.calibrate(
                config,
                markets,
                training_start=fold.train_start,
                cutoff_at=fold.train_end,
            )
            profiles_by_recipe = {
                (profile.model_name, profile.recipe_id): profile for profile in calibration_profiles
            }
            signals_by_time: dict[
                datetime,
                list[
                    tuple[
                        HistoricalMarketData, KalshiMarket, CryptoPriceContract, KalshiCandlestick
                    ]
                ],
            ] = {}
            considered_markets: set[str] = set()
            eligible_signals = 0
            missing_metadata = 0
            for historical_market in markets:
                identity = historical_market.market
                if identity.result not in {"yes", "no"}:
                    unresolved.add(identity.ticker)
                if not historical_market.candlesticks:
                    without_candles.add(identity.ticker)
                if not historical_market.candlestick_revisions:
                    missing = sum(
                        fold.test_start
                        <= datetime.fromtimestamp(c.end_period_ts, UTC)
                        < fold.test_end
                        for c in historical_market.candlesticks
                    )
                    eligible_signals += missing
                    missing_metadata += missing
                for signal_at, candle in _available_signals(historical_market):
                    if not fold.test_start <= signal_at < fold.test_end:
                        continue
                    market = _metadata_at(historical_market, signal_at)
                    if market is None:
                        eligible_signals += 1
                        missing_metadata += 1
                        continue
                    try:
                        contract = market.evaluation_contract(signal_at)
                        horizon = min(market.expiry, market.observation_end_at)
                    except (UnsupportedMarketError, ValueError):
                        eligible_signals += 1
                        missing_metadata += 1
                        unsupported.add(market.ticker)
                        continue
                    if signal_at >= horizon:
                        continue
                    eligible_signals += 1
                    considered_markets.add(market.ticker)
                    signals_by_time.setdefault(signal_at, []).append(
                        (historical_market, market, contract, candle)
                    )
            trades: list[BacktestTrade] = []
            forecasts: list[BacktestForecast] = []
            evaluated_signals = 0
            missing_context = 0
            missing_fee = 0
            missing_calibration = 0
            for signal_at, cycle_signals in sorted(signals_by_time.items()):
                candidates: list[_BacktestCandidate] = []
                for historical_market, market, contract, signal_candle in cycle_signals:
                    quote_age = (
                        signal_at - datetime.fromtimestamp(signal_candle.end_period_ts, UTC)
                    ).total_seconds()
                    if quote_age > self._engine_config.maximum_input_age_seconds:
                        missing_context += 1
                        continue
                    fee = self._fee_at(historical_market, signal_at, configured_fee)
                    if fee is None:
                        missing_fee += 1
                    # Costs do not generate probabilities. Retain the forecast
                    # population even when execution evidence is incomplete.
                    evaluation_fee = fee or configured_fee
                    try:
                        context = self._research_context(
                            config, as_of=signal_at, event_ticker=market.event_ticker
                        )
                        context.validate_at(
                            signal_at,
                            maximum_spot_age_seconds=self._engine_config.maximum_input_age_seconds,
                        )
                    except (ResearchDataUnavailable, ValueError):
                        missing_context += 1
                        continue
                    snapshot = _market_snapshot(market, signal_candle, config, signal_at=signal_at)
                    engine = CryptoThresholdEngine(
                        self._engine_config.model_copy(
                            update={
                                "binary_fee_type": evaluation_fee.fee_type,
                                "binary_fee_coefficient": evaluation_fee.multiplier,
                            }
                        )
                    )
                    crypto = context.to_crypto_snapshot(
                        strike_price=engine.reference_price(contract),
                        expected_annual_return=config.expected_annual_return,
                    )
                    recipe_id = engine.recipe_id(crypto)
                    calibration_profile = profiles_by_recipe.get(
                        (engine.model_name(contract), recipe_id)
                    )
                    if calibration_profile is None:
                        missing_calibration += 1
                    try:
                        forecast, opportunity = engine.evaluate(
                            snapshot, crypto, contract, calibration_profile
                        )
                    except ValueError:
                        missing_context += 1
                        continue
                    self._record_historical_evaluation(context, forecast, opportunity)
                    evaluated_signals += 1
                    label_available = (
                        historical_market.outcome_available_at is not None
                        and historical_market.outcome_available_at <= config.end
                        and historical_market.market.result in {"yes", "no"}
                    )
                    forecasts.append(
                        BacktestForecast(
                            fold_index=fold.index,
                            event_ticker=market.event_ticker,
                            observed_at=signal_at,
                            ticker=market.ticker,
                            model_name=forecast.model_name,
                            model_version=forecast.model_version,
                            recipe_id=recipe_id,
                            probability_yes=forecast.probability_yes,
                            market_probability_yes=(
                                float(signal_candle.yes_bid.close)
                                + float(signal_candle.yes_ask.close)
                            )
                            / 2,
                            outcome_yes=(
                                historical_market.market.result == "yes"
                                if label_available
                                else None
                            ),
                            outcome_available_at=historical_market.outcome_available_at,
                            uncertainty_source=forecast.uncertainty_source,
                            calibration_profile_id=forecast.calibration_profile_id,
                            forecast=forecast.model_copy(
                                update={"input_manifest": _compact_manifest(forecast)}
                            ),
                            input_population_id=_population_id(
                                snapshot, contract, crypto.model_dump(mode="json")
                            ),
                        )
                    )
                    if (
                        market.ticker in traded_markets
                        or fee is None
                        or not label_available
                        or (config.require_calibration and calibration_profile is None)
                    ):
                        continue
                    if opportunity.state not in {
                        RecommendationState.ENTER_YES,
                        RecommendationState.ENTER_NO,
                    }:
                        continue
                    if (
                        opportunity.side is None
                        or opportunity.executable_price is None
                        or opportunity.conservative_net_edge is None
                    ):
                        continue
                    execution = _execution_candle(
                        historical_market.candlestick_revisions,
                        signal_at,
                        config.latency_seconds,
                        min(market.expiry, market.observation_end_at),
                        config.period_minutes * 60,
                        self._engine_config.maximum_input_age_seconds,
                    )
                    if execution is None:
                        continue
                    execution_at, execution_candle = execution
                    execution_fee = self._fee_at(historical_market, execution_at, configured_fee)
                    if execution_fee is None:
                        missing_fee += 1
                        continue
                    candidates.append(
                        _BacktestCandidate(
                            historical_market=historical_market,
                            signal_market=market,
                            signal_at=signal_at,
                            execution_at=execution_at,
                            execution_candle=execution_candle,
                            opportunity=opportunity,
                            forecast=forecast,
                            execution_fee=execution_fee,
                        )
                    )

                candidates.sort(
                    key=lambda candidate: (
                        candidate.opportunity.conservative_net_edge
                        if candidate.opportunity.conservative_net_edge is not None
                        else float("-inf")
                    ),
                    reverse=True,
                )
                for candidate in candidates:
                    market = candidate.signal_market.model_copy(
                        update={"result": candidate.historical_market.market.result}
                    )
                    if market.ticker in traded_markets:
                        continue
                    opportunity = candidate.opportunity
                    remaining = remaining_by_event.setdefault(
                        market.event_ticker,
                        self._engine_config.paper_bankroll
                        * self._engine_config.max_event_bankroll_fraction,
                    )
                    allowed_exposure = min(opportunity.suggested_max_exposure, remaining)
                    side = opportunity.side
                    executable_price = opportunity.executable_price
                    conservative_edge = opportunity.conservative_net_edge
                    if side is None or executable_price is None or conservative_edge is None:
                        continue
                    if allowed_exposure <= 0:
                        continue
                    trade = _fill_trade(
                        fold.index,
                        market,
                        candidate.signal_at,
                        candidate.execution_at,
                        candidate.execution_candle,
                        side,
                        float(executable_price),
                        float(allowed_exposure),
                        candidate.forecast.probability_yes,
                        candidate.forecast.model_name,
                        candidate.forecast.uncertainty_margin,
                        candidate.forecast.uncertainty_source,
                        candidate.forecast.calibration_profile_id,
                        conservative_edge,
                        candidate.execution_fee,
                        config.max_volume_participation,
                        fee_rate=self._engine_config.fee_rate,
                        slippage_bps=self._engine_config.slippage_bps,
                        recipe_id=candidate.forecast.recipe_id,
                        model_version=candidate.forecast.model_version,
                    )
                    if trade is None:
                        continue
                    trades.append(trade)
                    traded_markets.add(market.ticker)
                    remaining_by_event[market.event_ticker] = max(
                        remaining - float(trade.cost_dollars),
                        0.0,
                    )

            fold_results.append(
                _fold_result(
                    fold,
                    len(considered_markets),
                    evaluated_signals,
                    missing_context,
                    missing_fee,
                    missing_calibration,
                    calibration_profiles,
                    trades,
                    forecasts=tuple(forecasts),
                    eligible_signals=eligible_signals,
                    missing_metadata=missing_metadata,
                )
            )

        all_trades = tuple(trade for fold in fold_results for trade in fold.trades)
        all_forecasts = tuple(record for fold in fold_results for record in fold.forecasts)
        total_cost = sum((trade.cost_dollars for trade in all_trades), Decimal("0.00"))
        total_pnl = sum((trade.pnl_dollars for trade in all_trades), Decimal("0.00"))
        deployment_profiles = self.calibrate(
            config,
            markets,
            training_start=max(config.start, config.end - timedelta(days=config.train_days)),
            cutoff_at=config.end,
        )
        validations = _model_validations(
            config,
            tuple(fold_results),
            deployment_profiles,
        )
        return BacktestResult(
            config=config,
            folds=tuple(fold_results),
            unsupported_markets=tuple(sorted(unsupported)),
            unresolved_markets=tuple(sorted(unresolved)),
            markets_without_candles=tuple(sorted(without_candles)),
            total_trades=len(all_trades),
            partial_fills=sum(trade.partial_fill for trade in all_trades),
            total_cost_dollars=total_cost,
            total_pnl_dollars=total_pnl,
            return_on_cost=_ratio(total_pnl, total_cost),
            brier_score=_forecast_score(all_forecasts),
            deployment_profiles=deployment_profiles,
            model_validations=validations,
            forecasts=all_forecasts,
            eligible_signals=sum(fold.eligible_signals for fold in fold_results),
            missing_metadata_signals=sum(fold.missing_metadata_signals for fold in fold_results),
            missing_outcome_signals=sum(fold.missing_outcome_signals for fold in fold_results),
            incomplete_forecast_signals=sum(
                fold.incomplete_forecast_signals for fold in fold_results
            ),
            log_loss=_forecast_score(all_forecasts, logarithmic=True),
            market_brier_score=_forecast_score(all_forecasts, market=True),
            market_log_loss=_forecast_score(all_forecasts, market=True, logarithmic=True),
            engine_config=self._engine_config,
            input_manifest={
                # Dataset membership by content identity; the raw point-in-time records
                # remain in the immutable venue/research history tables.
                "historical_market_count": len(markets),
                "historical_markets": [
                    {
                        "ticker": history.market.ticker,
                        "event_ticker": history.market.event_ticker,
                        "sha256": _content_sha256(history.model_dump(mode="json")),
                    }
                    for history in sorted(markets, key=lambda item: item.market.ticker)
                ],
                "dataset_sha256": _content_sha256(
                    sorted(_content_sha256(history.model_dump(mode="json")) for history in markets)
                ),
            },
        )

    def calibrate(
        self,
        config: BacktestConfig,
        markets: tuple[HistoricalMarketData, ...],
        *,
        training_start: datetime,
        cutoff_at: datetime,
    ) -> tuple[UncertaintyCalibrationProfile, ...]:
        samples: list[CalibrationSample] = []
        engine = CryptoThresholdEngine(self._engine_config)
        for history in markets:
            outcome_at = history.outcome_available_at
            if (
                history.market.result not in {"yes", "no"}
                or outcome_at is None
                or outcome_at > cutoff_at
            ):
                continue
            # Pick the last admissible signal using only metadata known then.
            # Final expiry/rules/settlement payload must never choose features.
            selected: (
                tuple[datetime, KalshiCandlestick, KalshiMarket, CryptoPriceContract] | None
            ) = None
            for signal_at, candle in _available_signals(history):
                if (
                    signal_at - datetime.fromtimestamp(candle.end_period_ts, UTC)
                ).total_seconds() > self._engine_config.maximum_input_age_seconds:
                    continue
                if not training_start <= signal_at < min(cutoff_at, outcome_at):
                    continue
                market = _metadata_at(history, signal_at)
                if market is None:
                    continue
                try:
                    contract = market.evaluation_contract(signal_at)
                    horizon = min(market.expiry, market.observation_end_at)
                except (UnsupportedMarketError, ValueError):
                    continue
                if signal_at >= horizon or signal_at > horizon - timedelta(
                    seconds=config.calibration_lead_seconds
                ):
                    continue
                selected = signal_at, candle, market, contract
            if selected is None:
                continue
            signal_at, candle, market, contract = selected
            try:
                context = self._research_context(
                    config, as_of=signal_at, event_ticker=market.event_ticker
                )
                context.validate_at(
                    signal_at,
                    maximum_spot_age_seconds=self._engine_config.maximum_input_age_seconds,
                )
                crypto = context.to_crypto_snapshot(
                    strike_price=engine.reference_price(contract),
                    expected_annual_return=config.expected_annual_return,
                )
                forecast, _ = engine.evaluate(
                    _market_snapshot(market, candle, config, signal_at=signal_at), crypto, contract
                )
            except (ResearchDataUnavailable, ValueError):
                continue
            samples.append(
                CalibrationSample(
                    market_id=market.ticker,
                    event_id=market.event_ticker,
                    symbol=config.symbol,
                    model_name=forecast.model_name,
                    model_version=forecast.model_version,
                    recipe_id=engine.recipe_id(crypto),
                    horizon_seconds=(
                        market.observation_end_at - crypto.observed_at
                    ).total_seconds(),
                    probability_yes=forecast.probability_yes,
                    outcome_yes=history.market.result == "yes",
                    observed_at=signal_at,
                    resolved_at=outcome_at,
                )
            )
        profiles = fit_uncertainty_profiles(
            tuple(samples),
            training_start=training_start,
            cutoff_at=cutoff_at,
            confidence_level=config.calibration_confidence,
            minimum_samples=config.minimum_calibration_samples,
            maximum_bins=config.maximum_calibration_bins,
        )
        if not config.require_calibration:
            return tuple(profile.model_copy(update={"research_only": True}) for profile in profiles)
        return profiles

    @staticmethod
    def _fee_at(
        history: HistoricalMarketData, as_of: datetime, configured: EffectiveFee
    ) -> EffectiveFee | None:
        fee = effective_fee_at(history, as_of)
        if fee is not None:
            return fee
        if (
            history.series_fee_changes
            or history.event_fee_changes
            or history.series_fee_revisions
            or history.event_fee_revisions
        ):
            return None
        return configured


def _market_snapshot(
    market: KalshiMarket,
    candle: KalshiCandlestick,
    config: BacktestConfig,
    *,
    signal_at: datetime | None = None,
) -> MarketSnapshot:
    observed_at = signal_at or datetime.fromtimestamp(candle.end_period_ts, UTC)
    yes_bid = float(candle.yes_bid.close)
    yes_ask = float(candle.yes_ask.close)
    volume = float(candle.volume) * config.max_volume_participation
    return MarketSnapshot(
        market_id=market.ticker,
        question=market.question,
        venue="kalshi",
        observed_at=observed_at,
        source_metadata=market.model_dump(mode="json"),
        expires_at=market.expiry,
        observation_end_at=market.observation_end_at,
        expected_settlement_at=market.expected_expiration_time,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        yes_ask_size=volume,
        no_ask_size=volume,
        resolution_rule=market.resolution_rule,
        series_id=market.normalized_series_ticker,
        event_id=market.event_ticker,
        contract_label=market.contract_label,
        market_url=market.market_url,
    )


def _execution_candle(
    revisions: tuple[tuple[datetime, KalshiCandlestick], ...],
    signal_at: datetime,
    latency_seconds: int,
    expiry: datetime,
    interval_seconds: int,
    maximum_age_seconds: int,
) -> tuple[datetime, KalshiCandlestick] | None:
    executable_after = signal_at + timedelta(seconds=latency_seconds)
    next_boundary = (int(executable_after.timestamp()) // interval_seconds + 1) * interval_seconds
    # Missing the next executable interval is not permission to jump arbitrarily
    # far ahead. Later provider revisions cannot improve the original fill.
    eligible = [
        (max(available, datetime.fromtimestamp(candle.end_period_ts, UTC)), candle)
        for available, candle in revisions
        if candle.end_period_ts == next_boundary
        and datetime.fromtimestamp(candle.end_period_ts, UTC) <= available < expiry
        and (available - datetime.fromtimestamp(candle.end_period_ts, UTC)).total_seconds()
        <= maximum_age_seconds
    ]
    if not eligible:
        return None
    first_at = min(item[0] for item in eligible)
    first = [item for item in eligible if item[0] == first_at]
    if len({item[1].model_dump_json() for item in first}) != 1:
        return None
    return first[0]


def _fill_trade(
    fold_index: int,
    market: KalshiMarket,
    signal_at: datetime,
    execution_at: datetime,
    execution_candle: KalshiCandlestick,
    side: MarketSide,
    signal_price: float,
    suggested_exposure: float,
    probability_yes: float,
    model_name: str,
    uncertainty_margin: float,
    uncertainty_source: Literal["fixed", "held_out"],
    calibration_profile_id: UUID | None,
    conservative_net_edge: float,
    fee: EffectiveFee,
    max_volume_participation: float,
    *,
    fee_rate: float = 0.0,
    slippage_bps: float = 0.0,
    recipe_id: str | None = None,
    model_version: str | None = None,
) -> BacktestTrade | None:
    if market.result not in {"yes", "no"}:
        return None
    if side is MarketSide.YES:
        execution_price = float(execution_candle.yes_ask.high)
    else:
        execution_price = 1.0 - float(execution_candle.yes_bid.low)
    if not 0.0 < execution_price < 1.0:
        return None
    execution_price_decimal = Decimal(str(execution_price))
    cost_multiplier = Decimal("1") + Decimal(str(fee_rate)) + Decimal(str(slippage_bps)) / 10000
    budget = Decimal(str(suggested_exposure))

    def all_in_cost(contracts: int) -> Decimal:
        return execution_price_decimal * contracts * cost_multiplier + kalshi_taker_fee(
            contracts, execution_price, fee
        )

    low, high = 0, math.floor(budget / (execution_price_decimal * cost_multiplier))
    while low < high:
        middle = (low + high + 1) // 2
        if all_in_cost(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    requested_contracts = low
    available_contracts = math.floor(float(execution_candle.volume) * max_volume_participation)
    filled_contracts = min(requested_contracts, available_contracts)
    if filled_contracts <= 0:
        return None

    fee_dollars = kalshi_taker_fee(filled_contracts, execution_price, fee)
    notional = market.notional_value_dollars
    cost = all_in_cost(filled_contracts)
    won = (side is MarketSide.YES and market.result == "yes") or (
        side is MarketSide.NO and market.result == "no"
    )
    payout = notional * filled_contracts if won else Decimal("0.00")
    pnl = payout - cost
    return BacktestTrade(
        fold_index=fold_index,
        event_ticker=market.event_ticker,
        ticker=market.ticker,
        signal_at=signal_at,
        executed_at=execution_at,
        side=side,
        signal_price=signal_price,
        execution_price=execution_price,
        requested_contracts=requested_contracts,
        filled_contracts=filled_contracts,
        partial_fill=filled_contracts < requested_contracts,
        probability_yes=probability_yes,
        model_name=model_name,
        uncertainty_margin=uncertainty_margin,
        uncertainty_source=uncertainty_source,
        calibration_profile_id=calibration_profile_id,
        conservative_net_edge=conservative_net_edge,
        fee_type=fee.fee_type,
        fee_multiplier=fee.multiplier,
        fee_dollars=fee_dollars,
        cost_dollars=cost,
        payout_dollars=payout,
        pnl_dollars=pnl,
        result=market.result,
        recipe_id=recipe_id,
        model_version=model_version,
    )


def _fold_result(
    fold: WalkForwardFold,
    considered: int,
    evaluated_signals: int,
    missing_context: int,
    missing_fee: int,
    missing_calibration: int,
    calibration_profiles: tuple[UncertaintyCalibrationProfile, ...],
    trades: list[BacktestTrade],
    *,
    forecasts: tuple[BacktestForecast, ...] = (),
    eligible_signals: int = 0,
    missing_metadata: int = 0,
) -> BacktestFoldResult:
    total_cost = sum((trade.cost_dollars for trade in trades), Decimal("0.00"))
    total_pnl = sum((trade.pnl_dollars for trade in trades), Decimal("0.00"))
    missing_outcomes = sum(record.outcome_yes is None for record in forecasts)
    return BacktestFoldResult(
        fold=fold,
        markets_considered=considered,
        evaluated_signals=evaluated_signals,
        missing_context_signals=missing_context,
        missing_fee_signals=missing_fee,
        missing_calibration_signals=missing_calibration,
        calibration_profiles=calibration_profiles,
        trades=tuple(trades),
        total_cost_dollars=total_cost,
        total_pnl_dollars=total_pnl,
        return_on_cost=_ratio(total_pnl, total_cost),
        brier_score=_forecast_score(forecasts),
        forecasts=forecasts,
        eligible_signals=eligible_signals,
        missing_metadata_signals=missing_metadata,
        missing_outcome_signals=missing_outcomes,
        incomplete_forecast_signals=max(0, eligible_signals - len(forecasts)) + missing_outcomes,
        log_loss=_forecast_score(forecasts, logarithmic=True),
        market_brier_score=_forecast_score(forecasts, market=True),
        market_log_loss=_forecast_score(forecasts, market=True, logarithmic=True),
    )


def _ratio(numerator: Decimal, denominator: Decimal) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _compact_manifest(forecast: ProbabilityForecast) -> dict[str, Any]:
    """Population records keep recipe/contract identity; bulky snapshots stay in the store.

    The market snapshot, calibration profile, and crypto inputs are reconstructible from
    the immutable venue/research history and the persisted profile ID, so the walk-forward
    result stays bounded for tens of thousands of forecasts.
    """
    manifest = forecast.input_manifest
    crypto = manifest.get("crypto", {})
    return {
        "recipe": manifest.get("recipe"),
        "contract": manifest.get("contract"),
        "engine_config": manifest.get("engine_config"),
        "crypto_provenance": crypto.get("input_provenance") if isinstance(crypto, dict) else None,
        "calibration_profile_id": (
            None
            if forecast.calibration_profile_id is None
            else str(forecast.calibration_profile_id)
        ),
    }


def _content_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _population_id(
    market: MarketSnapshot, contract: CryptoPriceContract, crypto: dict[str, Any]
) -> str:
    inputs = {
        "market": market.model_dump(mode="json"),
        "contract": contract.model_dump(mode="json"),
        "crypto": {key: value for key, value in crypto.items() if key != "expected_annual_return"},
    }
    return _content_sha256(inputs)


def _forecast_score(
    forecasts: tuple[BacktestForecast, ...],
    *,
    market: bool = False,
    logarithmic: bool = False,
) -> float | None:
    by_event: dict[str, list[float]] = {}
    for record in forecasts:
        if record.outcome_yes is None:
            continue
        probability = record.market_probability_yes if market else record.probability_yes
        outcome = float(record.outcome_yes)
        if logarithmic:
            # Fixed clipping applies equally to model and market at exact 0/1.
            probability = min(1 - 1e-15, max(1e-15, probability))
            score = -outcome * math.log(probability) - (1 - outcome) * math.log1p(-probability)
        else:
            score = (probability - outcome) ** 2
        by_event.setdefault(record.event_ticker, []).append(score)
    if not by_event:
        return None
    return sum(sum(scores) / len(scores) for scores in by_event.values()) / len(by_event)


def _model_validations(
    config: BacktestConfig,
    folds: tuple[BacktestFoldResult, ...],
    deployment_profiles: tuple[UncertaintyCalibrationProfile, ...],
) -> tuple[BacktestModelValidation, ...]:
    profiles_by_key = {
        (profile.model_name, profile.model_version, profile.recipe_id): profile
        for profile in deployment_profiles
    }
    trades = tuple(trade for fold in folds for trade in fold.trades)
    forecasts = tuple(record for fold in folds for record in fold.forecasts)
    keys = (
        set(profiles_by_key)
        | {(record.model_name, record.model_version, record.recipe_id) for record in forecasts}
        | {(trade.model_name, trade.model_version or "", trade.recipe_id) for trade in trades}
    )
    incomplete = sum(fold.incomplete_forecast_signals for fold in folds)
    validations: list[BacktestModelValidation] = []
    for key in sorted(keys, key=lambda value: tuple(part or "" for part in value)):
        model_name, model_version, recipe_id = key
        profile = profiles_by_key.get(key)
        model_forecasts = tuple(
            record
            for record in forecasts
            if (record.model_name, record.model_version, record.recipe_id) == key
        )
        model_trades = tuple(
            trade
            for trade in trades
            if (trade.model_name, trade.model_version or "", trade.recipe_id) == key
        )
        scored = tuple(record for record in model_forecasts if record.outcome_yes is not None)
        event_count = len({record.event_ticker for record in scored})
        fold_count = len({record.fold_index for record in scored})
        total_cost = sum((trade.cost_dollars for trade in model_trades), Decimal("0.00"))
        total_pnl = sum((trade.pnl_dollars for trade in model_trades), Decimal("0.00"))
        return_on_cost = _ratio(total_pnl, total_cost)
        brier_score = _forecast_score(model_forecasts)
        baseline_brier = _forecast_score(model_forecasts, market=True)
        log_loss = _forecast_score(model_forecasts, logarithmic=True)
        baseline_log = _forecast_score(model_forecasts, market=True, logarithmic=True)
        reasons: list[str] = []
        if not config.require_calibration:
            reasons.append("uncalibrated research override cannot authorize delivery")
        if recipe_id is None or model_version != CryptoThresholdEngine.model_version:
            reasons.append("legacy or incompatible forecast recipe evidence")
        if not model_forecasts or any(fold.eligible_signals == 0 for fold in folds if fold.trades):
            reasons.append("complete forecast population evidence is unavailable")
        if incomplete:
            reasons.append(f"forecast population has {incomplete} incomplete observations")
        if any(fold.missing_fee_signals for fold in folds):
            reasons.append("point-in-time execution fee coverage is incomplete")
        if any(
            record.uncertainty_source != "held_out" or record.calibration_profile_id is None
            for record in model_forecasts
        ) or any(trade.uncertainty_source != "held_out" for trade in model_trades):
            reasons.append("fixed-margin evidence cannot authorize delivery")
        for fold in folds:
            training_profiles = {item.profile_id: item for item in fold.calibration_profiles}
            for record in model_forecasts:
                if record.fold_index != fold.fold.index:
                    continue
                training = (
                    training_profiles.get(record.calibration_profile_id)
                    if record.calibration_profile_id is not None
                    else None
                )
                if (
                    training is None
                    or training.research_only
                    or (training.model_name, training.model_version, training.recipe_id) != key
                    or training.cutoff_at > min(record.observed_at, fold.fold.test_start)
                ):
                    reasons.append("forecast lacks exact admissible held-out calibration")
                    break
        if profile is None:
            reasons.append("no deployment calibration profile")
        elif profile.research_only:
            reasons.append("deployment calibration is research-only")
        elif profile.independent_event_count < config.minimum_calibration_samples:
            reasons.append("deployment calibration has insufficient independent events")
        if event_count < config.minimum_validation_events:
            reasons.append(
                f"held-out backtest has {event_count} forecast events; "
                f"minimum is {config.minimum_validation_events}"
            )
        if fold_count < config.minimum_validation_folds:
            reasons.append(
                f"held-out backtest has {fold_count} forecast folds; "
                f"minimum is {config.minimum_validation_folds}"
            )
        if len({trade.event_ticker for trade in model_trades}) < config.minimum_validation_events:
            reasons.append("held-out return evidence has insufficient independent traded events")
        if len({trade.fold_index for trade in model_trades}) < config.minimum_validation_folds:
            reasons.append("held-out return evidence has insufficient traded folds")
        if return_on_cost is None or return_on_cost <= config.minimum_return_on_cost:
            reasons.append(
                f"held-out return on cost does not exceed {config.minimum_return_on_cost:.2%}"
            )
        if brier_score is None or brier_score > config.maximum_brier_score:
            reasons.append(
                f"event-weighted held-out Brier score exceeds {config.maximum_brier_score:.4f}"
            )
        if (
            brier_score is None
            or baseline_brier is None
            or brier_score > baseline_brier
            or log_loss is None
            or baseline_log is None
            or log_loss > baseline_log
        ):
            reasons.append("full-population forecast scores do not meet the market-only baseline")
        validations.append(
            BacktestModelValidation(
                model_name=model_name,
                model_version=model_version,
                recipe_id=recipe_id,
                calibration_profile_id=None if profile is None else profile.profile_id,
                independent_calibration_events=0
                if profile is None
                else profile.independent_event_count,
                held_out_events=event_count,
                held_out_folds=fold_count,
                held_out_trades=len(model_trades),
                held_out_forecasts=len(scored),
                incomplete_forecast_signals=incomplete,
                total_cost_dollars=total_cost,
                total_pnl_dollars=total_pnl,
                return_on_cost=return_on_cost,
                brier_score=brier_score,
                log_loss=log_loss,
                market_brier_score=baseline_brier,
                market_log_loss=baseline_log,
                accepted_for_paper_alerts=not reasons,
                rejection_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return tuple(validations)
