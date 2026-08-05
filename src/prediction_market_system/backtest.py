from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from typing import Annotated, Literal, Protocol, Self, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prediction_market_system.domain import MarketSide, MarketSnapshot, RecommendationState
from prediction_market_system.engine import BinaryFeeType, CryptoThresholdEngine, EngineConfig
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
    realized_window_days: Annotated[int, Field(gt=0)] = 30
    train_days: Annotated[int, Field(gt=0)] = 90
    test_days: Annotated[int, Field(gt=0)] = 30
    step_days: Annotated[int, Field(gt=0)] = 30
    latency_seconds: Annotated[int, Field(ge=0)] = 30
    max_volume_participation: Annotated[float, Field(gt=0.0, le=1.0)] = 0.10
    expected_annual_return: float = 0.0

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


class BacktestTrade(BacktestModel):
    fold_index: int
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
    conservative_net_edge: float
    fee_type: BinaryFeeType
    fee_multiplier: float
    fee_dollars: Decimal
    cost_dollars: Decimal
    payout_dollars: Decimal
    pnl_dollars: Decimal
    result: Literal["yes", "no"]


class BacktestFoldResult(BacktestModel):
    fold: WalkForwardFold
    markets_considered: int
    evaluated_signals: int
    missing_context_signals: int
    missing_fee_signals: int
    trades: tuple[BacktestTrade, ...]
    total_cost_dollars: Decimal
    total_pnl_dollars: Decimal
    return_on_cost: float | None
    brier_score: float | None


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


class BacktestResearchSource(Protocol):
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
    series_change = _latest_change(market.series_fee_changes, as_of)
    event_change = _latest_change(market.event_fee_changes, as_of)
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


def _latest_change(
    changes: tuple[ChangeT, ...],
    as_of: datetime,
) -> ChangeT | None:
    eligible = (change for change in changes if change.scheduled_ts <= as_of)
    return max(eligible, key=lambda change: change.scheduled_ts, default=None)


class HistoricalBacktester:
    def __init__(
        self,
        research_source: BacktestResearchSource,
        engine_config: EngineConfig,
    ) -> None:
        self._research_source = research_source
        self._engine_config = engine_config

    def run(
        self,
        config: BacktestConfig,
        markets: tuple[HistoricalMarketData, ...],
    ) -> BacktestResult:
        unsupported: set[str] = set()
        unresolved: set[str] = set()
        without_candles: set[str] = set()
        traded_markets: set[str] = set()
        fold_results: list[BacktestFoldResult] = []

        for fold in walk_forward_folds(config):
            trades: list[BacktestTrade] = []
            considered = 0
            evaluated_signals = 0
            missing_context = 0
            missing_fee = 0
            for historical_market in markets:
                ticker = historical_market.market.ticker
                if ticker in traded_markets:
                    continue
                market = historical_market.market
                if market.result not in {"yes", "no"}:
                    unresolved.add(ticker)
                    continue
                if not historical_market.candlesticks:
                    without_candles.add(ticker)
                    continue
                try:
                    contract = market.threshold_contract()
                except UnsupportedMarketError:
                    unsupported.add(ticker)
                    continue

                signal_candles = tuple(
                    candle
                    for candle in historical_market.candlesticks
                    if fold.test_start
                    <= datetime.fromtimestamp(candle.end_period_ts, UTC)
                    < min(fold.test_end, market.expiry)
                )
                if not signal_candles:
                    continue
                considered += 1

                for signal_candle in signal_candles:
                    signal_at = datetime.fromtimestamp(signal_candle.end_period_ts, UTC)
                    fee = effective_fee_at(historical_market, signal_at)
                    if fee is None:
                        missing_fee += 1
                        continue
                    try:
                        context = self._research_source.research_context_as_of(
                            symbol=config.symbol,
                            as_of=signal_at,
                            event_ticker=market.event_ticker,
                            interval_seconds=config.period_minutes * 60,
                            realized_window_seconds=config.realized_window_days * 24 * 60 * 60,
                        )
                    except ResearchDataUnavailable:
                        missing_context += 1
                        continue

                    snapshot = _market_snapshot(market, signal_candle, config)
                    engine = CryptoThresholdEngine(
                        self._engine_config.model_copy(
                            update={
                                "binary_fee_type": fee.fee_type,
                                "binary_fee_coefficient": fee.multiplier,
                            }
                        )
                    )
                    forecast, opportunity = engine.evaluate(
                        snapshot,
                        context.to_crypto_snapshot(
                            strike_price=contract.strike_price,
                            expected_annual_return=config.expected_annual_return,
                        ),
                        contract,
                    )
                    evaluated_signals += 1
                    if opportunity.state not in {
                        RecommendationState.ENTER_YES,
                        RecommendationState.ENTER_NO,
                    }:
                        continue
                    if opportunity.side is None or opportunity.executable_price is None:
                        continue
                    if opportunity.conservative_net_edge is None:
                        continue

                    execution = _execution_candle(
                        historical_market.candlesticks,
                        signal_at,
                        config.latency_seconds,
                        market.expiry,
                    )
                    if execution is None:
                        continue
                    execution_at = datetime.fromtimestamp(execution.end_period_ts, UTC)
                    execution_fee = effective_fee_at(historical_market, execution_at)
                    if execution_fee is None:
                        missing_fee += 1
                        continue
                    trade = _fill_trade(
                        fold.index,
                        market,
                        signal_at,
                        execution_at,
                        execution,
                        opportunity.side,
                        float(opportunity.executable_price),
                        float(opportunity.suggested_max_exposure),
                        forecast.probability_yes,
                        forecast.model_name,
                        opportunity.conservative_net_edge,
                        execution_fee,
                        config.max_volume_participation,
                    )
                    if trade is None:
                        continue
                    trades.append(trade)
                    traded_markets.add(ticker)
                    break

            fold_results.append(
                _fold_result(
                    fold,
                    considered,
                    evaluated_signals,
                    missing_context,
                    missing_fee,
                    trades,
                )
            )

        all_trades = tuple(trade for fold in fold_results for trade in fold.trades)
        total_cost = sum((trade.cost_dollars for trade in all_trades), Decimal("0.00"))
        total_pnl = sum((trade.pnl_dollars for trade in all_trades), Decimal("0.00"))
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
            brier_score=_brier_score(all_trades),
        )


def _market_snapshot(
    market: KalshiMarket,
    candle: KalshiCandlestick,
    config: BacktestConfig,
) -> MarketSnapshot:
    observed_at = datetime.fromtimestamp(candle.end_period_ts, UTC)
    yes_bid = float(candle.yes_bid.close)
    yes_ask = float(candle.yes_ask.close)
    volume = float(candle.volume) * config.max_volume_participation
    return MarketSnapshot(
        market_id=market.ticker,
        question=market.question,
        venue="kalshi",
        observed_at=observed_at,
        expires_at=market.expiry,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        yes_ask_size=volume,
        no_ask_size=volume,
        resolution_rule=market.resolution_rule,
    )


def _execution_candle(
    candles: tuple[KalshiCandlestick, ...],
    signal_at: datetime,
    latency_seconds: int,
    expiry: datetime,
) -> KalshiCandlestick | None:
    executable_after = signal_at + timedelta(seconds=latency_seconds)
    return next(
        (
            candle
            for candle in candles
            if executable_after < datetime.fromtimestamp(candle.end_period_ts, UTC) < expiry
        ),
        None,
    )


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
    conservative_net_edge: float,
    fee: EffectiveFee,
    max_volume_participation: float,
) -> BacktestTrade | None:
    if market.result not in {"yes", "no"}:
        return None
    if side is MarketSide.YES:
        execution_price = float(execution_candle.yes_ask.high)
    else:
        execution_price = 1.0 - float(execution_candle.yes_bid.low)
    if not 0.0 < execution_price < 1.0:
        return None
    requested_contracts = math.floor(suggested_exposure / execution_price)
    available_contracts = math.floor(float(execution_candle.volume) * max_volume_participation)
    filled_contracts = min(requested_contracts, available_contracts)
    if filled_contracts <= 0:
        return None

    fee_dollars = kalshi_taker_fee(filled_contracts, execution_price, fee)
    execution_price_decimal = Decimal(str(execution_price))
    notional = market.notional_value_dollars
    contract_cost = execution_price_decimal * filled_contracts
    cost = contract_cost + fee_dollars
    won = (side is MarketSide.YES and market.result == "yes") or (
        side is MarketSide.NO and market.result == "no"
    )
    payout = notional * filled_contracts if won else Decimal("0.00")
    pnl = payout - cost
    return BacktestTrade(
        fold_index=fold_index,
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
        conservative_net_edge=conservative_net_edge,
        fee_type=fee.fee_type,
        fee_multiplier=fee.multiplier,
        fee_dollars=fee_dollars,
        cost_dollars=cost,
        payout_dollars=payout,
        pnl_dollars=pnl,
        result=market.result,
    )


def _fold_result(
    fold: WalkForwardFold,
    considered: int,
    evaluated_signals: int,
    missing_context: int,
    missing_fee: int,
    trades: list[BacktestTrade],
) -> BacktestFoldResult:
    total_cost = sum((trade.cost_dollars for trade in trades), Decimal("0.00"))
    total_pnl = sum((trade.pnl_dollars for trade in trades), Decimal("0.00"))
    return BacktestFoldResult(
        fold=fold,
        markets_considered=considered,
        evaluated_signals=evaluated_signals,
        missing_context_signals=missing_context,
        missing_fee_signals=missing_fee,
        trades=tuple(trades),
        total_cost_dollars=total_cost,
        total_pnl_dollars=total_pnl,
        return_on_cost=_ratio(total_pnl, total_cost),
        brier_score=_brier_score(tuple(trades)),
    )


def _ratio(numerator: Decimal, denominator: Decimal) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _brier_score(trades: tuple[BacktestTrade, ...]) -> float | None:
    if not trades:
        return None
    return sum(
        (trade.probability_yes - (1.0 if trade.result == "yes" else 0.0)) ** 2 for trade in trades
    ) / len(trades)
