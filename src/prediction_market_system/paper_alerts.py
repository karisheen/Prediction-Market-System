from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx

from prediction_market_system.domain import (
    MarketRegimeSnapshot,
    Opportunity,
    PriceTrendRegime,
    RecommendationState,
    VolatilityRegime,
)
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.research import ResearchContext, ResearchDataUnavailable, SpotCandle
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import (
    IncompleteOrderBookError,
    KalshiAPIError,
    KalshiMarket,
    KalshiOrderBook,
    UnsupportedMarketError,
    to_market_snapshot,
)


class PaperAlertMarketReader(Protocol):
    async def get_order_book(self, ticker: str, depth: int = 100) -> KalshiOrderBook: ...


class PaperAlertPublisher(Protocol):
    async def publish(self, opportunity: Opportunity) -> str: ...


@dataclass(frozen=True)
class PaperAlertCycleResult:
    discovered: int
    evaluated: int
    watch: int
    delivered: int
    unsupported: int
    uncalibrated: int
    failures: tuple[str, ...]


class PaperAlertRunner:
    """Evaluate one live series cycle and deliver only calibrated entry signals."""

    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        engine: CryptoThresholdEngine,
        market_reader: PaperAlertMarketReader,
        alert_service: PaperAlertPublisher,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.engine = engine
        self.market_reader = market_reader
        self.alert_service = alert_service
        self.clock = clock or (lambda: datetime.now(UTC))

    async def run(
        self,
        *,
        markets: Sequence[KalshiMarket],
        context: ResearchContext,
        regime: MarketRegimeSnapshot,
        expected_annual_return: float = 0.0,
    ) -> PaperAlertCycleResult:
        if context.symbol != regime.symbol:
            raise ValueError("research context and market regime symbols must match")

        evaluated = 0
        watch = 0
        delivered = 0
        unsupported = 0
        uncalibrated = 0
        failures: list[str] = []

        for market in markets:
            try:
                contract = market.threshold_contract()
            except UnsupportedMarketError:
                unsupported += 1
                continue

            profile = self.repository.latest_uncertainty_calibration(
                symbol=context.symbol,
                model_name=self.engine.model_name(contract),
                model_version=self.engine.model_version,
                as_of=context.as_of,
            )
            if profile is None:
                uncalibrated += 1
                continue

            try:
                order_book = await self.market_reader.get_order_book(market.ticker)
                observed_at = self.clock()
                snapshot = to_market_snapshot(market, order_book, observed_at=observed_at)
                crypto = context.to_crypto_snapshot(
                    strike_price=contract.strike_price,
                    expected_annual_return=expected_annual_return,
                )
                _, opportunity = self.engine.evaluate(snapshot, crypto, contract, profile)
                opportunity = opportunity.model_copy(update={"market_regime": regime})
                self.repository.save_evaluation(opportunity.forecast, opportunity)
                evaluated += 1
                if opportunity.state is RecommendationState.WATCH:
                    watch += 1
                    continue
                await self.alert_service.publish(opportunity)
                delivered += 1
            except (
                IncompleteOrderBookError,
                KalshiAPIError,
                httpx.HTTPError,
                RuntimeError,
                ValueError,
            ) as exc:
                failures.append(f"{market.ticker}: {exc}")

        return PaperAlertCycleResult(
            discovered=len(markets),
            evaluated=evaluated,
            watch=watch,
            delivered=delivered,
            unsupported=unsupported,
            uncalibrated=uncalibrated,
            failures=tuple(failures),
        )


def classify_market_regime(
    context: ResearchContext,
    candles: Sequence[SpotCandle],
    *,
    trend_threshold: float = 0.05,
    low_volatility_threshold: float = 0.40,
    high_volatility_threshold: float = 0.80,
) -> MarketRegimeSnapshot:
    """Classify an auditable trend/realized-volatility regime at a research cutoff."""
    if trend_threshold < 0.0:
        raise ValueError("trend threshold must be non-negative")
    if not 0.0 <= low_volatility_threshold < high_volatility_threshold:
        raise ValueError("volatility thresholds must be non-negative and increasing")

    eligible = sorted(
        (
            candle
            for candle in candles
            if candle.product_id == f"{context.symbol}-USD"
            and context.realized_volatility.source_start_at <= candle.end_at <= context.as_of
        ),
        key=lambda candle: candle.end_at,
    )
    if len(eligible) < 2:
        raise ResearchDataUnavailable(
            "at least two spot candles are required to classify the market regime"
        )

    trailing_return = float(eligible[-1].close / eligible[0].close) - 1.0
    if trailing_return >= trend_threshold:
        price_trend = PriceTrendRegime.UPTREND
    elif trailing_return <= -trend_threshold:
        price_trend = PriceTrendRegime.DOWNTREND
    else:
        price_trend = PriceTrendRegime.RANGE

    realized_volatility = context.realized_volatility.annualized_volatility
    if realized_volatility < low_volatility_threshold:
        volatility = VolatilityRegime.LOW
    elif realized_volatility >= high_volatility_threshold:
        volatility = VolatilityRegime.HIGH
    else:
        volatility = VolatilityRegime.TYPICAL

    return MarketRegimeSnapshot(
        symbol=context.symbol,
        observed_at=context.as_of,
        source_start_at=eligible[0].end_at,
        trailing_return=trailing_return,
        realized_volatility=realized_volatility,
        implied_volatility=(
            context.implied_volatility.annualized_volatility
            if context.implied_volatility is not None
            else None
        ),
        price_trend=price_trend,
        volatility=volatility,
        trend_threshold=trend_threshold,
        low_volatility_threshold=low_volatility_threshold,
        high_volatility_threshold=high_volatility_threshold,
    )
