from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

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
from prediction_market_system.storage import MarketCheckStatus, SQLiteRepository
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
        maximum_spot_age: timedelta = timedelta(minutes=2),
    ) -> None:
        if maximum_spot_age <= timedelta(0):
            raise ValueError("maximum spot age must be positive")
        self.repository = repository
        self.engine = engine
        self.market_reader = market_reader
        self.alert_service = alert_service
        self.clock = clock or (lambda: datetime.now(UTC))
        self.maximum_spot_age = maximum_spot_age

    async def run(
        self,
        *,
        markets: Sequence[KalshiMarket],
        context: ResearchContext,
        regime: MarketRegimeSnapshot,
        expected_annual_return: float = 0.0,
        cycle_id: str | None = None,
    ) -> PaperAlertCycleResult:
        if context.symbol != regime.symbol:
            raise ValueError("research context and market regime symbols must match")

        cycle = cycle_id or str(uuid4())
        cycle_observed_at = self.clock()
        spot_age = cycle_observed_at - context.spot.end_at
        if spot_age < timedelta(0) or spot_age > self.maximum_spot_age:
            reason = (
                f"live spot is {spot_age.total_seconds():.0f} seconds old; "
                f"maximum is {self.maximum_spot_age.total_seconds():.0f}"
            )
            for market in markets:
                self._record_check(
                    cycle,
                    market,
                    cycle_observed_at,
                    MarketCheckStatus.FAILED,
                    reason,
                )
            return PaperAlertCycleResult(
                discovered=len(markets),
                evaluated=0,
                watch=0,
                delivered=0,
                unsupported=0,
                uncalibrated=0,
                failures=(reason,),
            )

        unsupported = 0
        uncalibrated = 0
        failures: list[str] = []
        evaluated_opportunities: list[tuple[KalshiMarket, Opportunity]] = []

        for market in markets:
            try:
                contract = market.price_contract()
            except UnsupportedMarketError as exc:
                unsupported += 1
                self._record_check(
                    cycle,
                    market,
                    cycle_observed_at,
                    MarketCheckStatus.UNSUPPORTED,
                    str(exc),
                )
                continue

            profile = self.repository.latest_uncertainty_calibration(
                symbol=context.symbol,
                model_name=self.engine.model_name(contract),
                model_version=self.engine.model_version,
                as_of=context.as_of,
            )
            if profile is None:
                uncalibrated += 1
                self._record_check(
                    cycle,
                    market,
                    cycle_observed_at,
                    MarketCheckStatus.MISSING_CALIBRATION,
                    f"no held-out calibration for {self.engine.model_name(contract)}",
                )
                continue

            try:
                order_book = await self.market_reader.get_order_book(market.ticker)
                observed_at = self.clock()
                snapshot = to_market_snapshot(market, order_book, observed_at=observed_at)
                crypto = context.to_crypto_snapshot(
                    strike_price=self.engine.reference_price(contract),
                    expected_annual_return=expected_annual_return,
                )
                _, opportunity = self.engine.evaluate(snapshot, crypto, contract, profile)
                evaluated_opportunities.append(
                    (market, opportunity.model_copy(update={"market_regime": regime}))
                )
            except (
                IncompleteOrderBookError,
                KalshiAPIError,
                httpx.HTTPError,
                RuntimeError,
                ValueError,
            ) as exc:
                reason = str(exc)
                failures.append(f"{market.ticker}: {reason}")
                self._record_check(
                    cycle,
                    market,
                    cycle_observed_at,
                    MarketCheckStatus.FAILED,
                    reason,
                )

        opportunities = self._apply_event_exposure_caps(evaluated_opportunities)
        watch = 0
        delivered = 0
        for market, opportunity in opportunities:
            self.repository.save_evaluation(opportunity.forecast, opportunity)
            if opportunity.state is RecommendationState.WATCH:
                watch += 1
                self._record_check(
                    cycle,
                    market,
                    opportunity.market.observed_at,
                    MarketCheckStatus.WATCH,
                    "; ".join(opportunity.warnings) or None,
                    opportunity=opportunity,
                )
                continue
            try:
                self._record_check(
                    cycle,
                    market,
                    opportunity.market.observed_at,
                    MarketCheckStatus.ENTRY_CANDIDATE,
                    None,
                    opportunity=opportunity,
                )
                await self.alert_service.publish(opportunity)
                delivered += 1
                self._record_check(
                    cycle,
                    market,
                    opportunity.market.observed_at,
                    MarketCheckStatus.DELIVERED,
                    None,
                    opportunity=opportunity,
                )
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                reason = str(exc)
                failures.append(f"{market.ticker}: {reason}")
                self._record_check(
                    cycle,
                    market,
                    opportunity.market.observed_at,
                    MarketCheckStatus.FAILED,
                    reason,
                    opportunity=opportunity,
                )

        return PaperAlertCycleResult(
            discovered=len(markets),
            evaluated=len(opportunities),
            watch=watch,
            delivered=delivered,
            unsupported=unsupported,
            uncalibrated=uncalibrated,
            failures=tuple(failures),
        )

    def _apply_event_exposure_caps(
        self,
        evaluated: list[tuple[KalshiMarket, Opportunity]],
    ) -> list[tuple[KalshiMarket, Opportunity]]:
        adjusted = list(evaluated)
        remaining_by_event: dict[str, float] = {}

        def entry_edge(index: int) -> float:
            edge = adjusted[index][1].conservative_net_edge
            return edge if edge is not None else float("-inf")

        entry_indexes = sorted(
            (
                index
                for index, (_, opportunity) in enumerate(adjusted)
                if opportunity.state
                in {RecommendationState.ENTER_YES, RecommendationState.ENTER_NO}
            ),
            key=entry_edge,
            reverse=True,
        )
        for index in entry_indexes:
            market, opportunity = adjusted[index]
            event_id = opportunity.market.event_id or market.event_ticker
            remaining = remaining_by_event.setdefault(event_id, self.engine.event_exposure_cap)
            allowed = round(min(opportunity.suggested_max_exposure, remaining), 2)
            remaining_by_event[event_id] = max(remaining - allowed, 0.0)
            if allowed == opportunity.suggested_max_exposure:
                continue
            warning = (
                f"Event-level exposure cap reduced this entry from "
                f"${opportunity.suggested_max_exposure:,.2f} to ${allowed:,.2f}."
            )
            update: dict[str, object] = {
                "suggested_max_exposure": allowed,
                "warnings": (*opportunity.warnings, warning),
            }
            if allowed <= 0:
                update["state"] = RecommendationState.WATCH
            adjusted[index] = (market, opportunity.model_copy(update=update))
        return adjusted

    def _record_check(
        self,
        cycle_id: str,
        market: KalshiMarket,
        observed_at: datetime,
        status: MarketCheckStatus,
        reason: str | None,
        *,
        opportunity: Opportunity | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "market": market.model_dump(mode="json"),
            "app_mapping": {
                "series_ticker": market.normalized_series_ticker,
                "event_ticker": market.event_ticker,
                "contract_label": market.contract_label,
                "market_url": str(market.market_url),
            },
        }
        if opportunity is not None:
            payload["opportunity"] = opportunity.model_dump(mode="json")
        self.repository.save_paper_market_check(
            cycle_id=cycle_id,
            market_id=market.ticker,
            series_ticker=market.normalized_series_ticker,
            event_ticker=market.event_ticker,
            observed_at=observed_at,
            status=status,
            reason=reason,
            payload=payload,
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
