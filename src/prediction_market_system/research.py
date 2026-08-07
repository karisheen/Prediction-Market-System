from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prediction_market_system.domain import CryptoSnapshot

PositiveDecimal = Annotated[Decimal, Field(gt=0)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class ResearchDataUnavailable(RuntimeError):
    pass


class ResearchSyncStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpotCandle(ResearchModel):
    provider: str
    product_id: str
    interval_seconds: Annotated[int, Field(gt=0)]
    start_at: datetime
    end_at: datetime
    open: PositiveDecimal
    high: PositiveDecimal
    low: PositiveDecimal
    close: PositiveDecimal
    volume: NonNegativeDecimal
    retrieved_at: datetime
    raw_payload: dict[str, Any]

    @field_validator("start_at", "end_at", "retrieved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_candle(self) -> Self:
        if self.end_at <= self.start_at:
            raise ValueError("candle end must be after its start")
        if (self.end_at - self.start_at).total_seconds() != self.interval_seconds:
            raise ValueError("candle timestamps do not match interval_seconds")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("candle OHLC values are inconsistent")
        if self.low > self.high:
            raise ValueError("candle low cannot exceed high")
        return self


class VolatilityObservation(ResearchModel):
    provider: str
    symbol: str
    kind: Literal["realized", "implied"]
    window_seconds: Annotated[int, Field(gt=0)]
    source_start_at: datetime
    observed_at: datetime
    annualized_volatility: NonNegativeFloat
    retrieved_at: datetime
    raw_payload: dict[str, Any]

    @field_validator("source_start_at", "observed_at", "retrieved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.source_start_at > self.observed_at:
            raise ValueError("volatility source start cannot follow observation time")
        return self


class FundingObservation(ResearchModel):
    provider: str
    instrument_name: str
    observed_at: datetime
    index_price: float
    previous_index_price: float
    funding_rate_1h: float
    funding_rate_8h: float
    retrieved_at: datetime
    raw_payload: dict[str, Any]

    @field_validator("observed_at", "retrieved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)


class DerivativesSnapshot(ResearchModel):
    provider: str
    instrument_name: str
    observed_at: datetime
    index_price: Annotated[float, Field(gt=0.0)]
    mark_price: Annotated[float, Field(gt=0.0)]
    basis: float
    open_interest: NonNegativeFloat
    current_funding: float
    funding_rate_8h: float
    retrieved_at: datetime
    raw_payload: dict[str, Any]

    @field_validator("observed_at", "retrieved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)


class EventDataSnapshot(ResearchModel):
    provider: str
    event_ticker: str
    data_type: str
    observed_at: datetime
    retrieved_at: datetime
    is_historical: bool
    details: dict[str, Any]
    raw_payload: dict[str, Any]

    @field_validator("observed_at", "retrieved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)


class ResearchContext(ResearchModel):
    symbol: str
    event_ticker: str | None
    as_of: datetime
    spot: SpotCandle
    realized_volatility: VolatilityObservation
    implied_volatility: VolatilityObservation | None = None
    funding: FundingObservation | None = None
    derivatives: DerivativesSnapshot | None = None
    event_data: EventDataSnapshot | None = None
    warnings: tuple[str, ...] = ()

    @field_validator("as_of")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @property
    def selected_annualized_volatility(self) -> float:
        return (
            self.implied_volatility.annualized_volatility
            if self.implied_volatility is not None
            else self.realized_volatility.annualized_volatility
        )

    def to_crypto_snapshot(
        self,
        *,
        strike_price: float,
        expected_annual_return: float = 0.0,
    ) -> CryptoSnapshot:
        return CryptoSnapshot(
            symbol=self.symbol,
            observed_at=self.spot.end_at,
            spot_price=float(self.spot.close),
            strike_price=strike_price,
            annualized_volatility=self.selected_annualized_volatility,
            expected_annual_return=expected_annual_return,
        )


def calculate_realized_volatility(
    candles: list[SpotCandle],
    *,
    symbol: str,
    as_of: datetime,
    window_seconds: int,
) -> VolatilityObservation:
    as_of = _as_utc(as_of)
    if window_seconds <= 0:
        raise ValueError("realized-volatility window must be positive")
    if not candles:
        raise ResearchDataUnavailable("no spot candles are available for realized volatility")

    ordered = sorted(candles, key=lambda candle: candle.end_at)
    interval_seconds = ordered[0].interval_seconds
    product_id = ordered[0].product_id
    provider = ordered[0].provider
    if any(
        candle.interval_seconds != interval_seconds
        or candle.product_id != product_id
        or candle.provider != provider
        for candle in ordered
    ):
        raise ValueError("realized-volatility candles must share provider, product, and interval")

    window_start = as_of - timedelta(seconds=window_seconds)
    eligible = [
        candle
        for candle in ordered
        if candle.end_at <= as_of
        and candle.end_at >= window_start - timedelta(seconds=interval_seconds)
    ]
    if len(eligible) < 3:
        raise ResearchDataUnavailable(
            "at least three completed spot candles are required for realized volatility"
        )
    if eligible[0].end_at > window_start:
        raise ResearchDataUnavailable(
            "spot history does not cover the full realized-volatility window"
        )

    returns = [
        math.log(float(current.close / previous.close))
        for previous, current in zip(eligible, eligible[1:], strict=False)
        if current.end_at > window_start
    ]
    if len(returns) < 2:
        raise ResearchDataUnavailable("at least two returns are required for realized volatility")

    periods_per_year = (365 * 24 * 60 * 60) / interval_seconds
    annualized = statistics.stdev(returns) * math.sqrt(periods_per_year)
    return VolatilityObservation(
        provider=f"{provider}:calculated",
        symbol=symbol.upper(),
        kind="realized",
        window_seconds=window_seconds,
        source_start_at=eligible[0].end_at,
        observed_at=eligible[-1].end_at,
        annualized_volatility=annualized,
        retrieved_at=max(candle.retrieved_at for candle in eligible),
        raw_payload={
            "method": "sample standard deviation of log returns",
            "annualization_days": 365,
            "return_count": len(returns),
            "product_id": product_id,
            "interval_seconds": interval_seconds,
        },
    )
