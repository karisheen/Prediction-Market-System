from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketSide(StrEnum):
    YES = "YES"
    NO = "NO"


class ThresholdModelKind(StrEnum):
    TERMINAL = "terminal"
    BARRIER = "barrier"


class ThresholdDirection(StrEnum):
    ABOVE = "above"
    BELOW = "below"


class PriceTrendRegime(StrEnum):
    UPTREND = "uptrend"
    RANGE = "range"
    DOWNTREND = "downtrend"


class VolatilityRegime(StrEnum):
    LOW = "low-volatility"
    TYPICAL = "typical-volatility"
    HIGH = "high-volatility"


class ThresholdContract(FrozenModel):
    model_kind: ThresholdModelKind
    direction: ThresholdDirection
    strike_price: PositiveFloat


class TerminalRangeContract(FrozenModel):
    lower_bound: PositiveFloat
    upper_bound: PositiveFloat
    settlement_window_seconds: Annotated[int, Field(ge=0, le=3_600)] = 0

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.lower_bound >= self.upper_bound:
            raise ValueError("range lower bound must be below upper bound")
        return self


CryptoPriceContract = ThresholdContract | TerminalRangeContract


class MarketRegimeSnapshot(FrozenModel):
    symbol: Annotated[str, Field(min_length=1)]
    observed_at: datetime
    source_start_at: datetime
    trailing_return: float
    realized_volatility: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None = None
    price_trend: PriceTrendRegime
    volatility: VolatilityRegime
    trend_threshold: NonNegativeFloat
    low_volatility_threshold: NonNegativeFloat
    high_volatility_threshold: NonNegativeFloat

    @field_validator("observed_at", "source_start_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_regime(self) -> Self:
        if self.source_start_at >= self.observed_at:
            raise ValueError("regime source start must precede observation time")
        if self.low_volatility_threshold >= self.high_volatility_threshold:
            raise ValueError("low-volatility threshold must be below high-volatility threshold")
        return self

    @property
    def label(self) -> str:
        return f"{self.price_trend.value} / {self.volatility.value}"


class RecommendationState(StrEnum):
    WATCH = "WATCH"
    ENTER_YES = "ENTER YES"
    ENTER_NO = "ENTER NO"
    EDGE_WEAKENED = "EDGE WEAKENED"
    EXIT_REDUCE = "EXIT/REDUCE"
    EXPIRED = "EXPIRED"
    RESOLVED = "RESOLVED"


class MarketSnapshot(FrozenModel):
    market_id: Annotated[str, Field(min_length=1)]
    question: Annotated[str, Field(min_length=1)]
    venue: Annotated[str, Field(min_length=1)]
    observed_at: datetime
    expires_at: datetime
    yes_bid: Probability | None
    yes_ask: Probability | None
    no_bid: Probability | None
    no_ask: Probability | None
    yes_ask_size: NonNegativeFloat | None
    no_ask_size: NonNegativeFloat | None
    resolution_rule: Annotated[str, Field(min_length=1)]
    series_id: str | None = None
    event_id: str | None = None
    contract_label: str | None = None
    market_url: HttpUrl | None = None

    @field_validator("observed_at", "expires_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_market(self) -> Self:
        if self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be later than observed_at")
        if self.yes_bid is not None and self.yes_ask is not None and self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid cannot exceed yes_ask")
        if self.no_bid is not None and self.no_ask is not None and self.no_bid > self.no_ask:
            raise ValueError("no_bid cannot exceed no_ask")
        return self


class CryptoSnapshot(FrozenModel):
    symbol: Annotated[str, Field(min_length=1)]
    observed_at: datetime
    spot_price: PositiveFloat
    strike_price: PositiveFloat
    annualized_volatility: PositiveFloat
    expected_annual_return: float = 0.0

    @field_validator("observed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)


class ProbabilityForecast(FrozenModel):
    forecast_id: UUID = Field(default_factory=uuid4)
    market_id: Annotated[str, Field(min_length=1)]
    generated_at: datetime
    probability_yes: Probability
    lower_probability_yes: Probability
    upper_probability_yes: Probability
    structural_probability_yes: Probability
    market_probability_yes: Probability
    model_name: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]
    uncertainty_margin: Probability
    uncertainty_source: Literal["fixed", "held_out"]
    calibration_profile_id: UUID | None = None
    supporting_evidence: tuple[str, ...] = ()
    opposing_evidence: tuple[str, ...] = ()

    @field_validator("generated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not (self.lower_probability_yes <= self.probability_yes <= self.upper_probability_yes):
            raise ValueError("forecast probability must fall within its uncertainty interval")
        return self


class Opportunity(FrozenModel):
    opportunity_id: UUID = Field(default_factory=uuid4)
    market: MarketSnapshot
    forecast: ProbabilityForecast
    state: RecommendationState
    side: MarketSide | None = None
    executable_price: Probability | None = None
    conservative_probability: Probability | None = None
    conservative_net_edge: float | None = None
    suggested_max_exposure: NonNegativeFloat = 0.0
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    market_regime: MarketRegimeSnapshot | None = None

    @model_validator(mode="after")
    def validate_opportunity(self) -> Self:
        if self.forecast.market_id != self.market.market_id:
            raise ValueError("forecast and market IDs must match")

        entering = self.state in {
            RecommendationState.ENTER_YES,
            RecommendationState.ENTER_NO,
        }
        execution_values = (
            self.side,
            self.executable_price,
            self.conservative_probability,
            self.conservative_net_edge,
        )
        if entering and any(value is None for value in execution_values):
            raise ValueError("entry recommendations require side, price, probability, and edge")
        return self
