from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self
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
    yes_bid: Probability
    yes_ask: Probability
    no_bid: Probability
    no_ask: Probability
    yes_ask_size: NonNegativeFloat
    no_ask_size: NonNegativeFloat
    resolution_rule: Annotated[str, Field(min_length=1)]
    market_url: HttpUrl | None = None

    @field_validator("observed_at", "expires_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_market(self) -> Self:
        if self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be later than observed_at")
        if self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid cannot exceed yes_ask")
        if self.no_bid > self.no_ask:
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
