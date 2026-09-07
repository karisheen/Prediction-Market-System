from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
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


class RetrievedResearchModel(ResearchModel):
    """Provider observation with a local availability timestamp."""

    retrieved_at: datetime


class SpotCandle(RetrievedResearchModel):
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


class VolatilityObservation(RetrievedResearchModel):
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


class FundingObservation(RetrievedResearchModel):
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


class DerivativesSnapshot(RetrievedResearchModel):
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


class EventDataSnapshot(RetrievedResearchModel):
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
    optional_max_age_seconds: Annotated[int, Field(gt=0)] = 2 * 60 * 60
    event_max_age_seconds: Annotated[int, Field(gt=0)] = 6 * 60 * 60

    @field_validator("as_of")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        self._validate_required_at(self.as_of)
        if self.spot.product_id != f"{self.symbol}-USD":
            raise ResearchDataUnavailable("spot product does not match research symbol")
        warnings = list(self.warnings)
        for name, maximum_age in (
            ("implied_volatility", self.optional_max_age_seconds),
            ("funding", self.optional_max_age_seconds),
            ("derivatives", self.optional_max_age_seconds),
            ("event_data", self.event_max_age_seconds),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            age = (self.as_of - value.observed_at).total_seconds()
            invalid = age < 0 or age > maximum_age or value.retrieved_at > self.as_of
            if name == "implied_volatility":
                invalid = invalid or (
                    value.window_seconds
                    != self.realized_volatility.raw_payload.get("interval_seconds")
                    or (value.observed_at - value.source_start_at).total_seconds()
                    != value.window_seconds
                    or value.raw_payload.get("resolution_seconds", value.window_seconds)
                    != value.window_seconds
                )
            if invalid:
                object.__setattr__(self, name, None)
                warnings.append(f"Omitted inadmissible {name} at context boundary.")
        object.__setattr__(self, "warnings", tuple(warnings))
        return self

    def _validate_required_at(self, evaluated_at: datetime) -> None:
        if self.as_of > evaluated_at:
            raise ResearchDataUnavailable("research context is from the future")
        if self.spot.end_at > evaluated_at or self.spot.retrieved_at > evaluated_at:
            raise ResearchDataUnavailable("required spot is not yet available")
        realized = self.realized_volatility
        if (
            realized.observed_at > evaluated_at
            or realized.source_start_at > evaluated_at
            or realized.retrieved_at > evaluated_at
        ):
            raise ResearchDataUnavailable("required realized volatility is not yet available")
        interval = realized.raw_payload.get("interval_seconds")
        if not isinstance(interval, int) or interval <= 0:
            raise ResearchDataUnavailable("realized volatility lacks interval provenance")
        boundary = datetime.fromtimestamp(
            (int(evaluated_at.timestamp()) // interval) * interval, UTC
        )
        if realized.observed_at != boundary:
            raise ResearchDataUnavailable("realized volatility is stale at completed boundary")
        if (
            realized.kind != "realized"
            or realized.symbol != self.symbol
            or realized.window_seconds % interval
            or realized.source_start_at != boundary - timedelta(seconds=realized.window_seconds)
            or realized.raw_payload.get("return_count") != realized.window_seconds // interval
            or not isinstance(realized.raw_payload.get("source_revisions_sha256"), str)
        ):
            raise ResearchDataUnavailable("realized volatility lacks full-window provenance")

    def validate_at(self, evaluated_at: datetime, maximum_spot_age_seconds: int = 120) -> None:
        evaluated_at = _as_utc(evaluated_at)
        if maximum_spot_age_seconds < 0:
            raise ValueError("maximum spot age must be non-negative")
        self._validate_required_at(evaluated_at)
        if (evaluated_at - self.spot.end_at).total_seconds() > maximum_spot_age_seconds:
            raise ResearchDataUnavailable("required spot is stale at evaluation")
        for name, maximum_age in (
            ("implied_volatility", self.optional_max_age_seconds),
            ("funding", self.optional_max_age_seconds),
            ("derivatives", self.optional_max_age_seconds),
            ("event_data", self.event_max_age_seconds),
        ):
            value = getattr(self, name)
            if value is not None and (
                value.observed_at > evaluated_at
                or value.retrieved_at > evaluated_at
                or (evaluated_at - value.observed_at).total_seconds() > maximum_age
            ):
                raise ResearchDataUnavailable(f"{name} is inadmissible at evaluation")

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
        self._validate_required_at(self.as_of)
        selected = self.implied_volatility or self.realized_volatility
        return CryptoSnapshot(
            symbol=self.symbol,
            observed_at=self.spot.end_at,
            spot_price=float(self.spot.close),
            strike_price=strike_price,
            annualized_volatility=self.selected_annualized_volatility,
            expected_annual_return=expected_annual_return,
            feature_recipe={
                "semantics": "point-in-time-complete-windows-v2",
                "spot_provider": self.spot.provider,
                "spot_product": self.spot.product_id,
                "spot_interval_seconds": self.spot.interval_seconds,
                "volatility_selection": "matching-interval-dvol-else-realized",
                "selected_volatility_provider": selected.provider,
                "selected_volatility_kind": selected.kind,
                "selected_volatility_window_seconds": selected.window_seconds,
                "realized_interval_seconds": self.realized_volatility.raw_payload[
                    "interval_seconds"
                ],
                "realized_window_seconds": self.realized_volatility.window_seconds,
                "realized_estimator": "sample-log-return-standard-deviation-365d",
                "optional_max_age_seconds": self.optional_max_age_seconds,
            },
            input_provenance=self.provenance(),
        )

    def content_id(self) -> str:
        """Content address of the exact context, including source availability times."""
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def provenance(self) -> dict[str, Any]:
        """Compact, persistable identity of the research inputs behind a forecast.

        The full context (hundreds of kilobytes of source candles) is stored once,
        content-addressed, by the repository; forecasts carry only this reference.
        """
        selected = self.implied_volatility or self.realized_volatility
        realized = self.realized_volatility
        return {
            "research_context_id": self.content_id(),
            "as_of": self.as_of.isoformat(),
            "spot_end_at": self.spot.end_at.isoformat(),
            "spot_retrieved_at": self.spot.retrieved_at.isoformat(),
            "spot_revision": research_payload_hash(self.spot),
            "realized_source_start_at": realized.source_start_at.isoformat(),
            "realized_observed_at": realized.observed_at.isoformat(),
            "realized_revision": research_payload_hash(realized),
            "selected_volatility": {
                "provider": selected.provider,
                "kind": selected.kind,
                "observed_at": selected.observed_at.isoformat(),
                "revision": research_payload_hash(selected),
            },
            "optional_inputs_present": {
                name: getattr(self, name) is not None
                for name in ("implied_volatility", "funding", "derivatives", "event_data")
            },
        }


def calculate_realized_volatility(
    candles: Sequence[SpotCandle],
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

    if window_seconds % interval_seconds:
        raise ValueError("realized-volatility window must contain whole candle intervals")
    boundary = datetime.fromtimestamp(
        (int(as_of.timestamp()) // interval_seconds) * interval_seconds, UTC
    )
    window_start = boundary - timedelta(seconds=window_seconds)
    eligible = [
        candle
        for candle in ordered
        if window_start <= candle.end_at <= boundary and candle.retrieved_at <= as_of
    ]
    expected_count = window_seconds // interval_seconds + 1
    if expected_count < 3:
        raise ResearchDataUnavailable("at least two returns are required for realized volatility")
    if len(eligible) != expected_count or any(
        candle.end_at != window_start + timedelta(seconds=index * interval_seconds)
        for index, candle in enumerate(eligible)
    ):
        raise ResearchDataUnavailable(
            "spot history must contain a full contiguous completed realized-volatility window"
        )
    returns = [
        math.log(float(current.close / previous.close))
        for previous, current in zip(eligible, eligible[1:], strict=False)
    ]

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
            # Candles stay in the immutable spot store; the digest pins the exact
            # revisions used so any later reconstruction is verifiable, not assumed.
            "source_revisions_sha256": source_revisions_digest(eligible),
        },
    )


def source_revisions_digest(candles: Sequence[SpotCandle]) -> str:
    """Order-preserving digest of candle content identities (retrieval time excluded)."""
    digest = hashlib.sha256()
    for candle in candles:
        digest.update(research_payload_hash(candle).encode())
    return digest.hexdigest()


def research_payload_hash(value: ResearchModel) -> str:
    """Stable revision identity; local retrieval time is availability, not source content."""
    payload = value.model_dump(mode="json")
    payload.pop("retrieved_at", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
