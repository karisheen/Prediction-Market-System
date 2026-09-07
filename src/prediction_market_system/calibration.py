from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prediction_market_system.recipe import MODEL_VERSION


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationSample(CalibrationModel):
    market_id: Annotated[str, Field(min_length=1)]
    event_id: Annotated[str, Field(min_length=1)] | None = None
    symbol: Annotated[str, Field(min_length=1)]
    model_name: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]
    recipe_id: str | None = None
    horizon_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    probability_yes: Annotated[float, Field(ge=0.0, le=1.0)]
    outcome_yes: bool
    observed_at: datetime
    resolved_at: datetime

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("observed_at", "resolved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_resolution_order(self) -> Self:
        if self.resolved_at < self.observed_at:
            raise ValueError("calibration outcome cannot precede its forecast")
        return self


class CalibrationBin(CalibrationModel):
    lower_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    upper_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    mean_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    observed_frequency: Annotated[float, Field(ge=0.0, le=1.0)]
    outcome_interval_lower: Annotated[float, Field(ge=0.0, le=1.0)]
    outcome_interval_upper: Annotated[float, Field(ge=0.0, le=1.0)]
    uncertainty_margin: Annotated[float, Field(ge=0.0, le=1.0)]
    sample_count: Annotated[int, Field(gt=0)]
    minimum_horizon_seconds: Annotated[float, Field(gt=0)] | None = None
    maximum_horizon_seconds: Annotated[float, Field(gt=0)] | None = None


class UncertaintyCalibrationProfile(CalibrationModel):
    profile_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    symbol: Annotated[str, Field(min_length=1)]
    model_name: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]
    recipe_id: str | None = None
    research_only: bool = True
    training_start: datetime
    cutoff_at: datetime
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    sample_count: Annotated[int, Field(gt=0)]
    brier_score: Annotated[float, Field(ge=0.0, le=1.0)]
    bins: tuple[CalibrationBin, ...]
    independent_event_count: Annotated[int, Field(gt=0)] = 1
    method: str = "equal-frequency Wilson calibration envelope"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("generated_at", "training_start", "cutoff_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if self.cutoff_at <= self.training_start:
            raise ValueError("calibration cutoff must follow training start")
        if not self.bins:
            raise ValueError("calibration profile requires at least one bin")
        if sum(bin_.sample_count for bin_ in self.bins) != self.sample_count:
            raise ValueError("calibration bin counts do not match sample_count")
        if self.independent_event_count > self.sample_count:
            raise ValueError("independent event count cannot exceed calibration sample count")
        return self

    def margin_for(self, probability_yes: float, *, horizon_seconds: float | None = None) -> float:
        if not 0.0 <= probability_yes <= 1.0:
            raise ValueError("probability must be between zero and one")
        if horizon_seconds is not None and (
            not math.isfinite(horizon_seconds) or horizon_seconds <= 0
        ):
            raise ValueError("calibration horizon must be finite and positive")
        matching = tuple(
            bin_
            for bin_ in self.bins
            if bin_.lower_probability <= probability_yes <= bin_.upper_probability
            and (
                horizon_seconds is None
                or (
                    bin_.minimum_horizon_seconds is not None
                    and bin_.maximum_horizon_seconds is not None
                    and bin_.minimum_horizon_seconds
                    <= horizon_seconds
                    <= bin_.maximum_horizon_seconds
                )
            )
        )
        if matching:
            # The envelope includes within-bin heterogeneity. Never extrapolate
            # calibration into an unobserved probability region.
            return max(bin_.uncertainty_margin for bin_ in matching)
        return 1.0


def fit_uncertainty_profiles(
    samples: tuple[CalibrationSample, ...],
    *,
    training_start: datetime,
    cutoff_at: datetime,
    confidence_level: float = 0.95,
    minimum_samples: int = 30,
    maximum_bins: int = 5,
) -> tuple[UncertaintyCalibrationProfile, ...]:
    training_start = _as_utc(training_start)
    cutoff_at = _as_utc(cutoff_at)
    if cutoff_at <= training_start:
        raise ValueError("calibration cutoff must follow training start")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if minimum_samples <= 0 or maximum_bins <= 0:
        raise ValueError("calibration sample and bin limits must be positive")
    if any(sample.resolved_at > cutoff_at for sample in samples):
        raise ValueError("calibration samples include outcomes unavailable at cutoff")
    if any(
        sample.observed_at < training_start or sample.observed_at >= cutoff_at for sample in samples
    ):
        raise ValueError("calibration forecasts fall outside the training window")

    grouped: dict[tuple[str, str, str, str | None], list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(
            (sample.symbol, sample.model_name, sample.model_version, sample.recipe_id),
            [],
        ).append(sample)

    profiles: list[UncertaintyCalibrationProfile] = []
    for (symbol, model_name, model_version, recipe_id), group in sorted(
        grouped.items(), key=lambda item: tuple(value or "" for value in item[0])
    ):
        unique = _unique_market_samples(group)
        independent_events = {sample.event_id or sample.market_id for sample in unique}
        if len(independent_events) < minimum_samples:
            continue
        ordered = sorted(unique, key=lambda sample: sample.probability_yes)
        # Fixed probability regions cannot be merged merely to meet a sample
        # threshold: opposite conditional errors in a ladder would cancel.
        chunks: dict[tuple[int, int | None], list[CalibrationSample]] = {}
        for sample in ordered:
            index = min(int(sample.probability_yes * maximum_bins), maximum_bins - 1)
            horizon_bin = (
                math.floor(math.log2(sample.horizon_seconds))
                if sample.horizon_seconds is not None
                else None
            )
            chunks.setdefault((index, horizon_bin), []).append(sample)
        bins = tuple(
            _clustered_calibration_bin(
                tuple(chunk),
                confidence_level,
                min(sample.probability_yes for sample in chunk),
                max(sample.probability_yes for sample in chunk),
                len(chunks),
            )
            for _, chunk in sorted(
                chunks.items(),
                key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
            )
        )
        brier_score = _event_weighted_brier_score(ordered)
        profiles.append(
            UncertaintyCalibrationProfile(
                symbol=symbol,
                model_name=model_name,
                model_version=model_version,
                recipe_id=recipe_id,
                research_only=(
                    recipe_id is None
                    or model_version != MODEL_VERSION
                    or any(sample.horizon_seconds is None for sample in unique)
                ),
                training_start=training_start,
                cutoff_at=cutoff_at,
                confidence_level=confidence_level,
                sample_count=sum(bin_.sample_count for bin_ in bins),
                independent_event_count=len(independent_events),
                brier_score=brier_score,
                bins=bins,
                method="fixed-probability event-clustered Hoeffding envelope",
            )
        )
    return tuple(profiles)


def _unique_market_samples(samples: list[CalibrationSample]) -> list[CalibrationSample]:
    selected: dict[str, CalibrationSample] = {}
    for sample in sorted(samples, key=lambda value: value.observed_at):
        selected.setdefault(sample.market_id, sample)
    return list(selected.values())


def _clustered_calibration_bin(
    samples: tuple[CalibrationSample, ...],
    confidence_level: float,
    lower_probability: float,
    upper_probability: float,
    bin_count: int,
) -> CalibrationBin:
    by_event: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        by_event.setdefault(sample.event_id or sample.market_id, []).append(sample)
    event_observations = tuple(
        (
            sum(sample.probability_yes for sample in group) / len(group),
            sum(float(sample.outcome_yes) for sample in group) / len(group),
        )
        for group in by_event.values()
    )
    count = len(event_observations)
    mean_probability = sum(value[0] for value in event_observations) / count
    observed_frequency = sum(value[1] for value in event_observations) / count
    # Independent event-average outcomes are bounded in [0, 1]. Hoeffding
    # remains nondegenerate even for identical ladders/zero empirical variance.
    # Bonferroni supplies simultaneous coverage across the fixed regions.
    half_width = math.sqrt(math.log(2 * bin_count / (1 - confidence_level)) / (2 * count))
    lower = max(0.0, observed_frequency - half_width)
    upper = min(1.0, observed_frequency + half_width)
    margin = max(abs(lower_probability - upper), abs(upper_probability - lower))
    # A wide bin cannot establish conditional calibration across its interior.
    # Refuse pooling opposite-tail errors when a caller requests very few bins.
    if upper_probability - lower_probability > 0.2 + 1e-12:
        margin = 1.0
    return CalibrationBin(
        lower_probability=lower_probability,
        upper_probability=upper_probability,
        mean_probability=mean_probability,
        observed_frequency=observed_frequency,
        outcome_interval_lower=lower,
        outcome_interval_upper=upper,
        uncertainty_margin=min(margin, 1.0),
        sample_count=count,
        minimum_horizon_seconds=min(
            (sample.horizon_seconds for sample in samples if sample.horizon_seconds is not None),
            default=None,
        ),
        maximum_horizon_seconds=max(
            (sample.horizon_seconds for sample in samples if sample.horizon_seconds is not None),
            default=None,
        ),
    )


def _event_weighted_brier_score(samples: list[CalibrationSample]) -> float:
    by_event: dict[str, list[float]] = {}
    for sample in samples:
        by_event.setdefault(sample.event_id or sample.market_id, []).append(
            (sample.probability_yes - float(sample.outcome_yes)) ** 2
        )
    return sum(sum(scores) / len(scores) for scores in by_event.values()) / len(by_event)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)
