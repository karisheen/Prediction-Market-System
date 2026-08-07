from __future__ import annotations

import math
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationSample(CalibrationModel):
    market_id: Annotated[str, Field(min_length=1)]
    event_id: Annotated[str, Field(min_length=1)] | None = None
    symbol: Annotated[str, Field(min_length=1)]
    model_name: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]
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


class UncertaintyCalibrationProfile(CalibrationModel):
    profile_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    symbol: Annotated[str, Field(min_length=1)]
    model_name: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]
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

    def margin_for(self, probability_yes: float) -> float:
        if not 0.0 <= probability_yes <= 1.0:
            raise ValueError("probability must be between zero and one")
        matching = tuple(
            bin_
            for bin_ in self.bins
            if bin_.lower_probability <= probability_yes <= bin_.upper_probability
        )
        if matching:
            nearest_match = min(
                matching,
                key=lambda bin_: abs(bin_.mean_probability - probability_yes),
            )
            return nearest_match.uncertainty_margin
        nearest = min(self.bins, key=lambda bin_: abs(bin_.mean_probability - probability_yes))
        return nearest.uncertainty_margin


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

    grouped: dict[tuple[str, str, str], list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(
            (sample.symbol, sample.model_name, sample.model_version),
            [],
        ).append(sample)

    profiles: list[UncertaintyCalibrationProfile] = []
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    for (symbol, model_name, model_version), group in sorted(grouped.items()):
        unique = _unique_market_samples(group)
        independent_events = {sample.event_id or sample.market_id for sample in unique}
        if len(independent_events) < minimum_samples:
            continue
        ordered = sorted(unique, key=lambda sample: sample.probability_yes)
        clustered = any(sample.event_id is not None for sample in ordered)
        bin_count = min(maximum_bins, max(1, len(independent_events) // minimum_samples))
        while True:
            chunks = _equal_chunks(ordered, bin_count)
            if not clustered:
                bins = tuple(_calibration_bin(chunk, z_score) for chunk in chunks)
                break
            bins = tuple(_clustered_calibration_bin(chunk, z_score) for chunk in chunks)
            if all(bin_.sample_count >= minimum_samples for bin_ in bins) or bin_count == 1:
                break
            bin_count -= 1
        if any(bin_.sample_count < minimum_samples for bin_ in bins):
            continue
        brier_score = _event_weighted_brier_score(ordered)
        profiles.append(
            UncertaintyCalibrationProfile(
                symbol=symbol,
                model_name=model_name,
                model_version=model_version,
                training_start=training_start,
                cutoff_at=cutoff_at,
                confidence_level=confidence_level,
                sample_count=sum(bin_.sample_count for bin_ in bins),
                independent_event_count=len(independent_events),
                brier_score=brier_score,
                bins=bins,
                method=(
                    "equal-frequency event-clustered calibration envelope"
                    if clustered
                    else "equal-frequency Wilson calibration envelope"
                ),
            )
        )
    return tuple(profiles)


def _unique_market_samples(samples: list[CalibrationSample]) -> list[CalibrationSample]:
    selected: dict[str, CalibrationSample] = {}
    for sample in sorted(samples, key=lambda value: value.observed_at):
        selected.setdefault(sample.market_id, sample)
    return list(selected.values())


def _equal_chunks(
    samples: list[CalibrationSample],
    chunk_count: int,
) -> tuple[tuple[CalibrationSample, ...], ...]:
    base_size, remainder = divmod(len(samples), chunk_count)
    chunks: list[tuple[CalibrationSample, ...]] = []
    start = 0
    for index in range(chunk_count):
        size = base_size + int(index < remainder)
        chunks.append(tuple(samples[start : start + size]))
        start += size
    return tuple(chunks)


def _calibration_bin(
    samples: tuple[CalibrationSample, ...],
    z_score: float,
) -> CalibrationBin:
    count = len(samples)
    mean_probability = sum(sample.probability_yes for sample in samples) / count
    successes = sum(sample.outcome_yes for sample in samples)
    observed_frequency = successes / count
    lower, upper = _wilson_interval(successes, count, z_score)
    margin = max(abs(mean_probability - lower), abs(upper - mean_probability))
    return CalibrationBin(
        lower_probability=min(sample.probability_yes for sample in samples),
        upper_probability=max(sample.probability_yes for sample in samples),
        mean_probability=mean_probability,
        observed_frequency=observed_frequency,
        outcome_interval_lower=lower,
        outcome_interval_upper=upper,
        uncertainty_margin=min(margin, 1.0),
        sample_count=count,
    )


def _clustered_calibration_bin(
    samples: tuple[CalibrationSample, ...],
    z_score: float,
) -> CalibrationBin:
    by_event: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        by_event.setdefault(sample.event_id or sample.market_id, []).append(sample)
    event_observations = tuple(
        (
            sum(sample.probability_yes for sample in event_samples) / len(event_samples),
            sum(float(sample.outcome_yes) for sample in event_samples) / len(event_samples),
        )
        for event_samples in by_event.values()
    )
    count = len(event_observations)
    mean_probability = sum(value[0] for value in event_observations) / count
    observed_frequency = sum(value[1] for value in event_observations) / count
    residuals = tuple(observed - probability for probability, observed in event_observations)
    mean_residual = sum(residuals) / count
    if count == 1:
        residual_half_width = 1.0
    else:
        residual_variance = sum((residual - mean_residual) ** 2 for residual in residuals) / (
            count - 1
        )
        residual_half_width = z_score * math.sqrt(residual_variance / count)
    lower = max(0.0, mean_probability + mean_residual - residual_half_width)
    upper = min(1.0, mean_probability + mean_residual + residual_half_width)
    margin = max(abs(mean_probability - lower), abs(upper - mean_probability))
    return CalibrationBin(
        lower_probability=min(sample.probability_yes for sample in samples),
        upper_probability=max(sample.probability_yes for sample in samples),
        mean_probability=mean_probability,
        observed_frequency=observed_frequency,
        outcome_interval_lower=lower,
        outcome_interval_upper=upper,
        uncertainty_margin=min(margin, 1.0),
        sample_count=count,
    )


def _event_weighted_brier_score(samples: list[CalibrationSample]) -> float:
    by_event: dict[str, list[float]] = {}
    for sample in samples:
        by_event.setdefault(sample.event_id or sample.market_id, []).append(
            (sample.probability_yes - float(sample.outcome_yes)) ** 2
        )
    return sum(sum(scores) / len(scores) for scores in by_event.values()) / len(by_event)


def _wilson_interval(successes: int, count: int, z_score: float) -> tuple[float, float]:
    frequency = successes / count
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / count
    center = (frequency + z_squared / (2.0 * count)) / denominator
    half_width = (
        z_score
        * math.sqrt(frequency * (1.0 - frequency) / count + z_squared / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)
