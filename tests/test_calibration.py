from datetime import UTC, datetime, timedelta

import pytest

from prediction_market_system.calibration import (
    CalibrationSample,
    fit_uncertainty_profiles,
)

CUTOFF = datetime(2026, 1, 1, tzinfo=UTC)
MODEL_NAME = "crypto-terminal-above-threshold-market-anchor"
MODEL_VERSION = "0.3.0"


def sample(
    index: int,
    probability: float,
    outcome: bool,
    *,
    model_version: str = MODEL_VERSION,
) -> CalibrationSample:
    return CalibrationSample(
        market_id=f"market-{index}",
        symbol="btc",
        model_name=MODEL_NAME,
        model_version=model_version,
        probability_yes=probability,
        outcome_yes=outcome,
        observed_at=CUTOFF - timedelta(days=30 - index),
        resolved_at=CUTOFF - timedelta(days=20 - index),
    )


def test_fits_equal_frequency_wilson_uncertainty_profile() -> None:
    samples = tuple(
        sample(index, probability, outcome)
        for index, (probability, outcome) in enumerate(
            zip(
                (0.10, 0.20, 0.35, 0.60, 0.75, 0.90),
                (False, False, True, False, True, True),
                strict=True,
            )
        )
    )

    profiles = fit_uncertainty_profiles(
        samples,
        training_start=CUTOFF - timedelta(days=60),
        cutoff_at=CUTOFF,
        minimum_samples=3,
        maximum_bins=2,
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.symbol == "BTC"
    assert profile.model_version == MODEL_VERSION
    assert profile.sample_count == 6
    assert len(profile.bins) == 2
    assert tuple(bin_.sample_count for bin_ in profile.bins) == (3, 3)
    assert profile.brier_score == pytest.approx(0.1508333333)
    assert profile.margin_for(0.15) == profile.bins[0].uncertainty_margin
    assert profile.margin_for(0.85) == profile.bins[1].uncertainty_margin
    assert all(bin_.outcome_interval_lower <= bin_.observed_frequency for bin_ in profile.bins)
    assert all(bin_.observed_frequency <= bin_.outcome_interval_upper for bin_ in profile.bins)


def test_profiles_are_isolated_by_model_version() -> None:
    samples = (
        sample(0, 0.20, False, model_version="0.2.0"),
        sample(1, 0.80, True, model_version="0.3.0"),
    )

    profiles = fit_uncertainty_profiles(
        samples,
        training_start=CUTOFF - timedelta(days=60),
        cutoff_at=CUTOFF,
        minimum_samples=1,
        maximum_bins=1,
    )

    assert {profile.model_version for profile in profiles} == {"0.2.0", "0.3.0"}


def test_rejects_forecasts_or_outcomes_unavailable_in_training_window() -> None:
    future_outcome = sample(0, 0.50, True).model_copy(
        update={"resolved_at": CUTOFF + timedelta(seconds=1)}
    )
    outside_forecast = sample(1, 0.50, True).model_copy(update={"observed_at": CUTOFF})

    with pytest.raises(ValueError, match="outcomes unavailable"):
        fit_uncertainty_profiles(
            (future_outcome,),
            training_start=CUTOFF - timedelta(days=60),
            cutoff_at=CUTOFF,
            minimum_samples=1,
        )
    with pytest.raises(ValueError, match="outside the training window"):
        fit_uncertainty_profiles(
            (outside_forecast,),
            training_start=CUTOFF - timedelta(days=60),
            cutoff_at=CUTOFF,
            minimum_samples=1,
        )
