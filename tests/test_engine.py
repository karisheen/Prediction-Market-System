import math
from datetime import UTC, datetime, timedelta

import pytest

from prediction_market_system.calibration import (
    CalibrationBin,
    UncertaintyCalibrationProfile,
)
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSide,
    MarketSnapshot,
    RecommendationState,
    TerminalRangeContract,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.engine import (
    CryptoThresholdEngine,
    EngineConfig,
    barrier_hitting_probability,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)

TERMINAL_ABOVE = ThresholdContract(
    model_kind=ThresholdModelKind.TERMINAL,
    direction=ThresholdDirection.ABOVE,
    strike_price=100.0,
)


def market_snapshot(
    *,
    yes_bid: float | None = 0.39,
    yes_ask: float | None = 0.42,
    no_bid: float | None = 0.57,
    no_ask: float | None = 0.60,
    expires_in: timedelta = timedelta(days=30),
) -> MarketSnapshot:
    return MarketSnapshot(
        market_id="btc-threshold",
        question="Will BTC be above 100 USD at expiry?",
        venue="test",
        observed_at=NOW,
        expires_at=NOW + expires_in,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        yes_ask_size=1_000,
        no_ask_size=1_000,
        resolution_rule="Resolves from the test BTC index.",
    )


def crypto_snapshot(
    *,
    spot: float = 110.0,
    strike: float = 100.0,
    volatility: float = 0.50,
    expected_return: float = 0.0,
) -> CryptoSnapshot:
    return CryptoSnapshot(
        symbol="BTC",
        observed_at=NOW,
        spot_price=spot,
        strike_price=strike,
        annualized_volatility=volatility,
        expected_annual_return=expected_return,
    )


def test_recommends_yes_only_after_conservative_costs() -> None:
    engine = CryptoThresholdEngine(
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            fee_rate=0.0,
            slippage_bps=25,
            resolution_haircut=0.01,
        )
    )

    forecast, opportunity = engine.evaluate(
        market_snapshot(),
        crypto_snapshot(),
        TERMINAL_ABOVE,
    )

    assert forecast.structural_probability_yes > forecast.market_probability_yes
    assert forecast.lower_probability_yes <= forecast.probability_yes
    assert forecast.probability_yes <= forecast.upper_probability_yes
    assert forecast.uncertainty_margin == pytest.approx(0.03)
    assert forecast.uncertainty_source == "fixed"
    assert opportunity.state is RecommendationState.ENTER_YES
    assert opportunity.side is MarketSide.YES
    assert opportunity.conservative_net_edge is not None
    assert opportunity.conservative_net_edge > 0.03
    assert opportunity.suggested_max_exposure == pytest.approx(200.0)


def test_recommends_no_when_downside_is_underpriced() -> None:
    engine = CryptoThresholdEngine(EngineConfig(uncertainty_margin=0.03, structural_weight=0.80))
    market = market_snapshot(
        yes_bid=0.58,
        yes_ask=0.61,
        no_bid=0.38,
        no_ask=0.41,
    )

    _, opportunity = engine.evaluate(market, crypto_snapshot(spot=80.0), TERMINAL_ABOVE)

    assert opportunity.state is RecommendationState.ENTER_NO
    assert opportunity.side is MarketSide.NO
    assert opportunity.conservative_net_edge is not None
    assert opportunity.conservative_net_edge > 0.03


def test_evaluates_terminal_range_probability() -> None:
    engine = CryptoThresholdEngine(
        EngineConfig(
            uncertainty_margin=0.0,
            structural_weight=1.0,
            min_conservative_edge=0.0,
            binary_fee_coefficient=0.0,
            slippage_bps=0.0,
            resolution_haircut=0.0,
        )
    )
    contract = TerminalRangeContract(lower_bound=90.0, upper_bound=130.0)

    forecast, opportunity = engine.evaluate(
        market_snapshot(expires_in=timedelta(days=1)),
        crypto_snapshot(spot=110.0, strike=110.0, volatility=0.10),
        contract,
    )

    assert forecast.model_name == "crypto-terminal-range-market-anchor"
    assert forecast.structural_probability_yes > 0.99
    assert opportunity.side is MarketSide.YES


def test_models_sixty_second_terminal_settlement_average() -> None:
    engine = CryptoThresholdEngine(EngineConfig(structural_weight=1.0))
    market = market_snapshot(expires_in=timedelta(minutes=10))
    crypto = crypto_snapshot(spot=110.0, strike=110.0, volatility=2.0)
    instant = TerminalRangeContract(lower_bound=109.0, upper_bound=111.0)
    averaged = instant.model_copy(update={"settlement_window_seconds": 60})

    instant_forecast, _ = engine.evaluate(market, crypto, instant)
    averaged_forecast, _ = engine.evaluate(market, crypto, averaged)

    assert averaged_forecast.structural_probability_yes > (
        instant_forecast.structural_probability_yes
    )


def test_evaluates_only_the_executable_side_of_one_sided_book() -> None:
    engine = CryptoThresholdEngine(
        EngineConfig(
            min_conservative_edge=0.0,
            binary_fee_coefficient=0.0,
            slippage_bps=0.0,
            resolution_haircut=0.0,
        )
    )
    market = market_snapshot(
        yes_bid=None,
        yes_ask=0.42,
        no_bid=0.58,
        no_ask=None,
    ).model_copy(update={"no_ask_size": None})

    _, opportunity = engine.evaluate(market, crypto_snapshot(), TERMINAL_ABOVE)

    assert opportunity.side is MarketSide.YES
    assert opportunity.executable_price == pytest.approx(0.42)


def test_returns_watch_when_contract_is_too_close_to_expiry() -> None:
    engine = CryptoThresholdEngine(EngineConfig(minimum_seconds_to_expiry=300))

    _, opportunity = engine.evaluate(
        market_snapshot(expires_in=timedelta(seconds=120)),
        crypto_snapshot(),
        TERMINAL_ABOVE,
    )

    assert opportunity.state is RecommendationState.WATCH
    assert opportunity.suggested_max_exposure == 0.0
    assert any("too close to expiry" in warning for warning in opportunity.warnings)


def test_binary_fee_curve_reduces_conservative_edge() -> None:
    no_fee_engine = CryptoThresholdEngine(
        EngineConfig(uncertainty_margin=0.03, structural_weight=0.70)
    )
    fee_engine = CryptoThresholdEngine(
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            binary_fee_coefficient=0.07,
        )
    )

    _, no_fee = no_fee_engine.evaluate(market_snapshot(), crypto_snapshot(), TERMINAL_ABOVE)
    _, with_fee = fee_engine.evaluate(market_snapshot(), crypto_snapshot(), TERMINAL_ABOVE)

    assert no_fee.conservative_net_edge is not None
    assert with_fee.conservative_net_edge is not None
    assert with_fee.conservative_net_edge < no_fee.conservative_net_edge


def test_flat_binary_fee_is_charged_per_contract() -> None:
    quadratic_engine = CryptoThresholdEngine(
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            binary_fee_type="quadratic",
            binary_fee_coefficient=0.05,
        )
    )
    flat_engine = CryptoThresholdEngine(
        EngineConfig(
            uncertainty_margin=0.03,
            structural_weight=0.70,
            binary_fee_type="flat",
            binary_fee_coefficient=0.05,
        )
    )

    _, quadratic = quadratic_engine.evaluate(
        market_snapshot(),
        crypto_snapshot(),
        TERMINAL_ABOVE,
    )
    _, flat = flat_engine.evaluate(market_snapshot(), crypto_snapshot(), TERMINAL_ABOVE)

    assert quadratic.conservative_net_edge is not None
    assert flat.conservative_net_edge is not None
    assert flat.conservative_net_edge < quadratic.conservative_net_edge


def test_uses_only_matching_past_calibration_profile() -> None:
    engine = CryptoThresholdEngine()
    profile = UncertaintyCalibrationProfile(
        symbol="BTC",
        model_name=engine.model_name(TERMINAL_ABOVE),
        model_version=engine.model_version,
        recipe_id=engine.recipe_id(crypto_snapshot()),
        training_start=NOW - timedelta(days=60),
        cutoff_at=NOW - timedelta(seconds=1),
        confidence_level=0.95,
        sample_count=30,
        brier_score=0.20,
        bins=(
            CalibrationBin(
                lower_probability=0.0,
                upper_probability=1.0,
                mean_probability=0.5,
                observed_frequency=0.5,
                outcome_interval_lower=0.3,
                outcome_interval_upper=0.7,
                uncertainty_margin=0.20,
                sample_count=30,
                minimum_horizon_seconds=1,
                maximum_horizon_seconds=31557600,
            ),
        ),
    )

    forecast, _ = engine.evaluate(
        market_snapshot(),
        crypto_snapshot(),
        TERMINAL_ABOVE,
        profile,
    )

    assert forecast.uncertainty_margin == pytest.approx(0.20)
    assert forecast.uncertainty_source == "held_out"
    assert forecast.calibration_profile_id == profile.profile_id

    with pytest.raises(ValueError, match="outcomes unavailable"):
        engine.evaluate(
            market_snapshot(),
            crypto_snapshot(),
            TERMINAL_ABOVE,
            profile.model_copy(update={"cutoff_at": NOW + timedelta(seconds=1)}),
        )
    with pytest.raises(ValueError, match="model version"):
        engine.evaluate(
            market_snapshot(),
            crypto_snapshot(),
            TERMINAL_ABOVE,
            profile.model_copy(update={"model_version": "old"}),
        )
    with pytest.raises(ValueError, match="recipe"):
        CryptoThresholdEngine(EngineConfig(structural_weight=0.7)).evaluate(
            market_snapshot(), crypto_snapshot(), TERMINAL_ABOVE, profile
        )
    with pytest.raises(ValueError, match="recipe"):
        engine.evaluate(
            market_snapshot(),
            crypto_snapshot(),
            TERMINAL_ABOVE,
            profile.model_copy(update={"recipe_id": None}),
        )


def test_barrier_probability_matches_reflection_principle() -> None:
    sigma = 0.50
    years = 0.50
    distance = math.log(1.20)
    expected = math.erfc(distance / (sigma * math.sqrt(2.0 * years)))

    probability = barrier_hitting_probability(
        spot_price=100.0,
        strike_price=120.0,
        annualized_volatility=sigma,
        expected_annual_return=0.5 * sigma**2,
        years=years,
        direction=ThresholdDirection.ABOVE,
    )

    assert probability == pytest.approx(expected)


def test_barrier_probability_supports_both_directions_and_crossed_barriers() -> None:
    upper = barrier_hitting_probability(
        spot_price=100.0,
        strike_price=120.0,
        annualized_volatility=0.50,
        expected_annual_return=0.10,
        years=0.50,
        direction=ThresholdDirection.ABOVE,
    )
    lower = barrier_hitting_probability(
        spot_price=100.0,
        strike_price=80.0,
        annualized_volatility=0.50,
        expected_annual_return=0.10,
        years=0.50,
        direction=ThresholdDirection.BELOW,
    )
    crossed = barrier_hitting_probability(
        spot_price=121.0,
        strike_price=120.0,
        annualized_volatility=0.50,
        expected_annual_return=0.10,
        years=0.50,
        direction=ThresholdDirection.ABOVE,
    )

    assert upper == pytest.approx(0.5950035807)
    assert lower == pytest.approx(0.5397294383)
    assert crossed == 1.0


def test_barrier_probability_is_stable_for_short_expiry_and_extreme_tail() -> None:
    probability = barrier_hitting_probability(
        spot_price=100.0,
        strike_price=1000.0,
        annualized_volatility=0.10,
        expected_annual_return=-0.50,
        years=1.0 / (365.25 * 24.0 * 60.0),
        direction=ThresholdDirection.ABOVE,
    )

    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_barrier_model_exceeds_terminal_probability_for_same_upper_strike() -> None:
    engine = CryptoThresholdEngine(EngineConfig(structural_weight=1.0))
    crypto = crypto_snapshot(spot=100.0, strike=120.0, expected_return=0.10)
    terminal = ThresholdContract(
        model_kind=ThresholdModelKind.TERMINAL,
        direction=ThresholdDirection.ABOVE,
        strike_price=120.0,
    )
    barrier = terminal.model_copy(update={"model_kind": ThresholdModelKind.BARRIER})

    terminal_forecast, _ = engine.evaluate(
        market_snapshot(expires_in=timedelta(days=180)),
        crypto,
        terminal,
    )
    barrier_forecast, _ = engine.evaluate(
        market_snapshot(expires_in=timedelta(days=180)).model_copy(
            update={"observation_start_at": NOW}
        ),
        crypto,
        barrier,
    )

    assert barrier_forecast.structural_probability_yes > (
        terminal_forecast.structural_probability_yes
    )
    assert barrier_forecast.model_name == "crypto-barrier-above-threshold-market-anchor"


def test_rejects_crossed_order_book() -> None:
    with pytest.raises(ValueError, match="yes_bid cannot exceed yes_ask"):
        market_snapshot(yes_bid=0.50, yes_ask=0.49)


def test_rejects_future_crypto_input_at_engine_boundary() -> None:
    crypto = crypto_snapshot().model_copy(update={"observed_at": NOW + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="future"):
        CryptoThresholdEngine().evaluate(market_snapshot(), crypto, TERMINAL_ABOVE)


def test_rejects_stale_crypto_input_at_engine_boundary() -> None:
    crypto = crypto_snapshot().model_copy(update={"observed_at": NOW - timedelta(seconds=121)})
    with pytest.raises(ValueError, match="stale"):
        CryptoThresholdEngine().evaluate(market_snapshot(), crypto, TERMINAL_ABOVE)


def test_rejects_partially_elapsed_settlement_window() -> None:
    contract = TerminalRangeContract(lower_bound=99, upper_bound=101, settlement_window_seconds=60)
    with pytest.raises(ValueError, match="already started"):
        CryptoThresholdEngine().evaluate(
            market_snapshot(expires_in=timedelta(seconds=30)), crypto_snapshot(), contract
        )


def test_arithmetic_average_matches_independent_discrete_gbm_moments() -> None:
    # All 60 samples share diffusion before the trailing minute. A double sum is
    # an independent, intentionally slow oracle for the optimized moment formula.
    seconds_per_year = 365.25 * 24 * 3600
    horizon = 90 / seconds_per_year
    crypto = crypto_snapshot(spot=100, volatility=200, expected_return=0.12)
    times = [(90 - 60 + index) / seconds_per_year for index in range(60)]
    means = [100 * math.exp(0.12 * time) for time in times]
    mean = sum(means) / 60
    covariance = (
        sum(
            means[i] * means[j] * math.expm1(200**2 * min(times[i], times[j]))
            for i in range(60)
            for j in range(60)
        )
        / 60**2
    )
    log_variance = math.log1p(covariance / mean**2)
    expected = 0.5 * math.erfc(
        -(math.log(mean / 101) - log_variance / 2) / math.sqrt(2 * log_variance)
    )
    actual = CryptoThresholdEngine._averaged_terminal_above_probability(crypto, 101, horizon, 60)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_probability_uses_observation_not_trading_or_settlement_clock() -> None:
    engine = CryptoThresholdEngine(EngineConfig(structural_weight=1))
    market = market_snapshot(expires_in=timedelta(days=30)).model_copy(
        update={
            "observation_end_at": NOW + timedelta(hours=1),
            "expected_settlement_at": NOW + timedelta(days=60),
        }
    )
    forecast, _ = engine.evaluate(market, crypto_snapshot(), TERMINAL_ABOVE)
    reference, _ = engine.evaluate(
        market_snapshot(expires_in=timedelta(hours=1)), crypto_snapshot(), TERMINAL_ABOVE
    )
    assert forecast.probability_yes == pytest.approx(reference.probability_yes)


def test_diffusion_starts_at_actual_spot_observation() -> None:
    engine = CryptoThresholdEngine(EngineConfig(structural_weight=1))
    crypto = crypto_snapshot(spot=100).model_copy(
        update={"observed_at": NOW - timedelta(seconds=120)}
    )
    forecast, _ = engine.evaluate(
        market_snapshot(expires_in=timedelta(seconds=60)), crypto, TERMINAL_ABOVE
    )
    reference, _ = engine.evaluate(
        market_snapshot(expires_in=timedelta(seconds=180)),
        crypto_snapshot(spot=100),
        TERMINAL_ABOVE,
    )
    assert forecast.probability_yes == pytest.approx(reference.probability_yes)


def test_thresholds_and_ranges_use_identical_arithmetic_averaging() -> None:
    engine = CryptoThresholdEngine(EngineConfig(structural_weight=1))
    market = market_snapshot(expires_in=timedelta(seconds=60))
    lower = TERMINAL_ABOVE.model_copy(update={"settlement_window_seconds": 60})
    upper = lower.model_copy(update={"strike_price": 101})
    range_contract = TerminalRangeContract(
        lower_bound=100, upper_bound=101, settlement_window_seconds=60
    )
    lower_forecast, _ = engine.evaluate(market, crypto_snapshot(spot=100), lower)
    upper_forecast, _ = engine.evaluate(market, crypto_snapshot(spot=100, strike=101), upper)
    range_forecast, _ = engine.evaluate(market, crypto_snapshot(spot=100), range_contract)
    assert range_forecast.structural_probability_yes == pytest.approx(
        lower_forecast.structural_probability_yes - upper_forecast.structural_probability_yes,
        abs=1e-6,
    )


def test_recipe_identity_changes_only_with_probability_recipe() -> None:
    crypto = crypto_snapshot()
    engine = CryptoThresholdEngine()
    base = engine.recipe_id(crypto)
    assert engine.recipe_id(crypto.model_copy(update={"spot_price": 120})) == base
    assert engine.recipe_id(crypto.model_copy(update={"expected_annual_return": 0.1})) != base
    assert (
        engine.recipe_id(
            crypto.model_copy(update={"feature_recipe": {"source_selection": "different-provider"}})
        )
        != base
    )
    assert CryptoThresholdEngine(EngineConfig(structural_weight=0.7)).recipe_id(crypto) != base
    assert CryptoThresholdEngine(EngineConfig(fee_rate=0.1)).recipe_id(crypto) == base


def test_total_event_budget_and_rounding_never_expand_exposure() -> None:
    engine = CryptoThresholdEngine(
        EngineConfig(
            structural_weight=1,
            paper_bankroll=10,
            max_bankroll_fraction=1,
            max_event_bankroll_fraction=0.1004,
            fractional_kelly=1,
            uncertainty_margin=0,
            slippage_bps=0,
            resolution_haircut=0,
        )
    )
    _, opportunity = engine.evaluate(market_snapshot(), crypto_snapshot(spot=200), TERMINAL_ABOVE)
    assert opportunity.suggested_max_exposure <= 1.004
    assert opportunity.suggested_max_exposure == 1.0


def test_kalshi_budget_includes_rounded_fees_and_whole_contracts() -> None:
    engine = CryptoThresholdEngine(
        EngineConfig(
            structural_weight=1,
            paper_bankroll=10,
            max_bankroll_fraction=1,
            max_event_bankroll_fraction=0.1,
            fractional_kelly=1,
            uncertainty_margin=0,
            binary_fee_coefficient=0.07,
            fee_rate=0.02,
            slippage_bps=100,
            minimum_ask_size=0,
            resolution_haircut=0,
        )
    )
    market = market_snapshot().model_copy(
        update={
            "venue": "Kalshi",
            "observation_end_at": NOW + timedelta(days=30),
            "yes_ask_size": 2.9,
        }
    )
    _, opportunity = engine.evaluate(market, crypto_snapshot(spot=200), TERMINAL_ABOVE)
    # Two units: 2 * .42 * 1.03 + ceil_cent(2 * .07 * .42 * .58).
    assert opportunity.suggested_max_exposure == pytest.approx(0.9052)
    assert opportunity.suggested_max_exposure <= engine.event_exposure_cap
    assert engine.exposure_for_budget(0.42, 2.9, 0.45, whole_contracts=True) == 0


def test_forecast_manifest_reconstructs_exact_evaluation() -> None:
    engine = CryptoThresholdEngine()
    forecast, _ = engine.evaluate(market_snapshot(), crypto_snapshot(), TERMINAL_ABOVE)
    manifest = forecast.input_manifest
    replay, _ = CryptoThresholdEngine(
        EngineConfig.model_validate(manifest["engine_config"])
    ).evaluate(
        MarketSnapshot.model_validate(manifest["market"]),
        CryptoSnapshot.model_validate(manifest["crypto"]),
        ThresholdContract.model_validate(manifest["contract"]),
    )
    assert replay.probability_yes == forecast.probability_yes
    assert replay.recipe_id == forecast.recipe_id


def test_rejects_profile_after_probability_configuration_change() -> None:
    original = CryptoThresholdEngine(EngineConfig(structural_weight=0.5))
    crypto = crypto_snapshot()
    profile = UncertaintyCalibrationProfile(
        symbol="BTC",
        model_name=original.model_name(TERMINAL_ABOVE),
        model_version=original.model_version,
        training_start=NOW - timedelta(days=60),
        cutoff_at=NOW - timedelta(seconds=1),
        confidence_level=0.95,
        sample_count=30,
        brier_score=0.2,
        bins=(
            CalibrationBin(
                lower_probability=0,
                upper_probability=1,
                mean_probability=0.5,
                observed_frequency=0.5,
                outcome_interval_lower=0.3,
                outcome_interval_upper=0.7,
                uncertainty_margin=0.2,
                sample_count=30,
                minimum_horizon_seconds=1,
                maximum_horizon_seconds=31557600,
            ),
        ),
    )
    # model_copy also works on v1, letting the pristine-code reproduction reach
    # the old compatibility gate rather than failing on an unavailable new API.
    recipe = getattr(original, "recipe_id", lambda _: "v1-unfingerprinted")(crypto)
    profile = profile.model_copy(update={"recipe_id": recipe})
    changed = CryptoThresholdEngine(EngineConfig(structural_weight=0.9))
    with pytest.raises(ValueError, match="recipe"):
        changed.evaluate(market_snapshot(), crypto, TERMINAL_ABOVE, profile)
