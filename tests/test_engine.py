import math
from datetime import UTC, datetime, timedelta

import pytest

from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSide,
    MarketSnapshot,
    RecommendationState,
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
    yes_bid: float = 0.39,
    yes_ask: float = 0.42,
    no_bid: float = 0.57,
    no_ask: float = 0.60,
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
        market_snapshot(expires_in=timedelta(days=180)),
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
