from datetime import UTC, datetime, timedelta

import pytest

from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSide,
    MarketSnapshot,
    RecommendationState,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


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


def crypto_snapshot(*, spot: float = 110.0) -> CryptoSnapshot:
    return CryptoSnapshot(
        symbol="BTC",
        observed_at=NOW,
        spot_price=spot,
        strike_price=100.0,
        annualized_volatility=0.50,
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

    forecast, opportunity = engine.evaluate(market_snapshot(), crypto_snapshot())

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

    _, opportunity = engine.evaluate(market, crypto_snapshot(spot=80.0))

    assert opportunity.state is RecommendationState.ENTER_NO
    assert opportunity.side is MarketSide.NO
    assert opportunity.conservative_net_edge is not None
    assert opportunity.conservative_net_edge > 0.03


def test_returns_watch_when_contract_is_too_close_to_expiry() -> None:
    engine = CryptoThresholdEngine(EngineConfig(minimum_seconds_to_expiry=300))

    _, opportunity = engine.evaluate(
        market_snapshot(expires_in=timedelta(seconds=120)),
        crypto_snapshot(),
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

    _, no_fee = no_fee_engine.evaluate(market_snapshot(), crypto_snapshot())
    _, with_fee = fee_engine.evaluate(market_snapshot(), crypto_snapshot())

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

    _, quadratic = quadratic_engine.evaluate(market_snapshot(), crypto_snapshot())
    _, flat = flat_engine.evaluate(market_snapshot(), crypto_snapshot())

    assert quadratic.conservative_net_edge is not None
    assert flat.conservative_net_edge is not None
    assert flat.conservative_net_edge < quadratic.conservative_net_edge


def test_rejects_crossed_order_book() -> None:
    with pytest.raises(ValueError, match="yes_bid cannot exceed yes_ask"):
        market_snapshot(yes_bid=0.50, yes_ask=0.49)
