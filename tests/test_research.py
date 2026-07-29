from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from prediction_market_system.research import (
    DerivativesSnapshot,
    EventDataSnapshot,
    FundingObservation,
    ResearchDataUnavailable,
    SpotCandle,
    VolatilityObservation,
    calculate_realized_volatility,
)
from prediction_market_system.storage import SQLiteRepository


def spot_candle(end_at: datetime, close: str) -> SpotCandle:
    close_value = Decimal(close)
    return SpotCandle(
        provider="coinbase",
        product_id="BTC-USD",
        interval_seconds=3600,
        start_at=end_at - timedelta(hours=1),
        end_at=end_at,
        open=close_value,
        high=close_value + Decimal("1"),
        low=close_value - Decimal("1"),
        close=close_value,
        volume=Decimal("10"),
        retrieved_at=datetime(2035, 1, 1, tzinfo=UTC),
        raw_payload={"close": close},
    )


def implied_volatility(observed_at: datetime, value: float) -> VolatilityObservation:
    return VolatilityObservation(
        provider="deribit",
        symbol="BTC",
        kind="implied",
        window_seconds=3600,
        source_start_at=observed_at - timedelta(hours=1),
        observed_at=observed_at,
        annualized_volatility=value,
        retrieved_at=datetime(2035, 1, 1, tzinfo=UTC),
        raw_payload={"close": value * 100},
    )


def funding(observed_at: datetime, rate: float) -> FundingObservation:
    return FundingObservation(
        provider="deribit",
        instrument_name="BTC-PERPETUAL",
        observed_at=observed_at,
        index_price=100.0,
        previous_index_price=99.0,
        funding_rate_1h=rate,
        funding_rate_8h=rate * 8,
        retrieved_at=datetime(2035, 1, 1, tzinfo=UTC),
        raw_payload={"interest_1h": rate},
    )


def derivatives(observed_at: datetime, basis: float) -> DerivativesSnapshot:
    return DerivativesSnapshot(
        provider="deribit",
        instrument_name="BTC-PERPETUAL",
        observed_at=observed_at,
        index_price=100.0,
        mark_price=100.0 * (1 + basis),
        basis=basis,
        open_interest=1_000.0,
        current_funding=0.0,
        funding_rate_8h=0.0001,
        retrieved_at=datetime(2035, 1, 1, tzinfo=UTC),
        raw_payload={"basis": basis},
    )


def event_data(observed_at: datetime, label: str) -> EventDataSnapshot:
    return EventDataSnapshot(
        provider="kalshi",
        event_ticker="KXBTCTEST-30DEC31",
        data_type="crypto",
        observed_at=observed_at,
        retrieved_at=observed_at,
        is_historical=False,
        details={"label": label},
        raw_payload={"type": "crypto", "details": {"label": label}},
    )


def test_realized_volatility_uses_only_complete_as_of_window() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)) for offset in range(25)
    ]
    future = spot_candle(as_of + timedelta(hours=1), "100000")

    expected = calculate_realized_volatility(
        candles,
        symbol="BTC",
        as_of=as_of,
        window_seconds=24 * 60 * 60,
    )
    with_future = calculate_realized_volatility(
        [*candles, future],
        symbol="BTC",
        as_of=as_of,
        window_seconds=24 * 60 * 60,
    )

    assert with_future.annualized_volatility == pytest.approx(expected.annualized_volatility)
    assert with_future.observed_at == as_of
    assert with_future.raw_payload["return_count"] == 24


def test_realized_volatility_rejects_partial_window() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=2 - offset), str(100 + offset)) for offset in range(3)
    ]

    with pytest.raises(ResearchDataUnavailable, match="full"):
        calculate_realized_volatility(
            candles,
            symbol="BTC",
            as_of=as_of,
            window_seconds=24 * 60 * 60,
        )


def test_repository_never_selects_future_observations(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)) for offset in range(25)
    ]
    database_path = tmp_path / "research.db"
    repository = SQLiteRepository(database_path)
    repository.initialize()
    repository.save_research_data(
        spot_candles=[
            *candles,
            spot_candle(as_of + timedelta(seconds=1), "100000"),
        ],
        volatility_observations=[
            implied_volatility(as_of - timedelta(hours=1), 0.45),
            implied_volatility(as_of + timedelta(seconds=1), 9.99),
        ],
        funding_observations=[
            funding(as_of - timedelta(hours=1), 0.0001),
            funding(as_of + timedelta(seconds=1), 0.99),
        ],
        derivatives_snapshots=[
            derivatives(as_of - timedelta(hours=1), 0.001),
            derivatives(as_of + timedelta(seconds=1), 0.5),
        ],
        event_snapshots=[
            event_data(as_of - timedelta(hours=1), "available"),
            event_data(as_of + timedelta(seconds=1), "future"),
        ],
    )

    context = repository.research_context_as_of(
        symbol="BTC",
        event_ticker="KXBTCTEST-30DEC31",
        as_of=as_of,
        interval_seconds=3600,
        realized_window_seconds=24 * 60 * 60,
    )

    assert context.spot.end_at == as_of
    assert context.implied_volatility is not None
    assert context.implied_volatility.annualized_volatility == pytest.approx(0.45)
    assert context.funding is not None
    assert context.funding.funding_rate_1h == pytest.approx(0.0001)
    assert context.derivatives is not None
    assert context.derivatives.basis == pytest.approx(0.001)
    assert context.event_data is not None
    assert context.event_data.details["label"] == "available"
    crypto = context.to_crypto_snapshot(strike_price=120.0)
    assert crypto.observed_at == as_of
    assert crypto.spot_price == pytest.approx(124.0)
    assert crypto.annualized_volatility == pytest.approx(0.45)
