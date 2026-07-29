from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from prediction_market_system.cli import app
from prediction_market_system.venues.kalshi import (
    KalshiMarket,
    KalshiOrderBook,
    to_market_snapshot,
)

runner = CliRunner()


def kalshi_market(*, can_close_early: bool = False) -> KalshiMarket:
    return KalshiMarket.model_validate(
        {
            "ticker": "KXBTCTEST-30DEC31-T100000",
            "event_ticker": "KXBTCTEST-30DEC31",
            "market_type": "binary",
            "title": "Bitcoin price at year end?",
            "yes_sub_title": "Above $100,000",
            "no_sub_title": "$100,000 or below",
            "close_time": "2030-12-31T23:59:00Z",
            "expected_expiration_time": "2030-12-31T23:59:00Z",
            "latest_expiration_time": "2031-01-01T01:00:00Z",
            "status": "active",
            "notional_value_dollars": "1.0000",
            "can_close_early": can_close_early,
            "strike_type": "greater",
            "floor_strike": 100000,
            "rules_primary": "Resolves YES from the benchmark at expiry.",
            "rules_secondary": "",
            "yes_bid_dollars": "0.4200",
            "yes_ask_dollars": "0.4400",
            "no_bid_dollars": "0.5600",
            "no_ask_dollars": "0.5800",
            "yes_bid_size_fp": "13.00",
            "yes_ask_size_fp": "17.00",
        }
    )


def market_pair(*, can_close_early: bool = False) -> tuple[KalshiMarket, object]:
    market = kalshi_market(can_close_early=can_close_early)
    order_book = KalshiOrderBook(
        yes_dollars=[("0.4200", "13.00")],
        no_dollars=[("0.5600", "17.00")],
    )
    snapshot = to_market_snapshot(
        market,
        order_book,
        observed_at=datetime(2026, 7, 28, tzinfo=UTC),
    )
    return market, snapshot


def test_kalshi_evaluate_persists_live_snapshot(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    pair = market_pair()
    monkeypatch.setattr(
        "prediction_market_system.cli._load_kalshi_market",
        lambda ticker: pair,
    )
    database_path = tmp_path / "kalshi.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))

    result = runner.invoke(
        app,
        [
            "kalshi-evaluate",
            "--ticker",
            "KXBTCTEST-30DEC31-T100000",
            "--symbol",
            "BTC",
            "--spot",
            "110000",
            "--volatility",
            "0.55",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Bitcoin price at year end?" in result.output
    assert database_path.exists()


def test_kalshi_evaluate_rejects_early_close_contract(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    pair = market_pair(can_close_early=True)
    monkeypatch.setattr(
        "prediction_market_system.cli._load_kalshi_market",
        lambda ticker: pair,
    )

    result = runner.invoke(
        app,
        [
            "kalshi-evaluate",
            "--ticker",
            "KXBTCTEST-30DEC31-T100000",
            "--symbol",
            "BTC",
            "--spot",
            "110000",
            "--volatility",
            "0.55",
        ],
    )

    assert result.exit_code == 2
    assert "path-dependent" in result.output
