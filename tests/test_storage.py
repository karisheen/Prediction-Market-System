import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from prediction_market_system.domain import CryptoSnapshot, MarketSnapshot
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.storage import (
    AlertStatus,
    KalshiHistoryWriteResult,
    SQLiteRepository,
)
from prediction_market_system.venues.kalshi import (
    KalshiCandlestick,
    KalshiEventFeeChange,
    KalshiMarket,
    KalshiSeriesFeeChange,
)


def build_evaluation() -> tuple[object, object]:
    observed_at = datetime(2026, 7, 28, tzinfo=UTC)
    market = MarketSnapshot(
        market_id="btc-storage-test",
        question="Will BTC be above 100 USD?",
        venue="test",
        observed_at=observed_at,
        expires_at=observed_at + timedelta(days=30),
        yes_bid=0.39,
        yes_ask=0.42,
        no_bid=0.57,
        no_ask=0.60,
        yes_ask_size=1_000,
        no_ask_size=1_000,
        resolution_rule="Test index at expiry.",
    )
    crypto = CryptoSnapshot(
        symbol="BTC",
        observed_at=observed_at,
        spot_price=110,
        strike_price=100,
        annualized_volatility=0.50,
    )
    return CryptoThresholdEngine().evaluate(market, crypto)


def historical_market() -> KalshiMarket:
    return KalshiMarket.model_validate(
        {
            "ticker": "KXBTCTEST-30DEC31-T100000",
            "event_ticker": "KXBTCTEST-30DEC31",
            "market_type": "binary",
            "title": "Bitcoin price at year end?",
            "yes_sub_title": "Above $100,000",
            "no_sub_title": "$100,000 or below",
            "close_time": "2030-12-31T23:59:00Z",
            "latest_expiration_time": "2031-01-01T01:00:00Z",
            "status": "finalized",
            "notional_value_dollars": "1.0000",
            "can_close_early": False,
            "rules_primary": "Primary benchmark rule.",
            "rules_secondary": "Secondary correction rule.",
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "1.0000",
            "no_bid_dollars": "0.0000",
            "no_ask_dollars": "1.0000",
            "yes_bid_size_fp": "0.00",
            "yes_ask_size_fp": "0.00",
            "updated_time": "2031-01-01T00:01:00Z",
            "result": "yes",
            "settlement_value_dollars": "1.0000",
            "settlement_ts": "2031-01-01T00:00:00Z",
            "expiration_value": "101234.56",
        }
    )


def historical_candlestick() -> KalshiCandlestick:
    return KalshiCandlestick.model_validate(
        {
            "end_period_ts": 1_924_991_940,
            "yes_bid": {
                "open": "0.4000",
                "low": "0.3900",
                "high": "0.4300",
                "close": "0.4200",
            },
            "yes_ask": {
                "open": "0.4200",
                "low": "0.4100",
                "high": "0.4500",
                "close": "0.4400",
            },
            "price": {
                "open": "0.4100",
                "low": "0.4000",
                "high": "0.4400",
                "close": "0.4300",
                "mean": "0.4250",
                "previous": "0.4050",
            },
            "volume": "12.50",
            "open_interest": "200.00",
        }
    )


def test_persists_evaluation_and_idempotent_alert(tmp_path: Path) -> None:
    from prediction_market_system.domain import Opportunity, ProbabilityForecast

    forecast_object, opportunity_object = build_evaluation()
    assert isinstance(forecast_object, ProbabilityForecast)
    assert isinstance(opportunity_object, Opportunity)
    forecast = forecast_object
    opportunity = opportunity_object

    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    repository.save_evaluation(forecast, opportunity)

    history = repository.opportunity_history()
    assert len(history) == 1
    assert history[0]["market"]["market_id"] == "btc-storage-test"

    queued = repository.queue_alert(opportunity)
    assert queued.status is AlertStatus.QUEUED

    repository.mark_alert_delivered(opportunity, "discord-message-1")
    duplicate = repository.queue_alert(opportunity)
    assert duplicate.status is AlertStatus.DELIVERED
    assert duplicate.discord_message_id == "discord-message-1"
    assert repository.get_discord_delivery("btc-storage-test") == "discord-message-1"


def test_persists_complete_kalshi_history_idempotently(tmp_path: Path) -> None:
    market = historical_market()
    candle = historical_candlestick()
    series_fee = KalshiSeriesFeeChange(
        id="series-fee-1",
        series_ticker="KXBTCTEST",
        fee_type="quadratic",
        fee_multiplier=0.07,
        scheduled_ts=datetime(2030, 1, 1, tzinfo=UTC),
    )
    event_fee = KalshiEventFeeChange(
        id="event-fee-1",
        event_ticker=market.event_ticker,
        series_ticker="KXBTCTEST",
        fee_type_override=None,
        fee_multiplier_override=None,
        scheduled_ts=datetime(2030, 6, 1, tzinfo=UTC),
    )
    observed_at = datetime(2031, 1, 1, 0, 5, tzinfo=UTC)
    database_path = tmp_path / "history.db"
    repository = SQLiteRepository(database_path)
    repository.initialize()

    def save() -> KalshiHistoryWriteResult:
        return repository.save_kalshi_history(
            series_ticker="KXBTCTEST",
            observed_at=observed_at,
            markets=[market],
            candlesticks={market.ticker: [candle]},
            period_interval=60,
            series_fee_changes=[series_fee],
            event_fee_changes=[event_fee],
        )

    assert save() == KalshiHistoryWriteResult(
        market_snapshots=1,
        candlesticks=1,
        rule_snapshots=1,
        resolutions=1,
        series_fee_changes=1,
        event_fee_changes=1,
    )
    assert save() == KalshiHistoryWriteResult()

    with sqlite3.connect(database_path) as connection:
        resolution = connection.execute(
            """
            SELECT result, settlement_value_dollars, expiration_value
            FROM kalshi_resolutions
            """
        ).fetchone()
        rules = connection.execute(
            "SELECT rules_primary, rules_secondary FROM kalshi_rule_snapshots"
        ).fetchone()
        event_override = connection.execute(
            """
            SELECT fee_type_override, fee_multiplier_override
            FROM kalshi_event_fee_changes
            """
        ).fetchone()
        candle_count = connection.execute("SELECT count(*) FROM kalshi_candlesticks").fetchone()

    assert resolution == ("yes", "1.0000", "101234.56")
    assert rules == ("Primary benchmark rule.", "Secondary correction rule.")
    assert event_override == (None, None)
    assert candle_count == (1,)
