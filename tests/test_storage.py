from datetime import UTC, datetime, timedelta
from pathlib import Path

from prediction_market_system.domain import CryptoSnapshot, MarketSnapshot
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.storage import AlertStatus, SQLiteRepository


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
