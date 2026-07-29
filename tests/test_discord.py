import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from prediction_market_system.discord import DiscordAlertService, DiscordWebhookClient
from prediction_market_system.domain import CryptoSnapshot, MarketSnapshot
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.storage import SQLiteRepository


def evaluation() -> tuple[object, object]:
    observed_at = datetime(2026, 7, 28, tzinfo=UTC)
    market = MarketSnapshot(
        market_id="btc-discord-test",
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
        market_url="https://example.com/market",
    )
    crypto = CryptoSnapshot(
        symbol="BTC",
        observed_at=observed_at,
        spot_price=110,
        strike_price=100,
        annualized_volatility=0.50,
    )
    return CryptoThresholdEngine().evaluate(market, crypto)


@pytest.mark.asyncio
async def test_discord_alert_is_idempotent_and_updates_by_market(tmp_path: Path) -> None:
    from prediction_market_system.domain import Opportunity, ProbabilityForecast

    requests: list[tuple[str, str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.method, request.url.path, payload))
        return httpx.Response(200, json={"id": "message-1"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    discord = DiscordWebhookClient(
        "https://discord.com/api/webhooks/123/secret",
        client=http_client,
    )
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()

    forecast_object, opportunity_object = evaluation()
    assert isinstance(forecast_object, ProbabilityForecast)
    assert isinstance(opportunity_object, Opportunity)
    forecast = forecast_object
    opportunity = opportunity_object
    repository.save_evaluation(forecast, opportunity)
    service = DiscordAlertService(repository, discord)

    first_message_id = await service.publish(opportunity)
    duplicate_message_id = await service.publish(opportunity)

    updated_opportunity = opportunity.model_copy(
        update={"opportunity_id": uuid4()},
    )
    repository.save_evaluation(forecast, updated_opportunity)
    updated_message_id = await service.publish(updated_opportunity)

    assert first_message_id == "message-1"
    assert duplicate_message_id == "message-1"
    assert updated_message_id == "message-1"
    assert [method for method, _, _ in requests] == ["POST", "PATCH"]
    assert requests[0][2]["allowed_mentions"] == {"parse": []}
    assert requests[1][1].endswith("/messages/message-1")

    await http_client.aclose()


def test_rejects_non_discord_webhook_url() -> None:
    with pytest.raises(ValueError, match="official HTTPS webhook"):
        DiscordWebhookClient("https://example.com/api/webhooks/123/secret")
