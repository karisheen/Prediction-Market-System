import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from prediction_market_system.discord import DiscordAlertService, DiscordWebhookClient
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSnapshot,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.validation import ValidationCampaignReport, ValidationCampaignState


def evaluation() -> tuple[object, object]:
    observed_at = datetime(2026, 7, 28, tzinfo=UTC)
    market = MarketSnapshot(
        market_id="btc-discord-test",
        question="BTC price range on Aug 26, 2026 at 8pm EDT?",
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
        market_url=("https://kalshi.com/markets/kxbtctest/bitcoin-range/kxbtctest-30dec3117"),
        series_id="KXBTCTEST",
        event_id="KXBTCTEST-30DEC31",
        contract_label="$100 to 199.99",
    )
    crypto = CryptoSnapshot(
        symbol="BTC",
        observed_at=observed_at,
        spot_price=110,
        strike_price=100,
        annualized_volatility=0.50,
    )
    contract = ThresholdContract(
        model_kind=ThresholdModelKind.TERMINAL,
        direction=ThresholdDirection.ABOVE,
        strike_price=100.0,
    )
    return CryptoThresholdEngine().evaluate(market, crypto, contract)


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
    embed = requests[0][2]["embeds"][0]
    assert embed["title"] == "BTC price range on Aug 26, 2026 at 8 PM EDT?"
    assert embed["author"]["name"] == "ENTER YES • $100–$199.99"
    assert "description" not in embed
    fields = embed["fields"]
    identity = next(field["value"] for field in fields if field["name"] == "Exact Kalshi market")
    action = next(field["value"] for field in fields if field["name"] == "Manual-review action")
    settlement = next(field["value"] for field in fields if field["name"] == "Settlement time")
    assert "BUY YES" in action
    assert "YES condition: $100–$199.99" in action
    assert "Settlement: Wednesday, August 26 at 7:00 PM CDT" in action
    assert identity.startswith("BTC price range on Aug 26, 2026 at 8 PM EDT?\nYES: $100–$199.99")
    assert "Event: `KXBTCTEST-30DEC31`" in identity
    assert "Contract: `btc-discord-test`" in identity
    assert (
        "[OPEN THIS EXACT EVENT ON KALSHI]"
        "(https://kalshi.com/markets/kxbtctest/bitcoin-range/kxbtctest-30dec3117)" in identity
    )
    assert embed["url"] == (
        "https://kalshi.com/markets/kxbtctest/bitcoin-range/kxbtctest-30dec3117"
    )
    assert settlement == "Wednesday, August 26 at 7:00 PM CDT\n2026-08-27 00:00 UTC"
    await http_client.aclose()


@pytest.mark.asyncio
async def test_discord_validation_report_contains_readiness_gates() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "validation-message"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    discord = DiscordWebhookClient(
        "https://discord.com/api/webhooks/123/secret",
        client=http_client,
    )
    report = ValidationCampaignReport(
        series_ticker="KXBTC",
        symbol="BTC",
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
        state=ValidationCampaignState.COLLECTING,
        coverage_start=datetime(2026, 8, 7, tzinfo=UTC),
        coverage_end=datetime(2026, 8, 12, tzinfo=UTC),
        coverage_days=5,
        required_days=150,
    )

    message_id = await discord.send_validation_report(report)

    assert message_id == "validation-message"
    embed = payloads[0]["embeds"][0]
    assert embed["title"] == "Paper model validation: KXBTC / COLLECTING EVIDENCE"
    assert "5 / 150 required days" in embed["fields"][0]["value"]
    assert payloads[0]["allowed_mentions"] == {"parse": []}
    await http_client.aclose()


def test_rejects_non_discord_webhook_url() -> None:
    with pytest.raises(ValueError, match="official HTTPS webhook"):
        DiscordWebhookClient("https://example.com/api/webhooks/123/secret")
