from __future__ import annotations

from typing import Any

import httpx

from prediction_market_system.domain import Opportunity, RecommendationState
from prediction_market_system.storage import AlertStatus, SQLiteRepository

_DISCORD_HOSTS = {"discord.com", "www.discord.com", "discordapp.com", "www.discordapp.com"}


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


class DiscordWebhookClient:
    def __init__(
        self,
        webhook_url: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed_url = httpx.URL(webhook_url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.host not in _DISCORD_HOSTS
            or "/api/webhooks/" not in parsed_url.path
        ):
            raise ValueError("Discord webhook URL is not an official HTTPS webhook endpoint")

        self._webhook_url = webhook_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None

    async def send(self, opportunity: Opportunity) -> str:
        response = await self._client.post(
            self._webhook_url,
            params={"wait": "true"},
            json=self._payload(opportunity),
        )
        response.raise_for_status()
        message_id = response.json().get("id")
        if not message_id:
            raise RuntimeError("Discord did not return a message ID")
        return str(message_id)

    async def update(self, message_id: str, opportunity: Opportunity) -> str:
        response = await self._client.patch(
            f"{self._webhook_url}/messages/{message_id}",
            json=self._payload(opportunity),
        )
        response.raise_for_status()
        returned_id = response.json().get("id", message_id)
        return str(returned_id)

    async def send_health_check(self) -> str:
        response = await self._client.post(
            self._webhook_url,
            params={"wait": "true"},
            json={
                "content": "Prediction Market System Discord delivery is configured.",
                "allowed_mentions": {"parse": []},
            },
        )
        response.raise_for_status()
        message_id = response.json().get("id")
        if not message_id:
            raise RuntimeError("Discord did not return a message ID")
        return str(message_id)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _payload(opportunity: Opportunity) -> dict[str, Any]:
        forecast = opportunity.forecast
        market = opportunity.market
        side = opportunity.side.value if opportunity.side else "None"
        price = (
            f"{opportunity.executable_price:.1%}"
            if opportunity.executable_price is not None
            else "N/A"
        )
        edge = (
            f"{opportunity.conservative_net_edge:.2%}"
            if opportunity.conservative_net_edge is not None
            else "N/A"
        )
        evidence = "\n".join(f"• {item}" for item in forecast.supporting_evidence) or "None"
        counter_evidence = "\n".join(f"• {item}" for item in forecast.opposing_evidence) or "None"
        warnings = "\n".join(f"• {item}" for item in opportunity.warnings) or "None"

        colors = {
            RecommendationState.ENTER_YES: 0x2ECC71,
            RecommendationState.ENTER_NO: 0xE74C3C,
            RecommendationState.WATCH: 0xF1C40F,
        }
        embed: dict[str, Any] = {
            "title": f"{opportunity.state.value}: {side}",
            "description": _truncate(market.question, 4_096),
            "color": colors.get(opportunity.state, 0x95A5A6),
            "timestamp": forecast.generated_at.isoformat(),
            "fields": [
                {
                    "name": "Venue / Market",
                    "value": _truncate(f"{market.venue} • `{market.market_id}`", 1_024),
                    "inline": False,
                },
                {
                    "name": "Forecast",
                    "value": (
                        f"YES {forecast.probability_yes:.1%} "
                        f"({forecast.lower_probability_yes:.1%}–"
                        f"{forecast.upper_probability_yes:.1%})"
                    ),
                    "inline": True,
                },
                {"name": "Executable price", "value": price, "inline": True},
                {"name": "Conservative edge", "value": edge, "inline": True},
                {
                    "name": "Paper exposure cap",
                    "value": f"${opportunity.suggested_max_exposure:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Expires",
                    "value": market.expires_at.isoformat(),
                    "inline": True,
                },
                {
                    "name": "Supporting evidence",
                    "value": _truncate(evidence, 1_024),
                    "inline": False,
                },
                {
                    "name": "Counter-evidence",
                    "value": _truncate(counter_evidence, 1_024),
                    "inline": False,
                },
                {
                    "name": "Warnings",
                    "value": _truncate(warnings, 1_024),
                    "inline": False,
                },
                {
                    "name": "Resolution rule",
                    "value": _truncate(market.resolution_rule, 1_024),
                    "inline": False,
                },
            ],
            "footer": {
                "text": (
                    "Paper decision support • Manual review required • "
                    f"Alert {opportunity.opportunity_id}"
                )
            },
        }
        if opportunity.market_regime is not None:
            regime = opportunity.market_regime
            implied = (
                "N/A" if regime.implied_volatility is None else f"{regime.implied_volatility:.1%}"
            )
            embed["fields"].insert(
                5,
                {
                    "name": "Observed market regime",
                    "value": (
                        f"{regime.label} • trailing return {regime.trailing_return:+.1%} • "
                        f"realized vol {regime.realized_volatility:.1%} • implied vol {implied}"
                    ),
                    "inline": False,
                },
            )
        if market.market_url is not None:
            embed["url"] = str(market.market_url)

        return {
            "embeds": [embed],
            "allowed_mentions": {"parse": []},
        }


class DiscordAlertService:
    def __init__(
        self,
        repository: SQLiteRepository,
        discord: DiscordWebhookClient,
    ) -> None:
        self.repository = repository
        self.discord = discord

    async def publish(self, opportunity: Opportunity) -> str:
        alert = self.repository.queue_alert(opportunity)
        if alert.status is AlertStatus.DELIVERED and alert.discord_message_id:
            return alert.discord_message_id

        opportunity_id = str(opportunity.opportunity_id)
        previous_message_id = self.repository.get_discord_delivery(opportunity.market.market_id)
        try:
            if previous_message_id:
                try:
                    message_id = await self.discord.update(
                        previous_message_id,
                        opportunity,
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != httpx.codes.NOT_FOUND:
                        raise
                    message_id = await self.discord.send(opportunity)
            else:
                message_id = await self.discord.send(opportunity)
        except Exception as exc:
            self.repository.mark_alert_failed(opportunity_id, str(exc))
            raise

        self.repository.mark_alert_delivered(opportunity, message_id)
        return message_id
