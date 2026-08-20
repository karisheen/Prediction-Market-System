from __future__ import annotations

import re
from datetime import UTC
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from prediction_market_system.domain import Opportunity, RecommendationState
from prediction_market_system.storage import AlertStatus, SQLiteRepository
from prediction_market_system.validation import ValidationCampaignReport, ValidationCampaignState

_DISPLAY_TIME_ZONE = ZoneInfo("America/Chicago")
_AM_PM_PATTERN = re.compile(r"\b(\d{1,2})(am|pm)\b", re.IGNORECASE)
_RANGE_LABEL_PATTERN = re.compile(
    r"^\$?([\d,]+(?:\.\d+)?)\s+to\s+\$?([\d,]+(?:\.\d+)?)$",
    re.IGNORECASE,
)


_DISCORD_HOSTS = {"discord.com", "www.discord.com", "discordapp.com", "www.discordapp.com"}


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _display_market_title(value: str) -> str:
    title = value.strip()
    return _AM_PM_PATTERN.sub(
        lambda match: f"{match.group(1)} {match.group(2).upper()}",
        title,
    )


def _display_contract_label(value: str) -> str:
    label = value.strip()
    match = _RANGE_LABEL_PATTERN.fullmatch(label)
    if match is None:
        return label
    return f"${match.group(1)}–${match.group(2)}"


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

    async def send_validation_report(self, report: ValidationCampaignReport) -> str:
        response = await self._client.post(
            self._webhook_url,
            params={"wait": "true"},
            json=self._validation_payload(report),
        )
        response.raise_for_status()
        message_id = response.json().get("id")
        if not message_id:
            raise RuntimeError("Discord did not return a validation-status message ID")
        return str(message_id)

    async def update_validation_report(
        self,
        message_id: str,
        report: ValidationCampaignReport,
    ) -> str:
        response = await self._client.patch(
            f"{self._webhook_url}/messages/{message_id}",
            json=self._validation_payload(report),
        )
        response.raise_for_status()
        return str(response.json().get("id", message_id))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _payload(opportunity: Opportunity) -> dict[str, Any]:
        forecast = opportunity.forecast
        market = opportunity.market
        side = opportunity.side.value if opportunity.side else "None"
        market_title = _display_market_title(market.question)
        contract_label = _display_contract_label(market.contract_label or market.question)
        recommendation = opportunity.state.value
        if side != "None" and not recommendation.endswith(side):
            recommendation = f"{recommendation} {side}"
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
        local_expiry = market.expires_at.astimezone(_DISPLAY_TIME_ZONE)
        local_expiry_text = local_expiry.strftime("%A, %B %-d at %-I:%M %p %Z")
        utc_expiry_time = market.expires_at.astimezone(UTC).strftime("%H:%M UTC")
        market_url = None if market.market_url is None else str(market.market_url)
        market_identity = (
            f"{market_title}\n"
            f"YES: {contract_label}\n"
            f"Settles: {local_expiry_text} ({utc_expiry_time})\n"
            f"Event: `{market.event_id or 'N/A'}`\n"
            f"Contract: `{market.market_id}`"
        )
        if market_url is not None:
            market_identity += f"\n[OPEN THIS EXACT EVENT ON KALSHI]({market_url})"

        colors = {
            RecommendationState.ENTER_YES: 0x2ECC71,
            RecommendationState.ENTER_NO: 0xE74C3C,
            RecommendationState.WATCH: 0xF1C40F,
        }
        embed: dict[str, Any] = {
            "title": _truncate(market_title, 256),
            "author": {"name": f"{recommendation} • {contract_label}"},
            "color": colors.get(opportunity.state, 0x95A5A6),
            "timestamp": forecast.generated_at.isoformat(),
            "fields": [
                {
                    "name": "Exact Kalshi market",
                    "value": _truncate(market_identity, 1_024),
                    "inline": False,
                },
                {
                    "name": "Manual-review action",
                    "value": _truncate(
                        (
                            f"BUY {side} • do not pay above {price} • "
                            f"paper cap ${opportunity.suggested_max_exposure:,.2f}\n"
                            f"YES condition: {contract_label}\n"
                            f"Settlement: {local_expiry_text}"
                        ),
                        1_024,
                    ),
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
                    "name": "Settlement time",
                    "value": (
                        f"{local_expiry_text}\n"
                        f"{market.expires_at.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
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

    @staticmethod
    def _validation_payload(report: ValidationCampaignReport) -> dict[str, Any]:
        colors = {
            ValidationCampaignState.COLLECTING: 0xF1C40F,
            ValidationCampaignState.REJECTED: 0xE67E22,
            ValidationCampaignState.PARTIALLY_APPROVED: 0x3498DB,
            ValidationCampaignState.APPROVED: 0x2ECC71,
        }
        fields: list[dict[str, object]] = [
            {
                "name": "Chronological coverage",
                "value": (
                    f"{report.coverage_start.date()} → {report.coverage_end.date()}\n"
                    f"{report.coverage_days} / {report.required_days} required days "
                    f"({report.progress:.0%})"
                ),
                "inline": False,
            }
        ]
        for validation in report.validations:
            decision = (
                "APPROVED"
                if validation.accepted_for_paper_alerts
                else "\n".join(f"• {reason}" for reason in validation.rejection_reasons)
            )
            return_value = (
                "—" if validation.return_on_cost is None else f"{validation.return_on_cost:.2%}"
            )
            brier_value = "—" if validation.brier_score is None else f"{validation.brier_score:.4f}"
            fields.append(
                {
                    "name": validation.model_name,
                    "value": _truncate(
                        (
                            f"Calibration events: {validation.independent_calibration_events}\n"
                            f"Held-out events/folds/trades: {validation.held_out_events} / "
                            f"{validation.held_out_folds} / {validation.held_out_trades}\n"
                            f"Return: {return_value} • Brier: {brier_value}\n"
                            f"{decision}"
                        ),
                        1_024,
                    ),
                    "inline": False,
                }
            )
        description = (
            "Collecting settled, point-in-time evidence before walk-forward validation."
            if report.state is ValidationCampaignState.COLLECTING
            else "Latest chronological walk-forward validation result."
        )
        if report.run_id is not None:
            description = f"{description}\nBacktest run: `{report.run_id}`"
        return {
            "embeds": [
                {
                    "title": (
                        f"Paper model validation: {report.series_ticker} / {report.state.value}"
                    ),
                    "description": description,
                    "color": colors[report.state],
                    "timestamp": report.generated_at.isoformat(),
                    "fields": fields,
                    "footer": {
                        "text": ("Approval is automatic only when every configured gate passes")
                    },
                }
            ],
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
