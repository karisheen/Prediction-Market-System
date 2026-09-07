import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from prediction_market_system.calibration import CalibrationBin, UncertaintyCalibrationProfile
from prediction_market_system.discord import DiscordAlertService, DiscordWebhookClient
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSnapshot,
    Opportunity,
    ProbabilityForecast,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.validation import ValidationCampaignReport, ValidationCampaignState


def evaluation(
    *, repository: SQLiteRepository | None = None
) -> tuple[ProbabilityForecast, Opportunity]:
    observed_at = datetime.now(UTC)
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
    engine = CryptoThresholdEngine()
    profile = None
    if repository is not None:
        profile = UncertaintyCalibrationProfile(
            generated_at=observed_at - timedelta(days=1),
            symbol="BTC",
            model_name=engine.model_name(contract),
            model_version=engine.model_version,
            recipe_id=engine.recipe_id(crypto),
            research_only=False,
            training_start=observed_at - timedelta(days=60),
            cutoff_at=observed_at - timedelta(days=1),
            confidence_level=0.95,
            sample_count=30,
            brier_score=0.20,
            bins=(
                CalibrationBin(
                    lower_probability=0,
                    upper_probability=1,
                    mean_probability=0.5,
                    observed_frequency=0.5,
                    outcome_interval_lower=0.4,
                    outcome_interval_upper=0.6,
                    uncertainty_margin=0.05,
                    sample_count=30,
                    minimum_horizon_seconds=1,
                    maximum_horizon_seconds=31557600,
                ),
            ),
        )
        with sqlite3.connect(repository.database_path) as connection:
            connection.execute(
                "INSERT INTO uncertainty_calibrations "
                "(profile_id, symbol, model_name, model_version, cutoff_at, "
                "generated_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(profile.profile_id),
                    profile.symbol,
                    profile.model_name,
                    profile.model_version,
                    profile.cutoff_at.isoformat(),
                    profile.generated_at.isoformat(),
                    profile.model_dump_json(),
                ),
            )
    return engine.evaluate(market, crypto, contract, profile)


@pytest.mark.asyncio
async def test_discord_alert_is_idempotent_and_updates_by_market(tmp_path: Path) -> None:
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

    forecast_object, opportunity_object = evaluation(repository=repository)
    assert isinstance(forecast_object, ProbabilityForecast)
    assert isinstance(opportunity_object, Opportunity)
    forecast = forecast_object
    opportunity = opportunity_object
    repository.save_evaluation(forecast, opportunity)
    service = DiscordAlertService(repository, discord, allow_unapproved=True)

    first_message_id = await service.publish(opportunity)
    duplicate_message_id = await service.publish(opportunity)

    forecast = forecast.model_copy(update={"forecast_id": uuid4()})
    updated_opportunity = opportunity.model_copy(
        update={"opportunity_id": uuid4(), "forecast": forecast},
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


@pytest.mark.asyncio
async def test_direct_delivery_rejects_unapproved_before_queueing(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    repository.save_evaluation(forecast, opportunity)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "manual-message"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discord = DiscordWebhookClient(
            "https://discord.com/api/webhooks/123/fake-token", client=client
        )
        with pytest.raises(ValueError, match="not approved"):
            await DiscordAlertService(repository, discord).publish(opportunity)
        assert requests == []
        with sqlite3.connect(repository.database_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0] == 0
        assert (
            await DiscordAlertService(repository, discord, allow_unapproved=True).publish(
                opportunity
            )
            == "manual-message"
        )
        assert not repository.is_calibration_approved(forecast.calibration_profile_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["fixed", "legacy", "recipe", "future", "profile", "stale"])
async def test_manual_override_cannot_bypass_evidence_or_freshness(
    tmp_path: Path, invalid: str
) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    changes: dict[str, object] = {}
    if invalid == "fixed":
        changes["uncertainty_source"] = "fixed"
    elif invalid == "legacy":
        changes["recipe_id"] = None
    elif invalid == "recipe":
        changes["recipe_id"] = "different-recipe"
    elif invalid == "future":
        changes["generated_at"] = datetime.now(UTC) + timedelta(days=1)
    elif invalid == "profile":
        changes["calibration_profile_id"] = uuid4()
    elif invalid == "stale":
        changes["generated_at"] = datetime.now(UTC) - timedelta(minutes=3)
    forecast = forecast.model_copy(update=changes)
    opportunity = opportunity.model_copy(update={"forecast": forecast})
    repository.save_evaluation(forecast, opportunity)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("inadmissible forecast reached the webhook")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discord = DiscordWebhookClient(
            "https://discord.com/api/webhooks/123/fake-token", client=client
        )
        with pytest.raises(ValueError):
            await DiscordAlertService(repository, discord, allow_unapproved=True).publish(
                opportunity
            )
    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_error", [False, True])
async def test_webhook_failure_redacts_exception_and_persisted_error(
    tmp_path: Path, transport_error: bool
) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    repository.save_evaluation(forecast, opportunity)
    token = "fake-sensitive-webhook-token"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if transport_error:
            raise httpx.ReadTimeout(f"timeout for {request.url}", request=request)
        return httpx.Response(500, text="server failure")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discord = DiscordWebhookClient(
            f"https://discord.com/api/webhooks/123/{token}", client=client
        )
        with pytest.raises(RuntimeError) as error:
            await DiscordAlertService(repository, discord, allow_unapproved=True).publish(
                opportunity
            )
        assert token not in str(error.value)
        assert "[REDACTED]" in str(error.value)
    assert len(calls) == 1  # An ambiguous POST failure must not be automatically retried.
    with sqlite3.connect(repository.database_path) as connection:
        rows = connection.execute("SELECT * FROM alert_events").fetchall()
    assert token not in repr(rows)
    assert "[REDACTED]" in repr(rows)


@pytest.mark.asyncio
async def test_ambiguous_delivery_blocks_restarts_and_new_opportunities(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = SQLiteRepository(tmp_path / "uncertain.db")
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    repository.save_evaluation(forecast, opportunity)
    calls = 0
    token = "synthetic-token-not-for-logs"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    caplog.set_level("INFO", logger="httpx")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discord = DiscordWebhookClient(
            f"https://discord.com/api/webhooks/123/{token}",
            client=client,
        )
        service = DiscordAlertService(repository, discord, allow_unapproved=True)
        with pytest.raises(RuntimeError):
            await service.publish(opportunity)
        # A retry after process restart is not a fresh permission to POST.
        restarted = DiscordAlertService(
            SQLiteRepository(repository.database_path),
            discord,
            allow_unapproved=True,
        )
        with pytest.raises(RuntimeError, match="unresolved"):
            await restarted.publish(opportunity)
        next_forecast = forecast.model_copy(update={"forecast_id": uuid4()})
        next_opportunity = opportunity.model_copy(
            update={
                "opportunity_id": uuid4(),
                "forecast": next_forecast,
            }
        )
        repository.save_evaluation(next_forecast, next_opportunity)
        with pytest.raises(RuntimeError, match="unresolved"):
            await restarted.publish(next_opportunity)
    assert calls == 1
    assert token not in caplog.text
    with sqlite3.connect(repository.database_path) as connection:
        state = connection.execute(
            "SELECT status,attempts FROM alert_events WHERE opportunity_id=?",
            (str(opportunity.opportunity_id),),
        ).fetchone()
    assert state == ("uncertain", 1)


@pytest.mark.asyncio
async def test_uncertain_delivery_requires_operator_reconciliation(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "reconcile.db")
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    repository.save_evaluation(forecast, opportunity)
    responses: list[httpx.Response] = [httpx.Response(500)]
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        return responses.pop(0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discord = DiscordWebhookClient("https://discord.com/api/webhooks/123/tok", client=client)
        service = DiscordAlertService(repository, discord, allow_unapproved=True)
        with pytest.raises(RuntimeError):
            await service.publish(opportunity)

        unresolved = repository.unresolved_alert_attempts()
        assert [attempt.opportunity_id for attempt in unresolved] == [
            str(opportunity.opportunity_id)
        ]
        assert unresolved[0].market_id == opportunity.market.market_id
        assert unresolved[0].status.value == "uncertain"

        # The operator must state the observed outcome; the system never guesses.
        with pytest.raises(ValueError, match="note"):
            repository.resolve_alert_attempt(
                str(opportunity.opportunity_id), delivered=False, discord_message_id=None, note=" "
            )
        with pytest.raises(ValueError, match="message ID"):
            repository.resolve_alert_attempt(
                str(opportunity.opportunity_id),
                delivered=True,
                discord_message_id=None,
                note="seen in channel",
            )
        with pytest.raises(ValueError, match="unknown"):
            repository.resolve_alert_attempt(
                "missing", delivered=False, discord_message_id=None, note="n/a"
            )

        # Observed as delivered: the message becomes the market's update target.
        record = repository.resolve_alert_attempt(
            str(opportunity.opportunity_id),
            delivered=True,
            discord_message_id="msg-observed",
            note="message visible in channel after 500",
        )
        assert record.status.value == "delivered"
        assert repository.unresolved_alert_attempts() == ()
        assert repository.get_discord_delivery(opportunity.market.market_id) == "msg-observed"
        with pytest.raises(ValueError, match="not awaiting reconciliation"):
            repository.resolve_alert_attempt(
                str(opportunity.opportunity_id),
                delivered=True,
                discord_message_id="msg-observed",
                note="twice",
            )

        # A later opportunity on the same market edits the reconciled message.
        next_forecast = forecast.model_copy(update={"forecast_id": uuid4()})
        next_opportunity = opportunity.model_copy(
            update={"opportunity_id": uuid4(), "forecast": next_forecast}
        )
        repository.save_evaluation(next_forecast, next_opportunity)
        responses.append(httpx.Response(500))
        with pytest.raises(RuntimeError):
            await service.publish(next_opportunity)
        assert requests[-1] == ("PATCH", "/api/webhooks/123/tok/messages/msg-observed")

        # Observed as not delivered: the claim reopens and a retry may send again.
        record = repository.resolve_alert_attempt(
            str(next_opportunity.opportunity_id),
            delivered=False,
            discord_message_id=None,
            note="no edit visible; Discord status page reported outage",
        )
        assert record.status.value == "rejected"
        responses.append(httpx.Response(200, json={"id": "msg-observed"}))
        assert await service.publish(next_opportunity) == "msg-observed"

    with sqlite3.connect(repository.database_path) as connection:
        rows = connection.execute(
            "SELECT status, error FROM alert_events ORDER BY created_at"
        ).fetchall()
    assert [row[0] for row in rows] == ["delivered", "delivered"]
    assert "manual reconciliation" in str(rows[0][1])


def test_alert_reconciliation_cli_and_doctor_report_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from prediction_market_system.cli import app

    database_path = tmp_path / "audit.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))
    repository = SQLiteRepository(database_path)
    repository.initialize()
    forecast, opportunity = evaluation(repository=repository)
    repository.save_evaluation(forecast, opportunity)
    repository.queue_alert(opportunity)
    repository.claim_alert_attempt(opportunity)
    repository.mark_alert_outcome(
        str(opportunity.opportunity_id),
        "https://discord.com/api/webhooks/123/secret-token timeout",
        uncertain=True,
    )
    runner = CliRunner()

    health = runner.invoke(app, ["doctor"])
    assert health.exit_code == 0, health.output
    unresolved = json.loads(health.output)["unresolved_deliveries"]
    assert unresolved["count"] == 1
    assert unresolved["by_status"] == {"uncertain": 1}
    assert unresolved["markets"] == [opportunity.market.market_id]

    listed = runner.invoke(app, ["alerts-unresolved"])
    assert listed.exit_code == 0, listed.output
    assert "secret-token" not in listed.output
    payload = json.loads(listed.output)
    assert payload["count"] == 1
    assert payload["attempts"][0]["opportunity_id"] == str(opportunity.opportunity_id)

    rejected = runner.invoke(
        app,
        ["alerts-reconcile", str(opportunity.opportunity_id), "--delivered", "--note", "seen"],
    )
    assert rejected.exit_code != 0
    assert "message ID" in rejected.output

    resolved = runner.invoke(
        app,
        [
            "alerts-reconcile",
            str(opportunity.opportunity_id),
            "--not-delivered",
            "--note",
            "channel shows nothing; webhook https://discord.com/api/webhooks/123/secret-token",
        ],
    )
    assert resolved.exit_code == 0, resolved.output
    assert json.loads(resolved.output)["status"] == "rejected"
    assert repository.unresolved_alert_attempts() == ()
    with sqlite3.connect(database_path) as connection:
        (error,) = connection.execute("SELECT error FROM alert_events").fetchone()
    assert "secret-token" not in error
    assert "[REDACTED]" in error
    assert json.loads(runner.invoke(app, ["doctor"]).output)["unresolved_deliveries"]["count"] == 0


def test_settlement_time_renders_expected_settlement_not_trading_close() -> None:
    observed = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    trading_close = datetime(2026, 8, 10, 16, 0, tzinfo=UTC)
    observation_end = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)
    expected_settlement = datetime(2026, 8, 11, 14, 0, tzinfo=UTC)
    outcome_available = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)
    _, opportunity = evaluation()
    market = opportunity.market.model_copy(
        update={
            "observed_at": observed,
            "expires_at": trading_close,
            "observation_end_at": observation_end,
            "expected_settlement_at": expected_settlement,
            "outcome_available_at": outcome_available,
        }
    )
    opportunity = opportunity.model_copy(update={"market": market})
    client = DiscordWebhookClient("https://discord.com/api/webhooks/123/secret")
    payload = client._payload(opportunity)
    fields = {field["name"]: field["value"] for field in payload["embeds"][0]["fields"]}
    close_utc = trading_close.strftime("%Y-%m-%d %H:%M UTC")
    observation_utc = observation_end.strftime("%Y-%m-%d %H:%M UTC")
    settlement_utc = expected_settlement.strftime("%Y-%m-%d %H:%M UTC")
    outcome_utc = outcome_available.strftime("%Y-%m-%d %H:%M UTC")
    assert settlement_utc in fields["Settlement time"]
    assert close_utc not in fields["Settlement time"]
    assert observation_utc not in fields["Settlement time"]
    assert close_utc in fields["Trading close"]
    assert observation_utc in fields["Observation end"]
    assert outcome_utc in fields["Outcome available"]
    identity = fields["Exact Kalshi market"]
    assert "Trading close:" in identity
    assert "Expected settlement:" in identity
    assert "Settles:" not in identity
    action = fields["Manual-review action"]
    assert "Expected settlement:" in action
    assert "Settlement:" not in action.replace("Expected settlement:", "")
