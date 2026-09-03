import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prediction_market_system.calibration import CalibrationBin, UncertaintyCalibrationProfile
from prediction_market_system.cli import _ResearchDataBatch, app
from prediction_market_system.discord import DiscordWebhookClient
from prediction_market_system.domain import Opportunity, PriceTrendRegime, VolatilityRegime
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.paper_alerts import (
    PaperAlertCycleResult,
    PaperAlertRunner,
    classify_market_regime,
)
from prediction_market_system.research import ResearchContext, SpotCandle, VolatilityObservation
from prediction_market_system.sources import CoinbaseDataError
from prediction_market_system.storage import MarketCheckStatus, SQLiteRepository
from prediction_market_system.venues.kalshi import KalshiMarket, KalshiOrderBook

AS_OF = datetime(2026, 7, 28, 12, tzinfo=UTC)
cli_runner = CliRunner()


def spot_candle(
    end_at: datetime,
    close: str,
    *,
    interval_seconds: int = 3600,
) -> SpotCandle:
    return SpotCandle(
        provider="coinbase",
        product_id="BTC-USD",
        interval_seconds=interval_seconds,
        start_at=end_at - timedelta(seconds=interval_seconds),
        end_at=end_at,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("10"),
        retrieved_at=AS_OF,
        raw_payload={},
    )


def research_context(
    *,
    end_price: str,
    realized_volatility: float,
) -> tuple[ResearchContext, list[SpotCandle]]:
    start_at = AS_OF - timedelta(days=30)
    candles = [
        spot_candle(start_at, "100"),
        spot_candle(AS_OF - timedelta(hours=1), end_price),
    ]
    live_spot = spot_candle(AS_OF, end_price, interval_seconds=60)
    realized = VolatilityObservation(
        provider="coinbase:calculated",
        symbol="BTC",
        kind="realized",
        window_seconds=30 * 24 * 60 * 60,
        source_start_at=start_at,
        observed_at=AS_OF - timedelta(hours=1),
        annualized_volatility=realized_volatility,
        retrieved_at=AS_OF,
        raw_payload={},
    )
    implied = VolatilityObservation(
        provider="deribit",
        symbol="BTC",
        kind="implied",
        window_seconds=3600,
        source_start_at=AS_OF - timedelta(hours=1),
        observed_at=AS_OF - timedelta(minutes=1),
        annualized_volatility=realized_volatility + 0.10,
        retrieved_at=AS_OF,
        raw_payload={},
    )
    return (
        ResearchContext(
            symbol="BTC",
            event_ticker=None,
            as_of=AS_OF,
            spot=live_spot,
            realized_volatility=realized,
            implied_volatility=implied,
        ),
        candles,
    )


@pytest.mark.parametrize(
    ("end_price", "realized_volatility", "expected_trend", "expected_volatility"),
    [
        ("110", 0.30, PriceTrendRegime.UPTREND, VolatilityRegime.LOW),
        ("102", 0.60, PriceTrendRegime.RANGE, VolatilityRegime.TYPICAL),
        ("90", 0.90, PriceTrendRegime.DOWNTREND, VolatilityRegime.HIGH),
    ],
)
def test_classifies_auditable_market_regimes(
    end_price: str,
    realized_volatility: float,
    expected_trend: PriceTrendRegime,
    expected_volatility: VolatilityRegime,
) -> None:
    context, candles = research_context(
        end_price=end_price,
        realized_volatility=realized_volatility,
    )

    regime = classify_market_regime(context, candles)

    assert regime.price_trend is expected_trend
    assert regime.volatility is expected_volatility
    assert regime.implied_volatility == pytest.approx(realized_volatility + 0.10)


def kalshi_market(*, supported: bool = True) -> KalshiMarket:
    return KalshiMarket.model_validate(
        {
            "ticker": "KXBTCTEST-30DEC31-T100",
            "event_ticker": "KXBTCTEST-30DEC31",
            "market_type": "binary",
            "title": "Bitcoin price at year end?",
            "yes_sub_title": "Above $100",
            "no_sub_title": "$100 or below",
            "close_time": "2030-12-31T23:59:00Z",
            "expected_expiration_time": "2030-12-31T23:59:00Z",
            "latest_expiration_time": "2031-01-01T01:00:00Z",
            "status": "active",
            "notional_value_dollars": "1.0000",
            "can_close_early": False,
            "strike_type": "greater" if supported else None,
            "floor_strike": 100 if supported else None,
            "rules_primary": "Resolves YES from the benchmark at expiry.",
            "rules_secondary": "",
            "yes_bid_dollars": "0.2000",
            "yes_ask_dollars": "0.2200",
            "no_bid_dollars": "0.7800",
            "no_ask_dollars": "0.8000",
            "yes_bid_size_fp": "100.00",
            "yes_ask_size_fp": "100.00",
        }
    )


def calibration_profile() -> UncertaintyCalibrationProfile:
    return UncertaintyCalibrationProfile(
        symbol="BTC",
        model_name="crypto-terminal-above-threshold-market-anchor",
        model_version="0.3.0",
        training_start=datetime(2026, 1, 1, tzinfo=UTC),
        cutoff_at=datetime(2026, 7, 27, tzinfo=UTC),
        confidence_level=0.95,
        sample_count=30,
        brier_score=0.20,
        bins=(
            CalibrationBin(
                lower_probability=0.0,
                upper_probability=1.0,
                mean_probability=0.5,
                observed_frequency=0.5,
                outcome_interval_lower=0.4,
                outcome_interval_upper=0.6,
                uncertainty_margin=0.0,
                sample_count=30,
            ),
        ),
    )


class FakeMarketReader:
    def __init__(self) -> None:
        self.requested: list[str] = []

    async def get_order_book(self, ticker: str, depth: int = 100) -> KalshiOrderBook:
        self.requested.append(ticker)
        return KalshiOrderBook(
            yes_dollars=[("0.2000", "100.00")],
            no_dollars=[("0.7800", "100.00")],
        )


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[Opportunity] = []

    async def publish(self, opportunity: Opportunity) -> str:
        self.published.append(opportunity)
        return "message-1"


@pytest.mark.asyncio
async def test_runner_delivers_calibrated_entries_and_persists_regime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, candles = research_context(end_price="130", realized_volatility=0.60)
    regime = classify_market_regime(context, candles)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    profile = calibration_profile()
    monkeypatch.setattr(
        repository,
        "latest_uncertainty_calibration",
        lambda **kwargs: profile,
    )
    monkeypatch.setattr(repository, "is_calibration_approved", lambda profile_id: True)
    reader = FakeMarketReader()
    publisher = FakePublisher()
    runner = PaperAlertRunner(
        repository=repository,
        engine=CryptoThresholdEngine(
            EngineConfig(
                min_conservative_edge=0.0,
                binary_fee_coefficient=0.0,
                slippage_bps=0.0,
                resolution_haircut=0.0,
            )
        ),
        market_reader=reader,
        alert_service=publisher,
        clock=lambda: AS_OF,
    )

    unsupported_market = kalshi_market(supported=False).model_copy(
        update={"ticker": "KXBTCTEST-30DEC31-UNSUPPORTED"}
    )
    result = await runner.run(
        markets=[kalshi_market(), unsupported_market],
        context=context,
        regime=regime,
        cycle_id="cycle-delivery",
    )

    assert result.discovered == 2
    assert result.evaluated == 1
    assert result.delivered == 1
    assert result.unsupported == 1
    assert result.uncalibrated == 0
    assert reader.requested == ["KXBTCTEST-30DEC31-T100"]
    assert publisher.published[0].market_regime == regime
    payload = DiscordWebhookClient._payload(publisher.published[0])
    fields = payload["embeds"][0]["fields"]
    assert any(
        field["name"] == "Observed market regime"
        and "uptrend / typical-volatility" in field["value"]
        for field in fields
    )
    checks = {
        check["market_id"]: check["status"]
        for check in repository.paper_market_checks("cycle-delivery")
    }
    assert checks == {
        "KXBTCTEST-30DEC31-T100": "delivered",
        "KXBTCTEST-30DEC31-UNSUPPORTED": "unsupported",
    }


@pytest.mark.asyncio
async def test_runner_shadows_candidates_and_controls_unapproved_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, candles = research_context(end_price="130", realized_volatility=0.60)
    regime = classify_market_regime(context, candles)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    monkeypatch.setattr(
        repository,
        "latest_uncertainty_calibration",
        lambda **kwargs: calibration_profile(),
    )
    reader = FakeMarketReader()
    publisher = FakePublisher()
    runner = PaperAlertRunner(
        repository=repository,
        engine=CryptoThresholdEngine(
            EngineConfig(
                min_conservative_edge=0.0,
                binary_fee_coefficient=0.0,
                slippage_bps=0.0,
                resolution_haircut=0.0,
            )
        ),
        market_reader=reader,
        alert_service=publisher,
        clock=lambda: AS_OF,
    )

    shadow = await runner.run(
        markets=[kalshi_market()],
        context=context,
        regime=regime,
        cycle_id="cycle-shadow",
        deliver_entries=False,
    )
    blocked = await runner.run(
        markets=[kalshi_market()],
        context=context,
        regime=regime,
        cycle_id="cycle-unapproved",
        deliver_entries=True,
    )
    manual_review = await runner.run(
        markets=[kalshi_market()],
        context=context,
        regime=regime,
        cycle_id="cycle-unapproved-manual-review",
        deliver_entries=True,
        allow_unapproved_delivery=True,
    )

    assert shadow.evaluated == 1
    assert shadow.delivered == 0
    assert manual_review.delivered == 1
    assert publisher.published[0].warnings[-1] == (
        "UNAPPROVED MODEL: held-out approval is missing; manual review only"
    )
    assert repository.paper_market_checks("cycle-shadow")[0]["status"] == "entry_candidate"
    assert blocked.unapproved == 1
    assert blocked.evaluated == 0
    assert repository.paper_market_checks("cycle-unapproved")[0]["status"] == ("unapproved_model")
    assert repository.paper_market_checks("cycle-unapproved-manual-review")[0]["status"] == (
        "delivered"
    )


@pytest.mark.asyncio
async def test_runner_never_fetches_or_delivers_without_calibration(tmp_path: Path) -> None:
    context, candles = research_context(end_price="100", realized_volatility=0.60)
    regime = classify_market_regime(context, candles)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    reader = FakeMarketReader()
    publisher = FakePublisher()
    runner = PaperAlertRunner(
        repository=repository,
        engine=CryptoThresholdEngine(),
        market_reader=reader,
        alert_service=publisher,
        clock=lambda: AS_OF,
    )

    result = await runner.run(
        markets=[kalshi_market()],
        context=context,
        regime=regime,
        cycle_id="cycle-uncalibrated",
    )

    assert result.uncalibrated == 1
    assert result.evaluated == 0
    assert reader.requested == []
    assert publisher.published == []
    assert repository.paper_market_checks("cycle-uncalibrated")[0]["status"] == (
        "missing_calibration"
    )


@pytest.mark.asyncio
async def test_runner_fails_closed_and_audits_stale_spot(tmp_path: Path) -> None:
    context, candles = research_context(end_price="100", realized_volatility=0.60)
    context = context.model_copy(
        update={"spot": spot_candle(AS_OF - timedelta(minutes=3), "100", interval_seconds=60)}
    )
    regime = classify_market_regime(context, candles)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    runner = PaperAlertRunner(
        repository=repository,
        engine=CryptoThresholdEngine(),
        market_reader=FakeMarketReader(),
        alert_service=FakePublisher(),
        clock=lambda: AS_OF,
    )

    result = await runner.run(
        markets=[kalshi_market()],
        context=context,
        regime=regime,
        cycle_id="cycle-stale",
    )

    assert result.evaluated == 0
    assert result.failures == ("live spot is 180 seconds old; maximum is 120",)
    assert repository.paper_market_checks("cycle-stale")[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_runner_caps_combined_entries_for_one_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, candles = research_context(end_price="130", realized_volatility=0.60)
    regime = classify_market_regime(context, candles)
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    monkeypatch.setattr(
        repository,
        "latest_uncertainty_calibration",
        lambda **kwargs: calibration_profile(),
    )
    monkeypatch.setattr(repository, "is_calibration_approved", lambda profile_id: True)
    first = kalshi_market()
    second = first.model_copy(update={"ticker": "KXBTCTEST-30DEC31-T101"})
    publisher = FakePublisher()
    runner = PaperAlertRunner(
        repository=repository,
        engine=CryptoThresholdEngine(
            EngineConfig(
                min_conservative_edge=0.0,
                binary_fee_coefficient=0.0,
                slippage_bps=0.0,
                resolution_haircut=0.0,
                max_event_bankroll_fraction=0.002,
            )
        ),
        market_reader=FakeMarketReader(),
        alert_service=publisher,
        clock=lambda: AS_OF,
    )

    result = await runner.run(
        markets=[first, second],
        context=context,
        regime=regime,
        cycle_id="cycle-cap",
    )

    assert result.evaluated == 2
    assert result.delivered == 1
    assert result.watch == 1
    assert sum(item.suggested_max_exposure for item in publisher.published) == 20.0


def test_repository_reports_regime_coverage(tmp_path: Path) -> None:
    context, candles = research_context(end_price="110", realized_volatility=0.30)
    uptrend = classify_market_regime(context, candles)
    later = uptrend.model_copy(
        update={
            "observed_at": AS_OF + timedelta(days=1),
            "price_trend": PriceTrendRegime.DOWNTREND,
            "volatility": VolatilityRegime.HIGH,
        }
    )
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()

    repository.save_market_regime(series_ticker="KXBTC", regime=uptrend)
    repository.save_market_regime(series_ticker="KXBTC", regime=uptrend)
    repository.save_market_regime(series_ticker="KXBTC", regime=later)

    coverage = repository.market_regime_coverage(series_ticker="KXBTC", symbol="BTC")
    assert [(row["regime"], row["observation_count"]) for row in coverage] == [
        ("downtrend / high-volatility", 1),
        ("uptrend / low-volatility", 1),
    ]


def test_compacts_only_expired_watch_evaluations(tmp_path: Path) -> None:
    database_path = tmp_path / "audit.db"
    repository = SQLiteRepository(database_path)
    repository.initialize()
    old_at = AS_OF - timedelta(days=2)
    second_old_at = old_at + timedelta(seconds=1)
    recent_at = AS_OF + timedelta(minutes=1)

    with sqlite3.connect(database_path) as connection:
        for suffix, created_at in (
            ("old", old_at),
            ("old-second", second_old_at),
            ("recent", recent_at),
        ):
            connection.execute(
                """
                INSERT INTO forecasts (
                    forecast_id, market_id, generated_at, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (f"forecast-{suffix}", f"market-{suffix}", created_at.isoformat(), "{}"),
            )
            connection.execute(
                """
                INSERT INTO opportunities (
                    opportunity_id, forecast_id, market_id, state,
                    created_at, payload_json
                ) VALUES (?, ?, ?, 'WATCH', ?, ?)
                """,
                (
                    f"opportunity-{suffix}",
                    f"forecast-{suffix}",
                    f"market-{suffix}",
                    created_at.isoformat(),
                    "{}",
                ),
            )

    repository.save_paper_market_check(
        cycle_id="cycle-old-watch",
        market_id="market-old",
        series_ticker="KXBTC",
        event_ticker="event-old",
        observed_at=old_at,
        status=MarketCheckStatus.WATCH,
        reason=None,
        payload={"opportunity": {"opportunity_id": "opportunity-old"}},
    )
    repository.save_paper_market_check(
        cycle_id="cycle-old-watch-second",
        market_id="market-old-second",
        series_ticker="KXBTC",
        event_ticker="event-old",
        observed_at=second_old_at,
        status=MarketCheckStatus.WATCH,
        reason=None,
        payload={"opportunity": {"opportunity_id": "opportunity-old-second"}},
    )
    repository.save_paper_market_check(
        cycle_id="cycle-recent-watch",
        market_id="market-recent",
        series_ticker="KXBTC",
        event_ticker="event-recent",
        observed_at=recent_at,
        status=MarketCheckStatus.WATCH,
        reason=None,
        payload={"opportunity": {"opportunity_id": "opportunity-recent"}},
    )
    repository.save_paper_market_check(
        cycle_id="cycle-old-delivery",
        market_id="market-delivered",
        series_ticker="KXBTC",
        event_ticker="event-delivered",
        observed_at=old_at,
        status=MarketCheckStatus.DELIVERED,
        reason=None,
        payload={"opportunity": {"side": "YES"}},
    )

    preview = repository.compact_watch_history(
        series_ticker="KXBTC",
        cutoff_at=AS_OF,
    )
    assert preview.eligible_checks == 2
    assert preview.applied is False
    assert repository.paper_market_checks("cycle-old-watch")

    applied = repository.compact_watch_history(
        series_ticker="KXBTC",
        cutoff_at=AS_OF,
        apply=True,
        batch_size=1,
        compacted_at=AS_OF,
    )

    assert applied.eligible_checks == 2
    assert applied.rolled_up_checks == 2
    assert applied.deleted_checks == 2
    assert applied.deleted_opportunities == 2
    assert applied.deleted_forecasts == 2
    assert applied.deleted_cycles == 2
    assert applied.batches == 2
    assert repository.paper_market_checks("cycle-old-watch") == []
    assert repository.paper_market_checks("cycle-recent-watch")
    assert repository.paper_market_checks("cycle-old-delivery")
    assert repository.paper_watch_rollups("KXBTC") == [
        {
            "series_ticker": "KXBTC",
            "observed_day": old_at.date().isoformat(),
            "evaluation_count": 2,
            "first_observed_at": old_at.isoformat(),
            "last_observed_at": second_old_at.isoformat(),
            "compacted_at": AS_OF.isoformat(),
        }
    ]

    repeated = repository.compact_watch_history(
        series_ticker="KXBTC",
        cutoff_at=AS_OF,
        apply=True,
        compacted_at=AS_OF + timedelta(minutes=1),
    )
    assert repeated.eligible_checks == 0
    assert repeated.batches == 0
    assert repository.paper_watch_rollups("KXBTC")[0]["evaluation_count"] == 2


def test_legacy_cycle_backfill_runs_once(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-cycle.db"
    repository = SQLiteRepository(database_path)
    repository.initialize()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            DELETE FROM schema_migrations
            WHERE migration_name = 'paper_alert_cycles_backfill_v1'
            """
        )
        connection.execute(
            """
            INSERT INTO paper_alert_market_checks (
                cycle_id, market_id, series_ticker, event_ticker,
                observed_at, status, reason, payload_json
            ) VALUES (?, ?, ?, ?, ?, 'watch', NULL, '{}')
            """,
            (
                "legacy-cycle",
                "legacy-market",
                "KXBTC",
                "legacy-event",
                AS_OF.isoformat(),
            ),
        )

    repository.initialize()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_alert_cycles WHERE cycle_id = 'legacy-cycle'"
        ).fetchone() == (1,)
        connection.execute("DELETE FROM paper_alert_cycles WHERE cycle_id = 'legacy-cycle'")

    repository.initialize()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_alert_cycles WHERE cycle_id = 'legacy-cycle'"
        ).fetchone() == (0,)


def test_paper_alert_maintenance_previews_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "maintenance.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))

    result = cli_runner.invoke(
        app,
        ["paper-alert-maintain", "--series", "KXBTC", "--watch-retention-days", "14"],
    )

    assert result.exit_code == 0, result.output
    assert "Paper-alert WATCH maintenance" in result.output
    assert "preview" in result.output
    assert "Preview only; pass --apply" in result.output

def test_repository_reports_activity_since_previous_status_request(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()

    def save_check(
        cycle_id: str,
        ticker: str,
        observed_at: datetime,
        status: MarketCheckStatus,
        side: str | None = None,
    ) -> None:
        payload: dict[str, object] = {}
        if side is not None:
            payload["opportunity"] = {"side": side}
        repository.save_paper_market_check(
            cycle_id=cycle_id,
            market_id=ticker,
            series_ticker="KXBTC",
            event_ticker="KXBTCTEST-30DEC31",
            observed_at=observed_at,
            status=status,
            reason=None,
            payload=payload,
        )

    def save_resolution(ticker: str, result: str, settled_at: datetime) -> None:
        market = kalshi_market().model_copy(
            update={
                "ticker": ticker,
                "status": "settled",
                "result": result,
                "settlement_ts": settled_at,
                "settlement_value_dollars": Decimal("1.00"),
                "expiration_value": "101.00",
            }
        )
        repository.save_kalshi_history(
            series_ticker="KXBTC",
            observed_at=settled_at + timedelta(seconds=1),
            markets=[market],
            candlesticks={},
            period_interval=1,
            series_fee_changes=[],
            event_fee_changes=[],
        )

    save_check(
        "cycle-before-first-request",
        "KXBTCTEST-30DEC31-T100",
        AS_OF - timedelta(minutes=5),
        MarketCheckStatus.DELIVERED,
        "YES",
    )
    save_resolution(
        "KXBTCTEST-30DEC31-T100",
        "yes",
        AS_OF - timedelta(minutes=3),
    )

    first = repository.paper_alert_status_since_last_request(
        series_ticker="KXBTC",
        symbol="BTC",
        requested_at=AS_OF,
    )

    assert first.previous_requested_at is None
    assert first.cycles == 1
    assert first.delivered_alerts == 1
    assert first.resolved_alerts == 1
    assert first.profitable_alerts == 1
    assert first.unresolved_alerts == 0

    save_check(
        "cycle-winning-alert",
        "KXBTCTEST-30DEC31-T101",
        AS_OF + timedelta(minutes=1),
        MarketCheckStatus.DELIVERED,
        "YES",
    )
    save_check(
        "cycle-losing-alert",
        "KXBTCTEST-30DEC31-T102",
        AS_OF + timedelta(minutes=2),
        MarketCheckStatus.DELIVERED,
        "NO",
    )
    save_check(
        "cycle-without-alert",
        "KXBTCTEST-30DEC31-T103",
        AS_OF + timedelta(minutes=3),
        MarketCheckStatus.WATCH,
    )
    save_check(
        "cycle-unresolved-alert",
        "KXBTCTEST-30DEC31-T104",
        AS_OF + timedelta(minutes=3, seconds=30),
        MarketCheckStatus.DELIVERED,
        "YES",
    )
    save_resolution(
        "KXBTCTEST-30DEC31-T101",
        "yes",
        AS_OF + timedelta(minutes=4),
    )
    save_resolution(
        "KXBTCTEST-30DEC31-T102",
        "yes",
        AS_OF + timedelta(minutes=5),
    )

    second = repository.paper_alert_status_since_last_request(
        series_ticker="KXBTC",
        symbol="BTC",
        requested_at=AS_OF + timedelta(minutes=10),
    )

    assert second.previous_requested_at == AS_OF
    assert second.cycles == 4
    assert second.delivered_alerts == 4
    assert second.resolved_alerts == 3
    assert second.profitable_alerts == 2
    assert second.unresolved_alerts == 1

    third = repository.paper_alert_status_since_last_request(
        series_ticker="KXBTC",
        symbol="BTC",
        requested_at=AS_OF + timedelta(minutes=11),
    )
    assert third.cycles == 0
    assert third.delivered_alerts == 4
    assert third.profitable_alerts == 2


def test_paper_alert_status_displays_activity_since_previous_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "paper-alert-status.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))
    repository = SQLiteRepository(database_path)
    repository.initialize()
    repository.save_paper_market_check(
        cycle_id="cycle-status-display",
        market_id="KXBTCTEST-30DEC31-T100",
        series_ticker="KXBTC",
        event_ticker="KXBTCTEST-30DEC31",
        observed_at=AS_OF,
        status=MarketCheckStatus.WATCH,
        reason=None,
        payload={},
    )

    first = cli_runner.invoke(
        app,
        ["paper-alert-status", "--series", "KXBTC", "--symbol", "BTC"],
    )
    second = cli_runner.invoke(
        app,
        ["paper-alert-status", "--series", "KXBTC", "--symbol", "BTC"],
    )

    assert first.exit_code == 0, first.output
    assert "Cycles since last request" in first.output
    assert "Profitable alerts" in first.output
    assert "counts include all recorded activity" in first.output
    assert second.exit_code == 0, second.output
    assert "Previous status request:" in second.output


def test_paper_alert_command_records_regime_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fetch_research(
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        interval_seconds: int,
        event_ticker: str | None,
    ) -> _ResearchDataBatch:
        candles: list[SpotCandle] = []
        candle_start = start_at
        price = Decimal("100")
        while candle_start < end_at:
            candle_end = candle_start + timedelta(seconds=interval_seconds)
            candles.append(
                spot_candle(
                    candle_end,
                    str(price),
                    interval_seconds=interval_seconds,
                )
            )
            price += Decimal("1")
            candle_start = candle_end
        return _ResearchDataBatch(
            spot_candles=candles,
            volatility_observations=[],
            funding_observations=[],
            derivatives_snapshots=[],
            event_snapshots=[],
        )

    async def fetch_live_spot(symbol: str, as_of: datetime) -> SpotCandle:
        return spot_candle(as_of, "100", interval_seconds=60)

    cycle_arguments: dict[str, object] = {}

    async def run_cycle(**kwargs: object) -> PaperAlertCycleResult:
        cycle_arguments.update(kwargs)
        return PaperAlertCycleResult(
            discovered=0,
            evaluated=0,
            watch=0,
            delivered=0,
            unsupported=0,
            uncalibrated=0,
            failures=(),
        )

    database_path = tmp_path / "paper-alerts.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))
    monkeypatch.setenv(
        "PMS_DISCORD_WEBHOOK_URL",
        "https://discord.com/api/webhooks/123/secret",
    )
    monkeypatch.setattr("prediction_market_system.cli._fetch_research_data", fetch_research)
    monkeypatch.setattr("prediction_market_system.cli._fetch_live_spot", fetch_live_spot)
    monkeypatch.setattr("prediction_market_system.cli._load_kalshi_markets", lambda *args: [])
    monkeypatch.setattr("prediction_market_system.cli._run_paper_alert_cycle", run_cycle)

    research_result = cli_runner.invoke(
        app,
        [
            "paper-alert-research",
            "--series",
            "KXBTC",
            "--symbol",
            "BTC",
            "--interval",
            "1440",
        ],
    )
    assert research_result.exit_code == 0, research_result.output
    result = cli_runner.invoke(
        app,
        [
            "paper-alerts",
            "--series",
            "KXBTC",
            "--symbol",
            "BTC",
            "--interval",
            "1440",
            "--send-discord",
            "--allow-unapproved-discord",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Paper-alert cycle: KXBTC" in result.output
    assert cycle_arguments["send_discord"] is True
    assert cycle_arguments["allow_unapproved_discord"] is True
    coverage = SQLiteRepository(database_path).market_regime_coverage(
        series_ticker="KXBTC",
        symbol="BTC",
    )
    assert sum(int(row["observation_count"]) for row in coverage) == 1
    activity = SQLiteRepository(database_path).paper_alert_status_since_last_request(
        series_ticker="KXBTC",
        symbol="BTC",
        requested_at=datetime.now(UTC) + timedelta(seconds=1),
    )
    assert activity.cycles == 1


def _synthetic_research_batch(
    start_at: datetime,
    end_at: datetime,
    interval_seconds: int,
) -> _ResearchDataBatch:
    candles: list[SpotCandle] = []
    candle_start = start_at
    price = Decimal("100")
    while candle_start < end_at:
        candle_end = candle_start + timedelta(seconds=interval_seconds)
        candles.append(spot_candle(candle_end, str(price), interval_seconds=interval_seconds))
        price += Decimal("1")
        candle_start = candle_end
    return _ResearchDataBatch(
        spot_candles=candles,
        volatility_observations=[],
        funding_observations=[],
        derivatives_snapshots=[],
        event_snapshots=[],
    )


def test_paper_alert_command_refreshes_stale_research_before_evaluating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research_calls: list[str | None] = []

    async def fetch_research(
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        interval_seconds: int,
        event_ticker: str | None,
    ) -> _ResearchDataBatch:
        research_calls.append(event_ticker)
        return _synthetic_research_batch(start_at, end_at, interval_seconds)

    async def fetch_live_spot(symbol: str, as_of: datetime) -> SpotCandle:
        return spot_candle(as_of, "100", interval_seconds=60)

    async def run_cycle(**kwargs: object) -> PaperAlertCycleResult:
        return PaperAlertCycleResult(
            discovered=0,
            evaluated=0,
            watch=0,
            delivered=0,
            unsupported=0,
            uncalibrated=0,
            failures=(),
        )

    database_path = tmp_path / "paper-alerts.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))
    monkeypatch.setattr("prediction_market_system.cli._fetch_research_data", fetch_research)
    monkeypatch.setattr("prediction_market_system.cli._fetch_live_spot", fetch_live_spot)
    monkeypatch.setattr("prediction_market_system.cli._load_kalshi_markets", lambda *args: [])
    monkeypatch.setattr("prediction_market_system.cli._run_paper_alert_cycle", run_cycle)

    result = cli_runner.invoke(
        app,
        ["paper-alerts", "--series", "KXBTC", "--symbol", "BTC", "--interval", "1440"],
    )

    assert result.exit_code == 0, result.output
    assert "refreshing before evaluation" in result.output
    assert "Paper-alert cycle: KXBTC" in result.output
    assert research_calls == [None]
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT status, request_json FROM research_data_sync_runs"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "succeeded"
    assert '"purpose": "paper-alerts-recovery"' in rows[0][1]
    coverage = SQLiteRepository(database_path).market_regime_coverage(
        series_ticker="KXBTC",
        symbol="BTC",
    )
    assert sum(int(row["observation_count"]) for row in coverage) == 1


def test_paper_alert_command_reports_failed_research_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fetch_research(
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        interval_seconds: int,
        event_ticker: str | None,
    ) -> _ResearchDataBatch:
        raise CoinbaseDataError("Coinbase request failed for /products/BTC-USD/candles")

    async def fetch_live_spot(symbol: str, as_of: datetime) -> SpotCandle:
        return spot_candle(as_of, "100", interval_seconds=60)

    database_path = tmp_path / "paper-alerts.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))
    monkeypatch.setattr("prediction_market_system.cli._fetch_research_data", fetch_research)
    monkeypatch.setattr("prediction_market_system.cli._fetch_live_spot", fetch_live_spot)

    result = cli_runner.invoke(
        app,
        ["paper-alerts", "--series", "KXBTC", "--symbol", "BTC", "--interval", "1440"],
    )

    assert result.exit_code == 1
    assert "refreshing before evaluation" in result.output
    assert "Paper-alert evaluation data unavailable" in result.output
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT status, error FROM research_data_sync_runs").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "failed"
    assert "Coinbase request failed" in rows[0][1]
