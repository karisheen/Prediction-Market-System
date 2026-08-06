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
from prediction_market_system.storage import SQLiteRepository
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
            spot=candles[-1],
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

    result = await runner.run(
        markets=[kalshi_market(), kalshi_market(supported=False)],
        context=context,
        regime=regime,
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
    assert repository.opportunity_history(1)[0]["market_regime"]["price_trend"] == "uptrend"


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

    result = await runner.run(markets=[kalshi_market()], context=context, regime=regime)

    assert result.uncalibrated == 1
    assert result.evaluated == 0
    assert reader.requested == []
    assert publisher.published == []


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
    monkeypatch.setenv(
        "PMS_DISCORD_WEBHOOK_URL",
        "https://discord.com/api/webhooks/123/secret",
    )
    monkeypatch.setattr("prediction_market_system.cli._fetch_research_data", fetch_research)
    monkeypatch.setattr("prediction_market_system.cli._load_kalshi_markets", lambda *args: [])
    monkeypatch.setattr("prediction_market_system.cli._run_paper_alert_cycle", run_cycle)

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
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Paper-alert cycle: KXBTC" in result.output
    coverage = SQLiteRepository(database_path).market_regime_coverage(
        series_ticker="KXBTC",
        symbol="BTC",
    )
    assert sum(int(row["observation_count"]) for row in coverage) == 1
