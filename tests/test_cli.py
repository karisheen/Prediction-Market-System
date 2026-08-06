from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from typer.testing import CliRunner

from prediction_market_system.calibration import (
    CalibrationBin,
    UncertaintyCalibrationProfile,
)
from prediction_market_system.cli import _ResearchDataBatch, app
from prediction_market_system.research import (
    DerivativesSnapshot,
    FundingObservation,
    SpotCandle,
    VolatilityObservation,
)
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import (
    KalshiMarket,
    KalshiOrderBook,
    to_market_snapshot,
)

runner = CliRunner()


def kalshi_market(
    *,
    can_close_early: bool = False,
    touch_rule: bool = False,
) -> KalshiMarket:
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
            "rules_primary": (
                "Resolves YES if the benchmark reaches the threshold at any time before expiry."
                if touch_rule
                else "Resolves YES from the benchmark at expiry."
            ),
            "rules_secondary": "",
            "yes_bid_dollars": "0.4200",
            "yes_ask_dollars": "0.4400",
            "no_bid_dollars": "0.5600",
            "no_ask_dollars": "0.5800",
            "yes_bid_size_fp": "13.00",
            "yes_ask_size_fp": "17.00",
        }
    )


def market_pair(
    *,
    can_close_early: bool = False,
    touch_rule: bool = False,
) -> tuple[KalshiMarket, object]:
    market = kalshi_market(can_close_early=can_close_early, touch_rule=touch_rule)
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
            "--allow-uncalibrated",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Bitcoin price at year end?" in result.output
    assert database_path.exists()


def test_kalshi_evaluate_requires_held_out_calibration(
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
    monkeypatch.setenv("PMS_DATABASE_PATH", str(tmp_path / "missing-calibration.db"))

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
    assert "run a calibrated backtest first" in result.output


def test_kalshi_evaluate_uses_persisted_calibration(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    pair = market_pair()
    profile = UncertaintyCalibrationProfile(
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
                outcome_interval_lower=0.3,
                outcome_interval_upper=0.7,
                uncertainty_margin=0.20,
                sample_count=30,
            ),
        ),
    )
    monkeypatch.setattr(
        "prediction_market_system.cli._load_kalshi_market",
        lambda ticker: pair,
    )
    monkeypatch.setattr(
        "prediction_market_system.storage.SQLiteRepository.latest_uncertainty_calibration",
        lambda self, **kwargs: profile,
    )
    monkeypatch.setenv("PMS_DATABASE_PATH", str(tmp_path / "calibrated-live.db"))

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
    assert "(held_out)" in result.output


def test_kalshi_evaluate_rejects_uncalibrated_discord_alert(
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
    monkeypatch.setenv("PMS_DATABASE_PATH", str(tmp_path / "uncalibrated-alert.db"))

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
            "--allow-uncalibrated",
            "--send-discord",
        ],
    )

    assert result.exit_code == 2
    assert "Discord paper alerts require held-out" in result.output


def test_kalshi_evaluate_accepts_fixed_expiry_market_with_early_close_flag(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    pair = market_pair(can_close_early=True)
    monkeypatch.setattr(
        "prediction_market_system.cli._load_kalshi_market",
        lambda ticker: pair,
    )
    database_path = tmp_path / "kalshi-terminal.db"
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
            "--allow-uncalibrated",
        ],
    )

    assert result.exit_code == 0, result.output
    stored = SQLiteRepository(database_path).opportunity_history(1)
    assert stored[0]["forecast"]["model_name"] == ("crypto-terminal-above-threshold-market-anchor")


def test_kalshi_evaluate_accepts_explicit_touch_barrier(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    pair = market_pair(can_close_early=True, touch_rule=True)
    monkeypatch.setattr(
        "prediction_market_system.cli._load_kalshi_market",
        lambda ticker: pair,
    )
    database_path = tmp_path / "kalshi-barrier.db"
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
            "90000",
            "--volatility",
            "0.55",
            "--allow-uncalibrated",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Structural YES" in result.output
    assert database_path.exists()


def test_research_sync_and_context_commands(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    start_at = datetime(2030, 1, 1, tzinfo=UTC)
    end_at = start_at + timedelta(days=1)
    candles = []
    for offset in range(25):
        candle_start = start_at - timedelta(hours=1) + timedelta(hours=offset)
        price = Decimal(100 + offset)
        candles.append(
            SpotCandle(
                provider="coinbase",
                product_id="BTC-USD",
                interval_seconds=3600,
                start_at=candle_start,
                end_at=candle_start + timedelta(hours=1),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=Decimal("10"),
                retrieved_at=end_at,
                raw_payload={"offset": offset},
            )
        )
    batch = _ResearchDataBatch(
        spot_candles=candles,
        volatility_observations=[
            VolatilityObservation(
                provider="deribit",
                symbol="BTC",
                kind="implied",
                window_seconds=3600,
                source_start_at=end_at - timedelta(hours=1),
                observed_at=end_at,
                annualized_volatility=0.45,
                retrieved_at=end_at,
                raw_payload={"close": 45.0},
            )
        ],
        funding_observations=[
            FundingObservation(
                provider="deribit",
                instrument_name="BTC-PERPETUAL",
                observed_at=end_at,
                index_price=123.0,
                previous_index_price=122.0,
                funding_rate_1h=0.0001,
                funding_rate_8h=0.0008,
                retrieved_at=end_at,
                raw_payload={"interest_1h": 0.0001},
            )
        ],
        derivatives_snapshots=[
            DerivativesSnapshot(
                provider="deribit",
                instrument_name="BTC-PERPETUAL",
                observed_at=end_at,
                index_price=123.0,
                mark_price=123.1,
                basis=(123.1 / 123.0) - 1,
                open_interest=5000.0,
                current_funding=0.00001,
                funding_rate_8h=0.00008,
                retrieved_at=end_at,
                raw_payload={"open_interest": 5000.0},
            )
        ],
        event_snapshots=[],
    )

    async def fake_fetch(*args: object, **kwargs: object) -> _ResearchDataBatch:
        return batch

    monkeypatch.setattr(
        "prediction_market_system.cli._fetch_research_data",
        fake_fetch,
    )
    database_path = tmp_path / "research-cli.db"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(database_path))

    sync = runner.invoke(
        app,
        [
            "sync-research-data",
            "--symbol",
            "BTC",
            "--start",
            start_at.isoformat(),
            "--end",
            end_at.isoformat(),
            "--interval",
            "60",
            "--realized-window-days",
            "1",
        ],
    )
    context = runner.invoke(
        app,
        [
            "research-context",
            "--symbol",
            "BTC",
            "--as-of",
            end_at.isoformat(),
            "--interval",
            "60",
            "--realized-window-days",
            "1",
        ],
    )

    assert sync.exit_code == 0, sync.output
    assert "Coinbase spot candles" in sync.output
    assert context.exit_code == 0, context.output
    assert "Point-in-time research context" in context.output
    assert "45.00%" in context.output
