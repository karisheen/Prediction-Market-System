import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from test_backtest import FixedResearchSource, config, historical_market
from test_storage import build_evaluation
from typer.testing import CliRunner

from prediction_market_system.cli import app
from prediction_market_system.domain import (
    CryptoSnapshot,
    Opportunity,
    ProbabilityForecast,
    RecommendationState,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import EVIDENCE_FORWARD_SHADOW, content_id
from prediction_market_system.storage import SQLiteRepository


def test_manifest_replays_exact_inputs_and_refuses_changed_forecast(tmp_path: Path) -> None:
    from prediction_market_system.domain import MarketSnapshot, ThresholdContract

    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    forecast, opportunity = build_evaluation()
    assert isinstance(forecast, ProbabilityForecast) and isinstance(opportunity, Opportunity)
    repository.save_evaluation(forecast, opportunity)
    repository.save_evaluation(forecast, opportunity)
    explained = repository.explain_forecast(str(forecast.forecast_id))
    inputs = explained["inputs"]
    reproduced, _ = CryptoThresholdEngine(
        EngineConfig.model_validate(inputs["engine_config"])
    ).evaluate(
        MarketSnapshot.model_validate(inputs["market"]),
        CryptoSnapshot.model_validate(inputs["crypto"]),
        ThresholdContract.model_validate(inputs["contract"]),
    )
    assert reproduced.probability_yes == forecast.probability_yes
    assert reproduced.recipe_id == forecast.recipe_id
    assert explained["manifest"]["inputs"]["input_id"] == content_id(inputs)
    altered = forecast.model_copy(update={"probability_yes": 0.01})
    with pytest.raises(ValueError, match="immutable"):
        repository.save_evaluation(altered, opportunity.model_copy(update={"forecast": altered}))
    assert repository.explain_forecast(str(forecast.forecast_id)) == explained


def test_shadow_scoring_includes_watch_and_is_independent_of_deliveries(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    forecast, opportunity = build_evaluation()
    assert isinstance(forecast, ProbabilityForecast) and isinstance(opportunity, Opportunity)
    market = opportunity.market.model_copy(update={"series_id": "KXBTC", "event_id": "EVENT"})
    for probability in (0.2, 0.8):
        changed = forecast.model_copy(
            update={"forecast_id": uuid4(), "probability_yes": probability}
        )
        observation = opportunity.model_copy(
            update={
                "opportunity_id": uuid4(),
                "market": market,
                "forecast": changed,
                "state": RecommendationState.WATCH,
            }
        )
        repository.save_evaluation(changed, observation, ledger_kind=EVIDENCE_FORWARD_SHADOW)
    known = forecast.generated_at + timedelta(hours=1)
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "INSERT INTO kalshi_resolutions VALUES (?, ?, 'yes', '1', ?, '')",
            (market.market_id, known.isoformat(), known.isoformat()),
        )
    before = repository.shadow_report(series_ticker="KXBTC", as_of=known - timedelta(seconds=1))
    assert before["recipes"][0]["resolved_forecasts"] == 0
    scored = repository.shadow_report(series_ticker="KXBTC", as_of=known)["recipes"][0]
    assert scored["watch_forecasts"] == scored["resolved_forecasts"] == 2
    assert scored["independent_resolved_events"] == 1
    assert scored["event_weighted_brier"] == pytest.approx(0.34)


def test_reused_holdout_is_visible_and_cannot_approve_a_rerun(tmp_path: Path) -> None:
    from prediction_market_system.backtest import HistoricalBacktester

    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    result = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(
        config(), (historical_market(),)
    )
    first = repository.save_backtest_result(result)
    assert repository.save_backtest_result(first) == first
    rerun = result.model_copy(update={"run_id": uuid4()})
    second = repository.save_backtest_result(rerun)
    assert all(not item.accepted_for_paper_alerts for item in second.model_validations)
    assert all(
        any("holdout already consumed" in reason for reason in item.rejection_reasons)
        for item in second.model_validations
    )
    report = repository.campaign_report(series_ticker=config().series_ticker, symbol="BTC")
    assert report["reused_holdout_events"][0]["uses"] == 2
    comparison = repository.compare_runs(str(first.run_id), str(second.run_id))
    assert comparison["comparable_full_population"]
    assert comparison["event_weighted_brier_delta_second_minus_first"] == 0


def test_comparison_rejects_same_timestamp_with_different_source_revision(tmp_path: Path) -> None:
    from prediction_market_system.backtest import HistoricalBacktester

    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    original = HistoricalBacktester(FixedResearchSource(), EngineConfig()).run(
        config(), (historical_market(),)
    )
    repository.save_backtest_result(original)
    revised_folds = tuple(
        fold.model_copy(
            update={
                "forecasts": tuple(
                    item.model_copy(update={"input_population_id": "revised-data"})
                    for item in fold.forecasts
                )
            }
        )
        for fold in original.folds
    )
    revised = original.model_copy(update={"run_id": uuid4(), "folds": revised_folds})
    repository.save_backtest_result(revised)
    comparison = repository.compare_runs(str(original.run_id), str(revised.run_id))
    assert comparison["identical_observations"] == 0
    assert comparison["event_weighted_brier_delta_second_minus_first"] is None


def test_archive_coverage_stops_at_gap_and_repairs_old_incomplete_window(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for offset in (0, 2):
        archive = dict(
            series_ticker="KXBTC",
            symbol="BTC",
            start_at=start + timedelta(days=offset),
            end_at=start + timedelta(days=offset + 1),
            period_interval=1,
        )
        repository.begin_validation_archive(**archive)
        repository.complete_validation_archive(
            **archive, counts={"candlesticks": 100, "coverage_complete": 1}
        )
    assert repository.validation_archive_coverage(
        series_ticker="KXBTC",
        symbol="BTC",
        period_interval=1,
        campaign_start=start,
    ) == (1, start + timedelta(days=1))
    missing = dict(
        series_ticker="KXBTC",
        symbol="BTC",
        start_at=start + timedelta(days=1),
        end_at=start + timedelta(days=2),
        period_interval=1,
    )
    repository.begin_validation_archive(**missing)
    repository.complete_validation_archive(**missing, counts={"candlesticks": 100})
    assert not repository.validation_archive_succeeded(**missing)
    repository.begin_validation_archive(**missing)
    repository.complete_validation_archive(
        **missing, counts={"candlesticks": 100, "coverage_complete": 1}
    )
    assert repository.validation_archive_coverage(
        series_ticker="KXBTC",
        symbol="BTC",
        period_interval=1,
        campaign_start=start,
    ) == (3, start + timedelta(days=3))


def test_persistence_redacts_nested_failures(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    token = "fake-private-hook-value"
    failure = f"POST https://discord.com/api/webhooks/123/{token}?token=query-secret"
    run_id = repository.begin_research_sync(
        symbol="BTC", event_ticker=None, request={"api_key": token}
    )
    repository.complete_research_sync(run_id, error=failure)
    with sqlite3.connect(repository.database_path) as connection:
        payload = connection.execute(
            "SELECT request_json,error FROM research_data_sync_runs"
        ).fetchone()
    assert token not in repr(payload) and "query-secret" not in repr(payload)
    assert "[REDACTED]" in repr(payload)


def test_backup_restore_and_doctor_cli_do_not_migrate_legacy_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE immutable_evidence(value TEXT)")
        connection.execute("INSERT INTO immutable_evidence VALUES ('legacy research')")
    monkeypatch.setenv("PMS_DATABASE_PATH", str(source))
    runner = CliRunner()
    health = runner.invoke(app, ["doctor"])
    assert health.exit_code == 0, health.output
    assert json.loads(health.output)["integrity"] == "ok"
    backup = tmp_path / "backup.sqlite3"
    result = runner.invoke(app, ["db-backup", "--destination", str(backup)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["restore_verified"]
    restored = tmp_path / "restored.sqlite3"
    result = runner.invoke(
        app, ["db-restore", "--source", str(backup), "--destination", str(restored)]
    )
    assert result.exit_code == 0, result.output
    for path in (source, backup, restored):
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT * FROM immutable_evidence").fetchall() == [
                ("legacy research",)
            ]
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall() == [("immutable_evidence",)]


def test_schema_upgrade_keeps_legacy_payload_and_requires_verified_backup(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE legacy_evidence(id INTEGER PRIMARY KEY,payload TEXT)")
        connection.execute("INSERT INTO legacy_evidence VALUES (1,'original')")
    repository = SQLiteRepository(source)
    repository.initialize()
    backups = list((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    repository.initialize()
    assert list((tmp_path / "backups").glob("*.sqlite3")) == backups
    for path in (source, backups[0]):
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT * FROM legacy_evidence").fetchall() == [
                (1, "original")
            ]


def test_history_collection_records_receipt_not_requested_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prediction_market_system.cli import _fetch_kalshi_history
    from prediction_market_system.venues.kalshi import KalshiEventsResponse

    class EmptyClient:
        async def list_events(self, **kwargs: object) -> KalshiEventsResponse:
            return KalshiEventsResponse(events=[], cursor="")

        async def get_series_fee_changes(self, series: str) -> list[object]:
            return []

        async def close(self) -> None:
            pass

    monkeypatch.setattr("prediction_market_system.cli.KalshiClient", EmptyClient)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    before = datetime.now(UTC)
    batch = asyncio.run(
        _fetch_kalshi_history("KXBTC", start, start + timedelta(days=1), 60, 10, 3, 24)
    )
    assert before <= batch.observed_at <= datetime.now(UTC)


def test_archive_repair_reaches_gaps_older_than_catchup_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prediction_market_system.sources import CoinbaseDataError

    starts = []

    async def fail_after_selection(series: str, start: datetime, *args: object) -> object:
        starts.append(start)
        raise CoinbaseDataError("controlled unavailable provider")

    monkeypatch.setenv("PMS_DATABASE_PATH", str(tmp_path / "archive.db"))
    monkeypatch.setattr("prediction_market_system.cli._fetch_kalshi_history", fail_after_selection)
    result = CliRunner().invoke(
        app,
        [
            "paper-alert-archive",
            "--series",
            "KXBTC",
            "--symbol",
            "BTC",
            "--campaign-start",
            "2020-01-01T00:00:00Z",
            "--catch-up-days",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert starts == [datetime(2020, 1, 1, tzinfo=UTC)]


def test_campaign_configuration_is_frozen_until_explicitly_replaced(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audit.db")
    repository.initialize()
    configuration = {"maximum_brier_score": 0.25, "train_days": 90, "model_version": "v2"}
    campaign_id = repository.register_validation_campaign(
        series_ticker="kxbtc", symbol="btc", configuration=configuration
    )
    # Identical settings are the same preregistered campaign.
    assert (
        repository.register_validation_campaign(
            series_ticker="KXBTC", symbol="BTC", configuration=dict(configuration)
        )
        == campaign_id
    )
    # Loosening a gate after seeing results is refused.
    loosened = {**configuration, "maximum_brier_score": 0.30}
    with pytest.raises(ValueError, match="frozen"):
        repository.register_validation_campaign(
            series_ticker="KXBTC", symbol="BTC", configuration=loosened
        )
    # Other series/symbols are independent campaigns.
    other = repository.register_validation_campaign(
        series_ticker="KXETH", symbol="ETH", configuration=configuration
    )
    assert other != campaign_id

    replacement = repository.register_validation_campaign(
        series_ticker="KXBTC", symbol="BTC", configuration=loosened, replace=True
    )
    assert replacement != campaign_id
    report = repository.campaign_report(series_ticker="KXBTC", symbol="BTC")
    registrations = report["campaign_registrations"]
    assert [item["campaign_id"] for item in registrations] == [campaign_id, replacement]
    assert registrations[0]["superseded_at"] is not None
    assert registrations[1]["superseded_at"] is None
    assert registrations[0]["configuration"] == configuration
    assert registrations[1]["configuration"] == loosened
    # The original registration is preserved as legacy evidence, and the old
    # settings cannot silently become active again without another replacement.
    with pytest.raises(ValueError, match="frozen"):
        repository.register_validation_campaign(
            series_ticker="KXBTC", symbol="BTC", configuration=configuration
        )


def test_paper_alert_validate_cli_refuses_changed_campaign_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PMS_DATABASE_PATH", str(tmp_path / "audit.db"))
    runner = CliRunner()
    base = [
        "paper-alert-validate",
        "--series",
        "KXBTC",
        "--symbol",
        "BTC",
        "--campaign-start",
        "2030-01-01T00:00:00+00:00",
    ]
    first = runner.invoke(app, base)
    assert first.exit_code == 0, first.output
    assert "COLLECTING EVIDENCE" in first.output
    repeated = runner.invoke(app, base)
    assert repeated.exit_code == 0, repeated.output
    loosened = runner.invoke(app, [*base, "--maximum-brier-score", "0.5"])
    assert loosened.exit_code != 0
    assert "frozen" in loosened.output
    replaced = runner.invoke(app, [*base, "--maximum-brier-score", "0.5", "--replace-campaign"])
    assert replaced.exit_code == 0, replaced.output
    repository = SQLiteRepository(tmp_path / "audit.db")
    registrations = repository.campaign_report(series_ticker="KXBTC", symbol="BTC")[
        "campaign_registrations"
    ]
    assert len(registrations) == 2
    assert registrations[0]["superseded_at"] is not None
    assert registrations[1]["configuration"]["maximum_brier_score"] == 0.5
    assert "model_version" in registrations[1]["configuration"]
    assert "engine" in registrations[1]["configuration"]
