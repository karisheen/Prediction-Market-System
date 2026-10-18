import gc
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prediction_market_system.alert_outcomes import (
    DeliveredAlert,
    first_alert_per_market,
    load_delivered_alerts,
    resolve_outcomes,
    score_alerts,
    summarize,
)
from prediction_market_system.backtest import EffectiveFee
from prediction_market_system.cli import app
from prediction_market_system.storage import SQLiteRepository

FEE = EffectiveFee(fee_type="quadratic", multiplier=0.07)
START = datetime(2026, 8, 12, 15, 0, tzinfo=UTC)

runner = CliRunner()


def alert(
    market_id: str,
    side: str,
    price: float,
    *,
    minutes: int = 0,
    probability_yes: float = 0.5,
) -> DeliveredAlert:
    return DeliveredAlert(
        opportunity_id=f"{market_id}-{side}-{minutes}",
        market_id=market_id,
        side="YES" if side == "YES" else "NO",
        created_at=START + timedelta(minutes=minutes),
        entry_price=price,
        probability_yes=probability_yes,
        market_probability_yes=price if side == "YES" else 1 - price,
        conservative_net_edge=0.04,
    )


def test_pnl_uses_entry_price_and_rounded_lot_fee() -> None:
    scored = score_alerts(
        [alert("KXBTC-A", "NO", 0.90), alert("KXBTC-B", "YES", 0.10)],
        {"KXBTC-A": "no", "KXBTC-B": "no"},
        fee=FEE,
        lot_contracts=100,
    )

    no_win, yes_loss = scored
    assert no_win.won
    assert no_win.fee_per_contract == Decimal("0.0063")
    assert no_win.pnl == Decimal("0.0937")
    assert not yes_loss.won
    assert yes_loss.pnl == Decimal("-0.1063")


def test_void_and_missing_results_are_excluded_from_pnl() -> None:
    scored = score_alerts(
        [
            alert("KXBTC-A", "YES", 0.40),
            alert("KXBTC-B", "YES", 0.40),
            alert("KXBTC-C", "NO", 0.60),
        ],
        {"KXBTC-A": "yes", "KXBTC-B": "scalar", "KXBTC-C": None},
        fee=FEE,
        lot_contracts=100,
    )

    summary = summarize("All", scored)

    assert summary.alerts == 3
    assert summary.resolved == 1
    assert summary.unresolved == 2
    assert summary.wins == 1
    assert summary.total_cost == Decimal("0.4168")
    assert summary.win_rate == 1.0


def test_first_alert_per_market_keeps_earliest_alert_per_side() -> None:
    scored = score_alerts(
        [
            alert("KXBTC-A", "NO", 0.95, minutes=10),
            alert("KXBTC-A", "NO", 0.80, minutes=0),
            alert("KXBTC-A", "YES", 0.05, minutes=5),
            alert("KXBTC-B", "NO", 0.70, minutes=1),
        ],
        {"KXBTC-A": "no", "KXBTC-B": "yes"},
        fee=FEE,
        lot_contracts=100,
    )

    first = first_alert_per_market(scored)

    assert [(i.alert.market_id, i.alert.side, i.alert.entry_price) for i in first] == [
        ("KXBTC-A", "NO", 0.80),
        ("KXBTC-B", "NO", 0.70),
        ("KXBTC-A", "YES", 0.05),
    ]


def test_brier_scores_compare_model_with_market_price() -> None:
    scored = score_alerts(
        [alert("KXBTC-A", "YES", 0.30, probability_yes=0.60)],
        {"KXBTC-A": "yes"},
        fee=FEE,
        lot_contracts=100,
    )

    summary = summarize("All", scored)

    assert summary.model_brier == pytest.approx(0.16)
    assert summary.market_brier == pytest.approx(0.49)


@pytest.mark.asyncio
async def test_resolution_prefers_database_and_fetches_only_missing() -> None:
    requested: list[str] = []

    async def fetch(market_id: str) -> str | None:
        requested.append(market_id)
        return "no"

    results, fetched = await resolve_outcomes(
        ["KXBTC-A", "KXBTC-B", "KXBTC-B"],
        {"KXBTC-A": "yes"},
        fetch,
    )

    assert results == {"KXBTC-A": "yes", "KXBTC-B": "no"}
    assert fetched == 1
    assert requested == ["KXBTC-B"]


def seeded_database(tmp_path: Path) -> Path:
    path = tmp_path / "outcomes.db"
    SQLiteRepository(path).initialize()
    rows = [
        ("opp-1", "KXBTC-26AUG1215-B1", "ENTER NO", "NO", 0.90),
        ("opp-2", "KXBTC-26AUG1215-B1", "ENTER NO", "NO", 0.92),
        ("opp-3", "KXBTC-26AUG1215-B2", "ENTER YES", "YES", 0.20),
    ]
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        for index, (opportunity_id, market_id, state, side, price) in enumerate(rows):
            payload = {
                "side": side,
                "executable_price": price,
                "conservative_net_edge": 0.05,
                "forecast": {"probability_yes": 0.3, "market_probability_yes": 0.25},
            }
            created_at = (START + timedelta(minutes=index)).isoformat()
            connection.execute(
                """
                INSERT INTO alert_events (
                    opportunity_id, market_id, state, status, attempts,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, 'delivered', 1, ?, ?, ?)
                """,
                (opportunity_id, market_id, state, created_at, created_at, json.dumps(payload)),
            )
        connection.execute(
            """
            INSERT INTO kalshi_resolutions (ticker, observed_at, result, expiration_value)
            VALUES ('KXBTC-26AUG1215-B1', ?, 'no', '63000.00')
            """,
            (START.isoformat(),),
        )
    gc.collect()
    return path


def test_loads_alert_payload_fields(tmp_path: Path) -> None:
    alerts, stored = load_delivered_alerts(seeded_database(tmp_path), series_ticker="kxbtc")

    assert [(a.market_id, a.side, a.entry_price) for a in alerts] == [
        ("KXBTC-26AUG1215-B1", "NO", 0.90),
        ("KXBTC-26AUG1215-B1", "NO", 0.92),
        ("KXBTC-26AUG1215-B2", "YES", 0.20),
    ]
    assert alerts[0].probability_yes == 0.3
    assert stored == {"KXBTC-26AUG1215-B1": "no"}


def test_command_reports_without_writing_to_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = seeded_database(tmp_path)
    before = path.read_bytes()
    report_path = tmp_path / "report.json"
    monkeypatch.setenv("PMS_DATABASE_PATH", str(path))

    result = runner.invoke(
        app,
        [
            "paper-alert-outcomes",
            "--series",
            "KXBTC",
            "--no-fetch-missing",
            "--json",
            str(report_path),
        ],
    )

    output = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "First alert per market and side" in output
    assert "unresolved or void 1" in output
    assert path.read_bytes() == before
    report = json.loads(report_path.read_text())
    assert report["unresolved_markets"] == ["KXBTC-26AUG1215-B2"]
    first_view = report["views"]["First alert per market and side"][0]
    assert first_view["alerts"] == 2
    assert first_view["resolved"] == 1
    assert first_view["wins"] == 1
