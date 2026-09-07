from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from prediction_market_system.research import (
    DerivativesSnapshot,
    EventDataSnapshot,
    FundingObservation,
    ResearchContext,
    ResearchDataUnavailable,
    SpotCandle,
    VolatilityObservation,
    calculate_realized_volatility,
)
from prediction_market_system.storage import SQLiteRepository


def spot_candle(end_at: datetime, close: str) -> SpotCandle:
    close_value = Decimal(close)
    return SpotCandle(
        provider="coinbase",
        product_id="BTC-USD",
        interval_seconds=3600,
        start_at=end_at - timedelta(hours=1),
        end_at=end_at,
        open=close_value,
        high=close_value + Decimal("1"),
        low=close_value - Decimal("1"),
        close=close_value,
        volume=Decimal("10"),
        retrieved_at=end_at,
        raw_payload={"close": close},
    )


def implied_volatility(observed_at: datetime, value: float) -> VolatilityObservation:
    return VolatilityObservation(
        provider="deribit",
        symbol="BTC",
        kind="implied",
        window_seconds=3600,
        source_start_at=observed_at - timedelta(hours=1),
        observed_at=observed_at,
        annualized_volatility=value,
        retrieved_at=observed_at,
        raw_payload={"close": value * 100},
    )


def funding(observed_at: datetime, rate: float) -> FundingObservation:
    return FundingObservation(
        provider="deribit",
        instrument_name="BTC-PERPETUAL",
        observed_at=observed_at,
        index_price=100.0,
        previous_index_price=99.0,
        funding_rate_1h=rate,
        funding_rate_8h=rate * 8,
        retrieved_at=observed_at,
        raw_payload={"interest_1h": rate},
    )


def derivatives(observed_at: datetime, basis: float) -> DerivativesSnapshot:
    return DerivativesSnapshot(
        provider="deribit",
        instrument_name="BTC-PERPETUAL",
        observed_at=observed_at,
        index_price=100.0,
        mark_price=100.0 * (1 + basis),
        basis=basis,
        open_interest=1_000.0,
        current_funding=0.0,
        funding_rate_8h=0.0001,
        retrieved_at=observed_at,
        raw_payload={"basis": basis},
    )


def event_data(observed_at: datetime, label: str) -> EventDataSnapshot:
    return EventDataSnapshot(
        provider="kalshi",
        event_ticker="KXBTCTEST-30DEC31",
        data_type="crypto",
        observed_at=observed_at,
        retrieved_at=observed_at,
        is_historical=False,
        details={"label": label},
        raw_payload={"type": "crypto", "details": {"label": label}},
    )


def test_realized_volatility_uses_only_complete_as_of_window() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)) for offset in range(25)
    ]
    future = spot_candle(as_of + timedelta(hours=1), "100000")

    expected = calculate_realized_volatility(
        candles,
        symbol="BTC",
        as_of=as_of,
        window_seconds=24 * 60 * 60,
    )
    with_future = calculate_realized_volatility(
        [*candles, future],
        symbol="BTC",
        as_of=as_of,
        window_seconds=24 * 60 * 60,
    )

    assert with_future.annualized_volatility == pytest.approx(expected.annualized_volatility)
    assert with_future.observed_at == as_of
    assert with_future.raw_payload["return_count"] == 24


def test_realized_volatility_rejects_partial_window() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=2 - offset), str(100 + offset)) for offset in range(3)
    ]

    with pytest.raises(ResearchDataUnavailable, match="full"):
        calculate_realized_volatility(
            candles,
            symbol="BTC",
            as_of=as_of,
            window_seconds=24 * 60 * 60,
        )


def test_repository_never_selects_future_observations(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)) for offset in range(25)
    ]
    database_path = tmp_path / "research.db"
    repository = SQLiteRepository(database_path)
    repository.initialize()
    repository.save_research_data(
        spot_candles=[
            *candles,
            spot_candle(as_of + timedelta(seconds=1), "100000"),
        ],
        volatility_observations=[
            implied_volatility(as_of - timedelta(hours=1), 0.45),
            implied_volatility(as_of + timedelta(seconds=1), 9.99),
        ],
        funding_observations=[
            funding(as_of - timedelta(hours=1), 0.0001),
            funding(as_of + timedelta(seconds=1), 0.99),
        ],
        derivatives_snapshots=[
            derivatives(as_of - timedelta(hours=1), 0.001),
            derivatives(as_of + timedelta(seconds=1), 0.5),
        ],
        event_snapshots=[
            event_data(as_of - timedelta(hours=1), "available"),
            event_data(as_of + timedelta(seconds=1), "future"),
        ],
    )

    context = repository.research_context_as_of(
        symbol="BTC",
        event_ticker="KXBTCTEST-30DEC31",
        as_of=as_of,
        interval_seconds=3600,
        realized_window_seconds=24 * 60 * 60,
    )

    assert context.spot.end_at == as_of
    assert context.implied_volatility is not None
    assert context.implied_volatility.annualized_volatility == pytest.approx(0.45)
    assert context.funding is not None
    assert context.funding.funding_rate_1h == pytest.approx(0.0001)
    assert context.derivatives is not None
    assert context.derivatives.basis == pytest.approx(0.001)
    assert context.event_data is not None
    assert context.event_data.details["label"] == "available"
    crypto = context.to_crypto_snapshot(strike_price=120.0)
    assert crypto.observed_at == as_of
    assert crypto.spot_price == pytest.approx(124.0)
    assert crypto.annualized_volatility == pytest.approx(0.45)


def research_candles(as_of: datetime) -> list[SpotCandle]:
    return [
        spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)) for offset in range(25)
    ]


def research_context(as_of: datetime) -> ResearchContext:
    candles = research_candles(as_of)
    return ResearchContext(
        symbol="BTC",
        event_ticker=None,
        as_of=as_of,
        spot=candles[-1],
        realized_volatility=calculate_realized_volatility(
            candles, symbol="BTC", as_of=as_of, window_seconds=86400
        ),
    )


@pytest.mark.parametrize("defect", ["gap", "duplicate", "stale_end"])
def test_realized_volatility_rejects_incomplete_intervals(defect: str) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    candles = [
        spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)) for offset in range(25)
    ]
    if defect == "gap":
        del candles[12]
    elif defect == "duplicate":
        candles.insert(12, candles[12])
    else:
        candles.pop()
    with pytest.raises(ResearchDataUnavailable, match="full"):
        calculate_realized_volatility(candles, symbol="BTC", as_of=as_of, window_seconds=86400)


def test_window_aligns_to_latest_completed_boundary() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    context = research_context(as_of)
    candles = research_candles(as_of)
    later = calculate_realized_volatility(
        candles, symbol="BTC", as_of=as_of + timedelta(minutes=17), window_seconds=86400
    )
    assert later.annualized_volatility == context.realized_volatility.annualized_volatility
    assert later.observed_at == as_of


def test_backfilled_candles_are_not_historically_available(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    repository = SQLiteRepository(tmp_path / "research.db")
    repository.initialize()
    repository.save_research_data(
        spot_candles=[
            spot_candle(as_of - timedelta(hours=24 - offset), str(100 + offset)).model_copy(
                update={"retrieved_at": as_of + timedelta(days=1)}
            )
            for offset in range(25)
        ]
    )
    with pytest.raises(ResearchDataUnavailable):
        repository.research_context_as_of(symbol="BTC", as_of=as_of, realized_window_seconds=86400)


def test_required_future_context_fields_fail_closed() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    context = research_context(as_of)
    for field in ("spot", "realized_volatility"):
        value = getattr(context, field)
        for clock in (
            ("end_at", "retrieved_at")
            if field == "spot"
            else ("observed_at", "source_start_at", "retrieved_at")
        ):
            future = value.model_copy(update={clock: as_of + timedelta(seconds=1)})
            poisoned = context.model_copy(update={field: future})
            with pytest.raises(ResearchDataUnavailable):
                poisoned.validate_at(as_of)


def test_refreshed_spot_does_not_hide_stale_volatility() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    context = research_context(as_of)
    actual = as_of + timedelta(hours=1)
    refreshed = context.model_copy(update={"spot": spot_candle(actual, "125")})
    with pytest.raises(ResearchDataUnavailable, match="volatility"):
        refreshed.validate_at(actual)


def test_future_optional_inputs_are_omitted_with_warning() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    context = research_context(as_of)
    payload = context.model_dump()
    payload.update(
        implied_volatility=implied_volatility(as_of + timedelta(seconds=1), 0.8),
        funding=funding(as_of, 0.1).model_copy(
            update={"retrieved_at": as_of + timedelta(seconds=1)}
        ),
        derivatives=derivatives(as_of + timedelta(seconds=1), 0.01),
        event_data=event_data(as_of + timedelta(seconds=1), "future"),
    )
    guarded = ResearchContext.model_validate(payload)
    assert guarded.implied_volatility is None
    assert guarded.funding is None
    assert guarded.derivatives is None
    assert guarded.event_data is None
    assert guarded.warnings


def test_mismatched_dvol_interval_is_not_selected(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    context = research_context(as_of)
    repository = SQLiteRepository(tmp_path / "research.db")
    repository.initialize()
    repository.save_research_data(
        spot_candles=research_candles(as_of),
        volatility_observations=[
            implied_volatility(as_of, 0.9).model_copy(
                update={"window_seconds": 60, "source_start_at": as_of - timedelta(minutes=1)}
            )
        ],
    )
    selected = repository.research_context_as_of(
        symbol="BTC", as_of=as_of, interval_seconds=3600, realized_window_seconds=86400
    )
    assert selected.implied_volatility is None
    assert selected.selected_annualized_volatility == context.selected_annualized_volatility


def test_revisions_preserve_values_at_original_availability(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    repository = SQLiteRepository(tmp_path / "research.db")
    repository.initialize()
    old = spot_candle(as_of, "100")
    revised = spot_candle(as_of, "105").model_copy(
        update={"retrieved_at": as_of + timedelta(minutes=1)}
    )
    first = repository.save_research_data(spot_candles=[old])
    second = repository.save_research_data(spot_candles=[revised])
    repeated = repository.save_research_data(
        spot_candles=[revised.model_copy(update={"retrieved_at": as_of + timedelta(minutes=2)})]
    )
    assert first.spot_candles == second.spot_candles == 1
    assert repeated.spot_candles == 0
    before = repository.spot_candles_as_of(
        symbol="BTC", as_of=as_of, interval_seconds=3600, window_seconds=86400
    )
    after = repository.spot_candles_as_of(
        symbol="BTC",
        as_of=as_of + timedelta(minutes=2),
        interval_seconds=3600,
        window_seconds=86400,
    )
    assert before[-1].close == Decimal("100")
    assert after[-1].close == Decimal("105")
    assert after[-1].retrieved_at == as_of + timedelta(minutes=1)
    with repository._connect() as connection:
        legacy = connection.execute("SELECT payload_json FROM crypto_spot_candles").fetchone()
        rows = connection.execute(
            "SELECT payload_json FROM research_provider_revisions ORDER BY available_at"
        ).fetchall()
    assert SpotCandle.model_validate_json(legacy["payload_json"]).close == Decimal("100")
    assert [SpotCandle.model_validate_json(row["payload_json"]).close for row in rows] == [
        Decimal("100"),
        Decimal("105"),
    ]
    repository.save_research_data(
        spot_candles=[old.model_copy(update={"retrieved_at": as_of + timedelta(minutes=3)})]
    )
    reverted = repository.spot_candles_as_of(
        symbol="BTC",
        as_of=as_of + timedelta(minutes=3),
        interval_seconds=3600,
        window_seconds=86400,
    )
    assert reverted[-1].close == Decimal("100")
    assert reverted[-1].retrieved_at == old.retrieved_at


def test_backdated_current_snapshots_wait_for_retrieval(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    repository = SQLiteRepository(tmp_path / "research.db")
    repository.initialize()
    repository.save_research_data(
        spot_candles=research_candles(as_of),
        derivatives_snapshots=[
            derivatives(as_of - timedelta(minutes=5), 0.1).model_copy(
                update={"retrieved_at": as_of + timedelta(seconds=1)}
            )
        ],
        event_snapshots=[
            event_data(as_of - timedelta(minutes=5), "current").model_copy(
                update={"retrieved_at": as_of + timedelta(seconds=1)}
            )
        ],
    )
    selected = repository.research_context_as_of(
        symbol="BTC",
        event_ticker="KXBTCTEST-30DEC31",
        as_of=as_of,
        realized_window_seconds=86400,
    )
    assert selected.derivatives is None
    assert selected.event_data is None


@pytest.mark.parametrize(
    ("save_name", "context_name", "source_table"),
    [
        ("volatility_observations", "implied_volatility", "crypto_volatility_observations"),
        ("funding_observations", "funding", "crypto_funding_observations"),
        ("derivatives_snapshots", "derivatives", "crypto_derivatives_snapshots"),
        ("event_snapshots", "event_data", "kalshi_event_data_snapshots"),
    ],
)
def test_optional_provider_revisions_remain_point_in_time(
    tmp_path: Path, save_name: str, context_name: str, source_table: str
) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    original, revision = {
        "implied_volatility": (implied_volatility(as_of, 0.45), implied_volatility(as_of, 0.55)),
        "funding": (funding(as_of, 0.001), funding(as_of, 0.002)),
        "derivatives": (derivatives(as_of, 0.01), derivatives(as_of, 0.02)),
        "event_data": (event_data(as_of, "original"), event_data(as_of, "revised")),
    }[context_name]
    revision = revision.model_copy(update={"retrieved_at": as_of + timedelta(seconds=30)})
    repository = SQLiteRepository(tmp_path / "research.db")
    repository.initialize()
    repository.save_research_data(spot_candles=research_candles(as_of))
    repository.save_research_data(**{save_name: [original]})
    repository.save_research_data(**{save_name: [revision]})
    before = repository.research_context_as_of(
        symbol="BTC",
        event_ticker="KXBTCTEST-30DEC31",
        as_of=as_of,
        realized_window_seconds=86400,
    )
    after = repository.research_context_as_of(
        symbol="BTC",
        event_ticker="KXBTCTEST-30DEC31",
        as_of=as_of + timedelta(seconds=60),
        realized_window_seconds=86400,
    )
    assert getattr(before, context_name).raw_payload == original.raw_payload
    assert getattr(after, context_name).raw_payload == revision.raw_payload
    with repository._connect() as connection:
        evidence = connection.execute(
            "SELECT payload_json FROM research_provider_revisions WHERE source_table = ? "
            "ORDER BY available_at",
            (source_table,),
        ).fetchall()
    assert [
        type(original).model_validate_json(row["payload_json"]).raw_payload for row in evidence
    ] == [original.raw_payload, revision.raw_payload]


def test_migrated_legacy_payload_keeps_first_recorded_retrieval(tmp_path: Path) -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    original = spot_candle(as_of, "100")
    repository = SQLiteRepository(tmp_path / "research.db")
    repository.initialize()
    with repository._connect() as connection:
        connection.execute(
            "INSERT INTO crypto_spot_candles VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                original.provider,
                original.product_id,
                original.interval_seconds,
                original.start_at.isoformat(),
                original.end_at.isoformat(),
                original.retrieved_at.isoformat(),
                original.model_dump_json(),
            ),
        )
    repository.save_research_data(
        spot_candles=[original.model_copy(update={"retrieved_at": as_of + timedelta(minutes=1)})]
    )
    later = repository.spot_candles_as_of(
        symbol="BTC",
        as_of=as_of + timedelta(minutes=2),
        interval_seconds=3600,
        window_seconds=86400,
    )
    assert later[-1].retrieved_at == original.retrieved_at
    with repository._connect() as connection:
        revision = connection.execute(
            "SELECT available_at FROM research_provider_revisions"
        ).fetchone()
    assert datetime.fromisoformat(revision["available_at"]) == original.retrieved_at


def test_evaluation_rechecks_selected_regime_inputs_after_network_delay() -> None:
    as_of = datetime(2030, 1, 2, tzinfo=UTC)
    payload = research_context(as_of).model_dump()
    payload.update(
        optional_max_age_seconds=60,
        funding=funding(as_of - timedelta(seconds=30), 0.001),
    )
    context = ResearchContext.model_validate(payload)
    context.validate_at(as_of)
    with pytest.raises(ResearchDataUnavailable, match="funding"):
        context.validate_at(as_of + timedelta(seconds=45))
