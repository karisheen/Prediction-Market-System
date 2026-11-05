from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from prediction_market_system import forward_capture as capture
from prediction_market_system.config import Settings
from prediction_market_system.evidence import content_id
from prediction_market_system.forward_protocol import (
    DEFAULT_FORWARD_PROTOCOL_PATH,
    load_forward_protocol,
)
from prediction_market_system.research import SpotCandle
from prediction_market_system.sources import coinbase
from prediction_market_system.sources.coinbase import CoinbaseClient
from prediction_market_system.sources.deribit import DeribitClient
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import KalshiMarket

SLOT = datetime(2026, 11, 6, 0, 45, tzinfo=UTC)


def market_payload(
    ticker: str = "KXBTC-E-A",
    *,
    lower: float = 99950,
    end: datetime | None = None,
    event: str = "KXBTC-E",
) -> dict[str, Any]:
    end = end or SLOT + timedelta(minutes=15)
    return {
        "ticker": ticker,
        "event_ticker": event,
        "series_ticker": "KXBTC",
        "market_type": "binary",
        "title": "Bitcoin range at expiry",
        "yes_sub_title": "Inside range",
        "no_sub_title": "Outside range",
        "close_time": end.isoformat(),
        "latest_expiration_time": end.isoformat(),
        "expected_expiration_time": end.isoformat(),
        "status": "active",
        "notional_value_dollars": "1",
        "can_close_early": False,
        "strike_type": "between",
        "floor_strike": lower,
        "cap_strike": lower + 100,
        "rules_primary": "Resolves from the average of the sixty seconds before expiry.",
        "rules_secondary": "The benchmark at expiry controls the terminal range.",
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.60",
        "no_bid_dollars": "0.40",
        "no_ask_dollars": "0.60",
        "yes_bid_size_fp": "20",
        "yes_ask_size_fp": "20",
    }


def seed_history(repo: SQLiteRepository, at: datetime = SLOT) -> None:
    end = at.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=721)
    candles = []
    for index in range(721):
        clock = start + timedelta(hours=index)
        price = Decimal(100000 + index % 17)
        candles.append(
            SpotCandle(
                provider="coinbase",
                product_id="BTC-USD",
                interval_seconds=3600,
                start_at=clock,
                end_at=clock + timedelta(hours=1),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=Decimal(10),
                retrieved_at=clock + timedelta(hours=1),
                raw_payload={"index": index},
            )
        )
    repo.save_research_data(spot_candles=candles)


class MemoryStore(capture._DiagnosticSink):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


class PublicFixture:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = SLOT
        self.calls: list[tuple[str, datetime]] = []
        self.book_calls = 0
        self.fail_book: int | None = None
        self.fail_coinbase = False
        self.bad_fee = False
        self.event_fees: dict[str, Any] = {}
        self.event_fee_changes: list[dict[str, Any]] = []
        self.pages: dict[str, dict[str, Any]] = {"": {"markets": [market_payload()], "cursor": ""}}
        self.outcome: str | None = None
        self.network = httpx.MockTransport(self.handle)
        monkeypatch.setattr(capture, "_now", lambda: self.now)
        owner = self

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return owner.now

        monkeypatch.setattr(coinbase, "datetime", Clock)
        original_sleep = asyncio.sleep

        async def sleep(seconds: float) -> None:
            self.now += timedelta(seconds=max(seconds, 0))
            await original_sleep(0)

        monkeypatch.setattr(capture.asyncio, "sleep", sleep)
        monkeypatch.setattr(capture, "_Sources", self.sources)

    def sources(self) -> Any:
        # Preserve the actual public clients, parsers, timeout and no-retry policy.
        source = object.__new__(_SOURCE_CLASS)
        source.kalshi = capture._PublicKalshi(
            client=httpx.AsyncClient(
                base_url="https://kalshi.test/trade-api/v2", transport=self.network
            ),
            max_rate_limit_retries=0,
            max_transient_retries=0,
        )
        source.coinbase = CoinbaseClient(
            client=httpx.AsyncClient(base_url="https://coinbase.test", transport=self.network),
            max_transient_retries=0,
        )
        source.deribit = DeribitClient(
            client=httpx.AsyncClient(base_url="https://deribit.test", transport=self.network),
            max_transient_retries=0,
        )
        # They own the temporary test HTTP clients, so the real lifecycle closes them.
        source.kalshi._owns_client = True
        source.coinbase._owns_client = True
        source.deribit._owns_client = True
        return source

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        path = request.url.path
        self.calls.append((path, self.now))
        if path.endswith("/candles"):
            if self.fail_coinbase:
                raise httpx.ConnectError("token=not-for-persistence", request=request)
            start = int(request.url.params["start"])
            end = int(request.url.params["end"])
            step = 60 if request.url.params["granularity"] == "ONE_MINUTE" else 3600
            return httpx.Response(
                200,
                json={
                    "candles": [
                        {
                            "start": str(clock),
                            "open": "100000",
                            "high": "100000",
                            "low": "100000",
                            "close": "100000",
                            "volume": "10",
                        }
                        for clock in range(start, end + 1, step)
                    ]
                },
            )
        if path.endswith("/get_volatility_index_data"):
            return httpx.Response(200, json={"result": {"data": [], "continuation": None}})
        if path.endswith("/get_funding_rate_history"):
            return httpx.Response(200, json={"result": []})
        if path.endswith("/ticker"):
            return httpx.Response(503, json={"error": "optional ticker unavailable"})
        if path.endswith("/orderbook"):
            self.book_calls += 1
            if self.book_calls == self.fail_book:
                raise httpx.ConnectError("first book unavailable", request=request)
            # A unique native payload proves delayed books were not reused.
            return httpx.Response(
                200,
                json={
                    "orderbook_fp": {
                        "yes_dollars": [["0.40", str(2 + self.book_calls)]],
                        "no_dollars": [["0.40", str(2 + self.book_calls)]],
                    }
                },
            )
        if path.endswith("/series/KXBTC"):
            return httpx.Response(
                200,
                json={
                    "series": {
                        "ticker": "KXBTC",
                        "fee_type": "quadratic",
                        "fee_multiplier": 2 if self.bad_fee else 1,
                    }
                },
            )
        if path.endswith("/series/fee_changes"):
            return httpx.Response(200, json={"series_fee_change_arr": []})
        if path.endswith("/events/fee_changes"):
            return httpx.Response(
                200, json={"event_fee_changes": self.event_fee_changes, "cursor": ""}
            )
        if "/events/" in path:
            return httpx.Response(
                200, json={"event": {"event_ticker": "KXBTC-E", **self.event_fees}}
            )
        if path.endswith("/markets"):
            return httpx.Response(200, json=self.pages[request.url.params.get("cursor", "")])
        if "/markets/" in path and self.outcome:
            payload = market_payload()
            payload.update(status="finalized", result=self.outcome)
            return httpx.Response(200, json={"market": payload})
        raise AssertionError(f"unexpected public request: {request.url}")


_SOURCE_CLASS = capture._Sources


@pytest.fixture
def public(monkeypatch: pytest.MonkeyPatch) -> PublicFixture:
    return PublicFixture(monkeypatch)


@pytest.fixture
def protocol() -> dict[str, Any]:
    return load_forward_protocol(DEFAULT_FORWARD_PROTOCOL_PATH)


def test_selection_ignores_outcomes_quotes_and_state() -> None:
    first = market_payload("KXBTC-E-A", lower=99900)
    second = market_payload("KXBTC-E-B", lower=100000)
    later = market_payload("KXBTC-L-A", end=SLOT + timedelta(minutes=30), event="KXBTC-L")
    rows = [
        capture._candidate(KalshiMarket.model_validate(row), SLOT, SLOT)
        for row in (second, later, first)
    ]
    assert capture._choose(rows, 100000) == ("KXBTC-E", ["KXBTC-E-A", "KXBTC-E-B"])
    first.update(result="no", status="finalized", yes_ask_dollars="0.99")
    second.update(result="yes", status="settled", yes_ask_dollars="0.01")
    changed = [
        capture._candidate(KalshiMarket.model_validate(row), SLOT, SLOT)
        for row in (second, later, first)
    ]
    assert capture._choose(changed, 100000) == capture._choose(rows, 100000)


@pytest.mark.parametrize(
    "seconds,eligible", [(300, False), (301, True), (3600, True), (3601, False)]
)
def test_selection_expiry_boundaries(seconds: int, eligible: bool) -> None:
    market = KalshiMarket.model_validate(market_payload(end=SLOT + timedelta(seconds=seconds)))
    assert capture._candidate(market, SLOT, SLOT)["eligible"] is eligible


@pytest.mark.asyncio
async def test_complete_discovery_retains_every_page_and_rejects_cursor_cycle(
    public: PublicFixture,
) -> None:
    public.pages = {
        "": {"markets": [market_payload("KXBTC-E-A")], "cursor": "next"},
        "next": {"markets": [market_payload("KXBTC-E-B")], "cursor": "next"},
    }
    rows: list[dict[str, Any]] = []
    async with public.sources() as sources:
        with pytest.raises(ValueError, match="repeated pagination"):
            await capture._discover(sources, SLOT, rows)
    assert [row["market_id"] for row in rows] == ["KXBTC-E-A", "KXBTC-E-B"]
    assert public.book_calls == 0


@pytest.mark.asyncio
async def test_live_machinery_captures_watch_and_independent_delayed_books(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    repo = SQLiteRepository(tmp_path / "source.db")
    repo.initialize()
    seed_history(repo)
    sink = MemoryStore(repo.database_path)
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["status"] == "valid"
    rows = sink.records(protocol["protocol_id"])
    forecast = next(row for row in rows if row["record_type"] == "market_capture")
    assert forecast["observation"]["opportunity"]["state"] == "WATCH"
    assert forecast["observation"]["resolution"] is None
    assert forecast["observation"]["inputs"]["calibration_profile"] is None
    assert (
        forecast["observation"]["run_manifest"]["configuration"]["engine"]
        == protocol["engine_config"]
    )
    execution = [row for row in rows if row["record_type"] == "execution_capture"]
    assert len(execution) == 3
    assert public.book_calls == 4
    assert len({forecast["raw_book_id"], *(row["raw_book_id"] for row in execution)}) == 4
    for row in execution:
        assert row["status"] == "valid"
        assert datetime.fromisoformat(row["requested_at"]) >= datetime.fromisoformat(
            row["target_at"]
        )
    # Existing complete hourly source history was reused, not downloaded again.
    candle_calls = [path for path, _ in public.calls if path.endswith("/candles")]
    assert len(candle_calls) == 1
    before = len(public.calls)
    repeat = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert repeat["status"] == "already-attempted"
    assert len(public.calls) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["book", "coinbase", "fee"])
async def test_source_failures_are_retained_without_retry(
    failure: str,
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    repo = SQLiteRepository(tmp_path / "source.db")
    repo.initialize()
    seed_history(repo)
    public.fail_book = 1 if failure == "book" else None
    public.fail_coinbase = failure == "coinbase"
    public.bad_fee = failure == "fee"
    sink = MemoryStore(repo.database_path)
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["status"] == "failed"
    assert public.book_calls == (0 if failure == "coinbase" else 1)
    assert "not-for-persistence" not in str(sink.rows)
    assert any(row["record_type"] == "slot_attempt" for row in sink.rows)
    assert any(row["record_type"] == "selection" for row in sink.rows)


@pytest.mark.asyncio
async def test_late_attempt_does_not_fetch_and_cannot_be_replaced(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    public.now = SLOT + timedelta(seconds=61)
    sink = MemoryStore(tmp_path / "unused.db")
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=sink.path),
    )
    assert result["status"] == "late"
    assert not public.calls
    public.now = SLOT
    repeat = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=sink.path),
    )
    assert repeat["status"] == "already-attempted"


@pytest.mark.asyncio
@pytest.mark.parametrize("delay,valid", [(120, True), (120.001, False)])
async def test_execution_receipt_window_is_closed_at_120_seconds(
    delay: float,
    valid: bool,
    public: PublicFixture,
    protocol: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with public.sources() as sources:
        original = sources.kalshi.raw_book

        async def delayed(market_id: str) -> dict[str, Any]:
            raw = await original(market_id)
            public.now += timedelta(seconds=delay)
            return raw

        monkeypatch.setattr(sources.kalshi, "raw_book", delayed)
        sink = capture._DiagnosticSink()
        await capture._execution(
            sink=sink,
            sources=sources,
            protocol=protocol,
            attempt={"slot_id": "slot", "attempt_id": "attempt"},
            market=KalshiMarket.model_validate(market_payload()),
            observation={"observed_at": SLOT.isoformat(), "observation_id": "forecast"},
            scenario=protocol["cost_scenarios"][0],
        )
    assert (sink.rows[0]["status"] == "valid") is valid
    assert public.book_calls == 1


@pytest.mark.asyncio
async def test_fresh_spot_rejects_stale_and_future_receipts(
    public: PublicFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = SQLiteRepository(tmp_path / "spot.db")
    repo.initialize()
    async with public.sources() as sources:
        original = sources.coinbase.get_candles

        async def stale(*args: Any, **kwargs: Any) -> list[SpotCandle]:
            candles = await original(*args, **kwargs)
            public.now += timedelta(seconds=121)
            return candles

        monkeypatch.setattr(sources.coinbase, "get_candles", stale)
        with pytest.raises(ValueError, match="stale or future"):
            await capture._fresh_spot(sources, repo)


@pytest.mark.asyncio
async def test_diagnostic_uses_real_books_without_touching_study_database(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    repo = SQLiteRepository(tmp_path / "existing.db")
    repo.initialize()
    seed_history(repo)
    before = repo.database_path.read_bytes()
    frozen_id = content_id(protocol)
    result = await capture.probe_forward_capture(
        protocol=protocol,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["diagnostic"] is True and result["eligible"] is False
    assert result["status"] == "valid"
    assert result["valid_captures"] == 1 and result["valid_executions"] == 3
    assert public.book_calls == 4
    assert repo.database_path.read_bytes() == before
    assert content_id(protocol) == frozen_id


@pytest.mark.asyncio
async def test_resolver_preserves_actual_availability_and_conflicting_revision(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    sink = MemoryStore(tmp_path / "unused.db")
    sink.rows.append(
        {
            "record_type": "market_capture",
            "protocol_id": protocol["protocol_id"],
            "market_id": "KXBTC-E-A",
            "event_id": "KXBTC-E",
            "status": "valid",
            "observation": {
                "opportunity": {
                    "market": {
                        "observation_end_at": (SLOT + timedelta(minutes=15)).isoformat(),
                        "expires_at": (SLOT + timedelta(minutes=15)).isoformat(),
                    }
                }
            },
        }
    )
    public.outcome = "yes"
    # Not closed yet: no discovery, label, or other-market request is permitted.
    before_close = await capture.resolve_forward_outcomes(store=sink, protocol=protocol)
    assert before_close["queried"] == 0 and not public.calls
    public.now = SLOT + timedelta(hours=2)
    first = await capture.resolve_forward_outcomes(store=sink, protocol=protocol)
    assert first["resolved"] == 1 and "yes" not in str(first)
    resolutions = [row for row in sink.rows if row["record_type"] == "resolution"]
    assert resolutions[0]["observed_at"] == public.now.isoformat()
    await capture.resolve_forward_outcomes(store=sink, protocol=protocol)
    assert sum(row["record_type"] == "resolution" for row in sink.rows) == 1
    public.now += timedelta(minutes=1)
    public.outcome = "no"
    await capture.resolve_forward_outcomes(store=sink, protocol=protocol)
    assert {row["result"] for row in sink.rows if row["record_type"] == "resolution"} == {
        "yes",
        "no",
    }
    calls = len(public.calls)
    public.now = datetime.fromisoformat(protocol["report_as_of"])
    closed = await capture.resolve_forward_outcomes(store=sink, protocol=protocol)
    assert closed["status"] == "closed" and len(public.calls) == calls


@pytest.mark.asyncio
async def test_all_three_markets_get_each_latency_without_serial_waits(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    public.pages[""]["markets"] = [
        market_payload(f"KXBTC-E-{name}", lower=99900 + index * 100)
        for index, name in enumerate(("A", "B", "C"))
    ]
    repo = SQLiteRepository(tmp_path / "three.db")
    repo.initialize()
    seed_history(repo)
    sink = MemoryStore(repo.database_path)
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["status"] == "valid"
    execution = [row for row in sink.rows if row["record_type"] == "execution_capture"]
    assert len(execution) == 9
    assert public.book_calls == 12
    for row in execution:
        assert row["status"] == "valid"
        target = datetime.fromisoformat(row["target_at"])
        assert target <= datetime.fromisoformat(row["requested_at"])
        assert datetime.fromisoformat(row["received_at"]) <= target + timedelta(seconds=120)


@pytest.mark.asyncio
async def test_forecast_generation_cannot_hide_late_persistence(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = SQLiteRepository(tmp_path / "late.db")
    repo.initialize()
    seed_history(repo)
    original = capture._observation

    def slow_forecast(**kwargs: Any) -> dict[str, Any]:
        observation = original(**kwargs)
        public.now += timedelta(seconds=5.001)
        return observation

    monkeypatch.setattr(capture, "_observation", slow_forecast)
    sink = MemoryStore(repo.database_path)
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["status"] == "failed"
    forecast = next(row for row in sink.rows if row["record_type"] == "market_capture")
    assert "persistence allowance" in forecast["error"]
    assert forecast["forecast_persisted_at"] is None
    assert public.book_calls == 1


@pytest.mark.asyncio
async def test_incremental_hourly_tail_refresh_preserves_full_window(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    repo = SQLiteRepository(tmp_path / "tail.db")
    repo.initialize()
    seed_history(repo, SLOT - timedelta(hours=1))
    sink = MemoryStore(repo.database_path)
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["status"] == "valid"
    assert sum(path.endswith("/candles") for path, _ in public.calls) == 2
    observation = next(
        row["observation"] for row in sink.rows if row["record_type"] == "market_capture"
    )
    realized = observation["research_context"]["realized_volatility"]
    assert realized["window_seconds"] == 30 * 24 * 3600
    assert realized["raw_payload"]["return_count"] == 720


@pytest.mark.asyncio
async def test_incomplete_universe_persists_outage_and_cannot_retry(
    public: PublicFixture,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    public.pages = {
        "": {"markets": [market_payload("KXBTC-E-A")], "cursor": "cycle"},
        "cycle": {"markets": [market_payload("KXBTC-E-B")], "cursor": "cycle"},
    }
    repo = SQLiteRepository(tmp_path / "pages.db")
    repo.initialize()
    seed_history(repo)
    sink = MemoryStore(repo.database_path)
    result = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert result["status"] == "failed"
    selection = next(row for row in sink.rows if row["record_type"] == "selection")
    assert selection["discovery_complete"] is False
    assert len(selection["discovered_candidates"]) == 2
    assert selection["selected_market_ids"] == []
    assert public.book_calls == 0
    public.pages = {"": {"markets": [market_payload()], "cursor": ""}}
    before = len(public.calls)
    retry = await capture.capture_forward_slot(
        store=sink,
        protocol=protocol,
        scheduled_at=SLOT,
        settings=Settings(database_path=repo.database_path),
    )
    assert retry["status"] == "already-attempted" and len(public.calls) == before


@pytest.mark.asyncio
async def test_spot_receipt_cannot_be_future(
    public: PublicFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = SQLiteRepository(tmp_path / "future.db")
    repo.initialize()
    async with public.sources() as sources:
        original = sources.coinbase.get_candles

        async def future(*args: Any, **kwargs: Any) -> list[SpotCandle]:
            candles = await original(*args, **kwargs)
            return [
                candle.model_copy(update={"retrieved_at": public.now + timedelta(seconds=1)})
                for candle in candles
            ]

        monkeypatch.setattr(sources.coinbase, "get_candles", future)
        with pytest.raises(ValueError, match="receipt is from the future"):
            await capture._fresh_spot(sources, repo)


@pytest.mark.asyncio
@pytest.mark.parametrize("clear", [True, False])
async def test_current_event_null_clears_historical_fee_override(
    public: PublicFixture, clear: bool
) -> None:
    public.event_fee_changes = [
        {
            "id": "event-fee-1",
            "event_ticker": "KXBTC-E",
            "series_ticker": "KXBTC",
            "scheduled_ts": (SLOT - timedelta(hours=1)).isoformat(),
            "fee_type_override": "quadratic",
            "fee_multiplier_override": 2,
        }
    ]
    if clear:
        public.event_fees = {"fee_type_override": None, "fee_multiplier_override": None}
    async with public.sources() as sources:
        fees = await capture._fees(sources, "KXBTC-E")
    assert fees["fee_type"] == "quadratic"
    assert fees["fee_multiplier"] == (1 if clear else 2)


@pytest.mark.asyncio
async def test_resolver_queries_after_observation_end_before_later_expiry(
    public: PublicFixture, protocol: dict[str, Any], tmp_path: Path
) -> None:
    sink = MemoryStore(tmp_path / "unused.db")
    end = SLOT + timedelta(minutes=15)
    expiry = end + timedelta(hours=2)
    sink.rows.append(
        {
            "record_type": "market_capture",
            "protocol_id": protocol["protocol_id"],
            "market_id": "KXBTC-E-A",
            "event_id": "KXBTC-E",
            "status": "valid",
            "observation": {
                "opportunity": {
                    "market": {
                        "observation_end_at": end.isoformat(),
                        "expires_at": expiry.isoformat(),
                    }
                }
            },
        }
    )
    public.outcome = "yes"
    public.now = end - timedelta(microseconds=1)
    assert (await capture.resolve_forward_outcomes(store=sink, protocol=protocol))["queried"] == 0
    public.now = end + timedelta(seconds=1)
    result = await capture.resolve_forward_outcomes(store=sink, protocol=protocol)
    assert result["queried"] == result["resolved"] == 1
    resolution = next(row for row in sink.rows if row["record_type"] == "resolution")
    assert resolution["observed_at"] == resolution["received_at"] == public.now.isoformat()
    assert datetime.fromisoformat(resolution["observed_at"]) < expiry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current,reset_offset,expected_type,expected_multiplier",
    [
        ({"fee_type_override": None}, None, "quadratic", 2),
        ({"fee_multiplier_override": None}, None, "flat", 1),
        ({}, -1, "quadratic", 1),
        ({}, 0, "quadratic", 1),
        ({}, 1, "flat", 2),
    ],
)
async def test_event_fee_resets_are_field_specific_and_effective_at_receipt(
    public: PublicFixture, current, reset_offset, expected_type, expected_multiplier
) -> None:
    previous = {
        "id": "event-fee-previous",
        "event_ticker": "KXBTC-E",
        "series_ticker": "KXBTC",
        "scheduled_ts": (SLOT - timedelta(hours=1)).isoformat(),
        "fee_type_override": "flat",
        "fee_multiplier_override": 2,
    }
    public.event_fee_changes = [previous]
    if reset_offset is not None:
        public.event_fee_changes.append(
            {
                **previous,
                "id": "event-fee-reset",
                "scheduled_ts": (SLOT + timedelta(seconds=reset_offset)).isoformat(),
                "fee_type_override": None,
                "fee_multiplier_override": None,
            }
        )
    public.event_fees = current
    async with public.sources() as sources:
        fees = await capture._fees(sources, "KXBTC-E")
    assert fees["fee_type"] == expected_type
    assert fees["fee_multiplier"] == expected_multiplier
