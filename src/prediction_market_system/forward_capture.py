"""Public, first-attempt-only collection for the independently frozen forward study.

No trading, delivery, calibration approval, or outcome scoring occurs in this module.
The diagnostic sink is deliberately not a ForwardStudyStore and cannot mint study evidence.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any, Literal, Protocol, TypeVar
from uuid import uuid4

from prediction_market_system.config import Settings
from prediction_market_system.domain import TerminalRangeContract
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.evidence import code_identity, content_id, save_input_object
from prediction_market_system.forward_protocol import scheduled_slots, validate_forward_protocol
from prediction_market_system.forward_storage import ForwardStudyStore
from prediction_market_system.redaction import redact_secrets
from prediction_market_system.research import ResearchContext, SpotCandle
from prediction_market_system.research_storage import ResearchRepositoryMixin
from prediction_market_system.sources.coinbase import CoinbaseClient
from prediction_market_system.sources.deribit import DeribitClient
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import (
    KalshiClient,
    KalshiMarket,
    KalshiNotFoundError,
    KalshiOrderBook,
    to_market_snapshot,
)

_T = TypeVar("_T")
_MAX_PAGES = 100
_REQUEST_TIMEOUT = 15.0
_WINDOW = 30 * 24 * 3600


class _Sink(Protocol):
    def append(self, record: dict[str, Any]) -> bool: ...

    def records(self, protocol_id: str) -> list[dict[str, Any]]: ...


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _error(error: BaseException) -> str:
    return redact_secrets(f"{type(error).__name__}: {error}")


async def _bounded(awaitable: Awaitable[_T], seconds: float = _REQUEST_TIMEOUT) -> _T:
    return await asyncio.wait_for(awaitable, timeout=seconds)


class _PublicKalshi(KalshiClient):
    async def payload(
        self, path: str, params: dict[str, str | int | bool] | None = None
    ) -> dict[str, Any]:
        response = await self._get(path, params=params)
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Kalshi response is not an object")
        return payload

    async def resolution_payload(self, market_id: str) -> dict[str, Any]:
        try:
            return await self.payload(f"/markets/{market_id}")
        except KalshiNotFoundError:
            return await self.payload(f"/historical/markets/{market_id}")

    async def raw_book(self, market_id: str) -> dict[str, Any]:
        payload = await self.payload(f"/markets/{market_id}/orderbook", {"depth": 100})
        raw = payload.get("orderbook_fp")
        if not isinstance(raw, dict):
            raise ValueError("Kalshi response lacks the original fixed-point order book")
        return raw


class _Sources:
    def __init__(self) -> None:
        # A retry would be a different quote. Disable retries even for transient errors.
        self.kalshi = _PublicKalshi(max_rate_limit_retries=0, max_transient_retries=0)
        self.coinbase = CoinbaseClient(max_transient_retries=0)
        self.deribit = DeribitClient(max_transient_retries=0)

    async def __aenter__(self) -> _Sources:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await asyncio.gather(self.kalshi.close(), self.coinbase.close(), self.deribit.close())


class _ReadConnection(sqlite3.Connection):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            return super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


class _ReadOnlyResearch(ResearchRepositoryMixin):
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{self.path.resolve().as_uri()}?mode=ro", uri=True, factory=_ReadConnection
        )
        connection.row_factory = sqlite3.Row
        return connection


class _DiagnosticSink:
    """Ephemeral evidence-shaped observations, never a production registration."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> bool:
        row = dict(record, persisted_at=_iso(_now()))
        if row["record_type"] == "market_capture" and row["status"] == "valid":
            row["forecast_persisted_at"] = row["persisted_at"]
        row["record_id"] = content_id(row)
        self.rows.append(row)
        return True

    def records(self, protocol_id: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["protocol_id"] == protocol_id]

    def slot_attempt(self, protocol_id: str, scheduled_at: datetime) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in self.records(protocol_id)
                if row["record_type"] == "slot_attempt"
                and row["scheduled_at"] == _iso(scheduled_at)
            ),
            None,
        )

    def attempt_records(self, protocol_id: str, attempt_id: str) -> list[dict[str, Any]]:
        return [row for row in self.records(protocol_id) if row.get("attempt_id") == attempt_id]

    def resolution_candidates(self, protocol_id: str) -> Iterator[tuple[dict[str, Any], set[str]]]:
        rows = self.records(protocol_id)
        captures = {
            row["market_id"]: row
            for row in rows
            if row["record_type"] == "market_capture"
            and row["status"] == "valid"
            and row["observation"] is not None
        }
        for market_id in sorted(captures):
            yield (
                captures[market_id],
                {
                    row["result"]
                    for row in rows
                    if row["record_type"] == "resolution" and row["market_id"] == market_id
                },
            )


def _record(protocol: dict[str, Any], kind: str, **values: Any) -> dict[str, Any]:
    return {
        "schema": "pms-forward-evidence-v1",
        "record_type": kind,
        "protocol_id": protocol["protocol_id"],
        **values,
    }


def _candidate(market: KalshiMarket, slot: datetime, at: datetime) -> dict[str, Any]:
    contract: TerminalRangeContract | None = None
    reason = None
    try:
        parsed = market.price_contract()
        if not isinstance(parsed, TerminalRangeContract) or parsed.settlement_window_seconds != 60:
            raise ValueError("not an explicit 60-second arithmetic terminal range")
        contract = parsed
        if (
            market.normalized_series_ticker != "KXBTC"
            or market.market_type != "binary"
            or market.notional_value_dollars != 1
        ):
            raise ValueError("not a KXBTC binary one-dollar contract")
        if any(
            clock is not None and clock > at
            for clock in (market.created_time, market.updated_time, market.open_time)
        ):
            raise ValueError("market metadata is not available at selection")
        horizon = min(market.close_time, market.observation_end_at)
        if not slot + timedelta(seconds=300) < horizon <= slot + timedelta(seconds=3600):
            raise ValueError("contract outside the scheduled-slot expiry window")
    except ValueError as exc:
        reason = _error(exc)
    # Status, result, quotes, and later performance intentionally do not enter this rule.
    return {
        "event_id": market.event_ticker,
        "market_id": market.ticker,
        "market": market.model_dump(mode="json"),
        "contract": None if contract is None else contract.model_dump(mode="json"),
        "eligible": reason is None,
        "exclusion_reason": reason,
    }


def _choose(candidates: list[dict[str, Any]], spot: float) -> tuple[str | None, list[str]]:
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return None, []
    event = min(
        eligible,
        key=lambda row: (
            KalshiMarket.model_validate(row["market"]).observation_end_at,
            row["event_id"],
        ),
    )["event_id"]
    rows = sorted(
        (row for row in eligible if row["event_id"] == event),
        key=lambda row: (
            abs((row["contract"]["lower_bound"] + row["contract"]["upper_bound"]) / 2 - spot),
            row["market_id"],
        ),
    )
    return str(event), [row["market_id"] for row in rows[:3]]


async def _discover(sources: _Sources, slot: datetime, candidates: list[dict[str, Any]]) -> None:
    cursor: str | None = None
    seen: set[str] = set()
    identities: set[str] = set()
    for _ in range(_MAX_PAGES):
        page = await _bounded(
            sources.kalshi.list_markets(series_ticker="KXBTC", limit=1000, cursor=cursor)
        )
        for market in page.markets:
            candidates.append(_candidate(market, slot, _now()))
            if market.ticker in identities:
                raise ValueError("duplicate market identity in discovery pagination")
            identities.add(market.ticker)
        if not page.cursor:
            return
        if page.cursor in seen:
            raise ValueError("incomplete discovery: repeated pagination cursor")
        seen.add(page.cursor)
        cursor = page.cursor
    raise ValueError("incomplete discovery: page bound exceeded")


async def _fresh_spot(sources: _Sources, repo: SQLiteRepository) -> SpotCandle:
    end = _now().replace(second=0, microsecond=0)
    candles = await _bounded(
        sources.coinbase.get_candles(
            "BTC-USD", start_at=end - timedelta(minutes=3), end_at=end, interval_seconds=60
        )
    )
    if not candles:
        raise ValueError("fresh Coinbase minute spot unavailable")
    latest = max(candle.end_at for candle in candles)
    choices = [candle for candle in candles if candle.end_at == latest]
    if len({content_id(candle.model_dump(mode="json")) for candle in choices}) != 1:
        raise ValueError("conflicting fresh spot revisions")
    spot = choices[0]
    if not 0 <= (_now() - spot.end_at).total_seconds() <= 120:
        raise ValueError("fresh Coinbase minute spot is stale or future")
    if spot.retrieved_at > _now():
        raise ValueError("fresh Coinbase minute spot receipt is from the future")
    repo.save_research_data(spot_candles=candles)
    with repo._connect() as connection:
        save_input_object(connection, spot.model_dump(mode="json"))
    return spot


async def _refresh_history(
    sources: _Sources, repo: SQLiteRepository, history_path: Path
) -> list[str]:
    now = _now()
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(seconds=_WINDOW + 3600)
    if history_path != repo.database_path and history_path.exists():
        history = _ReadOnlyResearch(history_path).effective_spot_candles(
            symbol="BTC", start_at=start, end_at=end, interval_seconds=3600, known_as_of=now
        )
        repo.save_research_data(spot_candles=list(history))
    known = repo.effective_spot_candles(
        symbol="BTC", start_at=start, end_at=end, interval_seconds=3600, known_as_of=now
    )
    present = {candle.start_at for candle in known}
    missing = [start + timedelta(hours=i) for i in range(721)]
    missing = [clock for clock in missing if clock not in present]
    # Fetch only contiguous missing spans, not a new 30-day download on each slot.
    spans: list[tuple[datetime, datetime]] = []
    for clock in missing:
        if spans and spans[-1][1] == clock:
            spans[-1] = (spans[-1][0], clock + timedelta(hours=1))
        else:
            spans.append((clock, clock + timedelta(hours=1)))
    for lower, upper in spans:
        rows = await _bounded(
            sources.coinbase.get_candles(
                "BTC-USD", start_at=lower, end_at=upper, interval_seconds=3600
            ),
            45,
        )
        repo.save_research_data(spot_candles=rows)

    async def volatility() -> None:
        rows = await _bounded(
            sources.deribit.get_dvol_history(
                "BTC", start_at=end - timedelta(hours=2), end_at=now, resolution_seconds=3600
            )
        )
        repo.save_research_data(volatility_observations=rows)

    async def funding() -> None:
        rows = await _bounded(
            sources.deribit.get_funding_history(
                "BTC-PERPETUAL", start_at=now - timedelta(hours=2), end_at=now
            )
        )
        repo.save_research_data(funding_observations=rows)

    async def derivatives() -> None:
        row = await _bounded(sources.deribit.get_derivatives_snapshot("BTC-PERPETUAL"))
        repo.save_research_data(derivatives_snapshots=[row])

    optional = await asyncio.gather(volatility(), funding(), derivatives(), return_exceptions=True)
    return [
        f"{name} unavailable: {_error(result)}"
        for name, result in zip(("volatility", "funding", "derivatives"), optional, strict=True)
        if isinstance(result, BaseException)
    ]


async def _fees(sources: _Sources, event_id: str) -> dict[str, Any]:
    """Replay effective revisions, then apply authoritative current event fields.

    Explicit null overrides restore the effective series value; absent fields
    leave the revision-derived policy unchanged.
    """
    series, event, series_changes = await asyncio.gather(
        _bounded(sources.kalshi.payload("/series/KXBTC")),
        _bounded(sources.kalshi.payload(f"/events/{event_id}")),
        _bounded(sources.kalshi.get_series_fee_changes("KXBTC")),
    )
    event_changes: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(_MAX_PAGES):
        page = await _bounded(sources.kalshi.get_event_fee_changes(event_id, cursor=cursor))
        event_changes.extend(item.model_dump(mode="json") for item in page.event_fee_changes)
        if not page.cursor:
            break
        if page.cursor in seen:
            raise ValueError("incomplete event fee pagination")
        seen.add(page.cursor)
        cursor = page.cursor
    else:
        raise ValueError("event fee pagination exceeded bound")
    received = _now()
    base = series["series"]
    if base.get("ticker") != "KXBTC" or event["event"].get("event_ticker") != event_id:
        raise ValueError("fee policy response identity mismatch")
    fee_type = base.get("fee_type")
    multiplier = base.get("fee_multiplier")
    for change in sorted(series_changes, key=lambda value: value.scheduled_ts):
        if change.series_ticker != "KXBTC":
            raise ValueError("series fee change identity mismatch")
        if change.scheduled_ts <= received:
            fee_type, multiplier = change.fee_type, change.fee_multiplier
    series_fee_type, series_multiplier = fee_type, multiplier
    for change in sorted(
        event_changes, key=lambda value: datetime.fromisoformat(value["scheduled_ts"])
    ):
        if change["event_ticker"] != event_id or change["series_ticker"] != "KXBTC":
            raise ValueError("event fee change identity mismatch")
        if datetime.fromisoformat(change["scheduled_ts"]) <= received:
            fee_type = (
                series_fee_type
                if change["fee_type_override"] is None
                else change["fee_type_override"]
            )
            multiplier = (
                series_multiplier
                if change["fee_multiplier_override"] is None
                else change["fee_multiplier_override"]
            )
    current_event = event["event"]
    if "fee_type_override" in current_event:
        fee_type = (
            series_fee_type
            if current_event["fee_type_override"] is None
            else current_event["fee_type_override"]
        )
    if "fee_multiplier_override" in current_event:
        multiplier = (
            series_multiplier
            if current_event["fee_multiplier_override"] is None
            else current_event["fee_multiplier_override"]
        )
    return {
        "series_id": "KXBTC",
        "fee_type": str(fee_type),
        "fee_multiplier": float(multiplier),
        "received_at": _iso(received),
        "raw_payload": {
            "series": series,
            "event": event,
            "series_fee_changes": [item.model_dump(mode="json") for item in series_changes],
            "event_fee_changes": event_changes,
        },
    }


def _observation(
    *,
    market: KalshiMarket,
    raw: dict[str, Any],
    decision: datetime,
    repo: SQLiteRepository,
    engine: CryptoThresholdEngine,
    warnings: list[str],
) -> dict[str, Any]:
    context = repo.research_context_as_of(
        symbol="BTC",
        as_of=decision,
        event_ticker=market.event_ticker,
        interval_seconds=3600,
        realized_window_seconds=_WINDOW,
        spot_interval_seconds=60,
        spot_max_age_seconds=120,
    )
    if warnings:
        context = ResearchContext.model_validate(
            {**context.model_dump(mode="json"), "warnings": [*context.warnings, *warnings]}
        )
    context.validate_at(decision, maximum_spot_age_seconds=120)
    snapshot = to_market_snapshot(market, KalshiOrderBook.model_validate(raw), observed_at=decision)
    if snapshot.yes_ask is None or snapshot.yes_ask_size is None or snapshot.yes_ask_size < 1:
        raise ValueError("probability baseline requires an executable YES ask of size >=1")
    if min(market.close_time, market.observation_end_at) < decision + timedelta(seconds=300):
        raise ValueError("decision violates the minimum entry horizon")
    contract = market.evaluation_contract(decision)
    crypto = context.to_crypto_snapshot(strike_price=engine.reference_price(contract))
    forecast, opportunity = engine.evaluate(snapshot, crypto, contract)
    inputs = forecast.input_manifest
    input_id = content_id(inputs)
    run_id = str(uuid4())
    manifest = {
        "schema": "pms-run-manifest-v2",
        "kind": "forward-evaluation",
        "run_id": run_id,
        "recorded_at": _iso(decision),
        "code": code_identity(),
        "configuration": {
            "recipe_id": forecast.recipe_id,
            "engine": engine.config.model_dump(mode="json"),
        },
        "inputs": {"input_id": input_id},
    }
    # Full immutable input, context and run payloads live in the outer compact record;
    # this does not depend on retained ordinary WATCH rows or pollute the old ledger.
    return {
        "observation_id": str(forecast.forecast_id),
        "event_id": market.event_ticker,
        "market_id": market.ticker,
        "observed_at": _iso(decision),
        "opportunity": opportunity.model_dump(mode="json"),
        "inputs": inputs,
        "run_id": run_id,
        "run_manifest": manifest,
        "run_manifest_sha256": content_id(manifest),
        "input_id": input_id,
        "research_context": context.model_dump(mode="json"),
        "resolution": None,
    }


async def _execution(
    *,
    sink: _Sink,
    sources: _Sources,
    protocol: dict[str, Any],
    attempt: dict[str, Any],
    market: KalshiMarket,
    observation: dict[str, Any],
    scenario: dict[str, Any],
) -> None:
    decision = datetime.fromisoformat(observation["observed_at"])
    target = decision + timedelta(seconds=scenario["latency_seconds"])
    while _now() < target:
        await asyncio.sleep((target - _now()).total_seconds())
    requested = _now()
    received: datetime | None = None
    raw = None
    snapshot = None
    error = None
    try:
        raw = await _bounded(sources.kalshi.raw_book(market.ticker))
        received = _now()
        if received > target + timedelta(seconds=120):
            raise ValueError("execution receipt missed target+120s")
        if received > min(market.close_time, market.observation_end_at) - timedelta(seconds=300):
            raise ValueError("execution receipt violates minimum entry horizon")
        snapshot = to_market_snapshot(
            market, KalshiOrderBook.model_validate(raw), observed_at=received
        ).model_dump(mode="json")
    except Exception as exc:
        error = _error(exc)
    sink.append(
        _record(
            protocol,
            "execution_capture",
            slot_id=attempt["slot_id"],
            attempt_id=attempt["attempt_id"],
            market_id=market.ticker,
            event_id=market.event_ticker,
            observation_id=observation["observation_id"],
            scenario=scenario["name"],
            target_at=_iso(target),
            requested_at=_iso(requested),
            received_at=None if received is None else _iso(received),
            status="valid" if error is None else "failed",
            error=error,
            raw_book=raw,
            raw_book_id=None if raw is None else content_id(raw),
            snapshot=snapshot,
        )
    )


async def _capture_market(
    *,
    sink: _Sink,
    sources: _Sources,
    protocol: dict[str, Any],
    attempt: dict[str, Any],
    selection: dict[str, Any],
    market: KalshiMarket,
    repo: SQLiteRepository,
    fee: dict[str, Any] | None,
    source_error: str | None,
    warnings: list[str],
    execution_tasks: list[asyncio.Task[None]],
) -> bool:
    requested = _now()
    row = _record(
        protocol,
        "market_capture",
        slot_id=attempt["slot_id"],
        attempt_id=attempt["attempt_id"],
        scheduled_at=attempt["scheduled_at"],
        selection_inputs_id=selection["selection_inputs_id"],
        market_id=market.ticker,
        event_id=market.event_ticker,
        status="failed",
        error=None,
        book_requested_at=_iso(requested),
        book_received_at=None,
        raw_book=None,
        raw_book_id=None,
        decision_at=None,
        forecast_persisted_at=None,
        observation=None,
        fee_context=fee,
    )
    try:
        raw = await _bounded(sources.kalshi.raw_book(market.ticker))
        received = _now()
        row.update(book_received_at=_iso(received), raw_book=raw, raw_book_id=content_id(raw))
        if source_error:
            raise ValueError(source_error)
        if fee is None or fee["fee_type"] != "quadratic" or fee["fee_multiplier"] != 1.0:
            raise ValueError("public fee policy incompatible with frozen quadratic 0.07 policy")
        decision = _now()
        row["decision_at"] = _iso(decision)
        if decision > datetime.fromisoformat(attempt["scheduled_at"]) + timedelta(seconds=120):
            raise ValueError("decision missed scheduled slot+120s")
        if not 0 <= (decision - datetime.fromisoformat(fee["received_at"])).total_seconds() <= 120:
            raise ValueError("fee source receipt is stale or future")
        observation = _observation(
            market=market,
            raw=raw,
            decision=decision,
            repo=repo,
            engine=CryptoThresholdEngine(EngineConfig.model_validate(protocol["engine_config"])),
            warnings=warnings,
        )
        row.update(status="valid", observation=observation)
        if _now() > decision + timedelta(seconds=5):
            raise ValueError("forecast generation exceeded persistence allowance")
        if _now() >= market.observation_end_at - timedelta(seconds=60):
            raise ValueError("forecast persistence reached the observation window")
        sink.append(row)
    except Exception as exc:
        row.update(status="failed", error=_error(exc))
        sink.append(row)
        return False
    # Tasks wait independently, so a slow market never serializes the other markets' targets.
    for scenario in protocol["cost_scenarios"]:
        execution_tasks.append(
            asyncio.create_task(
                _execution(
                    sink=sink,
                    sources=sources,
                    protocol=protocol,
                    attempt=attempt,
                    market=market,
                    observation=observation,
                    scenario=scenario,
                )
            )
        )
    return True


async def _collect(
    *,
    sink: _Sink,
    protocol: dict[str, Any],
    attempt: dict[str, Any],
    repo: SQLiteRepository,
    history_path: Path,
) -> dict[str, Any]:
    slot = datetime.fromisoformat(attempt["scheduled_at"])
    candidates: list[dict[str, Any]] = []
    discovery_started = _now()
    discovery_received = discovery_started
    discovery_complete = False
    spot: SpotCandle | None = None
    error = None
    history_error = None
    warnings: list[str] = []
    selected: list[str] = []
    event_id = None
    captured: list[str] = []
    async with _Sources() as sources:

        async def discover() -> None:
            nonlocal discovery_received, discovery_complete
            try:
                await _bounded(_discover(sources, slot, candidates), 60)
                discovery_complete = True
            finally:
                discovery_received = _now()

        outcomes = await asyncio.gather(
            discover(),
            _fresh_spot(sources, repo),
            _bounded(_refresh_history(sources, repo, history_path), 80),
            return_exceptions=True,
        )
        errors = [_error(item) for item in outcomes[:2] if isinstance(item, BaseException)]
        if not isinstance(outcomes[1], BaseException):
            spot = outcomes[1]
        if isinstance(outcomes[2], BaseException):
            history_error = _error(outcomes[2])
        else:
            warnings = outcomes[2]
        selection_at = _now()
        candidates = [
            _candidate(KalshiMarket.model_validate(row["market"]), slot, selection_at)
            for row in candidates
        ]
        spot_payload = None
        if spot is not None:
            spot_payload = {
                "price": float(spot.close),
                "observed_at": _iso(spot.end_at),
                "received_at": _iso(spot.retrieved_at),
                "source": spot.provider,
                "input_id": content_id(spot.model_dump(mode="json")),
            }
            if not 0 <= (selection_at - spot.end_at).total_seconds() <= 120:
                errors.append("selection spot became stale or future")
        if discovery_complete and spot is not None and not errors:
            event_id, selected = _choose(candidates, float(spot.close))
        error = "; ".join(errors) or None
        identity = {
            "discovered_candidates": candidates,
            "spot": spot_payload,
            "selected_event_id": event_id,
            "selected_market_ids": selected,
        }
        selection = _record(
            protocol,
            "selection",
            slot_id=attempt["slot_id"],
            attempt_id=attempt["attempt_id"],
            scheduled_at=attempt["scheduled_at"],
            discovery_started_at=_iso(discovery_started),
            discovery_received_at=_iso(discovery_received),
            selection_at=_iso(selection_at),
            discovery_complete=discovery_complete,
            **identity,
            selection_inputs_id=content_id(identity),
            error=error,
        )
        sink.append(selection)
        if selected:
            fee = None
            try:
                fee = await _bounded(_fees(sources, str(event_id)), 25)
            except Exception as exc:
                history_error = "; ".join(filter(None, (history_error, _error(exc))))
            markets = {
                row["market_id"]: KalshiMarket.model_validate(row["market"]) for row in candidates
            }
            execution_tasks: list[asyncio.Task[None]] = []
            valid = await asyncio.gather(
                *(
                    _capture_market(
                        sink=sink,
                        sources=sources,
                        protocol=protocol,
                        attempt=attempt,
                        selection=selection,
                        market=markets[market_id],
                        repo=repo,
                        fee=fee,
                        source_error=history_error,
                        warnings=warnings,
                        execution_tasks=execution_tasks,
                    )
                    for market_id in selected
                )
            )
            captured = [
                market_id for market_id, success in zip(selected, valid, strict=True) if success
            ]
            await asyncio.gather(*execution_tasks)
    status = "failed"
    if not error and discovery_complete:
        status = (
            "no-market"
            if not selected
            else (
                "valid" if len(captured) == len(selected) else "partial" if captured else "failed"
            )
        )
    result = _record(
        protocol,
        "slot_result",
        slot_id=attempt["slot_id"],
        attempt_id=attempt["attempt_id"],
        scheduled_at=attempt["scheduled_at"],
        completed_at=_iso(_now()),
        status=status,
        selected_market_ids=selected,
        captured_market_ids=captured,
        error=error
        or history_error
        or ("selected capture failures" if selected and len(captured) != len(selected) else None),
    )
    sink.append(result)
    return result


async def capture_forward_slot(
    *,
    store: ForwardStudyStore,
    protocol: dict[str, Any],
    scheduled_at: datetime,
    settings: Settings,
) -> dict[str, Any]:
    validate_forward_protocol(protocol)
    if scheduled_at.tzinfo is None or scheduled_at.utcoffset() is None:
        raise ValueError("scheduled slot requires an aware UTC clock")
    scheduled_at = scheduled_at.astimezone(UTC)
    if scheduled_at not in scheduled_slots(
        protocol, scheduled_at, scheduled_at + timedelta(seconds=1)
    ):
        raise ValueError("scheduled_at is not a frozen study slot")
    stamp = _iso(scheduled_at)
    if store.slot_attempt(protocol["protocol_id"], scheduled_at) is not None:
        return {"status": "already-attempted", "scheduled_at": stamp, "eligible": False}
    started = _now()
    if started < scheduled_at:
        raise ValueError("cannot attempt a future slot")
    attempt = _record(
        protocol,
        "slot_attempt",
        slot_id=content_id({"protocol_id": protocol["protocol_id"], "scheduled_at": stamp}),
        scheduled_at=stamp,
        attempt_id=str(uuid4()),
        started_at=_iso(started),
    )
    try:
        store.append(attempt)
    except ValueError:
        if store.slot_attempt(protocol["protocol_id"], scheduled_at) is not None:
            return {"status": "already-attempted", "scheduled_at": stamp, "eligible": False}
        raise
    if started > scheduled_at + timedelta(seconds=60):
        result = _record(
            protocol,
            "slot_result",
            slot_id=attempt["slot_id"],
            attempt_id=attempt["attempt_id"],
            scheduled_at=stamp,
            completed_at=_iso(_now()),
            status="late",
            selected_market_ids=[],
            captured_market_ids=[],
            error="first attempt started after scheduled slot+60s",
        )
        store.append(result)
    else:
        try:
            repo = SQLiteRepository(store.path)
            repo.initialize()
            result = await _collect(
                sink=store,
                protocol=protocol,
                attempt=attempt,
                repo=repo,
                history_path=settings.database_path,
            )
        except Exception as exc:
            retained = store.attempt_records(protocol["protocol_id"], attempt["attempt_id"])
            selections = [row for row in retained if row["record_type"] == "selection"]
            selected = selections[0]["selected_market_ids"] if selections else []
            captured = [
                row["market_id"]
                for row in retained
                if row["record_type"] == "market_capture" and row["status"] == "valid"
            ]
            status = "failed"
            if captured:
                status = "valid" if set(captured) == set(selected) else "partial"
            elif (
                selections
                and selections[0]["discovery_complete"]
                and not selected
                and selections[0]["error"] is None
            ):
                status = "no-market"
            result = _record(
                protocol,
                "slot_result",
                slot_id=attempt["slot_id"],
                attempt_id=attempt["attempt_id"],
                scheduled_at=stamp,
                completed_at=_iso(_now()),
                status=status,
                selected_market_ids=selected,
                captured_market_ids=captured,
                error=_error(exc),
            )
            store.append(result)
    return next(
        row
        for row in store.attempt_records(protocol["protocol_id"], attempt["attempt_id"])
        if row["record_type"] == "slot_result"
    )


async def resolve_forward_outcomes(
    *,
    store: ForwardStudyStore,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    validate_forward_protocol(protocol)
    counts: dict[str, Any] = {
        "status": "valid",
        "queried": 0,
        "resolved": 0,
        "unavailable": 0,
        "failed": 0,
    }
    if _now() >= datetime.fromisoformat(protocol["report_as_of"]):
        return {**counts, "status": "closed"}
    async with _Sources() as sources:
        for row, known in store.resolution_candidates(protocol["protocol_id"]):
            market = row["observation"]["opportunity"]["market"]
            end = datetime.fromisoformat(market["observation_end_at"])
            if end > _now():
                continue
            market_id, event_id = row["market_id"], row["event_id"]
            counts["queried"] += 1
            try:
                payload = await _bounded(sources.kalshi.resolution_payload(market_id))
                received = _now()
                if received >= datetime.fromisoformat(protocol["report_as_of"]):
                    counts["unavailable"] += 1
                    break
                response_market = KalshiMarket.model_validate(payload["market"])
                if response_market.ticker != market_id or response_market.event_ticker != event_id:
                    raise ValueError("resolution response identity mismatch")
                if response_market.result not in {"yes", "no"}:
                    counts["unavailable"] += 1
                    continue
                if response_market.status not in {"closed", "settled", "finalized"}:
                    raise ValueError("outcome response is not a closed market")
                if any(
                    clock is not None and clock > received
                    for clock in (response_market.settlement_ts, response_market.updated_time)
                ):
                    raise ValueError("resolution metadata is from the future")
                if response_market.result not in known:
                    store.append(
                        _record(
                            protocol,
                            "resolution",
                            market_id=market_id,
                            event_id=event_id,
                            result=response_market.result,
                            observed_at=_iso(received),
                            received_at=_iso(received),
                            raw_payload=payload,
                        )
                    )
                counts["resolved"] += 1
            except Exception:
                counts["failed"] += 1
    if counts["failed"]:
        counts["status"] = "partial" if counts["failed"] < counts["queried"] else "failed"
    return counts


async def probe_forward_capture(
    *,
    protocol: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    validate_forward_protocol(protocol)
    started = _now()
    slot = started.replace(second=0, microsecond=0)
    sink = _DiagnosticSink()
    attempt = _record(
        protocol,
        "slot_attempt",
        slot_id=content_id({"diagnostic": True, "at": _iso(slot)}),
        scheduled_at=_iso(slot),
        attempt_id=str(uuid4()),
        started_at=_iso(started),
    )
    sink.append(attempt)
    try:
        with TemporaryDirectory(prefix="pms-forward-diagnostic-") as directory:
            repo = SQLiteRepository(Path(directory) / "diagnostic.db")
            repo.initialize()
            result = await _collect(
                sink=sink,
                protocol=protocol,
                attempt=attempt,
                repo=repo,
                history_path=settings.database_path,
            )
    except Exception as exc:
        result = {"status": "failed", "error": _error(exc)}
    rows = sink.records(protocol["protocol_id"])

    def count(kind: str, status: str) -> int:
        return sum(row["record_type"] == kind and row["status"] == status for row in rows)

    def clocks(kind: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
        return [{key: row[key] for key in fields} for row in rows if row["record_type"] == kind]

    return {
        "diagnostic": True,
        "eligible": False,
        "status": result["status"],
        "eligibility_reasons": [
            "diagnostic-only: no study registration or eligible evidence writes"
        ],
        "started_at": _iso(started),
        "completed_at": _iso(_now()),
        "scheduled_at": _iso(slot),
        "selected_markets": len(result.get("selected_market_ids", [])),
        "valid_captures": count("market_capture", "valid"),
        "failed_captures": count("market_capture", "failed"),
        "valid_executions": count("execution_capture", "valid"),
        "failed_executions": count("execution_capture", "failed"),
        "error": result.get("error"),
        "capture_clocks": clocks(
            "market_capture",
            (
                "market_id",
                "book_requested_at",
                "book_received_at",
                "decision_at",
                "forecast_persisted_at",
                "status",
                "error",
            ),
        ),
        "execution_clocks": clocks(
            "execution_capture",
            (
                "market_id",
                "scenario",
                "target_at",
                "requested_at",
                "received_at",
                "status",
                "error",
            ),
        ),
    }
