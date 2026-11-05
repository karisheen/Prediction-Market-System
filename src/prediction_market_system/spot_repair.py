"""Explicit, bounded reconstruction from freshly verified public Coinbase trades.

Audit receipts guide requests; they never authorize historical availability or supply
trusted candle values. Routine candle ingestion deliberately does not call this module.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import re
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from prediction_market_system.research import SpotCandle
from prediction_market_system.sources.coinbase import COINBASE_MARKET_API, CoinbaseDataError
from prediction_market_system.transport import DEFAULT_TRANSIENT_RETRIES, get_with_transient_retry

_EXCHANGE_API = "https://api.exchange.coinbase.com"
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_TRADES = 100_000
_MAX_SPANS = 16
_MAX_MINUTES = 350
_REQUEST_INTERVAL = 0.35
_FIELDS = ("open", "high", "low", "close", "volume")


class SpotRepairError(CoinbaseDataError):
    """The bounded repair cannot be independently verified; nothing may be applied."""


@dataclass(frozen=True)
class SpotCandleRepair:
    original: SpotCandle | None
    derived: SpotCandle


@dataclass(frozen=True)
class VerifiedSpotRepairBatch:
    repairs: tuple[SpotCandleRepair, ...]
    native_candles: tuple[SpotCandle, ...]
    proof: dict[str, Any]
    verified_at: datetime


@dataclass(frozen=True)
class _Trade:
    trade_id: int
    time: datetime
    price: Decimal
    size: Decimal
    side: str


def candle_ohlcv(candle: SpotCandle) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    return candle.open, candle.high, candle.low, candle.close, candle.volume


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]{1,19}", str(value)):
        raise SpotRepairError("Expected a bounded non-negative integer")
    return int(value)


def _decimal(value: Any) -> Decimal:
    if not isinstance(value, (str, int, Decimal)) or isinstance(value, bool):
        raise SpotRepairError("Price and size must be exact decimal values, not floats")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise SpotRepairError("Invalid decimal value") from exc
    exponent = result.as_tuple().exponent
    if (
        not result.is_finite()
        or not isinstance(exponent, int)
        or len(result.as_tuple().digits) > 60
        or abs(exponent) > 30
    ):
        raise SpotRepairError("Non-finite or unbounded decimal value")
    return result


def _timestamp(value: Any, *, minute: bool = False) -> datetime:
    if not isinstance(value, str):
        raise SpotRepairError("Expected an explicit UTC timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SpotRepairError("Invalid UTC timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise SpotRepairError("Timestamps must be explicitly UTC")
    if minute and (result.second or result.microsecond):
        raise SpotRepairError("Candle windows must be minute aligned")
    return result.astimezone(UTC)


def _trade(raw: Any) -> _Trade:
    if not isinstance(raw, dict):
        raise SpotRepairError("Trade must be an object")
    result = _Trade(
        _integer(raw["trade_id"]),
        _timestamp(raw["time"]),
        _decimal(raw["price"]),
        _decimal(raw["size"]),
        raw["side"],
    )
    if (
        result.trade_id <= 0
        or result.price <= 0
        or result.size <= 0
        or result.side not in ("buy", "sell")
    ):
        raise SpotRepairError("Trade has invalid ID, price, size, or side")
    return result


def _continuous(trades: list[_Trade]) -> None:
    for previous, current in zip(trades, trades[1:], strict=False):
        if current.trade_id == previous.trade_id:
            raise SpotRepairError("Duplicate or conflicting trade ID")
        if current.trade_id != previous.trade_id + 1:
            raise SpotRepairError("Trade ID continuity gap or order reversal")
        if current.time < previous.time:
            raise SpotRepairError("Trade timestamp order reversal")


def _ohlcv(trades: list[_Trade]) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    if not trades:
        raise SpotRepairError("No trades: zero-filled candle synthesis is forbidden")
    # Continuity validation establishes timestamp order and ID ordering of ties.
    with localcontext() as context:
        context.prec = 160
        return (
            trades[0].price,
            max(trade.price for trade in trades),
            min(trade.price for trade in trades),
            trades[-1].price,
            sum((trade.size for trade in trades), Decimal(0)),
        )


def _values(raw: Any) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    if not isinstance(raw, dict):
        raise SpotRepairError("OHLCV must be an object")
    return tuple(_decimal(raw[field]) for field in _FIELDS)  # type: ignore[return-value]


def _strings(values: tuple[Decimal, ...] | None) -> dict[str, str] | None:
    return None if values is None else dict(zip(_FIELDS, map(str, values), strict=True))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpotRepairError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise SpotRepairError(f"Non-finite JSON constant: {value}")


def _load_receipt(path: Path) -> tuple[dict[str, Any], str]:
    if not (path.name.endswith(".json") or path.name.endswith(".json.gz")):
        raise SpotRepairError("Receipt must be a .json or .json.gz file")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(_MAX_RECEIPT_BYTES + 1)
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise SpotRepairError("Receipt exceeds the bounded decompressed size")
    payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise SpotRepairError("Receipt must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _guide(
    payload: dict[str, Any], product: str
) -> tuple[dict[str, list[_Trade]], list[dict[str, Any]]]:
    provenance = payload["reconstruction_provenance"]
    if (
        not isinstance(provenance, dict)
        or payload.get("schema_version") != 1
        or provenance.get("product") != product
        or provenance.get("interval_seconds") != 60
        or provenance.get("not_provider_raw_candle") is not True
        or provenance.get("candidate_provider") != "coinbase_exchange_public_trades_derived"
    ):
        raise SpotRepairError("Missing or incompatible reconstruction provenance")
    reconstruction = payload["coinbase_independent_reconstruction"]
    raw_spans = reconstruction["raw_trade_spans"]
    bars = reconstruction["bars"]
    if not isinstance(raw_spans, dict) or not 1 <= len(raw_spans) <= _MAX_SPANS:
        raise SpotRepairError("Receipt has no bounded trade spans")
    if not isinstance(bars, list) or not 3 <= len(bars) <= _MAX_MINUTES:
        raise SpotRepairError("Receipt must name bounded minute slots and surrounding controls")
    spans: dict[str, list[_Trade]] = {}
    seen_ids: set[int] = set()
    for name, raw_trades in raw_spans.items():
        if not isinstance(raw_trades, list) or not 3 <= len(raw_trades) <= _MAX_TRADES:
            raise SpotRepairError("Invalid bounded trade span")
        trades = [_trade(raw) for raw in raw_trades]
        _continuous(trades)
        if trades[-1].time - trades[0].time > timedelta(minutes=_MAX_MINUTES):
            raise SpotRepairError("Trade span exceeds bounded time window")
        ids = {trade.trade_id for trade in trades}
        if seen_ids.intersection(ids):
            raise SpotRepairError("Overlapping receipt trade spans")
        seen_ids.update(ids)
        spans[name] = trades
    if len(seen_ids) > _MAX_TRADES:
        raise SpotRepairError("Receipt exceeds bounded total trade count")
    starts: set[datetime] = set()
    affected: set[datetime] = set()
    for bar in bars:
        start, end = _timestamp(bar["start"], minute=True), _timestamp(bar["end"], minute=True)
        if end - start != timedelta(minutes=1) or start in starts or end > datetime.now(UTC):
            raise SpotRepairError("Duplicate, unfinished, or invalid minute window")
        starts.add(start)
        if bar["full_ohlcv_matches_official"] is not True:
            affected.add(start)
    scope = provenance["scope"]
    partials = provenance["additional_nonmissing_partial_minutes"]
    if not isinstance(scope, list) or not isinstance(partials, list):
        raise SpotRepairError("Missing explicit repair scope")
    declared = [_timestamp(value, minute=True) for value in scope + partials]
    if len(set(declared)) != len(declared) or set(declared) != affected or not affected:
        raise SpotRepairError("Receipt repair scope differs from named minute evidence")
    if _integer(provenance["target_count"]) != len(scope):
        raise SpotRepairError("Receipt target count differs from scope")
    if {_timestamp(bar["start"]) for bar in bars if bar["official"] is None} != {
        _timestamp(value) for value in scope
    }:
        raise SpotRepairError("Missing-native scope disagrees with recorded observations")
    # Receipt URLs are evidence only; never pass them to an HTTP client.
    requests = reconstruction["requests"]
    if not isinstance(requests, list) or not requests:
        raise SpotRepairError("Missing original HTTP provenance")
    allowed = {
        ("api.exchange.coinbase.com", f"/products/{product}/trades"),
        ("api.exchange.coinbase.com", f"/products/{product}/candles"),
        ("api.coinbase.com", f"/api/v3/brokerage/market/products/{product}/candles"),
    }
    for request in requests:
        url = urlsplit(request["url"])
        if (
            url.scheme != "https"
            or (url.netloc, url.path) not in allowed
            or url.fragment
            or request.get("status") != 200
            or not re.fullmatch(r"[0-9a-f]{64}", request.get("sha256", ""))
        ):
            raise SpotRepairError("Untrusted URL or missing HTTP provenance in receipt")
    return spans, sorted(bars, key=lambda bar: _timestamp(bar["start"]))


async def verify_coinbase_repair_receipt(
    receipt: Path,
    *,
    symbol: str = "BTC",
    client: httpx.AsyncClient | None = None,
) -> VerifiedSpotRepairBatch:
    """Refetch and verify an explicit receipt without opening or mutating storage."""
    if not re.fullmatch(r"[A-Z0-9]{2,12}", symbol):
        raise SpotRepairError("Symbol must be an uppercase Coinbase base asset (for example BTC)")
    product = f"{symbol}-USD"
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=20.0,
            trust_env=False,
            follow_redirects=False,
            headers={"User-Agent": "prediction-market-system/0.1.0"},
        )
    try:
        if (
            client.auth is not None
            or client.cookies
            or client.params
            or client.follow_redirects
            or any(
                name.lower() in ("authorization", "proxy-authorization", "cookie")
                or name.lower().startswith("cb-access-")
                for name in client.headers
            )
        ):
            raise SpotRepairError(
                "Repair requires an anonymous client with redirects disabled "
                "and no default parameters"
            )
        payload, receipt_hash = _load_receipt(Path(receipt))
        spans, bars = _guide(payload, product)
        return await _verify(client, spans, bars, product, receipt_hash)
    except SpotRepairError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OverflowError,
        RecursionError,
        OSError,
        EOFError,
        zlib.error,
        httpx.HTTPError,
    ) as exc:
        raise SpotRepairError(f"Coinbase repair verification failed: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()


async def _verify(
    client: httpx.AsyncClient,
    spans: dict[str, list[_Trade]],
    bars: list[dict[str, Any]],
    product: str,
    receipt_hash: str,
) -> VerifiedSpotRepairBatch:
    requests: list[dict[str, Any]] = []

    async def fetch(url: str, params: dict[str, str | int]) -> tuple[Any, datetime]:
        if requests:
            await asyncio.sleep(_REQUEST_INTERVAL)
        requested_at = datetime.now(UTC)
        response = await get_with_transient_retry(
            client, url, params=params, max_retries=DEFAULT_TRANSIENT_RETRIES
        )
        received_at = datetime.now(UTC)
        response.raise_for_status()
        body = json.loads(
            response.content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        requests.append(
            {
                "url": str(response.request.url),
                "requested_at": requested_at.isoformat(),
                "retrieved_at": received_at.isoformat(),
                "status": response.status_code,
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "headers": {
                    key: response.headers[key]
                    for key in ("date", "content-type", "cb-before", "cb-after")
                    if key in response.headers
                },
                "body": body,
            }
        )
        return body, received_at

    fresh_spans: dict[str, list[_Trade]] = {}
    audits: list[dict[str, Any]] = []
    for name, guide_trades in spans.items():
        first_id, last_id = guide_trades[0].trade_id, guide_trades[-1].trade_id
        cursor = last_id + 1
        descending: list[_Trade] = []
        while cursor > first_id:
            limit = min(1000, cursor - first_id)
            body, _ = await fetch(
                f"{_EXCHANGE_API}/products/{product}/trades", {"after": cursor, "limit": limit}
            )
            if not isinstance(body, list) or len(body) != limit:
                raise SpotRepairError("Fresh trade page is incomplete")
            page = [_trade(raw) for raw in body]
            if [trade.trade_id for trade in page] != list(
                range(cursor - 1, cursor - limit - 1, -1)
            ):
                raise SpotRepairError(
                    "Fresh trade page has duplicate, conflicting, gapped, or unordered IDs"
                )
            descending.extend(page)
            cursor -= limit
        fresh = list(reversed(descending))
        _continuous(fresh)
        if fresh != guide_trades:
            raise SpotRepairError(
                "Fresh trades disagree with receipt: altered guide or changed trade history"
            )
        fresh_spans[name] = fresh
        audits.append(
            {
                "span": name,
                "first_id": first_id,
                "last_id": last_id,
                "trade_count": len(fresh),
                "contiguous_ids": True,
                "monotonic_utc_times": True,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in spans}
    for bar in bars:
        start, end = _timestamp(bar["start"]), _timestamp(bar["end"])
        covering = [
            name
            for name, trades in fresh_spans.items()
            if trades[0].time < start and trades[-1].time >= end
        ]
        if len(covering) != 1:
            raise SpotRepairError("Minute lacks unique complete before/after boundary witnesses")
        groups[covering[0]].append(bar)
    native_candles: list[SpotCandle] = []
    pending: list[tuple[SpotCandle | None, datetime, tuple[Decimal, ...], list[_Trade]]] = []
    comparisons: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for name, group in groups.items():
        if len(group) < 3:
            raise SpotRepairError(
                "Every span requires affected minutes and complete surrounding controls"
            )
        starts = [_timestamp(bar["start"]) for bar in group]
        if any(
            right - left != timedelta(minutes=1)
            for left, right in zip(starts, starts[1:], strict=False)
        ):
            raise SpotRepairError("All minute slots in a repair window must be explicitly named")
        if (
            group[0]["full_ohlcv_matches_official"] is not True
            or group[-1]["full_ohlcv_matches_official"] is not True
        ):
            raise SpotRepairError("Affected span lacks complete before/after controls")
        if all(bar["full_ohlcv_matches_official"] is True for bar in group):
            raise SpotRepairError("Trade span does not contain an affected minute")
        body, retrieved_at = await fetch(
            f"{COINBASE_MARKET_API}/products/{product}/candles",
            {
                "start": int(starts[0].timestamp()),
                "end": int(starts[-1].timestamp()),
                "granularity": "ONE_MINUTE",
                "limit": 350,
            },
        )
        if not isinstance(body, dict) or not isinstance(body.get("candles"), list):
            raise SpotRepairError("Invalid native candle response")
        natives: dict[datetime, SpotCandle] = {}
        for raw in body["candles"]:
            native_start = datetime.fromtimestamp(_integer(raw["start"]), UTC)
            if native_start not in starts or native_start in natives:
                raise SpotRepairError("Unexpected or duplicate native minute")
            values = _values(raw)
            candle = SpotCandle(
                provider="coinbase",
                product_id=product,
                interval_seconds=60,
                start_at=native_start,
                end_at=native_start + timedelta(minutes=1),
                **dict(zip(_FIELDS, values, strict=True)),
                retrieved_at=retrieved_at,
                raw_payload=raw,
            )
            natives[native_start] = candle
            native_candles.append(candle)
        windows.append(
            {
                "span": name,
                "start": starts[0].isoformat(),
                "end": (starts[-1] + timedelta(minutes=1)).isoformat(),
                "native_retrieved_at": retrieved_at.isoformat(),
            }
        )
        trades = fresh_spans[name]
        for bar, start in zip(group, starts, strict=True):
            end = start + timedelta(minutes=1)
            selected = [trade for trade in trades if start <= trade.time < end]
            values = _ohlcv(selected)
            before = next((trade for trade in reversed(trades) if trade.time < start), None)
            after = next((trade for trade in trades if trade.time >= end), None)
            if (
                before is None
                or after is None
                or before.trade_id + 1 != selected[0].trade_id
                or selected[-1].trade_id + 1 != after.trade_id
            ):
                raise SpotRepairError(
                    "Minute lacks adjacent inclusive/exclusive boundary witnesses"
                )
            for field, trade in (
                ("before", before),
                ("after", after),
                ("first_trade", selected[0]),
                ("last_trade", selected[-1]),
            ):
                if _trade(bar[field]) != trade:
                    raise SpotRepairError(
                        "Receipt boundary or endpoint trade disagrees with fresh evidence"
                    )
            if _integer(bar["trade_count"]) != len(selected) or _values(bar["ohlcv"]) != values:
                raise SpotRepairError(
                    "Recorded candidate OHLCV or count disagrees with fresh Decimal reconstruction"
                )
            recorded = None if bar["official"] is None else _values(bar["official"])
            native = natives.get(start)
            native_values = None if native is None else candle_ohlcv(native)
            control = bar["full_ohlcv_matches_official"] is True
            if control:
                if recorded != values or native_values != values or bar["exact_partial_matches"]:
                    raise SpotRepairError("Complete control mismatch")
                status = "control"
            else:
                expected_flag = None if recorded is None else False
                if bar["full_ohlcv_matches_official"] is not expected_flag:
                    raise SpotRepairError("Receipt partial/missing classification is inconsistent")
                if recorded == values:
                    raise SpotRepairError("Affected minute is mislabeled as a partial observation")
                matches = bar["exact_partial_matches"]
                if (
                    not isinstance(matches, list)
                    or (recorded is not None and not matches)
                    or (recorded is None and matches)
                ):
                    raise SpotRepairError("Missing independent partial-window evidence")
                for match in matches:
                    count = _integer(match["trade_count"])
                    if not 0 < count < len(selected) or match["direction"] not in (
                        "prefix",
                        "suffix",
                    ):
                        raise SpotRepairError("Invalid partial-window evidence")
                    partial = (
                        selected[:count] if match["direction"] == "prefix" else selected[-count:]
                    )
                    if _ohlcv(partial) != recorded:
                        raise SpotRepairError(
                            "Recorded partial candle is not explained by fresh trades"
                        )
                if native_values == values:
                    status = "already_corrected"
                elif native_values == recorded:
                    status = "repair"
                    pending.append((native, start, values, selected))
                else:
                    raise SpotRepairError(
                        "Native candle changed to an unknown revision; refusing repair"
                    )
            comparisons.append(
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "status": status,
                    "recorded_ohlcv": _strings(recorded),
                    "native_ohlcv": _strings(native_values),
                    "derived_ohlcv": _strings(values),
                    "native_retrieved_at": retrieved_at.isoformat(),
                    "trade_count": len(selected),
                    "first_trade_id": selected[0].trade_id,
                    "last_trade_id": selected[-1].trade_id,
                    "before_trade_id": before.trade_id,
                    "after_trade_id": after.trade_id,
                    "exact_partial_matches": bar["exact_partial_matches"],
                }
            )
    verified_at = datetime.now(UTC)
    proof: dict[str, Any] = {
        "schema_version": 1,
        "method": "coinbase_exchange_trades",
        "product": product,
        "interval_seconds": 60,
        "receipt_sha256": receipt_hash,
        "verified_at": verified_at.isoformat(),
        "historical_availability": False,
        "windows": windows,
        "requests": requests,
        "span_audits": audits,
        "comparisons": comparisons,
        "arithmetic": {
            "currency": "exact Decimal",
            "interval": "start <= time < end",
            "ordering": ["UTC timestamp ascending", "trade_id ascending for timestamp ties"],
        },
    }
    proof_digest = _digest(proof)
    proof["proof_digest"] = proof_digest
    repairs = tuple(
        SpotCandleRepair(
            original=native,
            derived=SpotCandle(
                provider="coinbase",
                product_id=product,
                interval_seconds=60,
                start_at=start,
                end_at=start + timedelta(minutes=1),
                **dict(zip(_FIELDS, values, strict=True)),
                retrieved_at=verified_at,
                raw_payload={
                    "method": "coinbase_exchange_trades",
                    "not_provider_raw_candle": True,
                    "proof_digest": proof_digest,
                    "first_trade_id": selected[0].trade_id,
                    "last_trade_id": selected[-1].trade_id,
                    "trade_count": len(selected),
                },
            ),
        )
        for native, start, values, selected in pending
    )
    return VerifiedSpotRepairBatch(
        repairs,
        tuple(sorted(native_candles, key=lambda candle: candle.start_at)),
        proof,
        verified_at,
    )
