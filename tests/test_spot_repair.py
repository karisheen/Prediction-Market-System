import copy
import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from prediction_market_system import spot_repair
from prediction_market_system.spot_repair import (
    SpotRepairError,
    candle_ohlcv,
    verify_coinbase_repair_receipt,
)

_FIELDS = ("open", "high", "low", "close", "volume")


def _values(rows: list[dict[str, Any]]) -> dict[str, str]:
    prices = [Decimal(row["price"]) for row in rows]
    return dict(
        zip(
            _FIELDS,
            map(
                str,
                (
                    prices[0],
                    max(prices),
                    min(prices),
                    prices[-1],
                    sum((Decimal(row["size"]) for row in rows), Decimal(0)),
                ),
            ),
            strict=True,
        )
    )


def _fixture(
    *, count: int = 2, day_offset: int = 0, id_offset: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day_offset)
    rows = []
    for index in range(4 * count + 2):
        if index == 0:
            timestamp = start - timedelta(seconds=1)
        elif index == 4 * count + 1:
            timestamp = start + timedelta(minutes=4)
        else:
            # Equal timestamps deliberately exercise the trade-ID tie breaker.
            timestamp = start + timedelta(minutes=(index - 1) // count)
        rows.append(
            {
                "trade_id": 100 + index + id_offset,
                "time": timestamp.isoformat(),
                "price": str(Decimal("100.1") + Decimal(index) / 100),
                "size": "0.00000001",
                "side": "buy",
            }
        )
    bars = []
    native = []
    for minute in range(4):
        first = 1 + minute * count
        selected = rows[first : first + count]
        values = _values(selected)
        official = _values(selected[:1]) if minute == 1 else None if minute == 2 else values.copy()
        minute_start = start + timedelta(minutes=minute)
        bars.append(
            {
                "start": minute_start.isoformat(),
                "end": (minute_start + timedelta(minutes=1)).isoformat(),
                "trade_count": count,
                "ohlcv": values,
                "first_trade": selected[0],
                "last_trade": selected[-1],
                "before": rows[first - 1],
                "after": rows[first + count],
                "official": official,
                "full_ohlcv_matches_official": True
                if minute in (0, 3)
                else False
                if minute == 1
                else None,
                "exact_partial_matches": [{"direction": "prefix", "trade_count": 1}]
                if minute == 1
                else [],
            }
        )
        if official is not None:
            native.append({"start": str(int(minute_start.timestamp())), **official})
    guide = {
        "schema_version": 1,
        "reconstruction_provenance": {
            "product": "BTC-USD",
            "interval_seconds": 60,
            "not_provider_raw_candle": True,
            "candidate_provider": "coinbase_exchange_public_trades_derived",
            "scope": [bars[2]["start"]],
            "target_count": 1,
            "additional_nonmissing_partial_minutes": [bars[1]["start"]],
        },
        "coinbase_independent_reconstruction": {
            "raw_trade_spans": {"window": rows},
            "bars": bars,
            "requests": [
                {
                    "url": "https://api.exchange.coinbase.com/products/BTC-USD/trades?after=999&limit=1000",
                    "status": 200,
                    "sha256": "0" * 64,
                }
            ],
        },
    }
    return guide, copy.deepcopy(rows), copy.deepcopy(native)


def _receipt(tmp_path: Path, guide: dict[str, Any], *, compressed: bool = False) -> Path:
    path = tmp_path / ("receipt.json.gz" if compressed else "receipt.json")
    data = json.dumps(guide).encode()
    path.write_bytes(gzip.compress(data) if compressed else data)
    return path


def _transport(
    rows: list[dict[str, Any]], native: list[dict[str, Any]], calls: list[httpx.Request]
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        if request.url.host == "api.exchange.coinbase.com":
            assert request.url.path == "/products/BTC-USD/trades"
            cursor, limit = int(request.url.params["after"]), int(request.url.params["limit"])
            assert 1 <= limit <= 1000
            body = [row for row in reversed(rows) if cursor - limit <= row["trade_id"] < cursor]
            return httpx.Response(200, json=body)
        assert request.url.host == "api.coinbase.com"
        assert request.url.path == "/api/v3/brokerage/market/products/BTC-USD/candles"
        assert request.url.params["granularity"] == "ONE_MINUTE"
        start, end = int(request.url.params["start"]), int(request.url.params["end"])
        return httpx.Response(
            200, json={"candles": [row for row in native if start <= int(row["start"]) <= end]}
        )

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spot_repair, "_REQUEST_INTERVAL", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_fresh_repair_is_exact_and_preserves_actual_availability(
    tmp_path: Path, compressed: bool
) -> None:
    guide, rows, native = _fixture()
    calls: list[httpx.Request] = []
    before = datetime.now(UTC)
    async with httpx.AsyncClient(transport=_transport(rows, native, calls)) as client:
        batch = await verify_coinbase_repair_receipt(
            _receipt(tmp_path, guide, compressed=compressed), client=client
        )
    assert len(batch.repairs) == 2
    partial, missing = batch.repairs
    assert partial.original is not None
    assert partial.original.volume == Decimal("0.00000001")
    assert missing.original is None
    assert partial.derived.volume == Decimal("0.00000002")
    assert partial.derived.open == Decimal("100.13")
    assert partial.derived.close == Decimal("100.14")
    assert candle_ohlcv(partial.derived) == (
        Decimal("100.13"),
        Decimal("100.14"),
        Decimal("100.13"),
        Decimal("100.14"),
        Decimal("0.00000002"),
    )
    assert before <= partial.derived.retrieved_at == batch.verified_at
    assert all(
        before <= candle.retrieved_at <= batch.verified_at for candle in batch.native_candles
    )
    assert partial.derived.raw_payload["method"] == "coinbase_exchange_trades"
    assert partial.derived.raw_payload["not_provider_raw_candle"] is True
    proof = dict(batch.proof)
    digest = proof.pop("proof_digest")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert partial.derived.raw_payload["proof_digest"] == digest
    assert len(batch.proof["requests"]) == 2
    assert len(batch.proof["requests"][0]["body"]) == len(rows)
    assert batch.proof["historical_availability"] is False
    assert [comparison["status"] for comparison in batch.proof["comparisons"]] == [
        "control",
        "repair",
        "repair",
        "control",
    ]
    assert batch.proof["comparisons"][2]["native_ohlcv"] is None
    assert batch.proof["comparisons"][2]["after_trade_id"] == rows[7]["trade_id"]
    assert client.is_closed


@pytest.mark.asyncio
async def test_pages_descend_with_bounded_after_cursor(tmp_path: Path) -> None:
    guide, rows, native = _fixture(count=400)
    calls: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(rows, native, calls)) as client:
        batch = await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)
    assert [
        (int(call.url.params["after"]), int(call.url.params["limit"])) for call in calls[:-1]
    ] == [(1702, 1000), (702, 602)]
    assert batch.proof["span_audits"][0]["trade_count"] == 1602


@pytest.mark.asyncio
async def test_corrected_native_is_not_repaired_again(tmp_path: Path) -> None:
    guide, rows, native = _fixture()
    bars = guide["coinbase_independent_reconstruction"]["bars"]
    native[1].update(bars[1]["ohlcv"])
    native.append(
        {
            "start": str(int(datetime.fromisoformat(bars[2]["start"]).timestamp())),
            **bars[2]["ohlcv"],
        }
    )
    async with httpx.AsyncClient(transport=_transport(rows, native, [])) as client:
        batch = await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)
    assert batch.repairs == ()
    assert [entry["status"] for entry in batch.proof["comparisons"]] == [
        "control",
        "already_corrected",
        "already_corrected",
        "control",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["gap", "duplicate", "conflicting_duplicate", "order", "time_reversal", "changed_trade"],
)
async def test_fresh_trade_integrity_fails_closed(tmp_path: Path, fault: str) -> None:
    guide, rows, native = _fixture()
    if fault == "gap":
        rows.pop(3)
    elif fault in ("duplicate", "conflicting_duplicate"):
        rows[3]["trade_id"] = rows[2]["trade_id"]
        if fault == "conflicting_duplicate":
            rows[3]["price"] = "999"
    elif fault == "order":
        rows[2], rows[3] = rows[3], rows[2]
    elif fault == "time_reversal":
        rows[3]["time"] = rows[0]["time"]
    else:
        rows[3]["price"] = "999"
    async with httpx.AsyncClient(transport=_transport(rows, native, [])) as client:
        with pytest.raises(SpotRepairError):
            await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "boundary",
        "candidate",
        "partial",
        "control",
        "scope",
        "provenance",
        "ssrf",
        "unbounded",
        "empty_minute",
    ],
)
async def test_tampered_or_incomplete_guide_is_rejected(tmp_path: Path, fault: str) -> None:
    guide, rows, native = _fixture()
    reconstruction = guide["coinbase_independent_reconstruction"]
    bars = reconstruction["bars"]
    if fault == "boundary":
        bars[1]["before"] = copy.deepcopy(rows[0])
    elif fault == "candidate":
        bars[1]["ohlcv"]["volume"] = "42"
    elif fault == "partial":
        bars[1]["exact_partial_matches"][0]["direction"] = "suffix"
    elif fault == "control":
        bars[0]["official"]["volume"] = "1"
    elif fault == "scope":
        guide["reconstruction_provenance"]["additional_nonmissing_partial_minutes"] = []
    elif fault == "provenance":
        del guide["reconstruction_provenance"]
    elif fault == "ssrf":
        reconstruction["requests"][0]["url"] = "http://127.0.0.1/latest/meta-data"
    elif fault == "unbounded":
        reconstruction["raw_trade_spans"]["window"][-1]["trade_id"] = 999999999999
    else:
        for collection in (reconstruction["raw_trade_spans"]["window"], rows):
            collection[5]["time"] = collection[7]["time"]
            collection[6]["time"] = collection[7]["time"]
    async with httpx.AsyncClient(transport=_transport(rows, native, [])) as client:
        with pytest.raises(SpotRepairError):
            await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "partial_changed",
        "partial_missing",
        "control_partial",
        "control_missing",
        "missing_unknown",
        "duplicate_native",
    ],
)
async def test_unknown_native_revisions_and_partial_controls_fail_closed(
    tmp_path: Path, fault: str
) -> None:
    guide, rows, native = _fixture()
    if fault == "partial_changed":
        native[1]["volume"] = "999"
    elif fault == "partial_missing":
        native.pop(1)
    elif fault == "control_partial":
        native[0]["volume"] = "0.00000001"
    elif fault == "control_missing":
        native.pop(0)
    elif fault == "duplicate_native":
        native.append(copy.deepcopy(native[0]))
    else:
        start = datetime.fromisoformat(
            guide["coinbase_independent_reconstruction"]["bars"][2]["start"]
        )
        native.append({"start": str(int(start.timestamp())), **{field: "1" for field in _FIELDS}})
    async with httpx.AsyncClient(transport=_transport(rows, native, [])) as client:
        with pytest.raises(SpotRepairError):
            await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)


@pytest.mark.asyncio
async def test_no_network_for_invalid_symbol_or_authenticated_client(tmp_path: Path) -> None:
    guide, rows, native = _fixture()
    calls: list[httpx.Request] = []
    receipt = _receipt(tmp_path, guide)
    async with httpx.AsyncClient(
        transport=_transport(rows, native, calls), headers={"Authorization": "secret"}
    ) as client:
        with pytest.raises(SpotRepairError, match="anonymous"):
            await verify_coinbase_repair_receipt(receipt, client=client)
    async with httpx.AsyncClient(transport=_transport(rows, native, calls)) as client:
        with pytest.raises(SpotRepairError, match="Symbol"):
            await verify_coinbase_repair_receipt(receipt, symbol="../admin", client=client)
    assert calls == []


@pytest.mark.asyncio
async def test_boundary_requires_earlier_witness_but_accepts_exact_end_timestamp(
    tmp_path: Path,
) -> None:
    guide, rows, native = _fixture()
    receipt = _receipt(tmp_path, guide)
    async with httpx.AsyncClient(transport=_transport(rows, native, [])) as client:
        assert len((await verify_coinbase_repair_receipt(receipt, client=client)).repairs) == 2
    guide["coinbase_independent_reconstruction"]["raw_trade_spans"]["window"].pop(0)
    rows.pop(0)
    async with httpx.AsyncClient(transport=_transport(rows, native, [])) as client:
        with pytest.raises(SpotRepairError, match="boundary witnesses"):
            await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)


@pytest.mark.asyncio
async def test_disjoint_spans_each_require_both_complete_controls(tmp_path: Path) -> None:
    guide, rows, native = _fixture()
    second, second_rows, second_native = _fixture(day_offset=1, id_offset=1000)
    reconstruction = guide["coinbase_independent_reconstruction"]
    other = second["coinbase_independent_reconstruction"]
    reconstruction["raw_trade_spans"]["second"] = other["raw_trade_spans"]["window"]
    reconstruction["bars"].extend(other["bars"])
    provenance = guide["reconstruction_provenance"]
    provenance["scope"].extend(second["reconstruction_provenance"]["scope"])
    provenance["additional_nonmissing_partial_minutes"].extend(
        second["reconstruction_provenance"]["additional_nonmissing_partial_minutes"]
    )
    provenance["target_count"] = 2
    calls: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=_transport(rows + second_rows, native + second_native, calls)
    ) as client:
        batch = await verify_coinbase_repair_receipt(_receipt(tmp_path, guide), client=client)
    assert len(batch.repairs) == 4
    assert len(batch.proof["windows"]) == 2
    assert sum(entry["status"] == "control" for entry in batch.proof["comparisons"]) == 4
    assert len(calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [b"[]", b"{", b'{"schema_version":1,"schema_version":2}', b'{"reconstruction_provenance": []}'],
)
async def test_malformed_receipt_fails_without_network(tmp_path: Path, raw: bytes) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    _, rows, native = _fixture()
    calls: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(rows, native, calls)) as client:
        with pytest.raises(SpotRepairError):
            await verify_coinbase_repair_receipt(receipt, client=client)
    assert calls == []
