from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from prediction_market_system.sources.coinbase import CoinbaseClient, CoinbaseDataError
from prediction_market_system.sources.deribit import DeribitClient


@pytest.mark.asyncio
async def test_coinbase_fetches_completed_candles_in_source_order() -> None:
    start_at = datetime(2030, 1, 1, tzinfo=UTC)
    end_at = start_at + timedelta(hours=2)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/products/BTC-USD/candles")
        assert request.url.params["granularity"] == "ONE_HOUR"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "candles": [
                    {
                        "start": str(int((start_at + timedelta(hours=1)).timestamp())),
                        "low": "101",
                        "high": "103",
                        "open": "101",
                        "close": "102",
                        "volume": "11",
                    },
                    {
                        "start": str(int(start_at.timestamp())),
                        "low": "99",
                        "high": "102",
                        "open": "100",
                        "close": "101",
                        "volume": "10",
                    },
                    {
                        "start": str(int(end_at.timestamp())),
                        "low": "102",
                        "high": "104",
                        "open": "102",
                        "close": "103",
                        "volume": "12",
                    },
                ]
            },
        )

    http_client = httpx.AsyncClient(
        base_url="https://api.coinbase.com/api/v3/brokerage/market",
        transport=httpx.MockTransport(handler),
    )
    client = CoinbaseClient(client=http_client)

    candles = await client.get_candles(
        "BTC-USD",
        start_at=start_at,
        end_at=end_at,
        interval_seconds=3600,
    )

    assert [candle.start_at for candle in candles] == [
        start_at,
        start_at + timedelta(hours=1),
    ]
    assert candles[-1].end_at == end_at
    assert candles[-1].close == Decimal("102")
    await http_client.aclose()


@pytest.mark.asyncio
async def test_deribit_normalizes_dvol_funding_and_current_snapshot() -> None:
    start_at = datetime(2030, 1, 1, tzinfo=UTC)
    end_at = start_at + timedelta(hours=1)
    start_ms = int(start_at.timestamp() * 1000)
    end_ms = int(end_at.timestamp() * 1000)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/get_volatility_index_data"):
            assert request.url.params["resolution"] == "3600"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "result": {
                        "data": [[start_ms, 44.0, 46.0, 43.0, 45.0]],
                        "continuation": None,
                    },
                },
            )
        if request.url.path.endswith("/get_funding_rate_history"):
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "result": [
                        {
                            "timestamp": end_ms,
                            "index_price": 101.0,
                            "prev_index_price": 100.0,
                            "interest_1h": 0.0001,
                            "interest_8h": 0.0008,
                        }
                    ],
                },
            )
        if request.url.path.endswith("/ticker"):
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "result": {
                        "timestamp": end_ms,
                        "instrument_name": "BTC-PERPETUAL",
                        "index_price": 100.0,
                        "mark_price": 101.0,
                        "open_interest": 5000.0,
                        "current_funding": 0.00001,
                        "funding_8h": 0.00008,
                    },
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    http_client = httpx.AsyncClient(
        base_url="https://www.deribit.com/api/v2/public",
        transport=httpx.MockTransport(handler),
    )
    client = DeribitClient(client=http_client)

    dvol = await client.get_dvol_history(
        "BTC",
        start_at=start_at,
        end_at=end_at,
        resolution_seconds=3600,
    )
    funding = await client.get_funding_history(
        "BTC-PERPETUAL",
        start_at=start_at,
        end_at=end_at,
    )
    snapshot = await client.get_derivatives_snapshot("BTC-PERPETUAL")

    assert dvol[0].observed_at == end_at
    assert dvol[0].annualized_volatility == pytest.approx(0.45)
    assert funding[0].observed_at == end_at
    assert funding[0].funding_rate_1h == pytest.approx(0.0001)
    assert snapshot.observed_at == end_at
    assert snapshot.basis == pytest.approx(0.01)
    assert snapshot.open_interest == pytest.approx(5000.0)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_deribit_paginates_dvol_backward_from_continuation() -> None:
    start_at = datetime(2030, 1, 1, tzinfo=UTC)
    midpoint = start_at + timedelta(hours=2)
    end_at = start_at + timedelta(hours=4)
    requested_ends: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        end_ms = int(request.url.params["end_timestamp"])
        requested_ends.append(end_ms)
        assert int(request.url.params["start_timestamp"]) == int(start_at.timestamp() * 1000)
        if end_ms == int(end_at.timestamp() * 1000):
            row_at = end_at - timedelta(hours=1)
            continuation = int(midpoint.timestamp() * 1000)
            close = 55.0
        else:
            assert end_ms == int(midpoint.timestamp() * 1000)
            row_at = start_at
            continuation = None
            close = 45.0
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "result": {
                    "data": [[int(row_at.timestamp() * 1000), close, close, close, close]],
                    "continuation": continuation,
                },
            },
        )

    http_client = httpx.AsyncClient(
        base_url="https://www.deribit.com/api/v2/public",
        transport=httpx.MockTransport(handler),
    )
    client = DeribitClient(client=http_client)

    observations = await client.get_dvol_history(
        "BTC",
        start_at=start_at,
        end_at=end_at,
        resolution_seconds=3600,
    )

    assert requested_ends == [
        int(end_at.timestamp() * 1000),
        int(midpoint.timestamp() * 1000),
    ]
    assert [observation.observed_at for observation in observations] == [
        start_at + timedelta(hours=1),
        end_at,
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_coinbase_retries_transient_transport_failures() -> None:
    start_at = datetime(2030, 1, 1, tzinfo=UTC)
    end_at = start_at + timedelta(hours=1)
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided", request=request)
        return httpx.Response(
            200,
            json={
                "candles": [
                    {
                        "start": str(int(start_at.timestamp())),
                        "low": "99",
                        "high": "102",
                        "open": "100",
                        "close": "101",
                        "volume": "10",
                    }
                ]
            },
        )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    http_client = httpx.AsyncClient(
        base_url="https://api.coinbase.com/api/v3/brokerage/market",
        transport=httpx.MockTransport(handler),
    )
    client = CoinbaseClient(client=http_client, max_transient_retries=2, sleep=record_sleep)

    candles = await client.get_candles(
        "BTC-USD",
        start_at=start_at,
        end_at=end_at,
        interval_seconds=3600,
    )

    assert len(candles) == 1
    assert attempts == 3
    assert delays == [1.0, 2.0]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_coinbase_surfaces_exhausted_transient_retries() -> None:
    start_at = datetime(2030, 1, 1, tzinfo=UTC)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided", request=request)

    async def no_sleep(delay: float) -> None:
        return None

    http_client = httpx.AsyncClient(
        base_url="https://api.coinbase.com/api/v3/brokerage/market",
        transport=httpx.MockTransport(handler),
    )
    client = CoinbaseClient(client=http_client, max_transient_retries=1, sleep=no_sleep)

    with pytest.raises(CoinbaseDataError, match="nodename nor servname"):
        await client.get_candles(
            "BTC-USD",
            start_at=start_at,
            end_at=start_at + timedelta(hours=1),
            interval_seconds=3600,
        )

    assert attempts == 2
    await http_client.aclose()


@pytest.mark.asyncio
async def test_deribit_retries_transient_transport_failures() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "result": {
                    "timestamp": int(datetime(2030, 1, 1, tzinfo=UTC).timestamp() * 1000),
                    "instrument_name": "BTC-PERPETUAL",
                    "index_price": 100.0,
                    "mark_price": 101.0,
                    "open_interest": 5000.0,
                    "current_funding": 0.0001,
                    "funding_8h": 0.0008,
                },
            },
        )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    http_client = httpx.AsyncClient(
        base_url="https://www.deribit.com/api/v2/public",
        transport=httpx.MockTransport(handler),
    )
    client = DeribitClient(client=http_client, sleep=record_sleep)

    snapshot = await client.get_derivatives_snapshot("BTC-PERPETUAL")

    assert snapshot.instrument_name == "BTC-PERPETUAL"
    assert attempts == 2
    assert delays == [1.0]
    await http_client.aclose()
