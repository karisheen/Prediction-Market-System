from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from prediction_market_system.venues.kalshi import (
    KalshiClient,
    KalshiMarket,
    KalshiOrderBook,
    UnsupportedMarketError,
    normalize_order_book,
)


def market_payload(*, can_close_early: bool = False) -> dict[str, object]:
    return {
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
        "rules_primary": "Resolves YES from the benchmark at expiry.",
        "rules_secondary": "",
        "yes_bid_dollars": "0.4200",
        "yes_ask_dollars": "0.4400",
        "no_bid_dollars": "0.5600",
        "no_ask_dollars": "0.5800",
        "yes_bid_size_fp": "13.00",
        "yes_ask_size_fp": "17.00",
    }


def order_book_payload() -> dict[str, object]:
    return {
        "orderbook_fp": {
            "yes_dollars": [["0.2000", "50.00"], ["0.4200", "13.00"]],
            "no_dollars": [["0.2500", "50.00"], ["0.5600", "17.00"]],
        }
    }


def test_normalizes_implied_asks_and_sizes() -> None:
    order_book = KalshiOrderBook.model_validate(order_book_payload()["orderbook_fp"])

    book = normalize_order_book(order_book)

    assert book.yes_bid == pytest.approx(0.42)
    assert book.yes_ask == pytest.approx(0.44)
    assert book.no_bid == pytest.approx(0.56)
    assert book.no_ask == pytest.approx(0.58)
    assert book.yes_ask_size == pytest.approx(17.0)
    assert book.no_ask_size == pytest.approx(13.0)


@pytest.mark.asyncio
async def test_fetches_public_market_snapshot_without_authentication() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/orderbook"):
            return httpx.Response(200, json=order_book_payload())
        return httpx.Response(200, json={"market": market_payload()})

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(client=http_client)

    market, snapshot = await client.get_market_snapshot("KXBTCTEST-30DEC31-T100000")

    assert market.terminal_threshold_strike() == 100000
    assert snapshot.market_id == market.ticker
    assert snapshot.yes_ask == pytest.approx(0.44)
    assert snapshot.no_ask == pytest.approx(0.58)
    assert all("KALSHI-ACCESS-KEY" not in request.headers for request in requests)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_lists_open_markets_by_series() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["status"] == "open"
        assert request.url.params["series_ticker"] == "KXBTCTEST"
        return httpx.Response(
            200,
            json={"markets": [market_payload()], "cursor": "next-page"},
        )

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(client=http_client)

    result = await client.list_markets(series_ticker="KXBTCTEST", limit=10)

    assert len(result.markets) == 1
    assert result.markets[0].ticker == "KXBTCTEST-30DEC31-T100000"
    assert result.cursor == "next-page"
    await http_client.aclose()


def test_rejects_path_dependent_market_for_terminal_model() -> None:
    market = KalshiMarket.model_validate(market_payload(can_close_early=True))

    with pytest.raises(UnsupportedMarketError, match="path-dependent"):
        market.terminal_threshold_strike()


def test_decimal_contract_values_are_preserved_during_parsing() -> None:
    market = KalshiMarket.model_validate(market_payload())

    assert market.notional_value_dollars == Decimal("1.0000")
    assert market.expiry == datetime(2030, 12, 31, 23, 59, tzinfo=UTC)
