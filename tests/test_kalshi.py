from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from prediction_market_system.domain import (
    TerminalRangeContract,
    ThresholdDirection,
    ThresholdModelKind,
)
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
        "rules_secondary": "The benchmark's final value controls.",
        "yes_bid_dollars": "0.4200",
        "yes_ask_dollars": "0.4400",
        "no_bid_dollars": "0.5600",
        "no_ask_dollars": "0.5800",
        "yes_bid_size_fp": "13.00",
        "yes_ask_size_fp": "17.00",
    }


def historical_market_payload() -> dict[str, object]:
    return {
        **market_payload(),
        "status": "finalized",
        "created_time": "2030-01-01T00:00:00Z",
        "updated_time": "2031-01-01T00:01:00Z",
        "open_time": "2030-01-01T00:00:00Z",
        "result": "yes",
        "settlement_value_dollars": "1.0000",
        "settlement_ts": "2031-01-01T00:00:00Z",
        "expiration_value": "101234.56",
    }


def candlestick_payload() -> dict[str, object]:
    return {
        "ticker": "KXBTCTEST-30DEC31-T100000",
        "candlesticks": [
            {
                "end_period_ts": 1_924_991_940,
                "yes_bid": {
                    "open": "0.4000",
                    "low": "0.3900",
                    "high": "0.4300",
                    "close": "0.4200",
                },
                "yes_ask": {
                    "open": "0.4200",
                    "low": "0.4100",
                    "high": "0.4500",
                    "close": "0.4400",
                },
                "price": {
                    "open": None,
                    "low": None,
                    "high": None,
                    "close": None,
                    "mean": None,
                    "previous": "0.4100",
                },
                "volume": "12.50",
                "open_interest": "200.00",
            }
        ],
    }


def live_candlestick_payload() -> dict[str, object]:
    return {
        "ticker": "KXBTCTEST-30DEC31-T100000",
        "candlesticks": [
            {
                "end_period_ts": 1_924_991_940,
                "yes_bid": {
                    "open_dollars": "0.4000",
                    "low_dollars": "0.3900",
                    "high_dollars": "0.4300",
                    "close_dollars": "0.4200",
                },
                "yes_ask": {
                    "open_dollars": "0.4200",
                    "low_dollars": "0.4100",
                    "high_dollars": "0.4500",
                    "close_dollars": "0.4400",
                },
                "price": {
                    "open_dollars": None,
                    "low_dollars": None,
                    "high_dollars": None,
                    "close_dollars": None,
                    "mean_dollars": None,
                    "previous_dollars": "0.4100",
                },
                "volume_fp": "12.50",
                "open_interest_fp": "200.00",
            }
        ],
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


def test_normalizes_one_sided_book_without_inventing_missing_liquidity() -> None:
    book = normalize_order_book(KalshiOrderBook(no_dollars=[("0.5600", "17.00")]))

    assert book.yes_bid is None
    assert book.yes_ask == pytest.approx(0.44)
    assert book.yes_ask_size == pytest.approx(17.0)
    assert book.no_ask is None
    assert book.no_ask_size is None


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

    contract = market.price_contract()
    assert contract.model_kind is ThresholdModelKind.TERMINAL
    assert contract.direction is ThresholdDirection.ABOVE
    assert contract.strike_price == 100000
    assert snapshot.market_id == market.ticker
    assert snapshot.yes_ask == pytest.approx(0.44)
    assert snapshot.no_ask == pytest.approx(0.58)
    assert all("KALSHI-ACCESS-KEY" not in request.headers for request in requests)
    assert snapshot.series_id == "KXBTCTEST"
    assert snapshot.event_id == "KXBTCTEST-30DEC31"
    assert snapshot.contract_label == "Above $100,000"
    assert str(snapshot.market_url) == (
        "https://kalshi.com/markets/kxbtctest/bitcoin-range/kxbtctest-30dec31"
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_retries_rate_limited_requests_with_bounded_backoff() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"market": market_payload()})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(
        client=http_client,
        max_rate_limit_retries=2,
        sleep=record_sleep,
    )

    market = await client.get_market("KXBTCTEST-30DEC31-T100000")

    assert market.ticker == "KXBTCTEST-30DEC31-T100000"
    assert attempts == 3
    assert delays == [1.0, 2.0]
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


@pytest.mark.asyncio
async def test_fetches_historical_research_inputs_and_fee_changes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/historical/markets"):
            assert request.url.params["series_ticker"] == "KXBTCTEST"
            return httpx.Response(
                200,
                json={"markets": [historical_market_payload()], "cursor": ""},
            )
        if path.endswith("/candlesticks"):
            assert request.url.params["period_interval"] == "60"
            return httpx.Response(200, json=candlestick_payload())
        if path.endswith("/series/fee_changes"):
            assert request.url.params["show_historical"] == "true"
            return httpx.Response(
                200,
                json={
                    "series_fee_change_arr": [
                        {
                            "id": "series-fee-1",
                            "series_ticker": "KXBTCTEST",
                            "fee_type": "quadratic",
                            "fee_multiplier": 0.07,
                            "scheduled_ts": "2030-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        if path.endswith("/events/fee_changes"):
            return httpx.Response(
                200,
                json={
                    "event_fee_changes": [
                        {
                            "id": "event-fee-1",
                            "event_ticker": "KXBTCTEST-30DEC31",
                            "series_ticker": "KXBTCTEST",
                            "fee_type_override": None,
                            "fee_multiplier_override": None,
                            "scheduled_ts": "2030-06-01T00:00:00Z",
                        }
                    ],
                    "cursor": "",
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(client=http_client)

    markets = await client.list_historical_markets(series_ticker="KXBTCTEST")
    candles = await client.get_historical_candlesticks(
        markets.markets[0].ticker,
        start_ts=1_924_988_400,
        end_ts=1_924_991_940,
        period_interval=60,
    )
    series_fees = await client.get_series_fee_changes("KXBTCTEST")
    event_fees = await client.get_event_fee_changes("KXBTCTEST-30DEC31")

    assert markets.markets[0].result == "yes"
    assert markets.markets[0].settlement_value_dollars == Decimal("1.0000")
    assert candles[0].price.close is None
    assert candles[0].price.previous == Decimal("0.4100")
    assert series_fees[0].fee_multiplier == pytest.approx(0.07)
    assert event_fees.event_fee_changes[0].fee_type_override is None
    await http_client.aclose()


@pytest.mark.asyncio
async def test_candlesticks_fall_back_to_recent_market_endpoint() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if "/historical/markets/" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json=live_candlestick_payload())

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(client=http_client)

    candles = await client.get_candlesticks(
        "KXBTCTEST",
        "KXBTCTEST-30DEC31-T100000",
        start_ts=1_924_988_400,
        end_ts=1_924_991_940,
        period_interval=60,
    )

    assert requested_paths == [
        "/trade-api/v2/historical/markets/KXBTCTEST-30DEC31-T100000/candlesticks",
        ("/trade-api/v2/series/KXBTCTEST/markets/KXBTCTEST-30DEC31-T100000/candlesticks"),
    ]
    assert candles[0].price.previous == Decimal("0.4100")
    await http_client.aclose()


@pytest.mark.asyncio
async def test_lists_settled_events_and_fetches_their_markets() -> None:
    event_payload = {
        "event_ticker": "KXBTCTEST-30DEC31",
        "series_ticker": "KXBTCTEST",
        "title": "BTC range",
        "sub_title": "At 5 PM",
        "mutually_exclusive": True,
        "strike_date": "2030-12-31T23:59:00Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            assert request.url.params["status"] == "settled"
            assert request.url.params["series_ticker"] == "KXBTCTEST"
            assert request.url.params["min_close_ts"] == "1924988400"
            return httpx.Response(200, json={"events": [event_payload], "cursor": "next"})
        if request.url.path.endswith("/events/KXBTCTEST-30DEC31"):
            return httpx.Response(
                200,
                json={
                    "event": event_payload,
                    "markets": [historical_market_payload()],
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(client=http_client)

    events = await client.list_events(
        status="settled",
        series_ticker="KXBTCTEST",
        min_close_ts=1_924_988_400,
    )
    event = await client.get_event(events.events[0].event_ticker)

    assert events.cursor == "next"
    assert events.events[0].mutually_exclusive is True
    assert event.markets[0].ticker == "KXBTCTEST-30DEC31-T100000"
    await http_client.aclose()


@pytest.mark.asyncio
async def test_fetches_flexible_kalshi_event_live_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/live_data/events/KXBTCTEST-30DEC31")
        assert request.url.params["range"] == "1h"
        return httpx.Response(
            200,
            json={
                "live_data": {
                    "type": "crypto",
                    "details": {
                        "coin": "BTC",
                        "maturity_ts_ms": 1_924_991_940_000,
                        "provider_specific": {"preserved": True},
                    },
                    "is_historical": True,
                    "default_range": "1h",
                    "range_options": ["15min", "1h"],
                }
            },
        )

    http_client = httpx.AsyncClient(
        base_url="https://external-api.kalshi.com/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    client = KalshiClient(client=http_client)

    live_data = await client.get_event_live_data(
        "KXBTCTEST-30DEC31",
        range_hint="1h",
    )

    assert live_data.type == "crypto"
    assert live_data.is_historical is True
    assert live_data.details["provider_specific"] == {"preserved": True}
    await http_client.aclose()


def test_classifies_fixed_expiry_rule_as_terminal_despite_early_close_flag() -> None:
    market = KalshiMarket.model_validate(market_payload(can_close_early=True))

    contract = market.price_contract()

    assert contract.model_kind is ThresholdModelKind.TERMINAL


def test_rejects_ambiguous_early_close_threshold() -> None:
    payload = market_payload(can_close_early=True)
    payload["rules_primary"] = "Resolves YES if the benchmark is above the threshold."
    payload["rules_secondary"] = ""
    market = KalshiMarket.model_validate(payload)

    with pytest.raises(UnsupportedMarketError, match="explicit terminal observation"):
        market.price_contract()


def test_fixed_time_rule_does_not_combine_cross_rule_markers_into_touch_barrier() -> None:
    payload = market_payload(can_close_early=True)
    payload["rules_primary"] = (
        "If the simple average of the sixty seconds of the Bitcoin Real-Time Index "
        "before 5 PM EDT is above 73749.99 at 5 PM EDT on Aug 6, 2026, "
        "then the market resolves to Yes."
    )
    payload["rules_secondary"] = (
        "At the last minute before expiration, 60 index prices are collected. "
        "The official and final value is the average of these prices."
    )
    market = KalshiMarket.model_validate(payload)

    contract = market.price_contract()

    assert contract.model_kind is ThresholdModelKind.TERMINAL


def test_parses_fixed_time_between_market_as_terminal_range() -> None:
    payload = market_payload(can_close_early=True)
    payload.update(
        {
            "ticker": "KXBTCTEST-30DEC31-B100050",
            "strike_type": "between",
            "floor_strike": 100000,
            "cap_strike": 100099.99,
            "yes_sub_title": "$100,000 to $100,099.99",
            "rules_primary": (
                "Resolves YES if the simple average of the sixty seconds before 5 PM EDT "
                "is between 100000 and 100099.99 at 5 PM EDT on Dec 31, 2030."
            ),
        }
    )

    contract = KalshiMarket.model_validate(payload).price_contract()

    assert isinstance(contract, TerminalRangeContract)
    assert contract.lower_bound == 100000
    assert contract.upper_bound == pytest.approx(100099.99)

    assert contract.settlement_window_seconds == 60


def test_rejects_nonterminal_between_market() -> None:
    payload = market_payload(can_close_early=True)
    payload.update(
        {
            "strike_type": "between",
            "floor_strike": 100000,
            "cap_strike": 100099.99,
            "rules_primary": "Resolves YES if the benchmark enters the range before expiry.",
            "rules_secondary": "",
        }
    )

    with pytest.raises(UnsupportedMarketError, match="fixed-time terminal"):
        KalshiMarket.model_validate(payload).price_contract()


def test_classifies_explicit_upper_and_lower_touch_barriers() -> None:
    upper_payload = market_payload(can_close_early=True)
    upper_payload["rules_primary"] = (
        "Resolves YES if the benchmark reaches the threshold at any time before expiry."
    )
    upper = KalshiMarket.model_validate(upper_payload).price_contract()

    lower_payload = {
        **upper_payload,
        "strike_type": "less_or_equal",
        "floor_strike": None,
        "cap_strike": 80000,
        "rules_primary": (
            "Resolves YES if the benchmark trades below the threshold "
            "at any point before expiration."
        ),
    }
    lower = KalshiMarket.model_validate(lower_payload).price_contract()

    assert upper.model_kind is ThresholdModelKind.BARRIER
    assert upper.direction is ThresholdDirection.ABOVE
    assert upper.strike_price == 100000
    assert lower.model_kind is ThresholdModelKind.BARRIER
    assert lower.direction is ThresholdDirection.BELOW
    assert lower.strike_price == 80000


def test_decimal_contract_values_are_preserved_during_parsing() -> None:
    market = KalshiMarket.model_validate(market_payload())

    assert market.notional_value_dollars == Decimal("1.0000")
    assert market.expiry == datetime(2030, 12, 31, 23, 59, tzinfo=UTC)
