from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prediction_market_system.domain import (
    MarketSnapshot,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)

KALSHI_PRODUCTION_API = "https://external-api.kalshi.com/trade-api/v2"
CandlestickPeriod = Literal[1, 60, 1440]
FeeType = Literal["quadratic", "quadratic_with_maker_fees", "flat"]


class KalshiAPIError(RuntimeError):
    pass


class IncompleteOrderBookError(ValueError):
    pass


class UnsupportedMarketError(ValueError):
    pass


def _has_explicit_touch_semantics(rule: str) -> bool:
    normalized = " ".join(rule.casefold().split())
    path_markers = (
        "at any time",
        "at any point",
        "before expiration",
        "before expiry",
        "before the market close",
        "prior to",
        "during the observation",
        "by expiration",
        "by expiry",
        "by the market close",
    )
    touch_markers = (
        "reach",
        "touch",
        "hit",
        "trade at",
        "cross",
        "exceed",
        "above",
        "below",
        "greater than",
        "less than",
    )
    return any(marker in normalized for marker in path_markers) and any(
        marker in normalized for marker in touch_markers
    )


class _KalshiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class KalshiMarket(_KalshiModel):
    ticker: str
    event_ticker: str
    market_type: str
    title: str = ""
    yes_sub_title: str
    no_sub_title: str
    close_time: datetime
    expected_expiration_time: datetime | None = None
    latest_expiration_time: datetime
    status: str
    notional_value_dollars: Decimal
    can_close_early: bool
    strike_type: str | None = None
    floor_strike: float | None = None
    cap_strike: float | None = None
    rules_primary: str
    rules_secondary: str = ""
    yes_bid_dollars: Decimal
    yes_ask_dollars: Decimal
    no_bid_dollars: Decimal
    no_ask_dollars: Decimal
    yes_bid_size_fp: Decimal
    yes_ask_size_fp: Decimal
    created_time: datetime | None = None
    updated_time: datetime | None = None
    open_time: datetime | None = None
    result: Literal["yes", "no", "scalar", ""] = ""
    settlement_value_dollars: Decimal | None = None
    settlement_ts: datetime | None = None
    expiration_value: str = ""

    @property
    def expiry(self) -> datetime:
        return self.expected_expiration_time or self.close_time

    @property
    def question(self) -> str:
        title = self.title.strip()
        yes_title = self.yes_sub_title.strip()
        if title and yes_title and yes_title.casefold() not in title.casefold():
            return f"{title} — {yes_title}"
        return title or yes_title

    @property
    def resolution_rule(self) -> str:
        rules = [rule.strip() for rule in (self.rules_primary, self.rules_secondary) if rule]
        return "\n\n".join(rules)

    def threshold_contract(self, override: float | None = None) -> ThresholdContract:
        if self.strike_type in {"greater", "greater_or_equal"}:
            direction = ThresholdDirection.ABOVE
            metadata_strike = self.floor_strike
        elif self.strike_type in {"less", "less_or_equal"}:
            direction = ThresholdDirection.BELOW
            metadata_strike = self.cap_strike
        else:
            raise UnsupportedMarketError(
                f"unsupported Kalshi strike type: {self.strike_type or 'missing'}"
            )

        strike = override if override is not None else metadata_strike
        if strike is None or strike <= 0:
            raise UnsupportedMarketError("a positive threshold strike is required for this market")

        model_kind = ThresholdModelKind.TERMINAL
        if self.can_close_early:
            if not _has_explicit_touch_semantics(self.resolution_rule):
                raise UnsupportedMarketError(
                    "early-close rules do not explicitly define a supported touch barrier"
                )
            model_kind = ThresholdModelKind.BARRIER

        return ThresholdContract(
            model_kind=model_kind,
            direction=direction,
            strike_price=strike,
        )


class KalshiMarketResponse(_KalshiModel):
    market: KalshiMarket


class KalshiMarketsResponse(_KalshiModel):
    markets: list[KalshiMarket]
    cursor: str = ""


class KalshiBidAskDistribution(_KalshiModel):
    open: Decimal
    low: Decimal
    high: Decimal
    close: Decimal


class KalshiPriceDistribution(_KalshiModel):
    open: Decimal | None = None
    low: Decimal | None = None
    high: Decimal | None = None
    close: Decimal | None = None
    mean: Decimal | None = None
    previous: Decimal | None = None


class KalshiCandlestick(_KalshiModel):
    end_period_ts: int
    yes_bid: KalshiBidAskDistribution
    yes_ask: KalshiBidAskDistribution
    price: KalshiPriceDistribution
    volume: Decimal
    open_interest: Decimal


class KalshiCandlesticksResponse(_KalshiModel):
    ticker: str
    candlesticks: list[KalshiCandlestick]


class KalshiSeriesFeeChange(_KalshiModel):
    id: str
    series_ticker: str
    fee_type: FeeType
    fee_multiplier: float
    scheduled_ts: datetime


class KalshiSeriesFeeChangesResponse(_KalshiModel):
    series_fee_change_arr: list[KalshiSeriesFeeChange]


class KalshiEventFeeChange(_KalshiModel):
    id: str
    event_ticker: str
    series_ticker: str
    fee_type_override: FeeType | None
    fee_multiplier_override: float | None
    scheduled_ts: datetime


class KalshiEventFeeChangesResponse(_KalshiModel):
    event_fee_changes: list[KalshiEventFeeChange]
    cursor: str = ""


class KalshiEventLiveData(_KalshiModel):
    type: str
    details: dict[str, Any]
    is_historical: bool = False
    default_range: str | None = None
    range_options: list[str] = Field(default_factory=list)


class KalshiEventLiveDataResponse(_KalshiModel):
    live_data: KalshiEventLiveData


class KalshiOrderBook(_KalshiModel):
    yes_dollars: list[tuple[Decimal, Decimal]] = Field(default_factory=list)
    no_dollars: list[tuple[Decimal, Decimal]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_levels(self) -> Self:
        for side in (self.yes_dollars, self.no_dollars):
            if any(price < 0 or price > 1 or quantity < 0 for price, quantity in side):
                raise ValueError("order-book prices and quantities are outside valid ranges")
        return self


class KalshiOrderBookResponse(_KalshiModel):
    orderbook_fp: KalshiOrderBook


class ExecutableBook(_KalshiModel):
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    yes_ask_size: float
    no_ask_size: float


def normalize_order_book(order_book: KalshiOrderBook) -> ExecutableBook:
    if not order_book.yes_dollars or not order_book.no_dollars:
        raise IncompleteOrderBookError(
            "both YES and NO bids are required to derive executable asks"
        )

    yes_bid_price, yes_bid_size = max(order_book.yes_dollars, key=lambda level: level[0])
    no_bid_price, no_bid_size = max(order_book.no_dollars, key=lambda level: level[0])
    yes_ask_price = Decimal("1") - no_bid_price
    no_ask_price = Decimal("1") - yes_bid_price

    if yes_bid_price > yes_ask_price or no_bid_price > no_ask_price:
        raise ValueError("Kalshi order book is crossed or internally inconsistent")

    return ExecutableBook(
        yes_bid=float(yes_bid_price),
        yes_ask=float(yes_ask_price),
        no_bid=float(no_bid_price),
        no_ask=float(no_ask_price),
        yes_ask_size=float(no_bid_size),
        no_ask_size=float(yes_bid_size),
    )


def to_market_snapshot(
    market: KalshiMarket,
    order_book: KalshiOrderBook,
    *,
    observed_at: datetime,
) -> MarketSnapshot:
    if market.market_type != "binary":
        raise UnsupportedMarketError("only binary Kalshi markets are supported")
    if market.notional_value_dollars != Decimal("1"):
        raise UnsupportedMarketError("only $1 payout contracts are supported")
    if market.status not in {"active", "initialized"}:
        raise UnsupportedMarketError(f"Kalshi market is not active: {market.status}")

    book = normalize_order_book(order_book)
    return MarketSnapshot(
        market_id=market.ticker,
        question=market.question,
        venue="Kalshi",
        observed_at=observed_at,
        expires_at=market.expiry,
        yes_bid=book.yes_bid,
        yes_ask=book.yes_ask,
        no_bid=book.no_bid,
        no_ask=book.no_ask,
        yes_ask_size=book.yes_ask_size,
        no_ask_size=book.no_ask_size,
        resolution_rule=market.resolution_rule,
    )


class KalshiClient:
    def __init__(
        self,
        *,
        base_url: str = KALSHI_PRODUCTION_API,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=10.0,
            headers={"User-Agent": "prediction-market-system/0.1.0"},
        )
        self._owns_client = client is None

    async def get_market(self, ticker: str) -> KalshiMarket:
        response = await self._get(f"/markets/{ticker}")
        return KalshiMarketResponse.model_validate(response.json()).market

    async def get_order_book(self, ticker: str, depth: int = 100) -> KalshiOrderBook:
        response = await self._get(
            f"/markets/{ticker}/orderbook",
            params={"depth": depth},
        )
        return KalshiOrderBookResponse.model_validate(response.json()).orderbook_fp

    async def get_market_snapshot(
        self,
        ticker: str,
    ) -> tuple[KalshiMarket, MarketSnapshot]:
        market, order_book = await asyncio.gather(
            self.get_market(ticker),
            self.get_order_book(ticker),
        )
        observed_at = datetime.now(UTC)
        return market, to_market_snapshot(market, order_book, observed_at=observed_at)

    async def list_markets(
        self,
        *,
        status: Literal["open", "closed", "settled", "unopened"] = "open",
        series_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiMarketsResponse:
        params: dict[str, str | int | bool] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        response = await self._get("/markets", params=params)
        return KalshiMarketsResponse.model_validate(response.json())

    async def get_historical_market(self, ticker: str) -> KalshiMarket:
        response = await self._get(f"/historical/markets/{ticker}")
        return KalshiMarketResponse.model_validate(response.json()).market

    async def list_historical_markets(
        self,
        *,
        tickers: list[str] | None = None,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiMarketsResponse:
        filters = (bool(tickers), event_ticker is not None, series_ticker is not None)
        if sum(filters) > 1:
            raise ValueError("historical market filters are mutually exclusive")
        params: dict[str, str | int | bool] = {"limit": limit}
        if tickers:
            params["tickers"] = ",".join(tickers)
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        response = await self._get("/historical/markets", params=params)
        return KalshiMarketsResponse.model_validate(response.json())

    async def get_historical_candlesticks(
        self,
        ticker: str,
        *,
        start_ts: int,
        end_ts: int,
        period_interval: CandlestickPeriod,
    ) -> list[KalshiCandlestick]:
        if start_ts > end_ts:
            raise ValueError("candlestick start timestamp must not exceed end timestamp")
        response = await self._get(
            f"/historical/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        payload = KalshiCandlesticksResponse.model_validate(response.json())
        if payload.ticker != ticker:
            raise KalshiAPIError(
                f"Kalshi returned candlesticks for {payload.ticker} while requesting {ticker}"
            )
        return payload.candlesticks

    async def get_series_fee_changes(
        self,
        series_ticker: str,
        *,
        show_historical: bool = True,
    ) -> list[KalshiSeriesFeeChange]:
        response = await self._get(
            "/series/fee_changes",
            params={
                "series_ticker": series_ticker,
                "show_historical": show_historical,
            },
        )
        return KalshiSeriesFeeChangesResponse.model_validate(response.json()).series_fee_change_arr

    async def get_event_fee_changes(
        self,
        event_ticker: str,
        *,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> KalshiEventFeeChangesResponse:
        params: dict[str, str | int | bool] = {
            "event_ticker": event_ticker,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        response = await self._get("/events/fee_changes", params=params)
        return KalshiEventFeeChangesResponse.model_validate(response.json())

    async def get_event_live_data(
        self,
        event_ticker: str,
        *,
        range_hint: str | None = None,
    ) -> KalshiEventLiveData:
        params: dict[str, str | int | bool] = {}
        if range_hint:
            params["range"] = range_hint
        response = await self._get(
            f"/live_data/events/{event_ticker}",
            params=params or None,
        )
        return KalshiEventLiveDataResponse.model_validate(response.json()).live_data

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str | int | bool] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise KalshiAPIError(f"Kalshi request failed for {path}: {exc}") from exc
        return response
