from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prediction_market_system.domain import MarketSnapshot

KALSHI_PRODUCTION_API = "https://external-api.kalshi.com/trade-api/v2"


class KalshiAPIError(RuntimeError):
    pass


class IncompleteOrderBookError(ValueError):
    pass


class UnsupportedMarketError(ValueError):
    pass


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

    def terminal_threshold_strike(self, override: float | None = None) -> float:
        if self.can_close_early:
            raise UnsupportedMarketError(
                "market can close early and may be path-dependent; "
                "the terminal threshold model must not evaluate it"
            )
        if self.strike_type not in {"greater", "greater_or_equal"}:
            raise UnsupportedMarketError(
                f"unsupported Kalshi strike type: {self.strike_type or 'missing'}"
            )
        strike = override if override is not None else self.floor_strike
        if strike is None or strike <= 0:
            raise UnsupportedMarketError("a positive threshold strike is required for this market")
        return strike


class KalshiMarketResponse(_KalshiModel):
    market: KalshiMarket


class KalshiMarketsResponse(_KalshiModel):
    markets: list[KalshiMarket]
    cursor: str = ""


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
        params: dict[str, str | int] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        response = await self._get("/markets", params=params)
        return KalshiMarketsResponse.model_validate(response.json())

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise KalshiAPIError(f"Kalshi request failed for {path}: {exc}") from exc
        return response
