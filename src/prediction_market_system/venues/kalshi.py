from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal, Self

import httpx
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from prediction_market_system.domain import (
    CryptoPriceContract,
    MarketSnapshot,
    TerminalRangeContract,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.transport import DEFAULT_TRANSIENT_RETRIES, get_with_transient_retry

KALSHI_PRODUCTION_API = "https://external-api.kalshi.com/trade-api/v2"
CandlestickPeriod = Literal[1, 60, 1440]
FeeType = Literal["quadratic", "quadratic_with_maker_fees", "flat"]


class KalshiAPIError(RuntimeError):
    pass


class KalshiNotFoundError(KalshiAPIError):
    pass


class IncompleteOrderBookError(ValueError):
    pass


class UnsupportedMarketError(ValueError):
    pass


def _has_explicit_touch_semantics(rule: str) -> bool:
    normalized = " ".join(rule.casefold().split())
    if ("average" in normalized or "averaging" in normalized) and not any(
        marker in normalized for marker in ("at any time", "at any point", "touch", "reach")
    ):
        # "Sixty seconds before expiry" describes a terminal benchmark window,
        # not the event that the underlying crosses a barrier before expiry.
        return False
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


_FIXED_OBSERVATION_TIME = re.compile(
    r"\bat\s+(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)"
    r"|\d{1,2}:\d{2}|noon|midnight)\b",
    re.IGNORECASE,
)
_DATED_OBSERVATION_TIME = re.compile(
    r"\bat\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<period>AM|PM)\s+(?P<zone>UTC|GMT|EST|EDT)\s+on\s+"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)


def _has_explicit_terminal_semantics(rule: str) -> bool:
    normalized = " ".join(rule.casefold().split())
    terminal_markers = (
        "at expiration",
        "at expiry",
        "at the market close",
        "when the market closes",
        "closing value at",
    )
    return any(marker in normalized for marker in terminal_markers) or bool(
        _FIXED_OBSERVATION_TIME.search(normalized)
    )


def _settlement_averaging_window_seconds(rules: tuple[str, ...]) -> int:
    normalized = " ".join(" ".join(rule.casefold().split()) for rule in rules)
    if not any(word in normalized for word in ("average", "averaging", "mean", "median")):
        return 0
    if any(word in normalized for word in ("weighted", "geometric", "median")):
        raise UnsupportedMarketError("only explicit simple arithmetic averaging is supported")
    one_minute = any(
        marker in normalized
        for marker in (
            "average of the sixty seconds",
            "average of sixty seconds",
            "average of the 60 seconds",
            "average of 60 seconds",
        )
    ) or (
        any(marker in normalized for marker in ("60 index prices", "60 rti prices"))
        and any(marker in normalized for marker in ("last minute", "sixty seconds", "60 seconds"))
    )
    durations = re.findall(
        r"\b(\d+|one|two|three|four|five|ten|thirty|sixty|ninety)\s+(seconds?|minutes?)\b",
        normalized,
    )
    conflicting_window = any(
        (unit.startswith("second") and count not in {"60", "sixty"})
        or (unit.startswith("minute") and count not in {"1", "one"})
        for count, unit in durations
    )
    if not one_minute or conflicting_window or "before" not in normalized:
        raise UnsupportedMarketError("ambiguous or unsupported settlement averaging window")
    return 60


class _KalshiModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class KalshiMarket(_KalshiModel):
    ticker: str
    event_ticker: str
    series_ticker: str = ""
    market_type: str
    title: str = ""
    yes_sub_title: str
    no_sub_title: str
    event_title: str = ""
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

    @field_validator(
        "close_time",
        "expected_expiration_time",
        "latest_expiration_time",
        "created_time",
        "updated_time",
        "open_time",
        "settlement_ts",
    )
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Kalshi timestamps must include a timezone")
        return value.astimezone(UTC)

    @property
    def expiry(self) -> datetime:
        return self.close_time

    @property
    def observation_end_at(self) -> datetime:
        rule = self.resolution_rule
        fixed = list(_FIXED_OBSERVATION_TIME.finditer(rule))
        if not fixed:
            if not (_has_explicit_terminal_semantics(rule) or _has_explicit_touch_semantics(rule)):
                raise UnsupportedMarketError("explicit terminal observation or touch end required")
            return self.close_time
        matches = list(_DATED_OBSERVATION_TIME.finditer(rule))
        if len(matches) != len(fixed):
            raise UnsupportedMarketError(
                "fixed observation clock requires explicit date and timezone"
            )
        instants: set[datetime] = set()
        for match in matches:
            hour = int(match["hour"])
            if not 1 <= hour <= 12:
                raise UnsupportedMarketError("invalid fixed observation hour")
            hour = hour % 12 + (12 if match["period"].upper() == "PM" else 0)
            offset = {"UTC": 0, "GMT": 0, "EST": -5, "EDT": -4}[match["zone"].upper()]
            try:
                month_text = match["month"]
                month_format = "%b" if len(month_text) == 3 else "%B"
                month = datetime.strptime(month_text, month_format).month
                instant = datetime(
                    int(match["year"]),
                    month,
                    int(match["day"]),
                    hour,
                    int(match["minute"] or 0),
                    tzinfo=timezone(timedelta(hours=offset)),
                ).astimezone(UTC)
            except ValueError as error:
                raise UnsupportedMarketError("invalid fixed observation date") from error
            instants.add(instant)
        if len(instants) != 1:
            raise UnsupportedMarketError("conflicting fixed observation clocks")
        return instants.pop()

    @property
    def question(self) -> str:
        event_title = self.event_title.strip()
        if event_title:
            return event_title
        title = self.title.strip()
        yes_title = self.yes_sub_title.strip()
        if title and yes_title and yes_title.casefold() not in title.casefold():
            return f"{title} — {yes_title}"
        return title or yes_title

    @property
    def resolution_rule(self) -> str:
        rules = [rule.strip() for rule in (self.rules_primary, self.rules_secondary) if rule]
        return "\n\n".join(rules)

    @property
    def normalized_series_ticker(self) -> str:
        return self.series_ticker.strip().upper() or self.event_ticker.split("-", 1)[0].upper()

    @property
    def market_url(self) -> HttpUrl:
        event_slug = self.event_ticker.lower()
        return HttpUrl(
            f"https://kalshi.com/markets/{self.normalized_series_ticker.lower()}/"
            f"bitcoin-range/{event_slug}"
        )

    @property
    def contract_label(self) -> str:
        return self.yes_sub_title.strip() or self.question

    def evaluation_contract(self, as_of: datetime) -> CryptoPriceContract:
        """Identical point-in-time eligibility for live reads and historical replay."""
        if self.market_type != "binary":
            raise UnsupportedMarketError("only binary Kalshi markets are supported")
        if self.notional_value_dollars != Decimal("1"):
            raise UnsupportedMarketError("only $1 payout contracts are supported")
        if self.status not in {"active", "initialized"} or self.result:
            raise UnsupportedMarketError(f"Kalshi market is not active: {self.status}")
        if any(
            value is not None and value > as_of for value in (self.created_time, self.updated_time)
        ):
            raise UnsupportedMarketError("market metadata is from the future")
        if self.open_time is not None and self.open_time > as_of:
            raise UnsupportedMarketError("market has not opened")
        if min(self.close_time, self.observation_end_at) <= as_of:
            raise UnsupportedMarketError("market trading or observation period has ended")
        contract = self.price_contract()
        if (
            isinstance(contract, ThresholdContract)
            and contract.model_kind is ThresholdModelKind.BARRIER
        ):
            # A spot quote cannot show whether the barrier was already crossed since the
            # contractual observation start. Kalshi exposes no benchmark path, so the
            # classification is preserved but the market is not evaluable.
            raise UnsupportedMarketError(
                "touch barrier requires benchmark path history from the contractual "
                "observation start; unavailable from Kalshi market data"
            )
        return contract

    def _validate_observation_clock(self) -> None:
        """Fail closed on missing, ambiguous, or conflicting fixed observation clocks."""
        _ = self.observation_end_at

    def price_contract(self) -> CryptoPriceContract:
        rules = tuple(rule for rule in (self.rules_primary, self.rules_secondary) if rule.strip())
        has_touch_semantics = any(_has_explicit_touch_semantics(rule) for rule in rules)
        has_terminal_semantics = any(_has_explicit_terminal_semantics(rule) for rule in rules)
        if has_touch_semantics and has_terminal_semantics:
            raise UnsupportedMarketError("conflicting terminal and touch resolution semantics")
        window_seconds = _settlement_averaging_window_seconds(rules)

        if self.strike_type == "between":
            if (
                self.floor_strike is None
                or self.cap_strike is None
                or self.floor_strike <= 0
                or self.floor_strike >= self.cap_strike
            ):
                raise UnsupportedMarketError(
                    "positive, increasing lower and upper bounds are required for a range market"
                )
            if has_touch_semantics or not has_terminal_semantics:
                raise UnsupportedMarketError(
                    "range markets require an explicit fixed-time terminal observation"
                )
            self._validate_observation_clock()
            return TerminalRangeContract(
                lower_bound=self.floor_strike,
                upper_bound=self.cap_strike,
                settlement_window_seconds=window_seconds,
            )

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

        strike = metadata_strike
        if strike is None or strike <= 0:
            raise UnsupportedMarketError("a positive threshold strike is required for this market")

        if has_touch_semantics:
            if window_seconds:
                raise UnsupportedMarketError("touch barriers cannot use terminal averaging")
            model_kind = ThresholdModelKind.BARRIER
        elif not has_terminal_semantics:
            raise UnsupportedMarketError(
                "rules define neither an explicit terminal observation "
                "nor a supported touch barrier"
            )
        else:
            model_kind = ThresholdModelKind.TERMINAL
        self._validate_observation_clock()

        return ThresholdContract(
            model_kind=model_kind,
            direction=direction,
            strike_price=strike,
            settlement_window_seconds=window_seconds,
        )


class KalshiMarketResponse(_KalshiModel):
    market: KalshiMarket


class KalshiMarketsResponse(_KalshiModel):
    markets: list[KalshiMarket]
    cursor: str = ""


class KalshiEvent(_KalshiModel):
    event_ticker: str
    series_ticker: str
    title: str
    sub_title: str
    mutually_exclusive: bool
    strike_date: datetime | None = None


class KalshiEventsResponse(_KalshiModel):
    events: list[KalshiEvent]
    cursor: str = ""


class KalshiEventResponse(_KalshiModel):
    event: KalshiEvent
    markets: list[KalshiMarket]


class KalshiBidAskDistribution(_KalshiModel):
    open: Decimal = Field(validation_alias=AliasChoices("open", "open_dollars"))
    low: Decimal = Field(validation_alias=AliasChoices("low", "low_dollars"))
    high: Decimal = Field(validation_alias=AliasChoices("high", "high_dollars"))
    close: Decimal = Field(validation_alias=AliasChoices("close", "close_dollars"))


class KalshiPriceDistribution(_KalshiModel):
    open: Decimal | None = Field(
        default=None, validation_alias=AliasChoices("open", "open_dollars")
    )
    low: Decimal | None = Field(default=None, validation_alias=AliasChoices("low", "low_dollars"))
    high: Decimal | None = Field(
        default=None, validation_alias=AliasChoices("high", "high_dollars")
    )
    close: Decimal | None = Field(
        default=None, validation_alias=AliasChoices("close", "close_dollars")
    )
    mean: Decimal | None = Field(
        default=None, validation_alias=AliasChoices("mean", "mean_dollars")
    )
    previous: Decimal | None = Field(
        default=None, validation_alias=AliasChoices("previous", "previous_dollars")
    )


class KalshiCandlestick(_KalshiModel):
    end_period_ts: int
    yes_bid: KalshiBidAskDistribution
    yes_ask: KalshiBidAskDistribution
    price: KalshiPriceDistribution
    volume: Decimal = Field(validation_alias=AliasChoices("volume", "volume_fp"))
    open_interest: Decimal = Field(
        validation_alias=AliasChoices("open_interest", "open_interest_fp")
    )


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
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    yes_ask_size: float | None
    no_ask_size: float | None


def normalize_order_book(order_book: KalshiOrderBook) -> ExecutableBook:
    if not order_book.yes_dollars and not order_book.no_dollars:
        raise IncompleteOrderBookError("at least one bid is required to derive an executable ask")

    yes_level = (
        max(order_book.yes_dollars, key=lambda level: level[0]) if order_book.yes_dollars else None
    )
    no_level = (
        max(order_book.no_dollars, key=lambda level: level[0]) if order_book.no_dollars else None
    )
    yes_bid_price, yes_bid_size = yes_level if yes_level is not None else (None, None)
    no_bid_price, no_bid_size = no_level if no_level is not None else (None, None)
    yes_ask_price = Decimal("1") - no_bid_price if no_bid_price is not None else None
    no_ask_price = Decimal("1") - yes_bid_price if yes_bid_price is not None else None

    if (
        yes_bid_price is not None and yes_ask_price is not None and yes_bid_price > yes_ask_price
    ) or (no_bid_price is not None and no_ask_price is not None and no_bid_price > no_ask_price):
        raise ValueError("Kalshi order book is crossed or internally inconsistent")

    return ExecutableBook(
        yes_bid=None if yes_bid_price is None else float(yes_bid_price),
        yes_ask=None if yes_ask_price is None else float(yes_ask_price),
        no_bid=None if no_bid_price is None else float(no_bid_price),
        no_ask=None if no_ask_price is None else float(no_ask_price),
        yes_ask_size=None if no_bid_size is None else float(no_bid_size),
        no_ask_size=None if yes_bid_size is None else float(yes_bid_size),
    )


def to_market_snapshot(
    market: KalshiMarket,
    order_book: KalshiOrderBook,
    *,
    observed_at: datetime,
) -> MarketSnapshot:
    contract = market.evaluation_contract(observed_at)
    observation_end = market.observation_end_at
    book = normalize_order_book(order_book)
    return MarketSnapshot(
        market_id=market.ticker,
        question=market.question,
        venue="Kalshi",
        observed_at=observed_at,
        source_metadata=market.model_dump(mode="json"),
        expires_at=market.close_time,
        observation_end_at=observation_end,
        observation_start_at=(
            observation_end - timedelta(seconds=contract.settlement_window_seconds)
            if contract.settlement_window_seconds
            else None
        ),
        expected_settlement_at=market.expected_expiration_time,
        yes_bid=book.yes_bid,
        yes_ask=book.yes_ask,
        no_bid=book.no_bid,
        no_ask=book.no_ask,
        yes_ask_size=book.yes_ask_size,
        no_ask_size=book.no_ask_size,
        resolution_rule=market.resolution_rule,
        series_id=market.normalized_series_ticker,
        event_id=market.event_ticker,
        contract_label=market.contract_label,
        market_url=market.market_url,
    )


class KalshiClient:
    def __init__(
        self,
        *,
        base_url: str = KALSHI_PRODUCTION_API,
        client: httpx.AsyncClient | None = None,
        max_rate_limit_retries: int = 5,
        max_transient_retries: int = DEFAULT_TRANSIENT_RETRIES,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=10.0,
            headers={"User-Agent": "prediction-market-system/0.1.0"},
        )
        self._owns_client = client is None
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries must be non-negative")
        if max_transient_retries < 0:
            raise ValueError("max_transient_retries must be non-negative")
        self._max_rate_limit_retries = max_rate_limit_retries
        self._max_transient_retries = max_transient_retries
        self._sleep = sleep

    async def list_events(
        self,
        *,
        status: Literal["unopened", "open", "closed", "settled"] | None = None,
        series_ticker: str | None = None,
        min_close_ts: int | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> KalshiEventsResponse:
        params: dict[str, str | int | bool] = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if cursor:
            params["cursor"] = cursor
        response = await self._get("/events", params=params)
        return KalshiEventsResponse.model_validate(response.json())

    async def get_event(self, event_ticker: str) -> KalshiEventResponse:
        response = await self._get(f"/events/{event_ticker}")
        return KalshiEventResponse.model_validate(response.json())

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

    async def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        *,
        start_ts: int,
        end_ts: int,
        period_interval: CandlestickPeriod,
    ) -> list[KalshiCandlestick]:
        try:
            return await self.get_historical_candlesticks(
                ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
            )
        except KalshiNotFoundError:
            response = await self._get(
                f"/series/{series_ticker}/markets/{ticker}/candlesticks",
                params={
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "period_interval": period_interval,
                },
            )
            return self._parse_candlesticks(response, ticker)

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
        return self._parse_candlesticks(response, ticker)

    @staticmethod
    def _parse_candlesticks(
        response: httpx.Response,
        ticker: str,
    ) -> list[KalshiCandlestick]:
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
        for attempt in range(self._max_rate_limit_retries + 1):
            try:
                response = await get_with_transient_retry(
                    self._client,
                    path,
                    params=params,
                    max_retries=self._max_transient_retries,
                    sleep=self._sleep,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == httpx.codes.NOT_FOUND:
                    raise KalshiNotFoundError(f"Kalshi request failed for {path}: {exc}") from exc
                retryable = exc.response.status_code == httpx.codes.TOO_MANY_REQUESTS
                if retryable and attempt < self._max_rate_limit_retries:
                    retry_after = exc.response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after is not None else 2**attempt
                    except ValueError:
                        delay = 2**attempt
                    await self._sleep(max(delay, 0.0))
                    continue
                raise KalshiAPIError(f"Kalshi request failed for {path}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise KalshiAPIError(f"Kalshi request failed for {path}: {exc}") from exc
        raise RuntimeError("unreachable Kalshi retry state")
