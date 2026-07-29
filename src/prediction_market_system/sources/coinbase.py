from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from prediction_market_system.research import SpotCandle

COINBASE_MARKET_API = "https://api.coinbase.com/api/v3/brokerage/market"
COINBASE_GRANULARITIES = {
    60: "ONE_MINUTE",
    300: "FIVE_MINUTE",
    900: "FIFTEEN_MINUTE",
    1800: "THIRTY_MINUTE",
    3600: "ONE_HOUR",
    7200: "TWO_HOUR",
    21600: "SIX_HOUR",
    86400: "ONE_DAY",
}


class CoinbaseDataError(RuntimeError):
    pass


class _CoinbaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _CoinbaseCandle(_CoinbaseModel):
    start: int
    low: Decimal
    high: Decimal
    open: Decimal
    close: Decimal
    volume: Decimal


class _CoinbaseCandlesResponse(_CoinbaseModel):
    candles: list[_CoinbaseCandle]


class CoinbaseClient:
    def __init__(
        self,
        *,
        base_url: str = COINBASE_MARKET_API,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=15.0,
            headers={"User-Agent": "prediction-market-system/0.1.0"},
        )
        self._owns_client = client is None

    async def get_candles(
        self,
        product_id: str,
        *,
        start_at: datetime,
        end_at: datetime,
        interval_seconds: int,
    ) -> list[SpotCandle]:
        start_at = _as_utc(start_at)
        end_at = _as_utc(end_at)
        if end_at <= start_at:
            raise ValueError("candle end must be after start")
        try:
            granularity = COINBASE_GRANULARITIES[interval_seconds]
        except KeyError as exc:
            supported = ", ".join(str(value) for value in COINBASE_GRANULARITIES)
            raise ValueError(f"unsupported Coinbase interval; use one of: {supported}") from exc

        requested_start = int(start_at.timestamp())
        requested_end = int(end_at.timestamp())
        latest_start = requested_end - interval_seconds
        page_start = requested_start
        candles_by_start: dict[int, SpotCandle] = {}
        while page_start <= latest_start:
            page_end = min(latest_start, page_start + (349 * interval_seconds))
            payload, retrieved_at = await self._get_candle_page(
                product_id.upper(),
                start=page_start,
                end=page_end,
                granularity=granularity,
            )
            for candle in payload.candles:
                candle_end = candle.start + interval_seconds
                if candle.start < requested_start or candle_end > requested_end:
                    continue
                raw_payload: dict[str, Any] = candle.model_dump(mode="json")
                candles_by_start[candle.start] = SpotCandle(
                    provider="coinbase",
                    product_id=product_id.upper(),
                    interval_seconds=interval_seconds,
                    start_at=datetime.fromtimestamp(candle.start, UTC),
                    end_at=datetime.fromtimestamp(candle_end, UTC),
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    retrieved_at=retrieved_at,
                    raw_payload=raw_payload,
                )
            page_start = page_end + interval_seconds

        return [candles_by_start[start] for start in sorted(candles_by_start)]

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_candle_page(
        self,
        product_id: str,
        *,
        start: int,
        end: int,
        granularity: str,
    ) -> tuple[_CoinbaseCandlesResponse, datetime]:
        path = f"/products/{product_id}/candles"
        try:
            response = await self._client.get(
                path,
                params={
                    "start": str(start),
                    "end": str(end),
                    "granularity": granularity,
                    "limit": 350,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CoinbaseDataError(f"Coinbase request failed for {path}: {exc}") from exc
        return _CoinbaseCandlesResponse.model_validate(response.json()), datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)
