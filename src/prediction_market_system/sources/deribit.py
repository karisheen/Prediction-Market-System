from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prediction_market_system.research import (
    DerivativesSnapshot,
    FundingObservation,
    VolatilityObservation,
)

DERIBIT_PUBLIC_API = "https://www.deribit.com/api/v2/public"
_DERIBIT_DVOL_RESOLUTIONS = {1, 60, 3600, 86400}
_FUNDING_PAGE = timedelta(days=30)


class DeribitDataError(RuntimeError):
    pass


class _DeribitModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _FundingRow(_DeribitModel):
    timestamp: int
    index_price: float = Field(gt=0.0)
    prev_index_price: float = Field(gt=0.0)
    interest_1h: float
    interest_8h: float


class _Ticker(_DeribitModel):
    timestamp: int
    instrument_name: str
    index_price: float = Field(gt=0.0)
    mark_price: float = Field(gt=0.0)
    open_interest: float = Field(ge=0.0)
    current_funding: float
    funding_8h: float


class DeribitClient:
    def __init__(
        self,
        *,
        base_url: str = DERIBIT_PUBLIC_API,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=15.0,
            headers={"User-Agent": "prediction-market-system/0.1.0"},
        )
        self._owns_client = client is None

    async def get_dvol_history(
        self,
        currency: str,
        *,
        start_at: datetime,
        end_at: datetime,
        resolution_seconds: int,
    ) -> list[VolatilityObservation]:
        start_at = _as_utc(start_at)
        end_at = _as_utc(end_at)
        if end_at <= start_at:
            raise ValueError("DVOL end must be after start")
        if resolution_seconds not in _DERIBIT_DVOL_RESOLUTIONS:
            supported = ", ".join(str(value) for value in sorted(_DERIBIT_DVOL_RESOLUTIONS))
            raise ValueError(f"unsupported DVOL resolution; use one of: {supported}")

        requested_end_ms = int(end_at.timestamp() * 1000)
        next_start_ms = int(start_at.timestamp() * 1000)
        observations: dict[datetime, VolatilityObservation] = {}
        while next_start_ms < requested_end_ms:
            result, retrieved_at = await self._get(
                "/get_volatility_index_data",
                params={
                    "currency": currency.upper(),
                    "start_timestamp": next_start_ms,
                    "end_timestamp": requested_end_ms,
                    "resolution": resolution_seconds,
                },
            )
            if not isinstance(result, dict):
                raise DeribitDataError("Deribit DVOL result is not an object")
            rows = result.get("data", [])
            if not isinstance(rows, list):
                raise DeribitDataError("Deribit DVOL data is not a list")
            for row in rows:
                if not isinstance(row, list) or len(row) != 5:
                    raise DeribitDataError("Deribit DVOL row has an unexpected shape")
                timestamp_ms, open_value, high, low, close = row
                source_start = datetime.fromtimestamp(float(timestamp_ms) / 1000, UTC)
                observed_at = source_start + timedelta(seconds=resolution_seconds)
                if observed_at > end_at:
                    continue
                observations[observed_at] = VolatilityObservation(
                    provider="deribit",
                    symbol=currency.upper(),
                    kind="implied",
                    window_seconds=resolution_seconds,
                    source_start_at=source_start,
                    observed_at=observed_at,
                    annualized_volatility=float(close) / 100,
                    retrieved_at=retrieved_at,
                    raw_payload={
                        "timestamp_ms": int(timestamp_ms),
                        "open": open_value,
                        "high": high,
                        "low": low,
                        "close": close,
                    },
                )

            continuation = result.get("continuation")
            if continuation is None:
                break
            continuation_ms = int(continuation)
            if continuation_ms <= next_start_ms:
                raise DeribitDataError("Deribit repeated a DVOL continuation timestamp")
            next_start_ms = continuation_ms

        return [observations[observed_at] for observed_at in sorted(observations)]

    async def get_funding_history(
        self,
        instrument_name: str,
        *,
        start_at: datetime,
        end_at: datetime,
    ) -> list[FundingObservation]:
        start_at = _as_utc(start_at)
        end_at = _as_utc(end_at)
        if end_at <= start_at:
            raise ValueError("funding end must be after start")

        observations: dict[datetime, FundingObservation] = {}
        page_start = start_at
        while page_start < end_at:
            page_end = min(end_at, page_start + _FUNDING_PAGE)
            result, retrieved_at = await self._get(
                "/get_funding_rate_history",
                params={
                    "instrument_name": instrument_name.upper(),
                    "start_timestamp": int(page_start.timestamp() * 1000),
                    "end_timestamp": int(page_end.timestamp() * 1000),
                },
            )
            if not isinstance(result, list):
                raise DeribitDataError("Deribit funding result is not a list")
            for payload in result:
                row = _FundingRow.model_validate(payload)
                observed_at = datetime.fromtimestamp(row.timestamp / 1000, UTC)
                if observed_at > end_at:
                    continue
                observations[observed_at] = FundingObservation(
                    provider="deribit",
                    instrument_name=instrument_name.upper(),
                    observed_at=observed_at,
                    index_price=row.index_price,
                    previous_index_price=row.prev_index_price,
                    funding_rate_1h=row.interest_1h,
                    funding_rate_8h=row.interest_8h,
                    retrieved_at=retrieved_at,
                    raw_payload=dict(payload),
                )
            page_start = page_end

        return [observations[observed_at] for observed_at in sorted(observations)]

    async def get_derivatives_snapshot(self, instrument_name: str) -> DerivativesSnapshot:
        result, retrieved_at = await self._get(
            "/ticker",
            params={"instrument_name": instrument_name.upper()},
        )
        ticker = _Ticker.model_validate(result)
        return DerivativesSnapshot(
            provider="deribit",
            instrument_name=ticker.instrument_name,
            observed_at=datetime.fromtimestamp(ticker.timestamp / 1000, UTC),
            index_price=ticker.index_price,
            mark_price=ticker.mark_price,
            basis=(ticker.mark_price / ticker.index_price) - 1,
            open_interest=ticker.open_interest,
            current_funding=ticker.current_funding,
            funding_rate_8h=ticker.funding_8h,
            retrieved_at=retrieved_at,
            raw_payload=dict(result),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str | int],
    ) -> tuple[Any, datetime]:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DeribitDataError(f"Deribit request failed for {path}: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise DeribitDataError(f"Deribit response for {path} is not an object")
        if payload.get("error") is not None:
            raise DeribitDataError(f"Deribit error for {path}: {payload['error']}")
        if "result" not in payload:
            raise DeribitDataError(f"Deribit response for {path} omitted result")
        return payload["result"], datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)
