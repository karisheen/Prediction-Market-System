"""Bounded retry for transient transport failures shared by every HTTP data client."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import httpx

DEFAULT_TRANSIENT_RETRIES = 3
_MAX_TRANSIENT_DELAY_SECONDS = 8.0

SleepFn = Callable[[float], Awaitable[None]]


def transient_retry_delay(attempt: int) -> float:
    return float(min(2**attempt, _MAX_TRANSIENT_DELAY_SECONDS))


async def get_with_transient_retry(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: Mapping[str, str | int | bool] | None,
    max_retries: int,
    sleep: SleepFn = asyncio.sleep,
) -> httpx.Response:
    """Issue a GET, retrying DNS, connection, and timeout failures with bounded backoff.

    ``httpx.TransportError`` covers failures that never produced an HTTP response, which
    is what a host sees for a few seconds after waking or booting before its network
    comes up. HTTP status errors are not retried here; callers keep their own policy.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    for attempt in range(max_retries + 1):
        try:
            return await client.get(path, params=params)
        except httpx.TransportError:
            if attempt >= max_retries:
                raise
            await sleep(transient_retry_delay(attempt))
    raise RuntimeError("unreachable transient retry state")
