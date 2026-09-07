"""Conservative archive interval completeness, not a claim of historical availability."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from prediction_market_system.research import SpotCandle
from prediction_market_system.venues.kalshi import KalshiCandlestick, KalshiMarket


def archive_window_complete(
    *,
    start: datetime,
    end: datetime,
    period_minutes: int,
    discovery_complete: bool,
    markets: Sequence[KalshiMarket],
    candlesticks: dict[str, list[KalshiCandlestick]],
    spot_candles: Sequence[SpotCandle],
    history_hours: int,
) -> bool:
    if not discovery_complete or end <= start:
        return False
    step = period_minutes * 60
    expected_spot = set(range(int(start.timestamp()) + step, int(end.timestamp()) + 1, step))
    actual_spot = [int(candle.end_at.timestamp()) for candle in spot_candles]
    if set(actual_spot) != expected_spot or len(actual_spot) != len(expected_spot):
        return False
    for market in markets:
        candle_start = max(start, market.close_time - timedelta(hours=history_hours))
        if market.open_time is not None:
            candle_start = max(candle_start, market.open_time)
        candle_end = min(end, market.close_time)
        first_boundary = (int(candle_start.timestamp()) // step + 1) * step
        last_boundary = int(candle_end.timestamp()) // step * step
        expected = set(range(first_boundary, last_boundary + 1, step))
        actual = [
            candle.end_period_ts
            for candle in candlesticks.get(market.ticker, [])
            if candle_start < datetime.fromtimestamp(candle.end_period_ts, UTC) <= candle_end
        ]
        if not expected or set(actual) != expected or len(actual) != len(expected):
            return False
    return True
