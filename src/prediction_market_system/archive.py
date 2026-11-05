"""Conservative archive interval completeness, not a claim of historical availability."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from prediction_market_system.research import SpotCandle
from prediction_market_system.venues.kalshi import KalshiCandlestick, KalshiMarket

ARCHIVE_COVERAGE_VERSION = 2


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
    return not archive_window_issues(
        start=start,
        end=end,
        period_minutes=period_minutes,
        discovery_complete=discovery_complete,
        markets=markets,
        candlesticks=candlesticks,
        spot_candles=spot_candles,
        history_hours=history_hours,
    )


def archive_window_issues(
    *,
    start: datetime,
    end: datetime,
    period_minutes: int,
    discovery_complete: bool,
    markets: Sequence[KalshiMarket],
    candlesticks: dict[str, list[KalshiCandlestick]],
    spot_candles: Sequence[SpotCandle],
    history_hours: int,
) -> list[str]:
    """Explain the same strict certificate used by ``archive_window_complete``.

    ``markets`` is the requested, outcome-independent history sample, never a
    universe inferred from the candles that happened to be returned.
    """
    issues: list[str] = []
    if not discovery_complete:
        issues.append("Kalshi event discovery incomplete")
    if end <= start or period_minutes <= 0:
        return [*issues, "Invalid archive interval or candle period"]
    step = period_minutes * 60
    expected_spot = set(range(int(start.timestamp()) + step, int(end.timestamp()) + 1, step))
    actual_spot = [int(candle.end_at.timestamp()) for candle in spot_candles]
    spot_issue = _interval_issue("Coinbase spot", expected_spot, actual_spot)
    if spot_issue is not None:
        issues.append(spot_issue)
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
        issue = _interval_issue(f"Kalshi {market.ticker}", expected, actual)
        if not expected:
            issues.append(f"Kalshi {market.ticker}: no completed intervals in requested window")
        elif issue is not None:
            issues.append(issue)
    return issues


def _interval_issue(provider: str, expected: set[int], actual: list[int]) -> str | None:
    actual_set = set(actual)
    if actual_set == expected and len(actual) == len(expected):
        return None
    missing = expected - actual_set
    unexpected = actual_set - expected
    details = [
        f"expected={len(expected)}",
        f"received={len(actual)}",
        f"missing={len(missing)}",
        f"unexpected={len(unexpected)}",
        f"duplicates={len(actual) - len(actual_set)}",
    ]
    if missing:
        details.extend(
            (
                f"first_missing_end={datetime.fromtimestamp(min(missing), UTC).isoformat()}",
                f"last_missing_end={datetime.fromtimestamp(max(missing), UTC).isoformat()}",
            )
        )
    return f"{provider}: " + ", ".join(details)
