from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from prediction_market_system.research import (
    DerivativesSnapshot,
    EventDataSnapshot,
    FundingObservation,
    ResearchContext,
    ResearchDataUnavailable,
    ResearchSyncStatus,
    SpotCandle,
    VolatilityObservation,
    calculate_realized_volatility,
)

ResearchObservation = TypeVar(
    "ResearchObservation",
    VolatilityObservation,
    FundingObservation,
    DerivativesSnapshot,
    EventDataSnapshot,
)

RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS crypto_spot_candles (
    provider TEXT NOT NULL,
    product_id TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (provider, product_id, interval_seconds, start_at)
);

CREATE INDEX IF NOT EXISTS idx_spot_candles_asof
ON crypto_spot_candles (provider, product_id, interval_seconds, end_at);

CREATE TABLE IF NOT EXISTS crypto_volatility_observations (
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    window_seconds INTEGER NOT NULL,
    source_start_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (provider, symbol, kind, window_seconds, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_volatility_asof
ON crypto_volatility_observations (symbol, kind, observed_at);

CREATE TABLE IF NOT EXISTS crypto_funding_observations (
    provider TEXT NOT NULL,
    instrument_name TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (provider, instrument_name, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_funding_asof
ON crypto_funding_observations (provider, instrument_name, observed_at);

CREATE TABLE IF NOT EXISTS crypto_derivatives_snapshots (
    provider TEXT NOT NULL,
    instrument_name TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (provider, instrument_name, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_derivatives_asof
ON crypto_derivatives_snapshots (provider, instrument_name, observed_at);

CREATE TABLE IF NOT EXISTS kalshi_event_data_snapshots (
    provider TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    data_type TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    is_historical INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (provider, event_ticker, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_event_data_asof
ON kalshi_event_data_snapshots (provider, event_ticker, observed_at);

CREATE TABLE IF NOT EXISTS research_data_sync_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_ticker TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    request_json TEXT NOT NULL,
    counts_json TEXT,
    error TEXT
);
"""


@dataclass(frozen=True)
class ResearchWriteResult:
    spot_candles: int = 0
    volatility_observations: int = 0
    funding_observations: int = 0
    derivatives_snapshots: int = 0
    event_snapshots: int = 0


class ResearchRepositoryMixin:
    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def begin_research_sync(
        self,
        *,
        symbol: str,
        event_ticker: str | None,
        request: dict[str, Any],
    ) -> str:
        run_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_data_sync_runs (
                    run_id, status, symbol, event_ticker, started_at, request_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    ResearchSyncStatus.RUNNING.value,
                    symbol.upper(),
                    event_ticker,
                    datetime.now(UTC).isoformat(),
                    json.dumps(request, sort_keys=True),
                ),
            )
        return run_id

    def complete_research_sync(
        self,
        run_id: str,
        *,
        result: ResearchWriteResult | None = None,
        error: str | None = None,
    ) -> None:
        status = ResearchSyncStatus.FAILED if error else ResearchSyncStatus.SUCCEEDED
        counts_json = json.dumps(result.__dict__, sort_keys=True) if result else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE research_data_sync_runs
                SET status = ?, completed_at = ?, counts_json = ?, error = ?
                WHERE run_id = ? AND status = ?
                """,
                (
                    status.value,
                    datetime.now(UTC).isoformat(),
                    counts_json,
                    error[:1000] if error else None,
                    run_id,
                    ResearchSyncStatus.RUNNING.value,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(f"research sync run is missing or already completed: {run_id}")

    def save_research_data(
        self,
        *,
        spot_candles: list[SpotCandle] | None = None,
        volatility_observations: list[VolatilityObservation] | None = None,
        funding_observations: list[FundingObservation] | None = None,
        derivatives_snapshots: list[DerivativesSnapshot] | None = None,
        event_snapshots: list[EventDataSnapshot] | None = None,
    ) -> ResearchWriteResult:
        inserted = {
            "spot_candles": 0,
            "volatility_observations": 0,
            "funding_observations": 0,
            "derivatives_snapshots": 0,
            "event_snapshots": 0,
        }
        with self._connect() as connection:
            for candle in spot_candles or []:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO crypto_spot_candles (
                        provider, product_id, interval_seconds, start_at, end_at,
                        retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candle.provider,
                        candle.product_id,
                        candle.interval_seconds,
                        candle.start_at.isoformat(),
                        candle.end_at.isoformat(),
                        candle.retrieved_at.isoformat(),
                        candle.model_dump_json(),
                    ),
                )
                inserted["spot_candles"] += cursor.rowcount

            for observation in volatility_observations or []:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO crypto_volatility_observations (
                        provider, symbol, kind, window_seconds, source_start_at,
                        observed_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.provider,
                        observation.symbol,
                        observation.kind,
                        observation.window_seconds,
                        observation.source_start_at.isoformat(),
                        observation.observed_at.isoformat(),
                        observation.retrieved_at.isoformat(),
                        observation.model_dump_json(),
                    ),
                )
                inserted["volatility_observations"] += cursor.rowcount

            for funding in funding_observations or []:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO crypto_funding_observations (
                        provider, instrument_name, observed_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        funding.provider,
                        funding.instrument_name,
                        funding.observed_at.isoformat(),
                        funding.retrieved_at.isoformat(),
                        funding.model_dump_json(),
                    ),
                )
                inserted["funding_observations"] += cursor.rowcount

            for snapshot in derivatives_snapshots or []:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO crypto_derivatives_snapshots (
                        provider, instrument_name, observed_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.provider,
                        snapshot.instrument_name,
                        snapshot.observed_at.isoformat(),
                        snapshot.retrieved_at.isoformat(),
                        snapshot.model_dump_json(),
                    ),
                )
                inserted["derivatives_snapshots"] += cursor.rowcount

            for event in event_snapshots or []:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_event_data_snapshots (
                        provider, event_ticker, data_type, observed_at,
                        retrieved_at, is_historical, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.provider,
                        event.event_ticker,
                        event.data_type,
                        event.observed_at.isoformat(),
                        event.retrieved_at.isoformat(),
                        event.is_historical,
                        event.model_dump_json(),
                    ),
                )
                inserted["event_snapshots"] += cursor.rowcount

        return ResearchWriteResult(**inserted)

    def spot_candles_as_of(
        self,
        *,
        symbol: str,
        as_of: datetime,
        interval_seconds: int,
        window_seconds: int,
    ) -> tuple[SpotCandle, ...]:
        as_of = _as_utc(as_of)
        lower_bound = as_of - timedelta(seconds=window_seconds + interval_seconds)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM crypto_spot_candles
                WHERE provider = 'coinbase'
                  AND product_id = ?
                  AND interval_seconds = ?
                  AND end_at <= ?
                  AND end_at >= ?
                ORDER BY end_at ASC
                """,
                (
                    f"{symbol.upper()}-USD",
                    interval_seconds,
                    as_of.isoformat(),
                    lower_bound.isoformat(),
                ),
            ).fetchall()
        return tuple(SpotCandle.model_validate_json(str(row["payload_json"])) for row in rows)

    def research_context_as_of(
        self,
        *,
        symbol: str,
        as_of: datetime,
        event_ticker: str | None = None,
        interval_seconds: int = 3600,
        realized_window_seconds: int = 30 * 24 * 60 * 60,
        spot_max_age_seconds: int | None = None,
        optional_max_age_seconds: int = 2 * 60 * 60,
        event_max_age_seconds: int = 6 * 60 * 60,
    ) -> ResearchContext:
        as_of = _as_utc(as_of)
        symbol = symbol.upper()
        product_id = f"{symbol}-USD"
        instrument_name = f"{symbol}-PERPETUAL"
        spot_max_age = spot_max_age_seconds or (2 * interval_seconds)
        lower_bound = as_of - timedelta(seconds=realized_window_seconds + interval_seconds)

        with self._connect() as connection:
            spot_rows = connection.execute(
                """
                SELECT payload_json
                FROM crypto_spot_candles
                WHERE provider = 'coinbase'
                  AND product_id = ?
                  AND interval_seconds = ?
                  AND end_at <= ?
                  AND end_at >= ?
                ORDER BY end_at ASC
                """,
                (
                    product_id,
                    interval_seconds,
                    as_of.isoformat(),
                    lower_bound.isoformat(),
                ),
            ).fetchall()
            implied_row = connection.execute(
                """
                SELECT payload_json
                FROM crypto_volatility_observations
                WHERE provider = 'deribit' AND symbol = ? AND kind = 'implied'
                  AND observed_at <= ?
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (symbol, as_of.isoformat()),
            ).fetchone()
            funding_row = connection.execute(
                """
                SELECT payload_json
                FROM crypto_funding_observations
                WHERE provider = 'deribit' AND instrument_name = ?
                  AND observed_at <= ?
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (instrument_name, as_of.isoformat()),
            ).fetchone()
            derivatives_row = connection.execute(
                """
                SELECT payload_json
                FROM crypto_derivatives_snapshots
                WHERE provider = 'deribit' AND instrument_name = ?
                  AND observed_at <= ?
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (instrument_name, as_of.isoformat()),
            ).fetchone()
            event_row = None
            if event_ticker:
                event_row = connection.execute(
                    """
                    SELECT payload_json
                    FROM kalshi_event_data_snapshots
                    WHERE provider = 'kalshi' AND event_ticker = ?
                      AND observed_at <= ?
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    (event_ticker.upper(), as_of.isoformat()),
                ).fetchone()

        candles = [SpotCandle.model_validate_json(str(row["payload_json"])) for row in spot_rows]
        if not candles:
            raise ResearchDataUnavailable(
                f"no completed Coinbase {product_id} candles are available at {as_of.isoformat()}"
            )
        spot = candles[-1]
        spot_age = (as_of - spot.end_at).total_seconds()
        if spot_age > spot_max_age:
            raise ResearchDataUnavailable(f"latest spot candle is stale by {int(spot_age)} seconds")
        realized = calculate_realized_volatility(
            candles,
            symbol=symbol,
            as_of=as_of,
            window_seconds=realized_window_seconds,
        )

        warnings: list[str] = []
        implied = _optional_as_of(
            implied_row,
            VolatilityObservation,
            as_of,
            optional_max_age_seconds,
            "implied volatility",
            warnings,
        )
        funding = _optional_as_of(
            funding_row,
            FundingObservation,
            as_of,
            optional_max_age_seconds,
            "funding",
            warnings,
        )
        derivatives = _optional_as_of(
            derivatives_row,
            DerivativesSnapshot,
            as_of,
            optional_max_age_seconds,
            "derivatives snapshot",
            warnings,
        )
        event_data = _optional_as_of(
            event_row,
            EventDataSnapshot,
            as_of,
            event_max_age_seconds,
            "event data",
            warnings,
        )

        return ResearchContext(
            symbol=symbol,
            event_ticker=event_ticker.upper() if event_ticker else None,
            as_of=as_of,
            spot=spot,
            realized_volatility=realized,
            implied_volatility=implied,
            funding=funding,
            derivatives=derivatives,
            event_data=event_data,
            warnings=tuple(warnings),
        )


def _optional_as_of(
    row: sqlite3.Row | None,
    model: type[ResearchObservation],
    as_of: datetime,
    max_age_seconds: int,
    label: str,
    warnings: list[str],
) -> ResearchObservation | None:
    if row is None:
        warnings.append(f"No point-in-time {label} is available.")
        return None
    value = model.model_validate_json(str(row["payload_json"]))
    age = (as_of - value.observed_at).total_seconds()
    if age > max_age_seconds:
        warnings.append(f"Latest {label} is stale by {int(age)} seconds.")
        return None
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)
