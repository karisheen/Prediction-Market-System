from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from prediction_market_system.redaction import redact_payload, redact_secrets
from prediction_market_system.research import (
    DerivativesSnapshot,
    EventDataSnapshot,
    FundingObservation,
    ResearchContext,
    ResearchDataUnavailable,
    ResearchSyncStatus,
    RetrievedResearchModel,
    SpotCandle,
    VolatilityObservation,
    calculate_realized_volatility,
    research_payload_hash,
)

ResearchObservation = TypeVar(
    "ResearchObservation",
    VolatilityObservation,
    FundingObservation,
    DerivativesSnapshot,
    EventDataSnapshot,
)
ResearchRow = TypeVar("ResearchRow", bound=RetrievedResearchModel)
SourceObservation = (
    SpotCandle
    | VolatilityObservation
    | FundingObservation
    | DerivativesSnapshot
    | EventDataSnapshot
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

CREATE TABLE IF NOT EXISTS research_provider_revisions (
    revision_id INTEGER PRIMARY KEY,
    source_table TEXT NOT NULL,
    series_key TEXT NOT NULL,
    provider_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (source_table, provider_key, payload_hash)
);
CREATE INDEX IF NOT EXISTS idx_research_revisions_asof
ON research_provider_revisions (source_table, series_key, observed_at, available_at);

CREATE TABLE IF NOT EXISTS research_revision_observations (
    observation_id INTEGER PRIMARY KEY,
    source_table TEXT NOT NULL,
    provider_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    UNIQUE (source_table, provider_key, payload_hash, retrieved_at)
);
CREATE INDEX IF NOT EXISTS idx_research_revision_observations
ON research_revision_observations (source_table, provider_key, retrieved_at);
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
                    json.dumps(redact_payload(request), sort_keys=True),
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
                    redact_secrets(error)[:1000] if error else None,
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

            batches: tuple[tuple[str, str, Sequence[SourceObservation]], ...] = (
                ("spot_candles", "crypto_spot_candles", spot_candles or []),
                (
                    "volatility_observations",
                    "crypto_volatility_observations",
                    volatility_observations or [],
                ),
                ("funding_observations", "crypto_funding_observations", funding_observations or []),
                (
                    "derivatives_snapshots",
                    "crypto_derivatives_snapshots",
                    derivatives_snapshots or [],
                ),
                ("event_snapshots", "kalshi_event_data_snapshots", event_snapshots or []),
            )
            for name, table, observations in batches:
                inserted[name] = 0
                series_columns, key_columns, clock = _RESEARCH_TABLES[table]
                for value in observations:
                    retrieved_at = value.retrieved_at
                    filters = " AND ".join(f"{column} = ?" for column in key_columns)
                    key_values = tuple(
                        field.isoformat() if isinstance(field, datetime) else field
                        for field in (getattr(value, column) for column in key_columns)
                    )
                    legacy = connection.execute(
                        f"SELECT payload_json FROM {table} WHERE {filters}", key_values
                    ).fetchone()
                    if legacy is not None:
                        original = type(value).model_validate_json(str(legacy["payload_json"]))
                        if research_payload_hash(original) == research_payload_hash(value):
                            value = original
                    payload = value.model_dump(mode="json")
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO research_provider_revisions (
                            source_table, series_key, provider_key, payload_hash,
                            observed_at, available_at, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            table,
                            _identity(payload, series_columns),
                            _identity(payload, key_columns),
                            research_payload_hash(value),
                            getattr(value, clock).isoformat(),
                            value.retrieved_at.isoformat(),
                            value.model_dump_json(),
                        ),
                    )
                    inserted[name] += cursor.rowcount
                    provider_key = _identity(payload, key_columns)
                    payload_hash = research_payload_hash(value)
                    previous = connection.execute(
                        """
                        SELECT payload_hash FROM research_revision_observations
                        WHERE source_table = ? AND provider_key = ? AND retrieved_at <= ?
                        ORDER BY retrieved_at DESC, observation_id DESC LIMIT 1
                        """,
                        (table, provider_key, retrieved_at.isoformat()),
                    ).fetchone()
                    # Keep content identity immutable, but retain A -> B -> A reversions.
                    # Identical repeated fetches do not grow the observation history.
                    if previous is None or previous["payload_hash"] != payload_hash:
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO research_revision_observations
                                (source_table, provider_key, payload_hash, retrieved_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (table, provider_key, payload_hash, retrieved_at.isoformat()),
                        )

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
        rows = self._research_rows_as_of(
            "crypto_spot_candles",
            SpotCandle,
            ("coinbase", f"{symbol.upper()}-USD", interval_seconds),
            as_of,
            lower_bound=as_of - timedelta(seconds=window_seconds + interval_seconds),
        )
        return tuple(rows)

    def research_context_as_of(
        self,
        *,
        symbol: str,
        as_of: datetime,
        event_ticker: str | None = None,
        interval_seconds: int = 3600,
        realized_window_seconds: int = 30 * 24 * 60 * 60,
        spot_interval_seconds: int | None = None,
        spot_max_age_seconds: int | None = None,
        optional_max_age_seconds: int = 2 * 60 * 60,
        event_max_age_seconds: int = 6 * 60 * 60,
    ) -> ResearchContext:
        as_of = _as_utc(as_of)
        symbol = symbol.upper()
        product_id = f"{symbol}-USD"
        instrument_name = f"{symbol}-PERPETUAL"
        decision_interval = (
            interval_seconds if spot_interval_seconds is None else spot_interval_seconds
        )
        spot_max_age = (
            spot_max_age_seconds if spot_max_age_seconds is not None else 2 * decision_interval
        )
        realized_candles = list(
            self.spot_candles_as_of(
                symbol=symbol,
                as_of=as_of,
                interval_seconds=interval_seconds,
                window_seconds=realized_window_seconds,
            )
        )
        if decision_interval == interval_seconds:
            candles = realized_candles
        else:
            spot_window = max(spot_max_age, decision_interval) + decision_interval
            candles = list(
                self.spot_candles_as_of(
                    symbol=symbol,
                    as_of=as_of,
                    interval_seconds=decision_interval,
                    window_seconds=spot_window,
                )
            )
        implied_rows = self._research_rows_as_of(
            "crypto_volatility_observations",
            VolatilityObservation,
            ("deribit", symbol, "implied", interval_seconds),
            as_of,
            lower_bound=as_of - timedelta(seconds=optional_max_age_seconds),
        )
        funding_rows = self._research_rows_as_of(
            "crypto_funding_observations",
            FundingObservation,
            ("deribit", instrument_name),
            as_of,
            lower_bound=as_of - timedelta(seconds=optional_max_age_seconds),
        )
        derivatives_rows = self._research_rows_as_of(
            "crypto_derivatives_snapshots",
            DerivativesSnapshot,
            ("deribit", instrument_name),
            as_of,
            lower_bound=as_of - timedelta(seconds=optional_max_age_seconds),
        )
        event_rows = (
            self._research_rows_as_of(
                "kalshi_event_data_snapshots",
                EventDataSnapshot,
                ("kalshi", event_ticker.upper()),
                as_of,
                lower_bound=as_of - timedelta(seconds=event_max_age_seconds),
            )
            if event_ticker
            else []
        )

        if not candles:
            raise ResearchDataUnavailable(
                f"no completed Coinbase {product_id} candles are available at {as_of.isoformat()}"
            )
        if not realized_candles:
            raise ResearchDataUnavailable(
                f"no completed Coinbase {product_id} research candles are available at "
                f"{as_of.isoformat()}"
            )
        spot = candles[-1]
        spot_age = (as_of - spot.end_at).total_seconds()
        if spot_age > spot_max_age:
            raise ResearchDataUnavailable(f"latest spot candle is stale by {int(spot_age)} seconds")
        realized = calculate_realized_volatility(
            realized_candles,
            symbol=symbol,
            as_of=as_of,
            window_seconds=realized_window_seconds,
        )

        warnings: list[str] = []
        implied = _optional_as_of(
            implied_rows[-1] if implied_rows else None,
            as_of,
            optional_max_age_seconds,
            "implied volatility",
            warnings,
        )
        funding = _optional_as_of(
            funding_rows[-1] if funding_rows else None,
            as_of,
            optional_max_age_seconds,
            "funding",
            warnings,
        )
        derivatives = _optional_as_of(
            derivatives_rows[-1] if derivatives_rows else None,
            as_of,
            optional_max_age_seconds,
            "derivatives snapshot",
            warnings,
        )
        event_data = _optional_as_of(
            event_rows[-1] if event_rows else None,
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
            optional_max_age_seconds=optional_max_age_seconds,
            event_max_age_seconds=event_max_age_seconds,
        )

    def _research_rows_as_of(
        self,
        table: str,
        model: type[ResearchRow],
        series: tuple[str | int, ...],
        as_of: datetime,
        *,
        lower_bound: datetime,
    ) -> list[ResearchRow]:
        series_columns, key_columns, clock = _RESEARCH_TABLES[table]
        # Legacy rows retain their recorded retrieval availability. They are never
        # backdated to provider observation time or rewritten during an upgrade.
        filters = " AND ".join(f"{column} = ?" for column in series_columns)
        with self._connect() as connection:
            legacy = connection.execute(
                f"SELECT payload_json, retrieved_at AS selected_at FROM {table} WHERE {filters} "
                # SQLite timestamp predicates are supplemented by parsed UTC checks below.
                f"AND {clock} >= ? AND {clock} <= ? AND retrieved_at <= ?",
                (*series, lower_bound.isoformat(), as_of.isoformat(), as_of.isoformat()),
            ).fetchall()
            revisions = connection.execute(
                """
                SELECT payload_json, available_at AS selected_at FROM research_provider_revisions
                WHERE source_table = ? AND series_key = ? AND observed_at >= ?
                  AND observed_at <= ? AND available_at <= ?
                ORDER BY available_at, revision_id
                """,
                (
                    table,
                    json.dumps(series, separators=(",", ":")),
                    lower_bound.isoformat(),
                    as_of.isoformat(),
                    as_of.isoformat(),
                ),
            ).fetchall()
            observations = connection.execute(
                """
                SELECT r.payload_json, o.retrieved_at AS selected_at
                FROM research_provider_revisions r
                JOIN research_revision_observations o
                  ON o.source_table = r.source_table AND o.provider_key = r.provider_key
                 AND o.payload_hash = r.payload_hash
                WHERE r.source_table = ? AND r.series_key = ? AND r.observed_at >= ?
                  AND r.observed_at <= ? AND r.available_at <= ? AND o.retrieved_at <= ?
                ORDER BY o.retrieved_at, o.observation_id
                """,
                (
                    table,
                    json.dumps(series, separators=(",", ":")),
                    lower_bound.isoformat(),
                    as_of.isoformat(),
                    as_of.isoformat(),
                    as_of.isoformat(),
                ),
            ).fetchall()
        latest: dict[str, ResearchRow] = {}
        selected_at: dict[str, datetime] = {}
        conflicting: set[str] = set()
        for row in [*legacy, *revisions, *observations]:
            value = model.model_validate_json(str(row["payload_json"]))
            if value.retrieved_at > as_of or getattr(value, clock) > as_of:
                continue
            key = _identity(value.model_dump(mode="json"), key_columns)
            available = datetime.fromisoformat(str(row["selected_at"]))
            if key not in latest or available > selected_at[key]:
                latest[key] = value
                selected_at[key] = available
                conflicting.discard(key)
            elif available == selected_at[key] and research_payload_hash(
                value
            ) != research_payload_hash(latest[key]):
                conflicting.add(key)
        if conflicting:
            raise ResearchDataUnavailable(
                "conflicting provider revisions have ambiguous availability order"
            )
        return sorted(latest.values(), key=lambda value: getattr(value, clock))


def _optional_as_of(
    row: ResearchObservation | None,
    as_of: datetime,
    max_age_seconds: int,
    label: str,
    warnings: list[str],
) -> ResearchObservation | None:
    if row is None:
        warnings.append(f"No point-in-time {label} is available.")
        return None
    value = row
    age = (as_of - value.observed_at).total_seconds()
    if age < 0 or age > max_age_seconds or value.retrieved_at > as_of:
        warnings.append(f"Latest {label} is stale by {int(age)} seconds.")
        return None
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


# Identifiers below are internal constants, never caller-supplied SQL.
_RESEARCH_TABLES: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "crypto_spot_candles": (
        ("provider", "product_id", "interval_seconds"),
        ("provider", "product_id", "interval_seconds", "start_at"),
        "end_at",
    ),
    "crypto_volatility_observations": (
        ("provider", "symbol", "kind", "window_seconds"),
        ("provider", "symbol", "kind", "window_seconds", "observed_at"),
        "observed_at",
    ),
    "crypto_funding_observations": (
        ("provider", "instrument_name"),
        ("provider", "instrument_name", "observed_at"),
        "observed_at",
    ),
    "crypto_derivatives_snapshots": (
        ("provider", "instrument_name"),
        ("provider", "instrument_name", "observed_at"),
        "observed_at",
    ),
    "kalshi_event_data_snapshots": (
        ("provider", "event_ticker"),
        ("provider", "event_ticker", "data_type", "observed_at"),
        "observed_at",
    ),
}


def _identity(payload: dict[str, Any], columns: tuple[str, ...]) -> str:
    return json.dumps([payload[column] for column in columns], separators=(",", ":"))
