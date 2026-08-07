from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from prediction_market_system.backtest import BacktestResult, HistoricalMarketData
from prediction_market_system.calibration import UncertaintyCalibrationProfile
from prediction_market_system.domain import MarketRegimeSnapshot, Opportunity, ProbabilityForecast
from prediction_market_system.research_storage import RESEARCH_SCHEMA, ResearchRepositoryMixin
from prediction_market_system.venues.kalshi import (
    CandlestickPeriod,
    KalshiCandlestick,
    KalshiEventFeeChange,
    KalshiMarket,
    KalshiSeriesFeeChange,
)


class AlertStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    FAILED = "failed"


class MarketCheckStatus(StrEnum):
    UNSUPPORTED = "unsupported"
    MISSING_CALIBRATION = "missing_calibration"
    UNAPPROVED_MODEL = "unapproved_model"
    FAILED = "failed"
    WATCH = "watch"
    ENTRY_CANDIDATE = "entry_candidate"
    DELIVERED = "delivered"


@dataclass(frozen=True)
class AlertRecord:
    opportunity_id: str
    status: AlertStatus
    discord_message_id: str | None


@dataclass(frozen=True)
class KalshiHistoryWriteResult:
    market_snapshots: int = 0
    candlesticks: int = 0
    rule_snapshots: int = 0
    resolutions: int = 0
    series_fee_changes: int = 0
    event_fee_changes: int = 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteRepository(ResearchRepositoryMixin):
    """Append-oriented forecast, alert, and venue-history audit storage."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_forecasts_market_generated
                ON forecasts (market_id, generated_at);

                CREATE TABLE IF NOT EXISTS opportunities (
                    opportunity_id TEXT PRIMARY KEY,
                    forecast_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (forecast_id) REFERENCES forecasts (forecast_id)
                );

                CREATE INDEX IF NOT EXISTS idx_opportunities_market_created
                ON opportunities (market_id, created_at);

                CREATE TABLE IF NOT EXISTS alert_events (
                    opportunity_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    discord_message_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (opportunity_id)
                        REFERENCES opportunities (opportunity_id)
                );

                CREATE TABLE IF NOT EXISTS discord_deliveries (
                    market_id TEXT PRIMARY KEY,
                    discord_message_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kalshi_market_snapshots (
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    event_ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_updated_at TEXT,
                    close_time TEXT NOT NULL,
                    result TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (ticker, observed_at)
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_markets_series_close
                ON kalshi_market_snapshots (series_ticker, close_time);

                CREATE TABLE IF NOT EXISTS kalshi_candlesticks (
                    ticker TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    period_interval_minutes INTEGER NOT NULL,
                    end_period_ts INTEGER NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (ticker, period_interval_minutes, end_period_ts)
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_candles_series_period
                ON kalshi_candlesticks (
                    series_ticker, period_interval_minutes, end_period_ts
                );

                CREATE TABLE IF NOT EXISTS kalshi_rule_snapshots (
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    rules_primary TEXT NOT NULL,
                    rules_secondary TEXT NOT NULL,
                    PRIMARY KEY (ticker, observed_at)
                );

                CREATE TABLE IF NOT EXISTS kalshi_resolutions (
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    result TEXT NOT NULL,
                    settlement_value_dollars TEXT,
                    settlement_ts TEXT,
                    expiration_value TEXT NOT NULL,
                    PRIMARY KEY (ticker, observed_at)
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_resolutions_settlement
                ON kalshi_resolutions (settlement_ts);

                CREATE TABLE IF NOT EXISTS kalshi_series_fee_changes (
                    change_id TEXT PRIMARY KEY,
                    series_ticker TEXT NOT NULL,
                    fee_type TEXT NOT NULL,
                    fee_multiplier REAL NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_series_fees_schedule
                ON kalshi_series_fee_changes (series_ticker, scheduled_at);

                CREATE TABLE IF NOT EXISTS kalshi_event_fee_changes (
                    change_id TEXT PRIMARY KEY,
                    event_ticker TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    fee_type_override TEXT,
                    fee_multiplier_override REAL,
                    scheduled_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kalshi_event_fees_schedule
                ON kalshi_event_fee_changes (event_ticker, scheduled_at);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    series_ticker TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_backtest_runs_series_generated
                ON backtest_runs (series_ticker, generated_at);

                CREATE TABLE IF NOT EXISTS uncertainty_calibrations (
                    profile_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    cutoff_at TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_calibrations_model_cutoff
                ON uncertainty_calibrations (
                    symbol, model_name, model_version, cutoff_at
                );

                CREATE TABLE IF NOT EXISTS paper_model_validations (
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    calibration_profile_id TEXT,
                    accepted INTEGER NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, model_name),
                    FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_model_validations_profile
                ON paper_model_validations (
                    calibration_profile_id, accepted, generated_at
                );

                CREATE TABLE IF NOT EXISTS market_regime_snapshots (
                    series_ticker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (series_ticker, symbol, observed_at)
                );

                CREATE INDEX IF NOT EXISTS idx_market_regimes_series_observed
                ON market_regime_snapshots (series_ticker, symbol, observed_at);

                CREATE TABLE IF NOT EXISTS paper_alert_market_checks (
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    event_ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (cycle_id, market_id)
                );

                CREATE INDEX IF NOT EXISTS idx_paper_checks_series_observed
                ON paper_alert_market_checks (series_ticker, observed_at);
                """
            )
            connection.executescript(RESEARCH_SCHEMA)

    def save_evaluation(
        self,
        forecast: ProbabilityForecast,
        opportunity: Opportunity,
    ) -> None:
        created_at = forecast.generated_at.isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO forecasts (
                    forecast_id, market_id, generated_at, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(forecast.forecast_id),
                    forecast.market_id,
                    created_at,
                    forecast.model_dump_json(),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO opportunities (
                    opportunity_id, forecast_id, market_id, state,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(opportunity.opportunity_id),
                    str(forecast.forecast_id),
                    opportunity.market.market_id,
                    opportunity.state.value,
                    created_at,
                    opportunity.model_dump_json(),
                ),
            )

    def save_paper_market_check(
        self,
        *,
        cycle_id: str,
        market_id: str,
        series_ticker: str,
        event_ticker: str,
        observed_at: datetime,
        status: MarketCheckStatus,
        reason: str | None,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_alert_market_checks (
                    cycle_id, market_id, series_ticker, event_ticker,
                    observed_at, status, reason, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cycle_id, market_id) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    status = excluded.status,
                    reason = excluded.reason,
                    payload_json = excluded.payload_json
                """,
                (
                    cycle_id,
                    market_id,
                    series_ticker.upper(),
                    event_ticker,
                    observed_at.isoformat(),
                    status.value,
                    None if reason is None else reason[:1_000],
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )

    def paper_market_checks(self, cycle_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT cycle_id, market_id, series_ticker, event_ticker,
                       observed_at, status, reason, payload_json
                FROM paper_alert_market_checks
                WHERE cycle_id = ?
                ORDER BY market_id
                """,
                (cycle_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(str(row["payload_json"])),
            }
            for row in rows
        ]

    def save_kalshi_history(
        self,
        *,
        series_ticker: str,
        observed_at: datetime,
        markets: list[KalshiMarket],
        candlesticks: dict[str, list[KalshiCandlestick]],
        period_interval: CandlestickPeriod,
        series_fee_changes: list[KalshiSeriesFeeChange],
        event_fee_changes: list[KalshiEventFeeChange],
    ) -> KalshiHistoryWriteResult:
        market_tickers = {market.ticker for market in markets}
        unexpected_tickers = candlesticks.keys() - market_tickers
        if unexpected_tickers:
            unexpected = ", ".join(sorted(unexpected_tickers))
            raise ValueError(f"candlesticks supplied for unknown markets: {unexpected}")

        observed = observed_at.isoformat()
        inserted = {
            "market_snapshots": 0,
            "candlesticks": 0,
            "rule_snapshots": 0,
            "resolutions": 0,
            "series_fee_changes": 0,
            "event_fee_changes": 0,
        }
        with self._connect() as connection:
            for market in markets:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_market_snapshots (
                        ticker, observed_at, series_ticker, event_ticker, status,
                        source_updated_at, close_time, result, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market.ticker,
                        observed,
                        series_ticker,
                        market.event_ticker,
                        market.status,
                        market.updated_time.isoformat() if market.updated_time else None,
                        market.close_time.isoformat(),
                        market.result,
                        market.model_dump_json(),
                    ),
                )
                inserted["market_snapshots"] += cursor.rowcount

                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_rule_snapshots (
                        ticker, observed_at, rules_primary, rules_secondary
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        market.ticker,
                        observed,
                        market.rules_primary,
                        market.rules_secondary,
                    ),
                )
                inserted["rule_snapshots"] += cursor.rowcount

                if market.result:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO kalshi_resolutions (
                            ticker, observed_at, result, settlement_value_dollars,
                            settlement_ts, expiration_value
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            market.ticker,
                            observed,
                            market.result,
                            (
                                str(market.settlement_value_dollars)
                                if market.settlement_value_dollars is not None
                                else None
                            ),
                            market.settlement_ts.isoformat() if market.settlement_ts else None,
                            market.expiration_value,
                        ),
                    )
                    inserted["resolutions"] += cursor.rowcount

                for candle in candlesticks.get(market.ticker, []):
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO kalshi_candlesticks (
                            ticker, series_ticker, period_interval_minutes,
                            end_period_ts, retrieved_at, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            market.ticker,
                            series_ticker,
                            period_interval,
                            candle.end_period_ts,
                            observed,
                            candle.model_dump_json(),
                        ),
                    )
                    inserted["candlesticks"] += cursor.rowcount

            for change in series_fee_changes:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_series_fee_changes (
                        change_id, series_ticker, fee_type, fee_multiplier,
                        scheduled_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        change.id,
                        change.series_ticker,
                        change.fee_type,
                        change.fee_multiplier,
                        change.scheduled_ts.isoformat(),
                        observed,
                        change.model_dump_json(),
                    ),
                )
                inserted["series_fee_changes"] += cursor.rowcount

            for event_change in event_fee_changes:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO kalshi_event_fee_changes (
                        change_id, event_ticker, series_ticker, fee_type_override,
                        fee_multiplier_override, scheduled_at, retrieved_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_change.id,
                        event_change.event_ticker,
                        event_change.series_ticker,
                        event_change.fee_type_override,
                        event_change.fee_multiplier_override,
                        event_change.scheduled_ts.isoformat(),
                        observed,
                        event_change.model_dump_json(),
                    ),
                )
                inserted["event_fee_changes"] += cursor.rowcount

        return KalshiHistoryWriteResult(**inserted)

    def load_kalshi_backtest_data(
        self,
        *,
        series_ticker: str,
        start: datetime,
        end: datetime,
        period_interval: CandlestickPeriod,
        max_events: int,
    ) -> tuple[HistoricalMarketData, ...]:
        with self._connect() as connection:
            market_rows = connection.execute(
                """
                WITH ranked_snapshots AS (
                    SELECT series_ticker, event_ticker, ticker, close_time, payload_json,
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker ORDER BY observed_at DESC
                           ) AS snapshot_rank
                    FROM kalshi_market_snapshots
                    WHERE series_ticker = ? AND close_time >= ? AND close_time <= ?
                ),
                latest_snapshots AS (
                    SELECT series_ticker, event_ticker, ticker, close_time, payload_json
                    FROM ranked_snapshots
                    WHERE snapshot_rank = 1
                ),
                selected_events AS (
                    SELECT event_ticker, MIN(close_time) AS event_close
                    FROM latest_snapshots
                    GROUP BY event_ticker
                    ORDER BY event_close ASC
                    LIMIT ?
                )
                SELECT latest.series_ticker, latest.ticker, latest.close_time,
                       latest.payload_json
                FROM latest_snapshots AS latest
                JOIN selected_events USING (event_ticker)
                ORDER BY latest.close_time ASC, latest.ticker ASC
                """,
                (
                    series_ticker.upper(),
                    start.isoformat(),
                    end.isoformat(),
                    max_events,
                ),
            ).fetchall()
            series_fee_rows = connection.execute(
                """
                SELECT payload_json
                FROM kalshi_series_fee_changes
                WHERE series_ticker = ?
                ORDER BY scheduled_at ASC
                """,
                (series_ticker.upper(),),
            ).fetchall()
            event_fee_rows = connection.execute(
                """
                SELECT payload_json
                FROM kalshi_event_fee_changes
                WHERE series_ticker = ?
                ORDER BY scheduled_at ASC
                """,
                (series_ticker.upper(),),
            ).fetchall()

            series_fees = tuple(
                KalshiSeriesFeeChange.model_validate_json(str(row["payload_json"]))
                for row in series_fee_rows
            )
            event_fees = tuple(
                KalshiEventFeeChange.model_validate_json(str(row["payload_json"]))
                for row in event_fee_rows
            )
            markets: list[HistoricalMarketData] = []
            for row in market_rows:
                market = KalshiMarket.model_validate_json(str(row["payload_json"]))
                if market.open_time is not None and market.open_time >= end:
                    continue
                resolution = connection.execute(
                    """
                    SELECT result, settlement_value_dollars, settlement_ts,
                           expiration_value
                    FROM kalshi_resolutions
                    WHERE ticker = ?
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    (market.ticker,),
                ).fetchone()
                if resolution is not None:
                    payload = market.model_dump()
                    payload.update(
                        {
                            "result": str(resolution["result"]),
                            "settlement_value_dollars": resolution["settlement_value_dollars"],
                            "settlement_ts": resolution["settlement_ts"],
                            "expiration_value": str(resolution["expiration_value"]),
                        }
                    )
                    market = KalshiMarket.model_validate(payload)

                candle_rows = connection.execute(
                    """
                    SELECT payload_json
                    FROM kalshi_candlesticks
                    WHERE ticker = ? AND period_interval_minutes = ?
                      AND end_period_ts >= ? AND end_period_ts <= ?
                    ORDER BY end_period_ts ASC
                    """,
                    (
                        market.ticker,
                        period_interval,
                        int(start.timestamp()),
                        int(market.expiry.timestamp()),
                    ),
                ).fetchall()
                markets.append(
                    HistoricalMarketData(
                        series_ticker=str(row["series_ticker"]),
                        market=market,
                        candlesticks=tuple(
                            KalshiCandlestick.model_validate_json(str(candle["payload_json"]))
                            for candle in candle_rows
                        ),
                        series_fee_changes=series_fees,
                        event_fee_changes=tuple(
                            change
                            for change in event_fees
                            if change.event_ticker == market.event_ticker
                        ),
                    )
                )
        return tuple(markets)

    def save_backtest_result(self, result: BacktestResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, series_ticker, generated_at, start_at, end_at,
                    config_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(result.run_id),
                    result.config.series_ticker,
                    result.generated_at.isoformat(),
                    result.config.start.isoformat(),
                    result.config.end.isoformat(),
                    result.config.model_dump_json(),
                    result.model_dump_json(),
                ),
            )
            profiles = (
                *(profile for fold in result.folds for profile in fold.calibration_profiles),
                *result.deployment_profiles,
            )
            for profile in profiles:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO uncertainty_calibrations (
                        profile_id, symbol, model_name, model_version,
                        cutoff_at, generated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(profile.profile_id),
                        profile.symbol,
                        profile.model_name,
                        profile.model_version,
                        profile.cutoff_at.isoformat(),
                        profile.generated_at.isoformat(),
                        profile.model_dump_json(),
                    ),
                )
            for validation in result.model_validations:
                connection.execute(
                    """
                    INSERT INTO paper_model_validations (
                        run_id, model_name, model_version, calibration_profile_id,
                        accepted, generated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(result.run_id),
                        validation.model_name,
                        validation.model_version,
                        (
                            None
                            if validation.calibration_profile_id is None
                            else str(validation.calibration_profile_id)
                        ),
                        int(validation.accepted_for_paper_alerts),
                        result.generated_at.isoformat(),
                        validation.model_dump_json(),
                    ),
                )

    def latest_uncertainty_calibration(
        self,
        *,
        symbol: str,
        model_name: str,
        model_version: str,
        as_of: datetime,
    ) -> UncertaintyCalibrationProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM uncertainty_calibrations
                WHERE symbol = ? AND model_name = ? AND model_version = ?
                  AND cutoff_at <= ?
                ORDER BY cutoff_at DESC, generated_at DESC
                LIMIT 1
                """,
                (symbol.upper(), model_name, model_version, as_of.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return UncertaintyCalibrationProfile.model_validate_json(str(row["payload_json"]))

    def is_calibration_approved(self, profile_id: UUID) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT accepted
                FROM paper_model_validations
                WHERE calibration_profile_id = ?
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                (str(profile_id),),
            ).fetchone()
        return row is not None and bool(row["accepted"])

    def save_market_regime(
        self,
        *,
        series_ticker: str,
        regime: MarketRegimeSnapshot,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO market_regime_snapshots (
                    series_ticker, symbol, observed_at, regime, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    series_ticker.upper(),
                    regime.symbol.upper(),
                    regime.observed_at.isoformat(),
                    regime.label,
                    regime.model_dump_json(),
                ),
            )

    def latest_market_regime(
        self,
        *,
        series_ticker: str,
        symbol: str,
        as_of: datetime,
    ) -> MarketRegimeSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM market_regime_snapshots
                WHERE series_ticker = ? AND symbol = ? AND observed_at <= ?
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (
                    series_ticker.upper(),
                    symbol.upper(),
                    as_of.isoformat(),
                ),
            ).fetchone()
        if row is None:
            return None
        return MarketRegimeSnapshot.model_validate_json(str(row["payload_json"]))

    def market_regime_coverage(
        self,
        *,
        series_ticker: str,
        symbol: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT regime, COUNT(*) AS observation_count,
                       MIN(observed_at) AS first_observed_at,
                       MAX(observed_at) AS last_observed_at
                FROM market_regime_snapshots
                WHERE series_ticker = ? AND symbol = ?
                GROUP BY regime
                ORDER BY regime
                """,
                (series_ticker.upper(), symbol.upper()),
            ).fetchall()
        return [dict(row) for row in rows]

    def queue_alert(self, opportunity: Opportunity) -> AlertRecord:
        now = _utc_now()
        opportunity_id = str(opportunity.opportunity_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO alert_events (
                    opportunity_id, market_id, state, status, created_at,
                    updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    opportunity_id,
                    opportunity.market.market_id,
                    opportunity.state.value,
                    AlertStatus.QUEUED.value,
                    now,
                    now,
                    opportunity.model_dump_json(),
                ),
            )
            row = connection.execute(
                """
                SELECT status, discord_message_id
                FROM alert_events
                WHERE opportunity_id = ?
                """,
                (opportunity_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError("failed to queue alert")
        return AlertRecord(
            opportunity_id=opportunity_id,
            status=AlertStatus(str(row["status"])),
            discord_message_id=row["discord_message_id"],
        )

    def get_discord_delivery(self, market_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT discord_message_id
                FROM discord_deliveries
                WHERE market_id = ?
                """,
                (market_id,),
            ).fetchone()
        return None if row is None else str(row["discord_message_id"])

    def mark_alert_delivered(
        self,
        opportunity: Opportunity,
        discord_message_id: str,
    ) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE alert_events
                SET status = ?, attempts = attempts + 1,
                    discord_message_id = ?, error = NULL, updated_at = ?
                WHERE opportunity_id = ?
                """,
                (
                    AlertStatus.DELIVERED.value,
                    discord_message_id,
                    now,
                    str(opportunity.opportunity_id),
                ),
            )
            connection.execute(
                """
                INSERT INTO discord_deliveries (
                    market_id, discord_message_id, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    discord_message_id = excluded.discord_message_id,
                    updated_at = excluded.updated_at
                """,
                (opportunity.market.market_id, discord_message_id, now),
            )

    def mark_alert_failed(self, opportunity_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE alert_events
                SET status = ?, attempts = attempts + 1,
                    error = ?, updated_at = ?
                WHERE opportunity_id = ?
                """,
                (
                    AlertStatus.FAILED.value,
                    error[:1_000],
                    _utc_now(),
                    opportunity_id,
                ),
            )

    def opportunity_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM opportunities
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection
