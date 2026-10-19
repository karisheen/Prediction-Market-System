"""Execute the frozen item-8 availability audit without touching production.

Run with uv run python experiments/item8/replay_audit.py --source BACKUP --output NEW_DIR.
The source is read-only. Raw selected rows and original clocks are preserved verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from prediction_market_system.backtest import BacktestConfig, HistoricalBacktester
from prediction_market_system.engine import EngineConfig
from prediction_market_system.evidence import code_identity, content_id
from prediction_market_system.storage import SQLiteRepository

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(source: Path, output: Path, protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    identity = protocol.pop("protocol_id")
    if identity != "sha256:" + content_id(protocol):
        raise ValueError("Frozen protocol fingerprint mismatch")
    protocol["protocol_id"] = identity
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "protocol.json", protocol)
    runner = Path(__file__).read_bytes()
    (output / "replay_audit.as_run.py").write_bytes(runner)
    start = protocol["windows"]["replay_start"]
    end = protocol["windows"]["end_exclusive"]
    warmup = protocol["windows"]["research_warmup_start"]
    source_stat = source.stat()
    original = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    original.execute("PRAGMA query_only = ON")
    original.execute("BEGIN")
    tables = {
        name: sql
        for name, sql in original.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")
    }
    # This extraction is deliberately specific to the legacy snapshot. Never silently
    # discard a newer revision store if a different source is supplied.
    for table in ("research_provider_revisions", "venue_source_revisions"):
        if table in tables and original.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone():
            raise ValueError(
                f"Source contains {table}; this legacy extractor cannot omit revisions"
            )
    target = output / "research.sqlite3"
    repository = SQLiteRepository(target)
    repository.initialize()
    destination = sqlite3.connect(target)
    universe = (
        "SELECT DISTINCT ticker FROM kalshi_market_snapshots "
        "WHERE series_ticker='KXBTC' AND julianday(close_time)>=julianday(:start) "
        "AND julianday(close_time)<julianday(:end) AND julianday(observed_at)<julianday(:end)"
    )
    params = {
        "start": start,
        "end": end,
        "warmup": warmup,
        "start_ts": int(datetime.fromisoformat(start).timestamp()),
        "end_ts": int(datetime.fromisoformat(end).timestamp()),
    }
    predicates = {
        "kalshi_market_snapshots": (
            f"ticker IN ({universe}) AND julianday(observed_at)<julianday(:end)"
        ),
        "kalshi_rule_snapshots": (
            f"ticker IN ({universe}) AND julianday(observed_at)<julianday(:end)"
        ),
        "kalshi_resolutions": (
            f"ticker IN ({universe}) AND julianday(observed_at)<julianday(:end) "
            "AND (settlement_ts IS NULL OR julianday(settlement_ts)<julianday(:end))"
        ),
        "kalshi_candlesticks": (
            f"ticker IN ({universe}) AND end_period_ts>=:start_ts AND end_period_ts<:end_ts "
            "AND julianday(retrieved_at)<julianday(:end)"
        ),
        "kalshi_series_fee_changes": (
            "series_ticker='KXBTC' AND julianday(retrieved_at)<julianday(:end)"
        ),
        "kalshi_event_fee_changes": (
            "series_ticker='KXBTC' AND julianday(retrieved_at)<julianday(:end)"
        ),
        "crypto_spot_candles": (
            "product_id='BTC-USD' AND interval_seconds IN (60,3600) "
            "AND julianday(end_at)>=julianday(:warmup) AND julianday(end_at)<julianday(:end) "
            "AND julianday(retrieved_at)<julianday(:end)"
        ),
        "crypto_volatility_observations": (
            "symbol='BTC' AND julianday(observed_at)>=julianday(:warmup) "
            "AND julianday(observed_at)<julianday(:end) "
            "AND julianday(retrieved_at)<julianday(:end)"
        ),
        "crypto_funding_observations": (
            "instrument_name LIKE 'BTC-%' AND julianday(observed_at)>=julianday(:warmup) "
            "AND julianday(observed_at)<julianday(:end) "
            "AND julianday(retrieved_at)<julianday(:end)"
        ),
        "crypto_derivatives_snapshots": (
            "instrument_name LIKE 'BTC-%' AND julianday(observed_at)>=julianday(:warmup) "
            "AND julianday(observed_at)<julianday(:end) "
            "AND julianday(retrieved_at)<julianday(:end)"
        ),
        "kalshi_event_data_snapshots": (
            "event_ticker LIKE 'KXBTC-%' AND julianday(observed_at)>=julianday(:warmup) "
            "AND julianday(observed_at)<julianday(:end) "
            "AND julianday(retrieved_at)<julianday(:end)"
        ),
    }
    extraction = {
        "protocol_id": identity,
        "code": code_identity(),
        "runner_sha256": hashlib.sha256(runner).hexdigest(),
        "source_file": source.name,
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_schema_sha256": content_id(tables),
        "parameters": params,
        "tables": {},
        "source_evidence_tables": {
            name: name in tables
            for name in (
                "forecast_ledger",
                "run_manifests",
                "input_objects",
                "research_provider_revisions",
                "research_revision_observations",
                "venue_source_revisions",
                "venue_revision_observations",
            )
        },
        "isolation": (
            "No production database/configuration read or written. No forward holdout rows copied."
        ),
    }
    for table, predicate in predicates.items():
        columns = list(original.execute(f'PRAGMA table_info("{table}")'))
        names = [row[1] for row in columns]
        order = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
        quoted = ",".join(f'"{name}"' for name in names)
        query = f'SELECT {quoted} FROM "{table}" WHERE {predicate} ORDER BY ' + ",".join(order)
        insert = f'INSERT INTO "{table}" ({quoted}) VALUES (' + ",".join("?" for _ in names) + ")"
        digest = hashlib.sha256()
        count = 0
        cursor = original.execute(query, params)
        while rows := cursor.fetchmany(1000):
            destination.executemany(insert, rows)
            for row in rows:
                digest.update(
                    (json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
                )
            count += len(rows)
        destination.commit()
        extraction["tables"][table] = {"query": query, "rows": count, "sha256": digest.hexdigest()}
        print(f"Copied {table}: {count}", flush=True)
    original.close()
    if (source.stat().st_size, source.stat().st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise RuntimeError("Immutable source changed during extraction")
    events = destination.execute(
        "SELECT count(DISTINCT event_ticker) FROM kalshi_market_snapshots"
    ).fetchone()[0]
    if events > protocol["walk_forward"]["max_events"]:
        raise ValueError("Event cap would truncate the predeclared universe")
    destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    destination.close()
    extraction["selected_events"] = events
    extraction["selected_source_sha256"] = content_id(extraction["tables"])
    dump(output / "extraction.json", extraction)

    settings = protocol["walk_forward"]
    config = BacktestConfig(
        series_ticker="KXBTC",
        symbol="BTC",
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        period_minutes=settings["market_period_minutes"],
        spot_interval_seconds=settings["spot_interval_seconds"],
        realized_interval_seconds=settings["research_interval_seconds"],
        realized_window_days=settings["realized_window_days"],
        train_days=settings["train_days"],
        test_days=settings["test_days"],
        step_days=settings["step_days"],
        latency_seconds=protocol["scenarios"][0]["latency_seconds"],
        max_volume_participation=0.10,
        require_calibration=True,
        minimum_calibration_samples=settings["minimum_calibration_samples"],
        maximum_calibration_bins=settings["calibration_bins"],
        calibration_confidence=settings["calibration_confidence"],
        calibration_lead_seconds=settings["calibration_lead_seconds"],
        minimum_validation_events=settings["minimum_validation_events"],
        minimum_validation_folds=settings["minimum_validation_folds"],
        minimum_return_on_cost=settings["minimum_return_on_cost"],
        maximum_brier_score=settings["maximum_brier_score"],
    )
    markets = repository.load_kalshi_backtest_data(
        series_ticker="KXBTC",
        start=config.start,
        end=config.end,
        period_interval=config.period_minutes,
        max_events=settings["max_events"],
    )
    funnel = Counter()
    by_month = {}
    statuses = Counter()
    for history in markets:
        funnel["loaded_contracts"] += 1
        month = history.market.close_time.strftime("%Y-%m")
        cell = by_month.setdefault(month, Counter())
        cell["contracts"] += 1
        for available, metadata in history.metadata_snapshots:
            statuses[metadata.status] += 1
            cell["metadata_snapshots"] += 1
            if available < metadata.close_time:
                cell["metadata_before_trading_close"] += 1
        for available, candle in history.candlestick_revisions:
            end_at = datetime.fromtimestamp(candle.end_period_ts, config.start.tzinfo)
            signal = max(available, end_at)
            funnel["candle_revisions"] += 1
            cell["candle_revisions"] += 1
            if available <= end_at:
                funnel["received_by_candle_end"] += 1
            if (signal - end_at).total_seconds() <= 120:
                funnel["received_within_120_seconds"] += 1
                cell["received_within_120_seconds"] += 1
            if signal < history.market.close_time:
                funnel["receipt_before_trading_close"] += 1
                cell["receipt_before_trading_close"] += 1
    coverage = {
        "protocol_id": identity,
        "counts": dict(funnel),
        "monthly": by_month,
        "metadata_status_counts": statuses,
        "events": events,
        "note": (
            "Receipt-before-close uses trading close, not a substitute for stricter "
            "benchmark and eligibility checks in replay."
        ),
    }
    dump(output / "coverage.json", coverage)
    print("Running original HistoricalBacktester with frozen configuration", flush=True)
    result = HistoricalBacktester(repository, EngineConfig(**protocol["engine"])).run(
        config, markets
    )
    result = repository.save_backtest_result(result)
    dump(output / "backtest.json", result.model_dump(mode="json"))
    summary = {
        "protocol_id": identity,
        "run_id": str(result.run_id),
        "folds": len(result.folds),
        "full_folds": sum(
            (fold.fold.test_end - fold.fold.test_start).days == config.test_days
            for fold in result.folds
        ),
        "forecasts": len(result.forecasts),
        "resolved_forecasts": sum(f.outcome_yes is not None for f in result.forecasts),
        "independent_resolved_events": len(
            {f.event_ticker for f in result.forecasts if f.outcome_yes is not None}
        ),
        "trades": result.total_trades,
        "return_on_cost": result.return_on_cost,
        "brier": result.brier_score,
        "log_loss": result.log_loss,
        "native_market_brier": result.market_brier_score,
        "native_market_log_loss": result.market_log_loss,
        "missing_metadata_signals": result.missing_metadata_signals,
        "missing_context_signals": sum(f.missing_context_signals for f in result.folds),
        "missing_calibration_signals": sum(f.missing_calibration_signals for f in result.folds),
        "status": "insufficient-evidence"
        if not result.forecasts
        else "requires-regime-score-analysis",
        "pnl_interpretation": (
            "Accounting zero with no trades is not an estimated strategy return "
            "or measured drawdown."
        ),
        "deployment_approval": False,
    }
    dump(output / "summary.json", summary)
    dump(
        output / "artifacts.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.json")},
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "experiments/item8/protocol-v1.json"
    )
    arguments = parser.parse_args()
    run(arguments.source, arguments.output, arguments.protocol)
