"""Current-retrieval price-path coverage, never point-in-time forecast evidence.

This deliberately does not forge ResearchContext/retrieval clocks or read outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from prediction_market_system.evidence import code_identity, content_id
from prediction_market_system.sources.coinbase import CoinbaseClient
from prediction_market_system.venues.kalshi import KalshiClient

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


async def run(output: Path, protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    identity = protocol.pop("protocol_id")
    if identity != "sha256:" + content_id(protocol):
        raise ValueError("Frozen protocol fingerprint mismatch")
    protocol["protocol_id"] = identity
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "protocol.json", protocol)
    runner = Path(__file__).read_bytes()
    (output / "price_coverage.as_run.py").write_bytes(runner)
    start = datetime.fromisoformat(protocol["windows"]["descriptive_price_start"])
    end = datetime.fromisoformat(protocol["windows"]["end_exclusive"])
    client = CoinbaseClient()
    candles = {}
    conflicts = set()
    receipts = []
    try:
        with gzip.open(output / "coinbase-hourly.jsonl.gz", "wt", encoding="utf-8") as raw:
            cursor = start
            while cursor < end:
                boundary = min(end, cursor + timedelta(hours=350))
                batch = await client.get_candles(
                    "BTC-USD", start_at=cursor, end_at=boundary, interval_seconds=3600
                )
                for candle in batch:
                    payload = candle.model_dump(mode="json")
                    raw.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    timestamp = candle.end_at
                    previous = candles.get(timestamp)
                    if previous is not None and previous.raw_payload != candle.raw_payload:
                        conflicts.add(timestamp)
                    candles[timestamp] = candle
                receipts.append(
                    {
                        "start": cursor.isoformat(),
                        "end": boundary.isoformat(),
                        "rows": len(batch),
                        "received_at": datetime.now(UTC).isoformat(),
                    }
                )
                print(
                    f"Downloaded {cursor.date()} through {boundary.date()}: "
                    f"{len(batch)} hourly candles",
                    flush=True,
                )
                cursor = boundary
                await asyncio.sleep(0.25)
    finally:
        await client.close()
        dump(output / "requests.json", receipts)

    days = []
    classifications = Counter()
    gaps = []
    cutoff = datetime.fromisoformat(protocol["windows"]["descriptive_decisions_start"])
    while cutoff < end:
        required = [cutoff - timedelta(hours=720 - index) for index in range(721)]
        missing = [
            time.isoformat() for time in required if time not in candles or time in conflicts
        ]
        row = {
            "date": cutoff.date().isoformat(),
            "regime": "unavailable",
            "trailing_return": None,
            "annualized_realized_volatility": None,
            "missing_or_conflicting_closes": len(missing),
        }
        if not missing:
            closes = [float(candles[time].close) for time in required]
            returns = [
                math.log(right / left) for left, right in zip(closes, closes[1:], strict=False)
            ]
            change = closes[-1] / closes[0] - 1.0
            vol = statistics.stdev(returns) * math.sqrt(365 * 24)
            trend = "downtrend" if change <= -0.05 else "uptrend" if change >= 0.05 else "range"
            volatility = (
                "high-volatility"
                if vol >= 0.80
                else "low-volatility"
                if vol < 0.40
                else "typical-volatility"
            )
            row.update(
                regime=f"{trend} / {volatility}",
                trailing_return=change,
                annualized_realized_volatility=vol,
            )
        else:
            gaps.append({"date": row["date"], "missing_or_conflicting_closes": missing})
        days.append(row)
        classifications[row["regime"]] += 1
        cutoff += timedelta(days=1)
    with (output / "daily-regimes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(days[0]))
        writer.writeheader()
        writer.writerows(days)
    dump(output / "gaps.json", gaps)
    monthly = {}
    for row in days:
        month = row["date"][:7]
        counts = monthly.setdefault(month, Counter())
        counts[row["regime"]] += 1
    report = {
        "protocol_id": identity,
        "code": code_identity(),
        "runner_sha256": hashlib.sha256(runner).hexdigest(),
        "evidence_class": "current-retrieval descriptive historical prices",
        "point_in_time_replay_eligible": False,
        "provider": "Coinbase BTC-USD; not Kalshi settlement benchmark",
        "hourly_candles": len(candles),
        "expected_hourly_candles": int((end - start).total_seconds() // 3600),
        "conflicting_closes": len(conflicts),
        "daily_cutoffs": len(days),
        "unclassifiable_days": len(gaps),
        "regime_days": classifications,
        "monthly_regime_days": monthly,
        "downtrend_days": sum(row["regime"].startswith("downtrend") for row in days),
        "high_volatility_days": sum(row["regime"].endswith("high-volatility") for row in days),
        "overlap_downtrend_high_days": sum(
            row["regime"] == "downtrend / high-volatility" for row in days
        ),
        "uncertainty": (
            "Overlapping trailing windows are dependent. Day counts are descriptive "
            "coverage, not independent forecast samples."
        ),
        "price_source_sha256": hashlib.sha256(
            (output / "coinbase-hourly.jsonl.gz").read_bytes()
        ).hexdigest(),
    }
    dump(output / "regime-coverage.json", report)

    # Fixed date/event probe, not a market-selection rule and not a complete venue inventory.
    venue = KalshiClient(min_request_interval=0.25, max_rate_limit_retries=2)
    probe = {
        "event_ticker": "KXBTC-25APR0717",
        "limit": 1,
        "purpose": "bounded public historical-data availability only",
    }
    try:
        response = await venue.list_historical_markets(event_ticker=probe["event_ticker"], limit=1)
        probe["returned_markets"] = len(response.markets)
        if response.markets:
            market = response.markets[0]
            if (
                not datetime(2025, 4, 6, tzinfo=UTC)
                <= market.close_time
                < datetime(2025, 4, 9, tzinfo=UTC)
            ):
                raise ValueError(
                    "Venue probe returned a market outside its predeclared historical date"
                )
            probe["ticker"] = market.ticker
            probe["close_time"] = market.close_time.isoformat()
            finish = int(market.close_time.timestamp())
            bars = await venue.get_candlesticks(
                "KXBTC", market.ticker, start_ts=finish - 3600, end_ts=finish, period_interval=1
            )
            probe["minute_candles_in_last_hour"] = len(bars)
            dump(
                output / "venue-probe-source.json",
                {
                    "received_at": datetime.now(UTC).isoformat(),
                    "market": market.model_dump(mode="json"),
                    "candles": [bar.model_dump(mode="json") for bar in bars],
                },
            )
        probe["conclusion"] = (
            "A returned archive is not evidence of receipt before historical decisions; "
            "this probe is not an exhaustive availability claim."
        )
    except Exception as error:
        probe["error"] = str(error)
        probe["conclusion"] = "Probe failed; no claim that all venue history is unavailable."
    finally:
        await venue.close()
    dump(output / "venue-probe.json", probe)
    dump(
        output / "artifacts.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "hourly_candles",
                    "daily_cutoffs",
                    "unclassifiable_days",
                    "regime_days",
                    "downtrend_days",
                    "high_volatility_days",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print(json.dumps(probe, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "experiments/item8/protocol-v1.json"
    )
    args = parser.parse_args()
    asyncio.run(run(args.output, args.protocol))
