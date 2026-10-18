"""Read-only scoring of delivered Discord alerts against settled Kalshi outcomes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from prediction_market_system.backtest import EffectiveFee, kalshi_taker_fee

Outcome = Literal["yes", "no"]
FetchResult = Callable[[str], Awaitable[str | None]]

PRICE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("< 0.25", 0.0, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-0.75", 0.50, 0.75),
    ("0.75-0.90", 0.75, 0.90),
    (">= 0.90", 0.90, 1.01),
)


@dataclass(frozen=True)
class DeliveredAlert:
    opportunity_id: str
    market_id: str
    side: Literal["YES", "NO"]
    created_at: datetime
    entry_price: float
    probability_yes: float | None
    market_probability_yes: float | None
    conservative_net_edge: float | None


@dataclass(frozen=True)
class ScoredAlert:
    alert: DeliveredAlert
    result: str | None
    fee_per_contract: Decimal

    @property
    def resolved(self) -> bool:
        return self.result in {"yes", "no"}

    @property
    def won(self) -> bool:
        return self.resolved and self.alert.side.casefold() == self.result

    @property
    def cost(self) -> Decimal:
        return Decimal(str(self.alert.entry_price)) + self.fee_per_contract

    @property
    def pnl(self) -> Decimal:
        return (Decimal("1") if self.won else Decimal("0")) - self.cost


@dataclass(frozen=True)
class OutcomeSummary:
    label: str
    alerts: int
    resolved: int
    unresolved: int
    wins: int
    total_cost: Decimal
    total_pnl: Decimal
    average_entry_price: float | None
    average_edge: float | None
    model_brier: float | None
    market_brier: float | None

    @property
    def win_rate(self) -> float | None:
        return None if self.resolved == 0 else self.wins / self.resolved

    @property
    def return_on_cost(self) -> float | None:
        if self.total_cost == 0:
            return None
        return float(self.total_pnl / self.total_cost)


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(str(value))


def connect_read_only(database_path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    if not database_path.exists():
        raise FileNotFoundError(f"database not found: {database_path}")
    mode = "immutable=1" if immutable else "mode=ro"
    connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?{mode}", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_delivered_alerts(
    database_path: Path,
    *,
    series_ticker: str,
    immutable: bool = False,
) -> tuple[list[DeliveredAlert], dict[str, str]]:
    with closing(connect_read_only(database_path, immutable=immutable)) as connection:
        rows = connection.execute(
            """
            SELECT opportunity_id, market_id, created_at, payload_json
            FROM alert_events
            WHERE status = 'delivered' AND market_id LIKE ?
            ORDER BY created_at, opportunity_id
            """,
            (f"{series_ticker.upper()}-%",),
        ).fetchall()
        stored_results = {
            str(row["ticker"]): str(row["result"])
            for row in connection.execute(
                """
                SELECT ticker, result
                FROM kalshi_resolutions
                WHERE ticker IN (
                    SELECT market_id FROM alert_events WHERE status = 'delivered'
                )
                ORDER BY observed_at
                """
            )
            if row["result"]
        }

    alerts: list[DeliveredAlert] = []
    for row in rows:
        payload: dict[str, Any] = json.loads(str(row["payload_json"]))
        side = str(payload.get("side") or "").upper()
        price = payload.get("executable_price")
        if side not in {"YES", "NO"} or price is None:
            continue
        forecast = payload.get("forecast") or {}
        alerts.append(
            DeliveredAlert(
                opportunity_id=str(row["opportunity_id"]),
                market_id=str(row["market_id"]),
                side="YES" if side == "YES" else "NO",
                created_at=_parse_time(row["created_at"]),
                entry_price=float(price),
                probability_yes=_optional_float(forecast.get("probability_yes")),
                market_probability_yes=_optional_float(forecast.get("market_probability_yes")),
                conservative_net_edge=_optional_float(payload.get("conservative_net_edge")),
            )
        )
    return alerts, stored_results


async def resolve_outcomes(
    market_ids: Iterable[str],
    stored_results: dict[str, str],
    fetch: FetchResult | None,
) -> tuple[dict[str, str | None], int]:
    results: dict[str, str | None] = {}
    fetched = 0
    for market_id in sorted(set(market_ids)):
        if market_id in stored_results:
            results[market_id] = stored_results[market_id]
        elif fetch is None:
            results[market_id] = None
        else:
            results[market_id] = await fetch(market_id)
            fetched += 1
    return results, fetched


def score_alerts(
    alerts: list[DeliveredAlert],
    results: dict[str, str | None],
    *,
    fee: EffectiveFee,
    lot_contracts: int,
) -> list[ScoredAlert]:
    scored: list[ScoredAlert] = []
    for alert in alerts:
        lot_fee = kalshi_taker_fee(lot_contracts, alert.entry_price, fee)
        scored.append(
            ScoredAlert(
                alert=alert,
                result=results.get(alert.market_id) or None,
                fee_per_contract=lot_fee / lot_contracts,
            )
        )
    return scored


def first_alert_per_market(scored: list[ScoredAlert]) -> list[ScoredAlert]:
    first: dict[tuple[str, str], ScoredAlert] = {}
    for item in sorted(scored, key=lambda value: value.alert.created_at):
        first.setdefault((item.alert.market_id, item.alert.side), item)
    return sorted(first.values(), key=lambda value: value.alert.created_at)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def summarize(label: str, scored: list[ScoredAlert]) -> OutcomeSummary:
    resolved = [item for item in scored if item.resolved]
    model_errors: list[float] = []
    market_errors: list[float] = []
    for item in resolved:
        outcome = 1.0 if item.result == "yes" else 0.0
        if item.alert.probability_yes is not None:
            model_errors.append((item.alert.probability_yes - outcome) ** 2)
        if item.alert.market_probability_yes is not None:
            market_errors.append((item.alert.market_probability_yes - outcome) ** 2)
    edges = [
        item.alert.conservative_net_edge
        for item in resolved
        if item.alert.conservative_net_edge is not None
    ]
    return OutcomeSummary(
        label=label,
        alerts=len(scored),
        resolved=len(resolved),
        unresolved=len(scored) - len(resolved),
        wins=sum(1 for item in resolved if item.won),
        total_cost=sum((item.cost for item in resolved), Decimal("0")),
        total_pnl=sum((item.pnl for item in resolved), Decimal("0")),
        average_entry_price=_mean([item.alert.entry_price for item in resolved]),
        average_edge=_mean(edges),
        model_brier=_mean(model_errors),
        market_brier=_mean(market_errors),
    )


def breakdown(scored: list[ScoredAlert]) -> list[OutcomeSummary]:
    rows = [summarize("All", scored)]
    for side in ("YES", "NO"):
        rows.append(summarize(f"Side {side}", [i for i in scored if i.alert.side == side]))
    for label, low, high in PRICE_BUCKETS:
        rows.append(
            summarize(
                f"Entry {label}",
                [i for i in scored if low <= i.alert.entry_price < high],
            )
        )
    return rows


def summary_payload(summary: OutcomeSummary) -> dict[str, object]:
    return {
        "label": summary.label,
        "alerts": summary.alerts,
        "resolved": summary.resolved,
        "unresolved": summary.unresolved,
        "wins": summary.wins,
        "win_rate": summary.win_rate,
        "total_cost": str(summary.total_cost),
        "total_pnl": str(summary.total_pnl),
        "return_on_cost": summary.return_on_cost,
        "average_entry_price": summary.average_entry_price,
        "average_conservative_edge": summary.average_edge,
        "model_brier": summary.model_brier,
        "market_brier": summary.market_brier,
    }
