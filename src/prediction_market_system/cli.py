from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, cast

import typer
from pydantic import HttpUrl
from rich.console import Console
from rich.table import Table

from prediction_market_system.config import Settings
from prediction_market_system.discord import DiscordAlertService, DiscordWebhookClient
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSnapshot,
    Opportunity,
    RecommendationState,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.storage import SQLiteRepository
from prediction_market_system.venues.kalshi import (
    CandlestickPeriod,
    KalshiAPIError,
    KalshiCandlestick,
    KalshiClient,
    KalshiEventFeeChange,
    KalshiMarket,
    KalshiSeriesFeeChange,
    UnsupportedMarketError,
)

app = typer.Typer(
    name="pms",
    help="Auditable prediction-market research and Discord decision support.",
    no_args_is_help=True,
)
console = Console()


@dataclass(frozen=True)
class _KalshiHistoryBatch:
    observed_at: datetime
    markets: list[KalshiMarket]
    candlesticks: dict[str, list[KalshiCandlestick]]
    series_fee_changes: list[KalshiSeriesFeeChange]
    event_fee_changes: list[KalshiEventFeeChange]


def _parse_timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise typer.BadParameter("use an ISO-8601 timestamp with a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _engine_config(settings: Settings) -> EngineConfig:
    return EngineConfig(
        paper_bankroll=settings.paper_bankroll,
        min_conservative_edge=settings.min_conservative_edge,
        uncertainty_margin=settings.uncertainty_margin,
        structural_weight=settings.structural_weight,
        fee_rate=settings.fee_rate,
        binary_fee_coefficient=settings.binary_fee_coefficient,
        slippage_bps=settings.slippage_bps,
        resolution_haircut=settings.resolution_haircut,
        minimum_ask_size=settings.minimum_ask_size,
        fractional_kelly=settings.fractional_kelly,
        max_bankroll_fraction=settings.max_bankroll_fraction,
        minimum_seconds_to_expiry=settings.minimum_seconds_to_expiry,
    )


@app.command("init-db")
def init_db() -> None:
    """Create the local append-oriented audit database."""
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    console.print(f"Initialized audit database at [bold]{settings.database_path}[/bold]")


@app.command()
def evaluate(
    market_id: Annotated[str, typer.Option(help="Venue-specific market identifier.")],
    question: Annotated[str, typer.Option(help="Exact contract question.")],
    venue: Annotated[str, typer.Option(help="Prediction-market venue name.")],
    expires_at: Annotated[str, typer.Option(help="ISO-8601 contract expiry.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    spot: Annotated[float, typer.Option(min=0.0, help="Current spot price.")],
    strike: Annotated[float, typer.Option(min=0.0, help="Contract price threshold.")],
    volatility: Annotated[
        float,
        typer.Option(min=0.0, help="Annualized volatility as a decimal."),
    ],
    yes_bid: Annotated[float, typer.Option(min=0.0, max=1.0)],
    yes_ask: Annotated[float, typer.Option(min=0.0, max=1.0)],
    no_bid: Annotated[float, typer.Option(min=0.0, max=1.0)],
    no_ask: Annotated[float, typer.Option(min=0.0, max=1.0)],
    resolution_rule: Annotated[str, typer.Option(help="Exact resolution rule.")],
    yes_ask_size: Annotated[float, typer.Option(min=0.0)] = 100.0,
    no_ask_size: Annotated[float, typer.Option(min=0.0)] = 100.0,
    expected_return: Annotated[
        float,
        typer.Option(help="Annualized physical drift assumption."),
    ] = 0.0,
    observed_at: Annotated[
        str | None,
        typer.Option(help="ISO-8601 observation time; defaults to now."),
    ] = None,
    market_url: Annotated[str | None, typer.Option(help="Direct market URL.")] = None,
    send_discord: Annotated[
        bool,
        typer.Option(help="Deliver actionable entries to the configured webhook."),
    ] = False,
) -> None:
    """Evaluate one crypto price-threshold contract and persist the result."""
    settings = Settings()
    observed = _parse_timestamp(observed_at)
    market = MarketSnapshot(
        market_id=market_id,
        question=question,
        venue=venue,
        observed_at=observed,
        expires_at=_parse_timestamp(expires_at),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        yes_ask_size=yes_ask_size,
        no_ask_size=no_ask_size,
        resolution_rule=resolution_rule,
        market_url=HttpUrl(market_url) if market_url else None,
    )
    crypto = CryptoSnapshot(
        symbol=symbol.upper(),
        observed_at=observed,
        spot_price=spot,
        strike_price=strike,
        annualized_volatility=volatility,
        expected_annual_return=expected_return,
    )

    engine = CryptoThresholdEngine(_engine_config(settings))
    _, opportunity = engine.evaluate(market, crypto)
    _persist_and_maybe_alert(settings, opportunity, send_discord)


@app.command("kalshi-markets")
def kalshi_markets(
    series: Annotated[
        str | None,
        typer.Option(help="Optional Kalshi series ticker filter."),
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
) -> None:
    """List currently open Kalshi markets using the public REST API."""
    markets = _load_kalshi_markets(series, limit)
    table = Table(title="Open Kalshi markets")
    table.add_column("Ticker")
    table.add_column("Question")
    table.add_column("YES bid / ask")
    table.add_column("Early close")
    for market in markets:
        table.add_row(
            market.ticker,
            market.question,
            f"{float(market.yes_bid_dollars):.1%} / {float(market.yes_ask_dollars):.1%}",
            str(market.can_close_early),
        )
    console.print(table)


@app.command("kalshi-inspect")
def kalshi_inspect(
    ticker: Annotated[str, typer.Option(help="Kalshi market ticker.")],
) -> None:
    """Inspect current public Kalshi metadata and executable quotes."""
    market, snapshot = _load_kalshi_market(ticker)

    table = Table(title=f"Kalshi {market.ticker}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Question", snapshot.question)
    table.add_row("Status", market.status)
    table.add_row("Expiry", snapshot.expires_at.isoformat())
    table.add_row("YES bid / ask", f"{snapshot.yes_bid:.2%} / {snapshot.yes_ask:.2%}")
    table.add_row("NO bid / ask", f"{snapshot.no_bid:.2%} / {snapshot.no_ask:.2%}")
    table.add_row("YES ask size", f"{snapshot.yes_ask_size:,.2f}")
    table.add_row("NO ask size", f"{snapshot.no_ask_size:,.2f}")
    table.add_row("Strike type", market.strike_type or "Missing")
    table.add_row("Floor strike", str(market.floor_strike or "Missing"))
    table.add_row("Can close early", str(market.can_close_early))
    console.print(table)


@app.command("kalshi-evaluate")
def kalshi_evaluate(
    ticker: Annotated[str, typer.Option(help="Kalshi market ticker.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    spot: Annotated[float, typer.Option(min=0.0, help="Current spot price.")],
    volatility: Annotated[
        float,
        typer.Option(min=0.0, help="Annualized volatility as a decimal."),
    ],
    strike: Annotated[
        float | None,
        typer.Option(min=0.0, help="Override missing Kalshi threshold metadata."),
    ] = None,
    expected_return: Annotated[
        float,
        typer.Option(help="Annualized physical drift assumption."),
    ] = 0.0,
    send_discord: Annotated[
        bool,
        typer.Option(help="Deliver actionable entries to the configured webhook."),
    ] = False,
) -> None:
    """Fetch and evaluate one live terminal-threshold Kalshi contract."""
    settings = Settings()
    market, snapshot = _load_kalshi_market(ticker)
    try:
        threshold = market.terminal_threshold_strike(strike)
    except UnsupportedMarketError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ticker") from exc

    crypto = CryptoSnapshot(
        symbol=symbol.upper(),
        observed_at=snapshot.observed_at,
        spot_price=spot,
        strike_price=threshold,
        annualized_volatility=volatility,
        expected_annual_return=expected_return,
    )
    engine = CryptoThresholdEngine(_engine_config(settings))
    _, opportunity = engine.evaluate(snapshot, crypto)
    _persist_and_maybe_alert(settings, opportunity, send_discord)


@app.command("kalshi-sync-history")
def kalshi_sync_history(
    series: Annotated[str, typer.Option(help="Kalshi series ticker to archive.")],
    start: Annotated[str, typer.Option(help="Inclusive ISO-8601 candlestick start.")],
    end: Annotated[str, typer.Option(help="Inclusive ISO-8601 candlestick end.")],
    period: Annotated[
        int,
        typer.Option(help="Candlestick length in minutes: 1, 60, or 1440."),
    ] = 60,
    max_markets: Annotated[
        int,
        typer.Option(min=1, max=10_000, help="Maximum archived markets to fetch."),
    ] = 100,
) -> None:
    """Persist archived Kalshi markets and their point-in-time research data."""
    if period not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--period")
    start_at = _parse_timestamp(start)
    end_at = _parse_timestamp(end)
    if start_at > end_at:
        raise typer.BadParameter("must not precede --start", param_hint="--end")

    series_ticker = series.upper()
    period_interval = cast(CandlestickPeriod, period)
    batch = _load_kalshi_history(
        series_ticker,
        start_at,
        end_at,
        period_interval,
        max_markets,
    )

    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    written = repository.save_kalshi_history(
        series_ticker=series_ticker,
        observed_at=batch.observed_at,
        markets=batch.markets,
        candlesticks=batch.candlesticks,
        period_interval=period_interval,
        series_fee_changes=batch.series_fee_changes,
        event_fee_changes=batch.event_fee_changes,
    )

    candle_count = sum(len(candles) for candles in batch.candlesticks.values())
    resolution_count = sum(bool(market.result) for market in batch.markets)
    table = Table(title=f"Kalshi historical sync: {series_ticker}")
    table.add_column("Dataset")
    table.add_column("Fetched", justify="right")
    table.add_column("Inserted", justify="right")
    table.add_row("Market snapshots", str(len(batch.markets)), str(written.market_snapshots))
    table.add_row("Candlesticks", str(candle_count), str(written.candlesticks))
    table.add_row("Rule snapshots", str(len(batch.markets)), str(written.rule_snapshots))
    table.add_row("Resolutions", str(resolution_count), str(written.resolutions))
    table.add_row(
        "Series fee changes",
        str(len(batch.series_fee_changes)),
        str(written.series_fee_changes),
    )
    table.add_row(
        "Event fee overrides",
        str(len(batch.event_fee_changes)),
        str(written.event_fee_changes),
    )
    console.print(table)
    console.print(f"Stored in [bold]{settings.database_path}[/bold]")


@app.command()
def history(
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
) -> None:
    """Show recent locally persisted recommendations."""
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    rows = repository.opportunity_history(limit)

    table = Table(title="Recent prediction-market evaluations")
    table.add_column("Generated (UTC)")
    table.add_column("Market")
    table.add_column("State")
    table.add_column("Side")
    table.add_column("Edge", justify="right")
    for row in rows:
        forecast = row["forecast"]
        edge = row.get("conservative_net_edge")
        table.add_row(
            str(forecast["generated_at"]),
            str(row["market"]["market_id"]),
            str(row["state"]),
            str(row.get("side") or "—"),
            "—" if edge is None else f"{float(edge):.2%}",
        )
    console.print(table)


@app.command("discord-test")
def discord_test() -> None:
    """Send a non-trading health message to verify webhook delivery."""
    settings = Settings()
    if settings.discord_webhook_url is None:
        raise typer.BadParameter("set PMS_DISCORD_WEBHOOK_URL in .env first")
    message_id = asyncio.run(
        _send_discord_health_check(
            settings.discord_webhook_url.get_secret_value(),
        )
    )
    console.print(f"Discord health check delivered as message [bold]{message_id}[/bold]")


def _load_kalshi_market(ticker: str) -> tuple[KalshiMarket, MarketSnapshot]:
    try:
        return asyncio.run(_fetch_kalshi_market(ticker.upper()))
    except KalshiAPIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def _load_kalshi_markets(series: str | None, limit: int) -> list[KalshiMarket]:
    try:
        return asyncio.run(_fetch_kalshi_markets(series, limit))
    except KalshiAPIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def _load_kalshi_history(
    series_ticker: str,
    start_at: datetime,
    end_at: datetime,
    period_interval: CandlestickPeriod,
    max_markets: int,
) -> _KalshiHistoryBatch:
    try:
        return asyncio.run(
            _fetch_kalshi_history(
                series_ticker,
                start_at,
                end_at,
                period_interval,
                max_markets,
            )
        )
    except KalshiAPIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


async def _fetch_kalshi_markets(
    series: str | None,
    limit: int,
) -> list[KalshiMarket]:
    client = KalshiClient()
    try:
        result = await client.list_markets(
            series_ticker=series.upper() if series else None,
            limit=limit,
        )
        return result.markets
    finally:
        await client.close()


async def _fetch_kalshi_market(ticker: str) -> tuple[KalshiMarket, MarketSnapshot]:
    client = KalshiClient()
    try:
        return await client.get_market_snapshot(ticker)
    finally:
        await client.close()


async def _fetch_kalshi_history(
    series_ticker: str,
    start_at: datetime,
    end_at: datetime,
    period_interval: CandlestickPeriod,
    max_markets: int,
) -> _KalshiHistoryBatch:
    client = KalshiClient()
    try:
        markets: list[KalshiMarket] = []
        cursor: str | None = None
        while len(markets) < max_markets:
            page = await client.list_historical_markets(
                series_ticker=series_ticker,
                limit=min(1000, max_markets - len(markets)),
                cursor=cursor,
            )
            markets.extend(page.markets[: max_markets - len(markets)])
            if not page.cursor or not page.markets:
                break
            if page.cursor == cursor:
                raise KalshiAPIError("Kalshi repeated a historical-markets cursor")
            cursor = page.cursor

        start_ts = int(start_at.timestamp())
        end_ts = int(end_at.timestamp())
        candlesticks: dict[str, list[KalshiCandlestick]] = {}
        for market in markets:
            candlesticks[market.ticker] = await client.get_historical_candlesticks(
                market.ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
            )

        series_fee_changes = await client.get_series_fee_changes(series_ticker)
        event_fee_changes: list[KalshiEventFeeChange] = []
        for event_ticker in sorted({market.event_ticker for market in markets}):
            event_cursor: str | None = None
            while True:
                fee_page = await client.get_event_fee_changes(
                    event_ticker,
                    cursor=event_cursor,
                )
                event_fee_changes.extend(fee_page.event_fee_changes)
                if not fee_page.cursor:
                    break
                if fee_page.cursor == event_cursor:
                    raise KalshiAPIError("Kalshi repeated an event-fee cursor")
                event_cursor = fee_page.cursor

        return _KalshiHistoryBatch(
            observed_at=datetime.now(UTC),
            markets=markets,
            candlesticks=candlesticks,
            series_fee_changes=series_fee_changes,
            event_fee_changes=event_fee_changes,
        )
    finally:
        await client.close()


def _persist_and_maybe_alert(
    settings: Settings,
    opportunity: Opportunity,
    send_discord: bool,
) -> None:
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    repository.save_evaluation(opportunity.forecast, opportunity)
    _print_evaluation(opportunity)

    actionable = opportunity.state in {
        RecommendationState.ENTER_YES,
        RecommendationState.ENTER_NO,
    }
    if send_discord and not actionable:
        console.print("[yellow]No Discord alert sent: recommendation is WATCH.[/yellow]")
    elif send_discord:
        if settings.discord_webhook_url is None:
            raise typer.BadParameter(
                "set PMS_DISCORD_WEBHOOK_URL in .env before using --send-discord"
            )
        message_id = asyncio.run(
            _publish(
                repository,
                settings.discord_webhook_url.get_secret_value(),
                opportunity,
            )
        )
        console.print(f"Discord alert delivered as message [bold]{message_id}[/bold]")


async def _publish(
    repository: SQLiteRepository,
    webhook_url: str,
    opportunity: Opportunity,
) -> str:
    client = DiscordWebhookClient(webhook_url)
    try:
        service = DiscordAlertService(repository, client)
        return await service.publish(opportunity)
    finally:
        await client.close()


async def _send_discord_health_check(webhook_url: str) -> str:
    client = DiscordWebhookClient(webhook_url)
    try:
        return await client.send_health_check()
    finally:
        await client.close()


def _print_evaluation(opportunity: Opportunity) -> None:
    forecast = opportunity.forecast
    table = Table(title=opportunity.market.question)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("State", opportunity.state.value)
    table.add_row("Best side", opportunity.side.value if opportunity.side else "None")
    table.add_row("Model YES", f"{forecast.probability_yes:.2%}")
    table.add_row(
        "YES interval",
        f"{forecast.lower_probability_yes:.2%}–{forecast.upper_probability_yes:.2%}",
    )
    table.add_row("Market YES", f"{forecast.market_probability_yes:.2%}")
    table.add_row("Structural YES", f"{forecast.structural_probability_yes:.2%}")
    table.add_row(
        "Conservative edge",
        (
            "N/A"
            if opportunity.conservative_net_edge is None
            else f"{opportunity.conservative_net_edge:.2%}"
        ),
    )
    table.add_row("Paper exposure cap", f"${opportunity.suggested_max_exposure:,.2f}")
    console.print(table)
    for warning in opportunity.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
