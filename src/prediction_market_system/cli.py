from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, cast

import typer
from pydantic import HttpUrl
from rich.console import Console
from rich.table import Table

from prediction_market_system.backtest import (
    BacktestConfig,
    HistoricalBacktester,
)
from prediction_market_system.config import Settings
from prediction_market_system.discord import DiscordAlertService, DiscordWebhookClient
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSnapshot,
    Opportunity,
    RecommendationState,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.research import (
    DerivativesSnapshot,
    EventDataSnapshot,
    FundingObservation,
    ResearchContext,
    ResearchDataUnavailable,
    SpotCandle,
    VolatilityObservation,
)
from prediction_market_system.research_storage import ResearchWriteResult
from prediction_market_system.sources import (
    CoinbaseClient,
    CoinbaseDataError,
    DeribitClient,
    DeribitDataError,
)
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


@dataclass(frozen=True)
class _ResearchDataBatch:
    spot_candles: list[SpotCandle]
    volatility_observations: list[VolatilityObservation]
    funding_observations: list[FundingObservation]
    derivatives_snapshots: list[DerivativesSnapshot]
    event_snapshots: list[EventDataSnapshot]


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
    contract = ThresholdContract(
        model_kind=ThresholdModelKind.TERMINAL,
        direction=ThresholdDirection.ABOVE,
        strike_price=strike,
    )
    _, opportunity = engine.evaluate(market, crypto, contract)
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
    """Fetch and evaluate one supported threshold Kalshi contract."""
    settings = Settings()
    market, snapshot = _load_kalshi_market(ticker)
    try:
        contract = market.threshold_contract(strike)
    except UnsupportedMarketError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ticker") from exc

    crypto = CryptoSnapshot(
        symbol=symbol.upper(),
        observed_at=snapshot.observed_at,
        spot_price=spot,
        strike_price=contract.strike_price,
        annualized_volatility=volatility,
        expected_annual_return=expected_return,
    )
    engine = CryptoThresholdEngine(_engine_config(settings))
    _, opportunity = engine.evaluate(snapshot, crypto, contract)
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


@app.command("sync-research-data")
def sync_research_data(
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    start: Annotated[str, typer.Option(help="Inclusive ISO-8601 data start.")],
    end: Annotated[str, typer.Option(help="Exclusive ISO-8601 data end.")],
    interval: Annotated[
        int,
        typer.Option(help="Candle and volatility interval in minutes: 1, 60, or 1440."),
    ] = 60,
    realized_window_days: Annotated[
        int,
        typer.Option(min=1, max=365, help="Trailing realized-volatility window."),
    ] = 30,
    event_ticker: Annotated[
        str | None,
        typer.Option(help="Optional Kalshi event live-data ticker."),
    ] = None,
) -> None:
    """Synchronize timestamped spot, volatility, derivatives, and event inputs."""
    if interval not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--interval")
    start_at = _parse_timestamp(start)
    end_at = _parse_timestamp(end)
    if start_at >= end_at:
        raise typer.BadParameter("must be after --start", param_hint="--end")

    normalized_symbol = symbol.upper()
    normalized_event = event_ticker.upper() if event_ticker else None
    interval_seconds = interval * 60
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run_id = repository.begin_research_sync(
        symbol=normalized_symbol,
        event_ticker=normalized_event,
        request={
            "start": start_at.isoformat(),
            "end": end_at.isoformat(),
            "interval_seconds": interval_seconds,
            "realized_window_days": realized_window_days,
        },
    )

    try:
        batch = asyncio.run(
            _fetch_research_data(
                normalized_symbol,
                start_at,
                end_at,
                interval_seconds,
                normalized_event,
            )
        )
        written = repository.save_research_data(
            spot_candles=batch.spot_candles,
            volatility_observations=batch.volatility_observations,
            funding_observations=batch.funding_observations,
            derivatives_snapshots=batch.derivatives_snapshots,
            event_snapshots=batch.event_snapshots,
        )
        context: ResearchContext | None = None
        with suppress(ResearchDataUnavailable):
            context = repository.research_context_as_of(
                symbol=normalized_symbol,
                event_ticker=normalized_event,
                as_of=end_at,
                interval_seconds=interval_seconds,
                realized_window_seconds=realized_window_days * 24 * 60 * 60,
                optional_max_age_seconds=2 * interval_seconds,
            )
        if context is not None:
            realized_written = repository.save_research_data(
                volatility_observations=[context.realized_volatility]
            )
            written = ResearchWriteResult(
                spot_candles=written.spot_candles,
                volatility_observations=(
                    written.volatility_observations + realized_written.volatility_observations
                ),
                funding_observations=written.funding_observations,
                derivatives_snapshots=written.derivatives_snapshots,
                event_snapshots=written.event_snapshots,
            )
        repository.complete_research_sync(run_id, result=written)
    except Exception as exc:
        repository.complete_research_sync(run_id, error=str(exc))
        if isinstance(
            exc,
            (CoinbaseDataError, DeribitDataError, KalshiAPIError, ValueError),
        ):
            console.print(f"[red]Research-data sync failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        raise

    table = Table(title=f"Research data sync: {normalized_symbol}")
    table.add_column("Dataset")
    table.add_column("Fetched", justify="right")
    table.add_column("Inserted", justify="right")
    table.add_row("Coinbase spot candles", str(len(batch.spot_candles)), str(written.spot_candles))
    table.add_row(
        "Volatility observations",
        str(len(batch.volatility_observations) + (1 if context else 0)),
        str(written.volatility_observations),
    )
    table.add_row(
        "Funding observations",
        str(len(batch.funding_observations)),
        str(written.funding_observations),
    )
    table.add_row(
        "Derivatives snapshots",
        str(len(batch.derivatives_snapshots)),
        str(written.derivatives_snapshots),
    )
    table.add_row(
        "Kalshi event snapshots",
        str(len(batch.event_snapshots)),
        str(written.event_snapshots),
    )
    console.print(table)
    if context is None:
        console.print(
            "[yellow]Realized volatility was not materialized: "
            "the database does not yet contain a complete trailing window.[/yellow]"
        )
    console.print(f"Sync run [bold]{run_id}[/bold] stored in {settings.database_path}")


@app.command("research-context")
def research_context(
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    as_of: Annotated[str, typer.Option(help="Point-in-time ISO-8601 cutoff.")],
    event_ticker: Annotated[
        str | None,
        typer.Option(help="Optional Kalshi event ticker."),
    ] = None,
    interval: Annotated[
        int,
        typer.Option(help="Stored spot-candle interval in minutes."),
    ] = 60,
    realized_window_days: Annotated[
        int,
        typer.Option(min=1, max=365, help="Trailing volatility window."),
    ] = 30,
    max_age_minutes: Annotated[
        int,
        typer.Option(min=1, help="Maximum optional-input age."),
    ] = 120,
) -> None:
    """Inspect provenance and staleness for one no-look-ahead research context."""
    if interval not in {1, 5, 15, 30, 60, 120, 360, 1440}:
        raise typer.BadParameter("unsupported stored candle interval", param_hint="--interval")
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    try:
        context = repository.research_context_as_of(
            symbol=symbol.upper(),
            event_ticker=event_ticker.upper() if event_ticker else None,
            as_of=_parse_timestamp(as_of),
            interval_seconds=interval * 60,
            realized_window_seconds=realized_window_days * 24 * 60 * 60,
            optional_max_age_seconds=max_age_minutes * 60,
        )
    except ResearchDataUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Point-in-time research context: {context.symbol}")
    table.add_column("Input")
    table.add_column("Value")
    table.add_column("Source time (UTC)")
    table.add_column("Provider")
    table.add_row("As of", context.as_of.isoformat(), context.as_of.isoformat(), "—")
    table.add_row(
        "Spot close",
        f"${float(context.spot.close):,.2f}",
        context.spot.end_at.isoformat(),
        context.spot.provider,
    )
    table.add_row(
        "Realized volatility",
        f"{context.realized_volatility.annualized_volatility:.2%}",
        context.realized_volatility.observed_at.isoformat(),
        context.realized_volatility.provider,
    )
    table.add_row(
        "Implied volatility",
        "—"
        if context.implied_volatility is None
        else f"{context.implied_volatility.annualized_volatility:.2%}",
        "—"
        if context.implied_volatility is None
        else context.implied_volatility.observed_at.isoformat(),
        "—" if context.implied_volatility is None else context.implied_volatility.provider,
    )
    table.add_row(
        "Funding (1h)",
        "—" if context.funding is None else f"{context.funding.funding_rate_1h:.6%}",
        "—" if context.funding is None else context.funding.observed_at.isoformat(),
        "—" if context.funding is None else context.funding.provider,
    )
    table.add_row(
        "Perpetual basis",
        "—" if context.derivatives is None else f"{context.derivatives.basis:.4%}",
        "—" if context.derivatives is None else context.derivatives.observed_at.isoformat(),
        "—" if context.derivatives is None else context.derivatives.provider,
    )
    table.add_row(
        "Open interest",
        "—" if context.derivatives is None else f"{context.derivatives.open_interest:,.2f}",
        "—" if context.derivatives is None else context.derivatives.observed_at.isoformat(),
        "—" if context.derivatives is None else context.derivatives.provider,
    )
    table.add_row(
        "Event data",
        "—" if context.event_data is None else context.event_data.data_type,
        "—" if context.event_data is None else context.event_data.observed_at.isoformat(),
        "—" if context.event_data is None else context.event_data.provider,
    )
    console.print(table)
    for warning in context.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


@app.command("backtest")
def backtest(
    series: Annotated[str, typer.Option(help="Stored Kalshi series ticker.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    start: Annotated[str, typer.Option(help="Walk-forward training-range start.")],
    end: Annotated[str, typer.Option(help="Exclusive final test-range end.")],
    period: Annotated[
        int,
        typer.Option(help="Stored Kalshi and spot candle interval: 1, 60, or 1440."),
    ] = 60,
    realized_window_days: Annotated[
        int,
        typer.Option(min=1, max=365, help="Trailing realized-volatility window."),
    ] = 30,
    train_days: Annotated[
        int,
        typer.Option(min=1, help="Rolling training-window length."),
    ] = 90,
    test_days: Annotated[
        int,
        typer.Option(min=1, help="Out-of-sample test-window length."),
    ] = 30,
    step_days: Annotated[
        int,
        typer.Option(min=1, help="Days between successive test windows."),
    ] = 30,
    latency_seconds: Annotated[
        int,
        typer.Option(min=0, help="Delay between signal and eligible execution."),
    ] = 30,
    max_volume_participation: Annotated[
        float,
        typer.Option(
            min=0.000001,
            max=1.0,
            help="Maximum fraction of execution-candle volume allowed to fill.",
        ),
    ] = 0.10,
    max_markets: Annotated[
        int,
        typer.Option(min=1, max=10_000, help="Maximum stored markets to replay."),
    ] = 100,
) -> None:
    """Replay resolved Kalshi markets with point-in-time walk-forward evaluation."""
    if period not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--period")
    try:
        config = BacktestConfig(
            series_ticker=series,
            symbol=symbol,
            start=_parse_timestamp(start),
            end=_parse_timestamp(end),
            period_minutes=cast(CandlestickPeriod, period),
            realized_window_days=realized_window_days,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            latency_seconds=latency_seconds,
            max_volume_participation=max_volume_participation,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    markets = repository.load_kalshi_backtest_data(
        series_ticker=config.series_ticker,
        start=config.start,
        end=config.end,
        period_interval=config.period_minutes,
        max_markets=max_markets,
    )
    if not markets:
        console.print("[red]No stored Kalshi markets match the requested backtest.[/red]")
        raise typer.Exit(code=1)

    result = HistoricalBacktester(repository, _engine_config(settings)).run(config, markets)
    repository.save_backtest_result(result)

    table = Table(title=f"Walk-forward backtest: {config.series_ticker}")
    table.add_column("Fold", justify="right")
    table.add_column("Test range (UTC)")
    table.add_column("Signals", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Partial", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")
    for fold_result in result.folds:
        fold = fold_result.fold
        table.add_row(
            str(fold.index + 1),
            f"{fold.test_start.date()} → {fold.test_end.date()}",
            str(fold_result.evaluated_signals),
            str(len(fold_result.trades)),
            str(sum(trade.partial_fill for trade in fold_result.trades)),
            f"${fold_result.total_pnl_dollars:,.2f}",
            ("—" if fold_result.return_on_cost is None else f"{fold_result.return_on_cost:.2%}"),
        )
    console.print(table)
    console.print(
        f"Trades: [bold]{result.total_trades}[/bold] "
        f"({result.partial_fills} partial) • "
        f"Cost: ${result.total_cost_dollars:,.2f} • "
        f"P&L: ${result.total_pnl_dollars:,.2f} • "
        f"Return: {'—' if result.return_on_cost is None else f'{result.return_on_cost:.2%}'} • "
        f"Brier: {'—' if result.brier_score is None else f'{result.brier_score:.4f}'}"
    )
    console.print(f"Backtest run [bold]{result.run_id}[/bold] stored in {settings.database_path}")
    if result.unsupported_markets:
        console.print(
            f"[yellow]Skipped {len(result.unsupported_markets)} unsupported markets.[/yellow]"
        )
    if result.unresolved_markets:
        console.print(
            f"[yellow]Skipped {len(result.unresolved_markets)} unresolved markets.[/yellow]"
        )
    if result.markets_without_candles:
        console.print(
            f"[yellow]Skipped {len(result.markets_without_candles)} "
            "markets without candles.[/yellow]"
        )


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


async def _fetch_research_data(
    symbol: str,
    start_at: datetime,
    end_at: datetime,
    interval_seconds: int,
    event_ticker: str | None,
) -> _ResearchDataBatch:
    coinbase = CoinbaseClient()
    deribit = DeribitClient()
    kalshi = KalshiClient()
    try:
        spot_candles = await coinbase.get_candles(
            f"{symbol}-USD",
            start_at=start_at,
            end_at=end_at,
            interval_seconds=interval_seconds,
        )
        volatility_observations = await deribit.get_dvol_history(
            symbol,
            start_at=start_at,
            end_at=end_at,
            resolution_seconds=interval_seconds,
        )
        funding_observations = await deribit.get_funding_history(
            f"{symbol}-PERPETUAL",
            start_at=start_at,
            end_at=end_at,
        )
        derivatives_snapshots = [await deribit.get_derivatives_snapshot(f"{symbol}-PERPETUAL")]
        event_snapshots: list[EventDataSnapshot] = []
        if event_ticker:
            live_data = await kalshi.get_event_live_data(event_ticker)
            retrieved_at = datetime.now(UTC)
            event_snapshots.append(
                EventDataSnapshot(
                    provider="kalshi",
                    event_ticker=event_ticker,
                    data_type=live_data.type,
                    observed_at=retrieved_at,
                    retrieved_at=retrieved_at,
                    is_historical=live_data.is_historical,
                    details=live_data.details,
                    raw_payload=live_data.model_dump(mode="json"),
                )
            )
        return _ResearchDataBatch(
            spot_candles=spot_candles,
            volatility_observations=volatility_observations,
            funding_observations=funding_observations,
            derivatives_snapshots=derivatives_snapshots,
            event_snapshots=event_snapshots,
        )
    finally:
        await asyncio.gather(
            coinbase.close(),
            deribit.close(),
            kalshi.close(),
        )


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
