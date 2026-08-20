from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, cast
from uuid import uuid4

import httpx
import typer
from pydantic import HttpUrl
from rich.console import Console
from rich.table import Table

from prediction_market_system.backtest import BacktestConfig, BacktestResult, HistoricalBacktester
from prediction_market_system.config import Settings
from prediction_market_system.discord import DiscordAlertService, DiscordWebhookClient
from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketRegimeSnapshot,
    MarketSnapshot,
    Opportunity,
    RecommendationState,
    TerminalRangeContract,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.engine import CryptoThresholdEngine, EngineConfig
from prediction_market_system.paper_alerts import (
    PaperAlertCycleResult,
    PaperAlertRunner,
    classify_market_regime,
)
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
from prediction_market_system.validation import ValidationCampaignReport, ValidationCampaignState
from prediction_market_system.venues.kalshi import (
    CandlestickPeriod,
    KalshiAPIError,
    KalshiCandlestick,
    KalshiClient,
    KalshiEvent,
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
        max_event_bankroll_fraction=settings.max_event_bankroll_fraction,
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
    yes_bid = "N/A" if snapshot.yes_bid is None else f"{snapshot.yes_bid:.2%}"
    yes_ask = "N/A" if snapshot.yes_ask is None else f"{snapshot.yes_ask:.2%}"
    no_bid = "N/A" if snapshot.no_bid is None else f"{snapshot.no_bid:.2%}"
    no_ask = "N/A" if snapshot.no_ask is None else f"{snapshot.no_ask:.2%}"
    yes_size = "N/A" if snapshot.yes_ask_size is None else f"{snapshot.yes_ask_size:,.2f}"
    no_size = "N/A" if snapshot.no_ask_size is None else f"{snapshot.no_ask_size:,.2f}"
    table.add_row("YES bid / ask", f"{yes_bid} / {yes_ask}")
    table.add_row("NO bid / ask", f"{no_bid} / {no_ask}")
    table.add_row("YES ask size", yes_size)
    table.add_row("NO ask size", no_size)
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
    allow_uncalibrated: Annotated[
        bool,
        typer.Option(help="Use the configured fixed margin when no held-out profile exists."),
    ] = False,
) -> None:
    """Fetch and evaluate one supported Kalshi crypto-price contract."""
    settings = Settings()
    market, snapshot = _load_kalshi_market(ticker)
    try:
        contract = market.price_contract(strike)
    except UnsupportedMarketError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ticker") from exc

    engine = CryptoThresholdEngine(_engine_config(settings))
    crypto = CryptoSnapshot(
        symbol=symbol.upper(),
        observed_at=snapshot.observed_at,
        spot_price=spot,
        strike_price=engine.reference_price(contract),
        annualized_volatility=volatility,
        expected_annual_return=expected_return,
    )
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    calibration_profile = repository.latest_uncertainty_calibration(
        symbol=crypto.symbol,
        model_name=engine.model_name(contract),
        as_of=snapshot.observed_at,
        model_version=engine.model_version,
    )
    if calibration_profile is None and not allow_uncalibrated:
        raise typer.BadParameter(
            "no held-out calibration profile is available; run a calibrated backtest first",
            param_hint="--ticker",
        )
    if calibration_profile is None and send_discord:
        raise typer.BadParameter(
            "Discord paper alerts require held-out uncertainty calibration",
            param_hint="--send-discord",
        )
    _, opportunity = engine.evaluate(snapshot, crypto, contract, calibration_profile)
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
    max_events: Annotated[
        int,
        typer.Option(min=1, max=5_000, help="Maximum settled events to fetch."),
    ] = 240,
    range_contracts_per_event: Annotated[
        int,
        typer.Option(
            min=1,
            max=500,
            help="Outcome-independent range-contract sample retained per event.",
        ),
    ] = 41,
    history_hours: Annotated[
        int,
        typer.Option(min=1, max=168, help="Pre-expiry candlestick history per contract."),
    ] = 24,
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
        max_events,
        range_contracts_per_event,
        history_hours,
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


@app.command("paper-alert-archive")
def paper_alert_archive(
    series: Annotated[str, typer.Option(help="Kalshi series ticker to archive.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    campaign_start: Annotated[
        str,
        typer.Option(help="Earliest UTC date eligible for automated archival."),
    ],
    period: Annotated[
        int,
        typer.Option(help="Candlestick length in minutes: 1, 60, or 1440."),
    ] = 1,
    catch_up_days: Annotated[
        int,
        typer.Option(min=1, max=31, help="Maximum completed UTC days checked per run."),
    ] = 7,
    max_events: Annotated[
        int,
        typer.Option(min=1, max=5_000, help="Maximum settled events fetched per day."),
    ] = 100,
    range_contracts_per_event: Annotated[
        int,
        typer.Option(min=1, max=500, help="Range contracts retained per event."),
    ] = 3,
    history_hours: Annotated[
        int,
        typer.Option(min=1, max=168, help="Pre-expiry candlestick history per contract."),
    ] = 24,
) -> None:
    """Archive completed UTC days for future paper-model validation."""
    if period not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--period")
    normalized_series = series.upper()
    normalized_symbol = symbol.upper()
    period_interval = cast(CandlestickPeriod, period)
    today = _utc_day(datetime.now(UTC))
    first_day = max(
        _utc_day(_parse_timestamp(campaign_start)), today - timedelta(days=catch_up_days)
    )
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()

    archived = 0
    skipped = 0
    window_start = first_day
    while window_start < today:
        window_end = window_start + timedelta(days=1)
        if repository.validation_archive_succeeded(
            series_ticker=normalized_series,
            symbol=normalized_symbol,
            start_at=window_start,
            end_at=window_end,
            period_interval=period_interval,
        ):
            skipped += 1
            window_start = window_end
            continue
        repository.begin_validation_archive(
            series_ticker=normalized_series,
            symbol=normalized_symbol,
            start_at=window_start,
            end_at=window_end,
            period_interval=period_interval,
        )
        try:
            market_batch = asyncio.run(
                _fetch_kalshi_history(
                    normalized_series,
                    window_start,
                    window_end,
                    period_interval,
                    max_events,
                    range_contracts_per_event,
                    history_hours,
                )
            )
            candle_count = sum(len(candles) for candles in market_batch.candlesticks.values())
            if market_batch.markets and candle_count == 0:
                raise ValueError(
                    "Kalshi returned settled markets but no point-in-time candlesticks"
                )
            market_written = repository.save_kalshi_history(
                series_ticker=normalized_series,
                observed_at=market_batch.observed_at,
                markets=market_batch.markets,
                candlesticks=market_batch.candlesticks,
                period_interval=period_interval,
                series_fee_changes=market_batch.series_fee_changes,
                event_fee_changes=market_batch.event_fee_changes,
            )
            research_batch = asyncio.run(
                _fetch_research_data(
                    normalized_symbol,
                    window_start,
                    window_end,
                    period_interval * 60,
                    None,
                )
            )
            research_written = repository.save_research_data(
                spot_candles=research_batch.spot_candles,
                volatility_observations=research_batch.volatility_observations,
                funding_observations=research_batch.funding_observations,
                derivatives_snapshots=research_batch.derivatives_snapshots,
            )
            counts = {
                "market_snapshots": market_written.market_snapshots,
                "candlesticks": candle_count,
                "resolutions": market_written.resolutions,
                "spot_candles": research_written.spot_candles,
                "volatility_observations": research_written.volatility_observations,
                "funding_observations": research_written.funding_observations,
            }
            repository.complete_validation_archive(
                series_ticker=normalized_series,
                symbol=normalized_symbol,
                start_at=window_start,
                end_at=window_end,
                period_interval=period_interval,
                counts=counts,
            )
            archived += 1
            console.print(
                f"Archived {window_start.date()} • "
                f"{counts['candlesticks']} market candles • "
                f"{counts['spot_candles']} spot candles"
            )
        except Exception as exc:
            repository.complete_validation_archive(
                series_ticker=normalized_series,
                symbol=normalized_symbol,
                start_at=window_start,
                end_at=window_end,
                period_interval=period_interval,
                error=str(exc),
            )
            if isinstance(
                exc,
                (CoinbaseDataError, DeribitDataError, KalshiAPIError, ValueError),
            ):
                console.print(
                    f"[red]Validation archive failed for {window_start.date()}: {exc}[/red]"
                )
                raise typer.Exit(code=1) from exc
            raise
        window_start = window_end

    console.print(f"Validation archive complete • {archived} archived • {skipped} already stored")


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
    allow_uncalibrated: Annotated[
        bool,
        typer.Option(help="Permit fixed-margin folds when training outcomes are insufficient."),
    ] = False,
    minimum_calibration_samples: Annotated[
        int,
        typer.Option(min=1, help="Minimum independent resolved events per model."),
    ] = 30,
    calibration_bins: Annotated[
        int,
        typer.Option(min=1, max=20, help="Maximum equal-frequency calibration bins."),
    ] = 5,
    calibration_confidence: Annotated[
        float,
        typer.Option(min=0.50, max=0.999, help="Wilson calibration confidence level."),
    ] = 0.95,
    calibration_lead_minutes: Annotated[
        int,
        typer.Option(min=0, max=1440, help="Fixed pre-expiry calibration decision lead."),
    ] = 5,
    minimum_validation_events: Annotated[
        int,
        typer.Option(min=1, help="Minimum traded held-out events for approval."),
    ] = 20,
    minimum_validation_folds: Annotated[
        int,
        typer.Option(min=1, help="Minimum held-out folds with trades for approval."),
    ] = 2,
    minimum_return_on_cost: Annotated[
        float,
        typer.Option(help="Held-out return must strictly exceed this threshold."),
    ] = 0.0,
    maximum_brier_score: Annotated[
        float,
        typer.Option(min=0.000001, max=1.0, help="Maximum event-weighted held-out Brier."),
    ] = 0.25,
    max_events: Annotated[
        int,
        typer.Option(min=1, max=5_000, help="Maximum stored events to replay."),
    ] = 500,
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
            require_calibration=not allow_uncalibrated,
            minimum_calibration_samples=minimum_calibration_samples,
            maximum_calibration_bins=calibration_bins,
            calibration_confidence=calibration_confidence,
            calibration_lead_seconds=calibration_lead_minutes * 60,
            minimum_validation_events=minimum_validation_events,
            minimum_validation_folds=minimum_validation_folds,
            minimum_return_on_cost=minimum_return_on_cost,
            maximum_brier_score=maximum_brier_score,
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
        max_events=max_events,
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
    table.add_column("Calibration", justify="right")
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
            str(sum(profile.sample_count for profile in fold_result.calibration_profiles)),
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
    validation_table = Table(title="Paper-alert model approval")
    validation_table.add_column("Model")
    validation_table.add_column("Calibration events", justify="right")
    validation_table.add_column("Held-out events", justify="right")
    validation_table.add_column("Return", justify="right")
    validation_table.add_column("Brier", justify="right")
    validation_table.add_column("Decision")
    for validation in result.model_validations:
        validation_table.add_row(
            validation.model_name,
            str(validation.independent_calibration_events),
            str(validation.held_out_events),
            ("—" if validation.return_on_cost is None else f"{validation.return_on_cost:.2%}"),
            "—" if validation.brier_score is None else f"{validation.brier_score:.4f}",
            (
                "APPROVED"
                if validation.accepted_for_paper_alerts
                else "; ".join(validation.rejection_reasons)
            ),
        )
    console.print(validation_table)
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
    missing_calibration = sum(fold.missing_calibration_signals for fold in result.folds)
    if missing_calibration:
        console.print(
            f"[yellow]Skipped {missing_calibration} signals without held-out calibration.[/yellow]"
        )


@app.command("paper-alert-validate")
def paper_alert_validate(
    series: Annotated[str, typer.Option(help="Stored Kalshi series ticker.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    campaign_start: Annotated[str, typer.Option(help="Validation campaign UTC start.")],
    period: Annotated[int, typer.Option(help="Stored candle interval in minutes.")] = 1,
    train_days: Annotated[int, typer.Option(min=1)] = 90,
    test_days: Annotated[int, typer.Option(min=1)] = 30,
    step_days: Annotated[int, typer.Option(min=1)] = 30,
    realized_window_days: Annotated[int, typer.Option(min=1, max=365)] = 30,
    minimum_calibration_samples: Annotated[int, typer.Option(min=1)] = 30,
    minimum_validation_events: Annotated[int, typer.Option(min=1)] = 20,
    minimum_validation_folds: Annotated[int, typer.Option(min=1)] = 2,
    minimum_return_on_cost: float = 0.0,
    maximum_brier_score: Annotated[float, typer.Option(min=0.000001, max=1.0)] = 0.25,
    max_events: Annotated[int, typer.Option(min=1, max=5_000)] = 5_000,
    send_discord: Annotated[bool, typer.Option(help="Update Discord validation status.")] = True,
) -> None:
    """Run validation when coverage is ready; otherwise report collection progress."""
    if period not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--period")
    settings = Settings()
    if send_discord and settings.discord_webhook_url is None:
        raise typer.BadParameter("set PMS_DISCORD_WEBHOOK_URL in .env first")
    normalized_series = series.upper()
    normalized_symbol = symbol.upper()
    period_interval = cast(CandlestickPeriod, period)
    start_at = _utc_day(_parse_timestamp(campaign_start))
    required_days = train_days + test_days + step_days * (minimum_validation_folds - 1)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    coverage_days, coverage_end = repository.validation_archive_coverage(
        series_ticker=normalized_series,
        symbol=normalized_symbol,
        period_interval=period_interval,
        campaign_start=start_at,
    )
    result: BacktestResult | None = None
    if coverage_days >= required_days:
        end_at = start_at + timedelta(days=coverage_days)
        config = BacktestConfig(
            series_ticker=normalized_series,
            symbol=normalized_symbol,
            start=start_at,
            end=end_at,
            period_minutes=period_interval,
            realized_window_days=realized_window_days,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            minimum_calibration_samples=minimum_calibration_samples,
            minimum_validation_events=minimum_validation_events,
            minimum_validation_folds=minimum_validation_folds,
            minimum_return_on_cost=minimum_return_on_cost,
            maximum_brier_score=maximum_brier_score,
        )
        markets = repository.load_kalshi_backtest_data(
            series_ticker=normalized_series,
            start=start_at,
            end=end_at,
            period_interval=period_interval,
            max_events=max_events,
        )
        if markets:
            result = HistoricalBacktester(repository, _engine_config(settings)).run(config, markets)
            repository.save_backtest_result(result)

    validations = () if result is None else result.model_validations
    approved = sum(item.accepted_for_paper_alerts for item in validations)
    if coverage_days < required_days or result is None:
        state = ValidationCampaignState.COLLECTING
    elif approved == len(validations) and validations:
        state = ValidationCampaignState.APPROVED
    elif approved:
        state = ValidationCampaignState.PARTIALLY_APPROVED
    else:
        state = ValidationCampaignState.REJECTED
    report = ValidationCampaignReport(
        series_ticker=normalized_series,
        symbol=normalized_symbol,
        generated_at=datetime.now(UTC),
        state=state,
        coverage_start=start_at,
        coverage_end=coverage_end,
        coverage_days=coverage_days,
        required_days=required_days,
        validations=validations,
        run_id=None if result is None else str(result.run_id),
    )
    message_id: str | None = None
    if send_discord and settings.discord_webhook_url is not None:
        message_id = asyncio.run(
            _publish_validation_report(
                settings.discord_webhook_url.get_secret_value(),
                repository,
                report,
            )
        )
    repository.save_validation_campaign(
        series_ticker=normalized_series,
        symbol=normalized_symbol,
        state=state.value,
        payload=_validation_report_payload(report),
        discord_message_id=message_id,
    )
    console.print(
        f"Validation campaign: [bold]{state.value}[/bold] • "
        f"{coverage_days}/{required_days} archived days"
    )
    for validation in validations:
        decision = (
            "APPROVED"
            if validation.accepted_for_paper_alerts
            else "; ".join(validation.rejection_reasons)
        )
        console.print(f"{validation.model_name}: {decision}")


@app.command("paper-alert-research")
def paper_alert_research(
    series: Annotated[str, typer.Option(help="Kalshi series ticker for regime records.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    interval: Annotated[
        int,
        typer.Option(help="Research candle interval in minutes: 1, 60, or 1440."),
    ] = 60,
    realized_window_days: Annotated[
        int,
        typer.Option(min=1, max=365, help="Trailing regime and volatility window."),
    ] = 30,
    trend_threshold: Annotated[
        float,
        typer.Option(min=0.0, help="Absolute trailing return defining an up/down trend."),
    ] = 0.05,
    low_volatility: Annotated[
        float,
        typer.Option(min=0.0, help="Realized volatility below this is low."),
    ] = 0.40,
    high_volatility: Annotated[
        float,
        typer.Option(min=0.0, help="Realized volatility at or above this is high."),
    ] = 0.80,
) -> None:
    """Refresh hourly research data and persist one auditable regime snapshot."""
    if interval not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--interval")
    if low_volatility >= high_volatility:
        raise typer.BadParameter(
            "must be greater than --low-volatility",
            param_hint="--high-volatility",
        )
    normalized_series = series.upper()
    normalized_symbol = symbol.upper()
    interval_seconds = interval * 60
    now = datetime.now(UTC)
    boundary_timestamp = int(now.timestamp()) // interval_seconds * interval_seconds
    research_end = datetime.fromtimestamp(boundary_timestamp, UTC)
    research_start = research_end - timedelta(
        days=realized_window_days,
        seconds=interval_seconds,
    )
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run_id = repository.begin_research_sync(
        symbol=normalized_symbol,
        event_ticker=None,
        request={
            "purpose": "paper-alert-research",
            "series_ticker": normalized_series,
            "start": research_start.isoformat(),
            "end": research_end.isoformat(),
            "interval_seconds": interval_seconds,
            "realized_window_days": realized_window_days,
        },
    )
    try:
        batch = asyncio.run(
            _fetch_research_data(
                normalized_symbol,
                research_start,
                research_end,
                interval_seconds,
                None,
            )
        )
        written = repository.save_research_data(
            spot_candles=batch.spot_candles,
            volatility_observations=batch.volatility_observations,
            funding_observations=batch.funding_observations,
            derivatives_snapshots=batch.derivatives_snapshots,
        )
        context = repository.research_context_as_of(
            symbol=normalized_symbol,
            as_of=research_end,
            interval_seconds=interval_seconds,
            realized_window_seconds=realized_window_days * 24 * 60 * 60,
            optional_max_age_seconds=2 * interval_seconds,
        )
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
        regime = classify_market_regime(
            context,
            batch.spot_candles,
            trend_threshold=trend_threshold,
            low_volatility_threshold=low_volatility,
            high_volatility_threshold=high_volatility,
        )
        repository.save_market_regime(series_ticker=normalized_series, regime=regime)
        repository.complete_research_sync(run_id, result=written)
    except Exception as exc:
        repository.complete_research_sync(run_id, error=str(exc))
        if isinstance(
            exc,
            (
                CoinbaseDataError,
                DeribitDataError,
                KalshiAPIError,
                ResearchDataUnavailable,
                ValueError,
            ),
        ):
            console.print(f"[red]Paper-alert research sync failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        raise
    console.print(
        f"Research sync [bold]{run_id}[/bold] • {regime.label} • "
        f"{written.spot_candles} spot • {written.volatility_observations} volatility"
    )


@app.command("paper-alerts")
def paper_alerts(
    series: Annotated[str, typer.Option(help="Open Kalshi series ticker to scan.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
    interval: Annotated[
        int,
        typer.Option(help="Stored research candle interval: 1, 60, or 1440 minutes."),
    ] = 60,
    realized_window_days: Annotated[
        int,
        typer.Option(min=1, max=365, help="Trailing realized-volatility window."),
    ] = 30,
    max_markets: Annotated[
        int,
        typer.Option(min=1, max=5_000, help="Maximum open markets to evaluate."),
    ] = 1_000,
    expected_return: Annotated[
        float,
        typer.Option(help="Annualized physical drift assumption."),
    ] = 0.0,
    send_discord: Annotated[
        bool,
        typer.Option(help="Deliver approved entry candidates; default is shadow mode."),
    ] = False,
    allow_unapproved_discord: Annotated[
        bool,
        typer.Option(
            help=(
                "Deliver calibrated candidates without held-out approval for manual "
                "review; requires --send-discord."
            )
        ),
    ] = False,
) -> None:
    """Evaluate open markets from stored research; default to delivery-free shadow mode."""
    if interval not in {1, 60, 1440}:
        raise typer.BadParameter("must be 1, 60, or 1440", param_hint="--interval")
    if allow_unapproved_discord and not send_discord:
        raise typer.BadParameter(
            "requires --send-discord",
            param_hint="--allow-unapproved-discord",
        )
    settings = Settings()
    if send_discord and settings.discord_webhook_url is None:
        raise typer.BadParameter("set PMS_DISCORD_WEBHOOK_URL in .env first")

    normalized_series = series.upper()
    normalized_symbol = symbol.upper()
    interval_seconds = interval * 60
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    decision_at = datetime.now(UTC)
    try:
        live_spot = asyncio.run(_fetch_live_spot(normalized_symbol, decision_at))
        repository.save_research_data(spot_candles=[live_spot])
        context = repository.research_context_as_of(
            symbol=normalized_symbol,
            as_of=decision_at,
            interval_seconds=interval_seconds,
            realized_window_seconds=realized_window_days * 24 * 60 * 60,
            optional_max_age_seconds=2 * interval_seconds,
        ).model_copy(update={"spot": live_spot})
    except (CoinbaseDataError, ResearchDataUnavailable, ValueError) as exc:
        console.print(f"[red]Paper-alert evaluation data unavailable: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    regime = repository.latest_market_regime(
        series_ticker=normalized_series,
        symbol=normalized_symbol,
        as_of=decision_at,
    )
    if regime is None:
        console.print("[red]Run paper-alert-research before market evaluation.[/red]")
        raise typer.Exit(code=1)

    markets = _load_kalshi_markets(normalized_series, max_markets)
    cycle_id = str(uuid4())
    repository.save_paper_alert_cycle(
        cycle_id=cycle_id,
        series_ticker=normalized_series,
        observed_at=decision_at,
    )
    result = asyncio.run(
        _run_paper_alert_cycle(
            repository=repository,
            settings=settings,
            markets=markets,
            cycle_id=cycle_id,
            context=context,
            regime=regime,
            expected_annual_return=expected_return,
            send_discord=send_discord,
            allow_unapproved_discord=allow_unapproved_discord,
        )
    )

    mode = "delivery enabled" if send_discord else "shadow"
    table = Table(title=f"Paper-alert cycle: {normalized_series} / {regime.label} / {mode}")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Open markets discovered", str(result.discovered))
    table.add_row("Calibrated evaluations", str(result.evaluated))
    table.add_row("WATCH evaluations", str(result.watch))
    table.add_row("Discord entries delivered", str(result.delivered))
    table.add_row("Unsupported markets", str(result.unsupported))
    table.add_row("Missing calibration", str(result.uncalibrated))
    table.add_row("Unapproved model", str(result.unapproved))
    table.add_row("Failures", str(len(result.failures)))
    console.print(table)
    console.print(
        f"Regime: {regime.label} • trailing return {regime.trailing_return:+.2%} • "
        f"realized volatility {regime.realized_volatility:.2%}"
    )
    for error in result.failures:
        console.print(f"[red]{error}[/red]")
    if result.failures or result.uncalibrated or result.unapproved:
        raise typer.Exit(code=1)


@app.command("paper-alert-maintain")
def paper_alert_maintain(
    series: Annotated[str, typer.Option(help="Kalshi series ticker to compact.")],
    watch_retention_days: Annotated[
        int,
        typer.Option(min=1, max=365, help="Days of detailed WATCH evaluations to retain."),
    ] = 14,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Apply compaction; without this flag, preview only."),
    ] = False,
) -> None:
    """Roll up and prune old WATCH-only audit records while preserving signals."""
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    cutoff_at = datetime.now(UTC) - timedelta(days=watch_retention_days)
    result = repository.compact_watch_history(
        series_ticker=series,
        cutoff_at=cutoff_at,
        apply=apply,
    )

    mode = "applied" if result.applied else "preview"
    table = Table(title=f"Paper-alert WATCH maintenance: {series.upper()} / {mode}")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Eligible WATCH checks", str(result.eligible_checks))
    table.add_row("Rolled-up WATCH checks", str(result.rolled_up_checks))
    table.add_row("Deleted detailed checks", str(result.deleted_checks))
    table.add_row("Deleted WATCH opportunities", str(result.deleted_opportunities))
    table.add_row("Deleted orphaned forecasts", str(result.deleted_forecasts))
    table.add_row("Deleted empty cycles", str(result.deleted_cycles))
    console.print(table)
    console.print(f"Detailed WATCH cutoff: {result.cutoff_at.isoformat()}")
    if not result.applied:
        console.print("[yellow]Preview only; pass --apply to compact stored history.[/yellow]")


@app.command("paper-alert-status")
def paper_alert_status(
    series: Annotated[str, typer.Option(help="Kalshi series ticker.")],
    symbol: Annotated[str, typer.Option(help="Crypto symbol, such as BTC.")],
) -> None:
    """Show activity since the last status request and forward regime coverage."""
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    activity = repository.paper_alert_status_since_last_request(
        series_ticker=series,
        symbol=symbol,
    )
    activity_table = Table(
        title=f"Paper-alert activity: {series.upper()} / {symbol.upper()}"
    )
    activity_table.add_column("Metric")
    activity_table.add_column("Count", justify="right")
    activity_table.add_row("Cycles since last request", str(activity.cycles))
    activity_table.add_row(
        "Discord alerts delivered (all time)",
        str(activity.delivered_alerts),
    )
    activity_table.add_row("Resolved alerts (all time)", str(activity.resolved_alerts))
    activity_table.add_row("Profitable alerts (all time)", str(activity.profitable_alerts))
    activity_table.add_row("Unresolved alerts (all time)", str(activity.unresolved_alerts))
    console.print(activity_table)
    if activity.previous_requested_at is None:
        console.print(
            "[yellow]No previous status request; counts include all recorded activity.[/yellow]"
        )
    else:
        console.print(
            "Previous status request: "
            f"{activity.previous_requested_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    coverage = repository.market_regime_coverage(
        series_ticker=series,
        symbol=symbol,
    )
    table = Table(title=f"Paper-alert regime coverage: {series.upper()} / {symbol.upper()}")
    table.add_column("Regime")
    table.add_column("Observations", justify="right")
    table.add_column("First observed (UTC)")
    table.add_column("Last observed (UTC)")
    for row in coverage:
        table.add_row(
            str(row["regime"]),
            str(row["observation_count"]),
            str(row["first_observed_at"]),
            str(row["last_observed_at"]),
        )
    console.print(table)
    if not coverage:
        console.print("[yellow]No paper-alert regime observations recorded yet.[/yellow]")


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
    max_events: int,
    range_contracts_per_event: int,
    history_hours: int,
) -> _KalshiHistoryBatch:
    try:
        return asyncio.run(
            _fetch_kalshi_history(
                series_ticker,
                start_at,
                end_at,
                period_interval,
                max_events,
                range_contracts_per_event,
                history_hours,
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
        markets: list[KalshiMarket] = []
        event_titles: dict[str, str] = {}
        cursor: str | None = None
        while len(markets) < limit:
            page = await client.list_markets(
                series_ticker=series.upper() if series else None,
                limit=min(100, limit - len(markets)),
                cursor=cursor,
            )
            for market in page.markets:
                event_title = event_titles.get(market.event_ticker)
                if event_title is None:
                    event = await client.get_event(market.event_ticker)
                    event_title = event.event.title
                    event_titles[market.event_ticker] = event_title
                markets.append(market.model_copy(update={"event_title": event_title}))
            if not page.cursor:
                break
            if page.cursor == cursor:
                raise KalshiAPIError("Kalshi repeated a live-market cursor")
            cursor = page.cursor
        return markets
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
    max_events: int,
    range_contracts_per_event: int,
    history_hours: int,
) -> _KalshiHistoryBatch:
    client = KalshiClient()
    try:
        events: list[KalshiEvent] = []
        cursor: str | None = None
        while len(events) < max_events:
            page = await client.list_events(
                status="settled",
                series_ticker=series_ticker,
                min_close_ts=int(start_at.timestamp()),
                limit=200,
                cursor=cursor,
            )
            for event in page.events:
                if event.strike_date is None:
                    continue
                if start_at <= event.strike_date <= end_at:
                    events.append(event)
                    if len(events) == max_events:
                        break
            if not page.cursor or not page.events:
                break
            if page.cursor == cursor:
                raise KalshiAPIError("Kalshi repeated an event cursor")
            cursor = page.cursor

        markets: list[KalshiMarket] = []
        selected_by_event: dict[str, list[KalshiMarket]] = {}
        for event in events:
            response = await client.get_event(event.event_ticker)
            event_markets = [
                market
                for market in response.markets
                if market.result in {"yes", "no"} and market.expiry <= end_at
            ]
            selected_markets = _select_history_markets(
                event_markets,
                range_contracts_per_event=range_contracts_per_event,
            )
            markets.extend(selected_markets)
            selected_by_event[event.event_ticker] = selected_markets

        semaphore = asyncio.Semaphore(8)

        async def fetch_candles(market: KalshiMarket) -> tuple[str, list[KalshiCandlestick]]:
            async with semaphore:
                candle_start = max(start_at, market.expiry - timedelta(hours=history_hours))
                candles = await client.get_candlesticks(
                    series_ticker,
                    market.ticker,
                    start_ts=int(candle_start.timestamp()),
                    end_ts=int(min(end_at, market.expiry).timestamp()),
                    period_interval=period_interval,
                )
                return market.ticker, candles

        candle_results = await asyncio.gather(
            *(
                fetch_candles(market)
                for event_markets in selected_by_event.values()
                for market in event_markets
            )
        )
        candlesticks = dict(candle_results)
        series_fee_changes = await client.get_series_fee_changes(series_ticker)
        event_fee_changes: list[KalshiEventFeeChange] = []
        for event in events:
            event_cursor: str | None = None
            while True:
                fee_page = await client.get_event_fee_changes(
                    event.event_ticker,
                    cursor=event_cursor,
                )
                event_fee_changes.extend(fee_page.event_fee_changes)
                if not fee_page.cursor:
                    break
                if fee_page.cursor == event_cursor:
                    raise KalshiAPIError("Kalshi repeated an event-fee cursor")
                event_cursor = fee_page.cursor

        return _KalshiHistoryBatch(
            observed_at=end_at,
            markets=markets,
            candlesticks=candlesticks,
            series_fee_changes=series_fee_changes,
            event_fee_changes=event_fee_changes,
        )
    finally:
        await client.close()


def _select_history_markets(
    markets: list[KalshiMarket],
    *,
    range_contracts_per_event: int,
) -> list[KalshiMarket]:
    ranges: list[tuple[float, KalshiMarket]] = []
    thresholds_by_model: dict[str, list[tuple[float, KalshiMarket]]] = {}
    for market in markets:
        try:
            contract = market.price_contract()
        except UnsupportedMarketError:
            continue
        if isinstance(contract, TerminalRangeContract):
            ranges.append(((contract.lower_bound + contract.upper_bound) / 2.0, market))
        else:
            thresholds_by_model.setdefault(
                CryptoThresholdEngine.model_name(contract),
                [],
            ).append((contract.strike_price, market))

    selected_thresholds: list[KalshiMarket] = []
    for model_name in sorted(thresholds_by_model):
        model_markets = sorted(
            thresholds_by_model[model_name],
            key=lambda item: (item[0], item[1].ticker),
        )
        selected_thresholds.append(model_markets[len(model_markets) // 2][1])

    ranges.sort(key=lambda item: item[0])
    if len(ranges) <= range_contracts_per_event:
        return [market for _, market in ranges] + selected_thresholds

    centered_count = max(1, range_contracts_per_event * 3 // 4)
    center = len(ranges) // 2
    start = max(0, center - centered_count // 2)
    end = min(len(ranges), start + centered_count)
    start = max(0, end - centered_count)
    selected_indexes = set(range(start, end))
    remaining = range_contracts_per_event - len(selected_indexes)
    if remaining > 0:
        denominator = max(remaining - 1, 1)
        selected_indexes.update(
            round(index * (len(ranges) - 1) / denominator) for index in range(remaining)
        )
    selected_ranges = [ranges[index][1] for index in sorted(selected_indexes)]
    return selected_ranges + selected_thresholds


async def _fetch_live_spot(symbol: str, as_of: datetime) -> SpotCandle:
    boundary_timestamp = int(as_of.timestamp()) // 60 * 60
    end_at = datetime.fromtimestamp(boundary_timestamp, UTC)
    start_at = end_at - timedelta(minutes=5)
    client = CoinbaseClient()
    try:
        candles = await client.get_candles(
            f"{symbol}-USD",
            start_at=start_at,
            end_at=end_at,
            interval_seconds=60,
        )
    finally:
        await client.close()
    if not candles:
        raise ResearchDataUnavailable("Coinbase returned no completed one-minute spot candle")
    return candles[-1]


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


async def _run_paper_alert_cycle(
    *,
    repository: SQLiteRepository,
    settings: Settings,
    markets: list[KalshiMarket],
    context: ResearchContext,
    regime: MarketRegimeSnapshot,
    expected_annual_return: float,
    cycle_id: str,
    send_discord: bool,
    allow_unapproved_discord: bool = False,
) -> PaperAlertCycleResult:
    kalshi = KalshiClient()
    discord: DiscordWebhookClient | None = None
    alert_service: DiscordAlertService | None = None
    if send_discord:
        if settings.discord_webhook_url is None:
            raise ValueError("Discord webhook URL is required when delivery is enabled")
        discord = DiscordWebhookClient(settings.discord_webhook_url.get_secret_value())
        alert_service = DiscordAlertService(repository, discord)
    try:
        runner = PaperAlertRunner(
            repository=repository,
            engine=CryptoThresholdEngine(_engine_config(settings)),
            market_reader=kalshi,
            alert_service=alert_service,
            maximum_spot_age=timedelta(seconds=settings.maximum_live_spot_age_seconds),
        )
        return await runner.run(
            markets=markets,
            context=context,
            regime=regime,
            expected_annual_return=expected_annual_return,
            cycle_id=cycle_id,
            deliver_entries=send_discord,
            allow_unapproved_delivery=allow_unapproved_discord,
        )
    finally:
        await kalshi.close()
        if discord is not None:
            await discord.close()


async def _publish_validation_report(
    webhook_url: str,
    repository: SQLiteRepository,
    report: ValidationCampaignReport,
) -> str:
    client = DiscordWebhookClient(webhook_url)
    previous_message_id = repository.validation_campaign_message_id(
        series_ticker=report.series_ticker,
        symbol=report.symbol,
    )
    try:
        if previous_message_id is not None:
            try:
                return await client.update_validation_report(previous_message_id, report)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != httpx.codes.NOT_FOUND:
                    raise
        return await client.send_validation_report(report)
    finally:
        await client.close()


def _validation_report_payload(report: ValidationCampaignReport) -> dict[str, object]:
    return {
        "series_ticker": report.series_ticker,
        "symbol": report.symbol,
        "generated_at": report.generated_at.isoformat(),
        "state": report.state.value,
        "coverage_start": report.coverage_start.isoformat(),
        "coverage_end": report.coverage_end.isoformat(),
        "coverage_days": report.coverage_days,
        "required_days": report.required_days,
        "run_id": report.run_id,
        "validations": [validation.model_dump(mode="json") for validation in report.validations],
    }


def _utc_day(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


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
        "Uncertainty",
        (f"±{forecast.uncertainty_margin:.2%} ({forecast.uncertainty_source})"),
    )
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
