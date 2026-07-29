from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from statistics import NormalDist

from pydantic import BaseModel, ConfigDict, Field

from prediction_market_system.domain import (
    CryptoSnapshot,
    MarketSide,
    MarketSnapshot,
    Opportunity,
    ProbabilityForecast,
    RecommendationState,
)

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_EPSILON = 1e-6


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_bankroll: float = Field(default=10_000.0, gt=0.0)
    min_conservative_edge: float = Field(default=0.03, ge=0.0, le=1.0)
    uncertainty_margin: float = Field(default=0.05, ge=0.0, le=0.5)
    structural_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    fee_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    binary_fee_coefficient: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=25.0, ge=0.0)
    resolution_haircut: float = Field(default=0.01, ge=0.0, le=1.0)
    minimum_ask_size: float = Field(default=10.0, ge=0.0)
    fractional_kelly: float = Field(default=0.25, ge=0.0, le=1.0)
    max_bankroll_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    minimum_seconds_to_expiry: int = Field(default=300, ge=0)


@dataclass(frozen=True)
class _Candidate:
    side: MarketSide
    ask: float
    ask_size: float
    conservative_probability: float
    effective_cost: float
    conservative_net_edge: float


def _clamp_probability(value: float) -> float:
    return min(max(value, _EPSILON), 1.0 - _EPSILON)


def _logit(probability: float) -> float:
    probability = _clamp_probability(probability)
    return math.log(probability / (1.0 - probability))


def _logistic(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


class CryptoThresholdEngine:
    """Evaluate binary crypto price-threshold contracts.

    The structural forecast uses a lognormal terminal-price distribution. It is
    blended in log-odds space with the current market estimate, then widened by
    a configurable uncertainty margin before any recommendation is considered.
    """

    model_name = "crypto-threshold-market-anchor"
    model_version = "0.1.0"

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def evaluate(
        self,
        market: MarketSnapshot,
        crypto: CryptoSnapshot,
    ) -> tuple[ProbabilityForecast, Opportunity]:
        structural_probability = self._structural_probability(market, crypto)
        market_probability = self._market_probability(market)
        final_probability = self._blend_probabilities(
            market_probability,
            structural_probability,
        )

        lower_probability = max(0.0, final_probability - self.config.uncertainty_margin)
        upper_probability = min(1.0, final_probability + self.config.uncertainty_margin)
        supporting, opposing = self._evidence(
            market_probability,
            structural_probability,
            crypto,
        )

        forecast = ProbabilityForecast(
            market_id=market.market_id,
            generated_at=market.observed_at,
            probability_yes=final_probability,
            lower_probability_yes=lower_probability,
            upper_probability_yes=upper_probability,
            structural_probability_yes=structural_probability,
            market_probability_yes=market_probability,
            model_name=self.model_name,
            model_version=self.model_version,
            supporting_evidence=tuple(supporting),
            opposing_evidence=tuple(opposing),
        )
        opportunity = self._recommend(market, forecast, crypto)
        return forecast, opportunity

    def _structural_probability(
        self,
        market: MarketSnapshot,
        crypto: CryptoSnapshot,
    ) -> float:
        time_to_expiry = (market.expires_at - market.observed_at).total_seconds()
        years = max(time_to_expiry / _SECONDS_PER_YEAR, _EPSILON)
        sigma = crypto.annualized_volatility
        numerator = (
            math.log(crypto.spot_price / crypto.strike_price)
            + (crypto.expected_annual_return - 0.5 * sigma**2) * years
        )
        z_score = numerator / (sigma * math.sqrt(years))
        return _clamp_probability(NormalDist().cdf(z_score))

    @staticmethod
    def _market_probability(market: MarketSnapshot) -> float:
        yes_midpoint = (market.yes_bid + market.yes_ask) / 2.0
        no_implied_yes = 1.0 - (market.no_bid + market.no_ask) / 2.0
        return _clamp_probability((yes_midpoint + no_implied_yes) / 2.0)

    def _blend_probabilities(
        self,
        market_probability: float,
        structural_probability: float,
    ) -> float:
        weight = self.config.structural_weight
        blended_log_odds = (1.0 - weight) * _logit(market_probability) + weight * _logit(
            structural_probability
        )
        return _clamp_probability(_logistic(blended_log_odds))

    @staticmethod
    def _evidence(
        market_probability: float,
        structural_probability: float,
        crypto: CryptoSnapshot,
    ) -> tuple[list[str], list[str]]:
        supporting: list[str] = []
        opposing: list[str] = []
        difference = structural_probability - market_probability
        distance_percent = (crypto.spot_price / crypto.strike_price - 1.0) * 100.0

        if difference >= 0.02:
            supporting.append(
                f"Structural YES probability is {difference:.1%} above the market estimate."
            )
        elif difference <= -0.02:
            opposing.append(
                f"Structural YES probability is {abs(difference):.1%} below the market estimate."
            )
        else:
            supporting.append("Structural and market probabilities broadly agree.")

        position = "above" if distance_percent >= 0 else "below"
        evidence = (
            f"{crypto.symbol} spot is {abs(distance_percent):.2f}% {position} the contract strike."
        )
        if distance_percent >= 0:
            supporting.append(evidence)
        else:
            opposing.append(evidence)
        return supporting, opposing

    def _candidate(
        self,
        side: MarketSide,
        ask: float,
        ask_size: float,
        conservative_probability: float,
    ) -> _Candidate:
        trading_cost_multiplier = 1.0 + self.config.fee_rate + self.config.slippage_bps / 10_000.0
        binary_contract_fee = self.config.binary_fee_coefficient * ask * (1.0 - ask)
        effective_cost = ask * trading_cost_multiplier + binary_contract_fee
        edge = conservative_probability - effective_cost - self.config.resolution_haircut
        return _Candidate(
            side=side,
            ask=ask,
            ask_size=ask_size,
            conservative_probability=conservative_probability,
            effective_cost=effective_cost,
            conservative_net_edge=edge,
        )

    def _recommend(
        self,
        market: MarketSnapshot,
        forecast: ProbabilityForecast,
        crypto: CryptoSnapshot,
    ) -> Opportunity:
        yes = self._candidate(
            MarketSide.YES,
            market.yes_ask,
            market.yes_ask_size,
            forecast.lower_probability_yes,
        )
        no = self._candidate(
            MarketSide.NO,
            market.no_ask,
            market.no_ask_size,
            1.0 - forecast.upper_probability_yes,
        )
        best = max((yes, no), key=lambda candidate: candidate.conservative_net_edge)

        warnings: list[str] = []
        reasons = [
            (
                f"{best.side} conservative edge is "
                f"{best.conservative_net_edge:.2%} after modeled costs."
            )
        ]

        if abs(market.observed_at - crypto.observed_at) > timedelta(minutes=1):
            warnings.append("Market and crypto observations are more than one minute apart.")
        if market.yes_ask + market.no_ask < 0.99:
            warnings.append("Complementary asks appear incoherent; verify quote freshness.")
        if market.yes_bid + market.no_bid > 1.01:
            warnings.append("Complementary bids appear incoherent; verify quote freshness.")

        seconds_to_expiry = (market.expires_at - market.observed_at).total_seconds()
        enough_time = seconds_to_expiry >= self.config.minimum_seconds_to_expiry
        enough_liquidity = best.ask_size >= self.config.minimum_ask_size
        cost_is_valid = best.effective_cost < 1.0
        edge_is_large_enough = best.conservative_net_edge >= self.config.min_conservative_edge

        if not enough_time:
            warnings.append("Contract is too close to expiry for a new entry.")
        if not enough_liquidity:
            warnings.append(
                f"Displayed size is below the {self.config.minimum_ask_size:g} unit minimum."
            )
        if not cost_is_valid:
            warnings.append("Modeled all-in cost is at least the maximum payout.")
        if not edge_is_large_enough:
            warnings.append(
                f"Edge is below the {self.config.min_conservative_edge:.2%} alert threshold."
            )

        should_enter = all((enough_time, enough_liquidity, cost_is_valid, edge_is_large_enough))
        if should_enter:
            state = (
                RecommendationState.ENTER_YES
                if best.side is MarketSide.YES
                else RecommendationState.ENTER_NO
            )
            exposure = self._suggested_exposure(best)
        else:
            state = RecommendationState.WATCH
            exposure = 0.0

        return Opportunity(
            market=market,
            forecast=forecast,
            state=state,
            side=best.side,
            executable_price=best.ask,
            conservative_probability=best.conservative_probability,
            conservative_net_edge=best.conservative_net_edge,
            suggested_max_exposure=exposure,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def _suggested_exposure(self, candidate: _Candidate) -> float:
        denominator = max(1.0 - candidate.effective_cost, _EPSILON)
        full_kelly_fraction = max(
            0.0,
            (candidate.conservative_probability - candidate.effective_cost) / denominator,
        )
        bankroll_fraction = min(
            self.config.fractional_kelly * full_kelly_fraction,
            self.config.max_bankroll_fraction,
        )
        bankroll_cap = self.config.paper_bankroll * bankroll_fraction
        displayed_liquidity_cap = candidate.ask_size * candidate.ask
        return round(min(bankroll_cap, displayed_liquidity_cap), 2)
