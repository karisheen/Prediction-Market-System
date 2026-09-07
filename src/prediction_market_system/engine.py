from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prediction_market_system.calibration import UncertaintyCalibrationProfile
from prediction_market_system.domain import (
    CryptoPriceContract,
    CryptoSnapshot,
    MarketSide,
    MarketSnapshot,
    Opportunity,
    ProbabilityForecast,
    RecommendationState,
    TerminalRangeContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.recipe import MODEL_VERSION, recipe_fingerprint

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_EPSILON = 1e-6

BinaryFeeType = Literal["quadratic", "quadratic_with_maker_fees", "flat"]


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_bankroll: float = Field(default=10_000.0, gt=0.0)
    min_conservative_edge: float = Field(default=0.03, ge=0.0, le=1.0)
    uncertainty_margin: float = Field(default=0.05, ge=0.0, le=0.5)
    structural_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    fee_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    binary_fee_type: BinaryFeeType = "quadratic"
    binary_fee_coefficient: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=25.0, ge=0.0)
    resolution_haircut: float = Field(default=0.01, ge=0.0, le=1.0)
    minimum_ask_size: float = Field(default=10.0, ge=0.0)
    fractional_kelly: float = Field(default=0.25, ge=0.0, le=1.0)
    max_bankroll_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    max_event_bankroll_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    minimum_seconds_to_expiry: int = Field(default=300, ge=0)
    maximum_input_age_seconds: int = Field(default=120, ge=0)


def deployment_policy(config: EngineConfig) -> dict[str, Any]:
    """Execution, cost, freshness, and sizing assumptions used by a delivery decision.

    Probability-recipe fields such as ``structural_weight`` are intentionally omitted.
    Changing them invalidates forecast identity, not this operational fingerprint.
    """
    return {
        "paper_bankroll": config.paper_bankroll,
        "min_conservative_edge": config.min_conservative_edge,
        "uncertainty_margin": config.uncertainty_margin,
        "fee_rate": config.fee_rate,
        "binary_fee_type": config.binary_fee_type,
        "binary_fee_coefficient": config.binary_fee_coefficient,
        "slippage_bps": config.slippage_bps,
        "resolution_haircut": config.resolution_haircut,
        "minimum_ask_size": config.minimum_ask_size,
        "fractional_kelly": config.fractional_kelly,
        "max_bankroll_fraction": config.max_bankroll_fraction,
        "max_event_bankroll_fraction": config.max_event_bankroll_fraction,
        "minimum_seconds_to_expiry": config.minimum_seconds_to_expiry,
        "maximum_input_age_seconds": config.maximum_input_age_seconds,
    }


def deployment_policy_id(config: EngineConfig) -> str:
    return recipe_fingerprint(deployment_policy(config))


def entry_horizon_at(market: MarketSnapshot) -> datetime:
    """Latest instant a new entry can still be about trading *and* the benchmark."""
    return min(market.expires_at, market.effective_observation_end_at)


def remaining_seconds_to_entry_horizon(market: MarketSnapshot, as_of: datetime) -> float:
    return (entry_horizon_at(market) - as_of).total_seconds()


def assert_time_sensitive_entry_controls(
    config: EngineConfig,
    market: MarketSnapshot,
    contract: CryptoPriceContract,
    *,
    as_of: datetime,
) -> None:
    """Fail closed when time-sensitive entry conditions are invalid at ``as_of``.

    Trading close, benchmark observation end, expected settlement, and outcome
    availability are distinct. This revalidates only controls that gate a new
    entry: trading still open, the observation window still open, averaging has
    not begun, and remaining time still meets ``minimum_seconds_to_expiry``.
    """
    if market.expires_at <= as_of:
        raise ValueError("trading has closed")
    observation_end = market.effective_observation_end_at
    if observation_end <= as_of:
        raise ValueError("benchmark observation has already ended")
    if remaining_seconds_to_entry_horizon(market, as_of) < config.minimum_seconds_to_expiry:
        raise ValueError("contract is inside the minimum-time-to-expiry exclusion window")
    if contract.settlement_window_seconds:
        window_start = observation_end - timedelta(seconds=contract.settlement_window_seconds)
        if as_of > window_start:
            raise ValueError("averaging window has already started; observed prefix is required")


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


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _log_normal_cdf(value: float) -> float:
    if value < -10.0:
        inverse_square = 1.0 / (value * value)
        correction = 1.0 - inverse_square + 3.0 * inverse_square**2 - 15.0 * inverse_square**3
        return (
            -0.5 * value * value
            - math.log(-value)
            - 0.5 * math.log(2.0 * math.pi)
            + math.log(correction)
        )
    return math.log(_normal_cdf(value))


def _scaled_normal_cdf(log_scale: float, value: float) -> float:
    log_value = log_scale + _log_normal_cdf(value)
    if log_value <= -745.0:
        return 0.0
    return min(math.exp(log_value), 1.0)


def barrier_hitting_probability(
    *,
    spot_price: float,
    strike_price: float,
    annualized_volatility: float,
    expected_annual_return: float,
    years: float,
    direction: ThresholdDirection,
) -> float:
    if spot_price <= 0.0 or strike_price <= 0.0:
        raise ValueError("spot and barrier prices must be positive")
    if annualized_volatility <= 0.0:
        raise ValueError("annualized volatility must be positive")

    direction_sign = 1.0 if direction is ThresholdDirection.ABOVE else -1.0
    distance = direction_sign * math.log(strike_price / spot_price)
    if distance <= 0.0:
        return 1.0
    if years <= 0.0:
        return 0.0

    sigma = annualized_volatility
    transformed_drift = direction_sign * (expected_annual_return - 0.5 * sigma**2)
    scaled_time = sigma * math.sqrt(years)
    first_z = (transformed_drift * years - distance) / scaled_time
    second_z = (-transformed_drift * years - distance) / scaled_time
    reflection_scale = 2.0 * transformed_drift * distance / sigma**2
    probability = _normal_cdf(first_z) + _scaled_normal_cdf(
        reflection_scale,
        second_z,
    )
    return min(max(probability, 0.0), 1.0)


class CryptoThresholdEngine:
    """Evaluate terminal ranges plus terminal and first-passage crypto thresholds.

    The selected structural forecast is blended in log-odds space with the
    current market estimate, then widened by a configurable uncertainty margin
    before any recommendation is considered.
    """

    model_version = MODEL_VERSION

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def recipe(self, crypto: CryptoSnapshot) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "structural_weight": self.config.structural_weight,
            "expected_annual_return": crypto.expected_annual_return,
            "features": crypto.feature_recipe,
            "terminal_estimator": "gbm-lognormal",
            "barrier_estimator": "gbm-reflection-first-passage",
            "barrier_history": "explicit-observation-start-no-unobserved-prefix",
            "averaging_estimator": "arithmetic-1hz-pre-end-exact-moments-lognormal",
            "horizon": "spot-observation-to-explicit-benchmark-end",
            "market_anchor": "available-bid-ask-midpoints-log-odds",
            "seconds_per_year": _SECONDS_PER_YEAR,
            "probability_clamp": _EPSILON,
        }

    def recipe_id(self, crypto: CryptoSnapshot) -> str:
        return recipe_fingerprint(self.recipe(crypto))

    @staticmethod
    def model_name(contract: CryptoPriceContract) -> str:
        if isinstance(contract, TerminalRangeContract):
            name = "crypto-terminal-range-market-anchor"
        else:
            name = (
                f"crypto-{contract.model_kind.value}-{contract.direction.value}"
                "-threshold-market-anchor"
            )
        if contract.settlement_window_seconds:
            name += f"-arithmetic-{contract.settlement_window_seconds}s"
        return name

    @property
    def event_exposure_cap(self) -> float:
        return self.config.paper_bankroll * self.config.max_event_bankroll_fraction

    @staticmethod
    def reference_price(contract: CryptoPriceContract) -> float:
        if isinstance(contract, TerminalRangeContract):
            return (contract.lower_bound + contract.upper_bound) / 2.0
        return contract.strike_price

    def evaluate(
        self,
        market: MarketSnapshot,
        crypto: CryptoSnapshot,
        contract: CryptoPriceContract,
        calibration_profile: UncertaintyCalibrationProfile | None = None,
    ) -> tuple[ProbabilityForecast, Opportunity]:
        structural_probability = self._structural_probability(market, crypto, contract)
        market_probability = self._market_probability(market)
        final_probability = self._blend_probabilities(
            market_probability,
            structural_probability,
        )

        model_name = self.model_name(contract)
        uncertainty_source: Literal["fixed", "held_out"]
        if calibration_profile is None:
            uncertainty_margin = self.config.uncertainty_margin
            uncertainty_source = "fixed"
            calibration_profile_id = None
        else:
            if calibration_profile.model_name != model_name:
                raise ValueError("calibration profile does not match structural model")
            if calibration_profile.model_version != self.model_version:
                raise ValueError("calibration profile does not match model version")
            if calibration_profile.symbol != crypto.symbol.upper():
                raise ValueError("calibration profile does not match crypto symbol")
            if calibration_profile.recipe_id != self.recipe_id(crypto):
                raise ValueError("calibration profile does not match forecast recipe")
            if calibration_profile.cutoff_at > market.observed_at:
                raise ValueError("calibration profile contains outcomes unavailable at evaluation")
            uncertainty_margin = calibration_profile.margin_for(
                final_probability,
                horizon_seconds=(
                    market.effective_observation_end_at - crypto.observed_at
                ).total_seconds(),
            )
            uncertainty_source = "held_out"
            calibration_profile_id = calibration_profile.profile_id
        lower_probability = max(0.0, final_probability - uncertainty_margin)
        upper_probability = min(1.0, final_probability + uncertainty_margin)
        supporting, opposing = self._evidence(
            market_probability,
            structural_probability,
            crypto,
            contract,
        )

        forecast = ProbabilityForecast(
            market_id=market.market_id,
            generated_at=market.observed_at,
            probability_yes=final_probability,
            lower_probability_yes=lower_probability,
            upper_probability_yes=upper_probability,
            structural_probability_yes=structural_probability,
            market_probability_yes=market_probability,
            model_name=model_name,
            model_version=self.model_version,
            uncertainty_margin=uncertainty_margin,
            uncertainty_source=uncertainty_source,
            calibration_profile_id=calibration_profile_id,
            recipe_id=self.recipe_id(crypto),
            input_manifest={
                "market": market.model_dump(mode="json"),
                "crypto": crypto.model_dump(mode="json"),
                "contract": contract.model_dump(mode="json"),
                "engine_config": self.config.model_dump(mode="json"),
                "recipe": self.recipe(crypto),
                "calibration_profile": (
                    calibration_profile.model_dump(mode="json")
                    if calibration_profile is not None
                    else None
                ),
            },
            supporting_evidence=tuple(supporting),
            opposing_evidence=tuple(opposing),
        )
        opportunity = self._recommend(market, forecast)
        return forecast, opportunity

    def _structural_probability(
        self,
        market: MarketSnapshot,
        crypto: CryptoSnapshot,
        contract: CryptoPriceContract,
    ) -> float:
        age = (market.observed_at - crypto.observed_at).total_seconds()
        if age < 0:
            raise ValueError("crypto input is from the future")
        if age > self.config.maximum_input_age_seconds:
            raise ValueError("crypto input is stale at evaluation")
        spot_end = crypto.input_provenance.get("spot_end_at")
        if spot_end is not None and datetime.fromisoformat(spot_end) != crypto.observed_at:
            # Research-derived snapshots must be observed at their spot candle boundary;
            # the full context is re-validated by callers and at delivery authorization.
            raise ValueError("crypto input does not match its research provenance")
        if market.venue.casefold() == "kalshi" and market.observation_end_at is None:
            raise ValueError("Kalshi evaluation requires explicit observation provenance")
        observation_end = market.effective_observation_end_at
        if observation_end <= market.observed_at:
            raise ValueError("benchmark observation has already ended")
        if contract.settlement_window_seconds:
            window_start = observation_end - timedelta(seconds=contract.settlement_window_seconds)
            if market.observed_at > window_start:
                raise ValueError(
                    "averaging window has already started; observed prefix is required"
                )
            if market.observation_start_at not in (None, window_start):
                raise ValueError("observation start does not match averaging window")
        time_to_observation = (observation_end - crypto.observed_at).total_seconds()
        years = time_to_observation / _SECONDS_PER_YEAR
        sigma = crypto.annualized_volatility

        def probability_above(strike: float) -> float:
            if contract.settlement_window_seconds:
                return self._averaged_terminal_above_probability(
                    crypto, strike, years, contract.settlement_window_seconds
                )
            return self._terminal_above_probability(crypto, strike, years)

        if isinstance(contract, TerminalRangeContract):
            lower_cdf = 1.0 - probability_above(contract.lower_bound)
            upper_cdf = 1.0 - probability_above(contract.upper_bound)
            return _clamp_probability(max(upper_cdf - lower_cdf, 0.0))

        if not math.isclose(crypto.strike_price, contract.strike_price):
            raise ValueError("crypto snapshot strike does not match threshold contract")
        if contract.model_kind is ThresholdModelKind.BARRIER:
            if contract.settlement_window_seconds:
                raise ValueError("touch barriers cannot use terminal settlement averaging")
            # A spot quote is not evidence that a barrier was never crossed earlier.
            # Without a benchmark path source, only the exact contractual start is
            # modelable. Unknown or elapsed prefixes remain unsupported.
            if market.observation_start_at != crypto.observed_at:
                raise ValueError("touch history unavailable for the contractual observation period")
            probability = barrier_hitting_probability(
                spot_price=crypto.spot_price,
                strike_price=contract.strike_price,
                annualized_volatility=sigma,
                expected_annual_return=crypto.expected_annual_return,
                years=years,
                direction=contract.direction,
            )
            return _clamp_probability(probability)

        above_probability = probability_above(contract.strike_price)
        probability = (
            above_probability
            if contract.direction is ThresholdDirection.ABOVE
            else 1.0 - above_probability
        )
        return _clamp_probability(probability)

    @staticmethod
    def _terminal_above_probability(
        crypto: CryptoSnapshot,
        strike_price: float,
        years: float,
    ) -> float:
        numerator = (
            math.log(crypto.spot_price / strike_price)
            + (crypto.expected_annual_return - 0.5 * crypto.annualized_volatility**2) * years
        )
        denominator = crypto.annualized_volatility * math.sqrt(years)
        return _normal_cdf(numerator / denominator)

    @staticmethod
    def _averaged_terminal_above_probability(
        crypto: CryptoSnapshot,
        strike_price: float,
        years: float,
        window_seconds: int,
    ) -> float:
        if window_seconds <= 0 or years < window_seconds / _SECONDS_PER_YEAR:
            raise ValueError("full future averaging window is required")
        # Exact first two moments of the arithmetic mean of one price per second
        # in [end - window, end). GBM cov(S_t,S_u) = E[S_t]E[S_u]expm1(sigma²min(t,u)).
        # The subsequent lognormal moment match approximates the distribution, not
        # its moments. Prefix sums exploit ordered sample times for linear work.
        drift = crypto.expected_annual_return
        variance = crypto.annualized_volatility**2
        total_weight = 0.0
        prefix_covariance = 0.0
        covariance_sum = 0.0
        for index in range(window_seconds):
            time = years - (window_seconds - index) / _SECONDS_PER_YEAR
            weight = math.exp(drift * time)
            covariance_factor = math.expm1(variance * time)
            covariance_sum += weight * weight * covariance_factor + 2 * weight * prefix_covariance
            prefix_covariance += weight * covariance_factor
            total_weight += weight
        first_moment = crypto.spot_price * total_weight / window_seconds
        log_variance = math.log1p(covariance_sum / total_weight**2)
        if log_variance == 0.0:
            return float(first_moment > strike_price)
        log_mean = math.log(first_moment) - 0.5 * log_variance
        z_score = (log_mean - math.log(strike_price)) / math.sqrt(log_variance)
        return _normal_cdf(z_score)

    @staticmethod
    def _market_probability(market: MarketSnapshot) -> float:
        estimates: list[float] = []
        if market.yes_bid is not None and market.yes_ask is not None:
            estimates.append((market.yes_bid + market.yes_ask) / 2.0)
        elif market.yes_bid is not None:
            estimates.append(market.yes_bid)
        elif market.yes_ask is not None:
            estimates.append(market.yes_ask)

        if market.no_bid is not None and market.no_ask is not None:
            estimates.append(1.0 - (market.no_bid + market.no_ask) / 2.0)
        elif market.no_bid is not None:
            estimates.append(1.0 - market.no_bid)
        elif market.no_ask is not None:
            estimates.append(1.0 - market.no_ask)

        if not estimates:
            raise ValueError("at least one market quote is required")
        return _clamp_probability(sum(estimates) / len(estimates))

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
        contract: CryptoPriceContract,
    ) -> tuple[list[str], list[str]]:
        supporting: list[str] = []
        opposing: list[str] = []
        difference = structural_probability - market_probability

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

        if isinstance(contract, TerminalRangeContract):
            if crypto.spot_price < contract.lower_bound:
                position = f"below the ${contract.lower_bound:,.2f} lower bound"
                opposing.append(f"{crypto.symbol} spot is {position}.")
            elif crypto.spot_price > contract.upper_bound:
                position = f"above the ${contract.upper_bound:,.2f} upper bound"
                opposing.append(f"{crypto.symbol} spot is {position}.")
            else:
                supporting.append(
                    f"{crypto.symbol} spot is inside the "
                    f"${contract.lower_bound:,.2f}–${contract.upper_bound:,.2f} range."
                )
            return supporting, opposing

        distance_percent = (crypto.spot_price / contract.strike_price - 1.0) * 100.0
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
        binary_contract_fee = (
            self.config.binary_fee_coefficient
            if self.config.binary_fee_type == "flat"
            else self.config.binary_fee_coefficient * ask * (1.0 - ask)
        )
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

    def all_in_cost(self, ask: float, units: float, *, round_fee: bool) -> float:
        """Maximum modeled spend, including configured slippage and rounded taker fees."""
        price = Decimal(str(ask))
        quantity = Decimal(str(units))
        coefficient = Decimal(str(self.config.binary_fee_coefficient))
        fee = quantity * coefficient
        if self.config.binary_fee_type != "flat":
            fee *= price * (1 - price)
        if round_fee:
            fee = fee.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        multiplier = (
            1 + Decimal(str(self.config.fee_rate)) + Decimal(str(self.config.slippage_bps)) / 10_000
        )
        return float(quantity * price * multiplier + fee)

    def exposure_for_budget(
        self, ask: float, ask_size: float, budget: float, *, whole_contracts: bool
    ) -> float:
        if budget <= 0 or ask_size <= 0:
            return 0.0
        if not whole_contracts:
            cap = min(budget, self.all_in_cost(ask, ask_size, round_fee=False))
            return float(Decimal(str(cap)).quantize(Decimal("0.01"), rounding="ROUND_FLOOR"))
        lower, upper = 0, math.floor(ask_size)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if self.all_in_cost(ask, middle, round_fee=True) <= budget:
                lower = middle
            else:
                upper = middle - 1
        return self.all_in_cost(ask, lower, round_fee=True)

    def _recommend(
        self,
        market: MarketSnapshot,
        forecast: ProbabilityForecast,
    ) -> Opportunity:
        candidates: list[_Candidate] = []
        if market.yes_ask is not None and market.yes_ask_size is not None:
            candidates.append(
                self._candidate(
                    MarketSide.YES,
                    market.yes_ask,
                    market.yes_ask_size,
                    forecast.lower_probability_yes,
                )
            )
        if market.no_ask is not None and market.no_ask_size is not None:
            candidates.append(
                self._candidate(
                    MarketSide.NO,
                    market.no_ask,
                    market.no_ask_size,
                    1.0 - forecast.upper_probability_yes,
                )
            )
        whole_contracts = market.venue.casefold() == "kalshi"
        if whole_contracts:
            candidates = [
                replace(
                    candidate,
                    effective_cost=self.all_in_cost(candidate.ask, 1, round_fee=True),
                    conservative_net_edge=(
                        candidate.conservative_probability
                        - self.all_in_cost(candidate.ask, 1, round_fee=True)
                        - self.config.resolution_haircut
                    ),
                )
                for candidate in candidates
            ]
        if not candidates:
            raise ValueError("no executable side is available")
        best = max(candidates, key=lambda candidate: candidate.conservative_net_edge)

        warnings: list[str] = []
        reasons = [
            (
                f"{best.side} conservative edge is "
                f"{best.conservative_net_edge:.2%} after modeled costs."
            )
        ]

        if (
            market.yes_ask is not None
            and market.no_ask is not None
            and market.yes_ask + market.no_ask < 0.99
        ):
            warnings.append("Complementary asks appear incoherent; verify quote freshness.")
        if (
            market.yes_bid is not None
            and market.no_bid is not None
            and market.yes_bid + market.no_bid > 1.01
        ):
            warnings.append("Complementary bids appear incoherent; verify quote freshness.")

        enough_time = (
            remaining_seconds_to_entry_horizon(market, market.observed_at)
            >= self.config.minimum_seconds_to_expiry
        )
        enough_liquidity = best.ask_size >= max(
            self.config.minimum_ask_size, 1.0 if whole_contracts else 0.0
        )
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
            exposure = self._suggested_exposure(best, whole_contracts=whole_contracts)
            if exposure <= 0:
                state = RecommendationState.WATCH
                warnings.append("Risk budget cannot fund an executable contract.")
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

    def _suggested_exposure(self, candidate: _Candidate, *, whole_contracts: bool) -> float:
        denominator = max(1.0 - candidate.effective_cost, _EPSILON)
        full_kelly_fraction = max(
            0.0,
            candidate.conservative_net_edge / denominator,
        )
        bankroll_fraction = min(
            self.config.fractional_kelly * full_kelly_fraction,
            self.config.max_bankroll_fraction,
            self.config.max_event_bankroll_fraction,
        )
        bankroll_cap = self.config.paper_bankroll * bankroll_fraction
        return self.exposure_for_budget(
            candidate.ask, candidate.ask_size, bankroll_cap, whole_contracts=whole_contracts
        )
