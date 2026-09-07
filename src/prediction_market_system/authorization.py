"""One fail-closed authorization boundary for every opportunity delivery path."""

from __future__ import annotations

from datetime import UTC, datetime

from prediction_market_system.domain import (
    CryptoSnapshot,
    Opportunity,
    RecommendationState,
    TerminalRangeContract,
    ThresholdContract,
)
from prediction_market_system.engine import (
    CryptoThresholdEngine,
    EngineConfig,
    assert_time_sensitive_entry_controls,
    deployment_policy_id,
)
from prediction_market_system.storage import SQLiteRepository


class UnapprovedDeliveryError(ValueError):
    """Compatible held-out evidence exists, but deployment approval is absent."""


def authorize_delivery(
    repository: SQLiteRepository,
    opportunity: Opportunity,
    *,
    allow_unapproved: bool = False,
    as_of: datetime | None = None,
) -> None:
    now = as_of or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("delivery timestamp must include a timezone")
    forecast = opportunity.forecast
    if opportunity.state not in {RecommendationState.ENTER_YES, RecommendationState.ENTER_NO}:
        raise ValueError("only entry candidates may be delivered")
    if forecast.uncertainty_source != "held_out" or forecast.calibration_profile_id is None:
        raise ValueError("delivery requires held-out calibration")
    if not forecast.recipe_id or forecast.model_version != CryptoThresholdEngine.model_version:
        raise ValueError("legacy or missing forecast recipe cannot authorize delivery")
    manifest = forecast.input_manifest
    if not manifest.get("crypto") or not manifest.get("engine_config"):
        raise ValueError("delivery requires the exact forecast input manifest")
    crypto = CryptoSnapshot.model_validate(manifest["crypto"])
    config = EngineConfig.model_validate(manifest["engine_config"])
    if CryptoThresholdEngine(config).recipe_id(crypto) != forecast.recipe_id:
        raise ValueError("forecast recipe does not match its inputs")
    if manifest.get("market") != opportunity.market.model_dump(mode="json"):
        raise ValueError("delivery market does not match the forecast manifest")
    for observed_at in (forecast.generated_at, opportunity.market.observed_at, crypto.observed_at):
        age = (now - observed_at).total_seconds()
        if age < 0 or age > config.maximum_input_age_seconds:
            raise ValueError("delivery inputs are future-dated or stale")
    if opportunity.market.expires_at <= now:
        raise ValueError("trading has closed")
    context_id = crypto.input_provenance.get("research_context_id")
    if context_id is not None:
        # The forecast references its research context by content identity; delivery
        # requires that exact context to be persisted and still admissible now.
        context = repository.research_context_by_id(str(context_id))
        if context is None:
            raise ValueError("delivery research context is not persisted")
        context.validate_at(now, maximum_spot_age_seconds=config.maximum_input_age_seconds)
        if (
            context.to_crypto_snapshot(
                strike_price=crypto.strike_price,
                expected_annual_return=crypto.expected_annual_return,
            )
            != crypto
        ):
            raise ValueError("delivery crypto does not match recorded research provenance")
    profile = repository.uncertainty_calibration(forecast.calibration_profile_id)
    if (
        profile is None
        or profile.research_only
        or profile.recipe_id != forecast.recipe_id
        or profile.model_name != forecast.model_name
        or profile.model_version != forecast.model_version
        or profile.symbol != crypto.symbol
        or profile.generated_at > forecast.generated_at
        or profile.cutoff_at > forecast.generated_at
        or manifest.get("calibration_profile") != profile.model_dump(mode="json")
    ):
        raise ValueError("delivery requires the exact available persisted calibration profile")
    contract_payload = manifest.get("contract")
    if not isinstance(contract_payload, dict):
        raise ValueError("delivery requires exact contract semantics")
    contract = (
        TerminalRangeContract.model_validate(contract_payload)
        if "lower_bound" in contract_payload
        else ThresholdContract.model_validate(contract_payload)
    )
    engine = CryptoThresholdEngine(config)
    reproduced, recommendation = engine.evaluate(opportunity.market, crypto, contract, profile)
    if reproduced.model_dump(exclude={"forecast_id"}) != forecast.model_dump(
        exclude={"forecast_id"}
    ):
        raise ValueError("forecast does not reproduce from recorded inputs")
    if (
        recommendation.state != opportunity.state
        or recommendation.side != opportunity.side
        or recommendation.executable_price != opportunity.executable_price
        or recommendation.conservative_probability != opportunity.conservative_probability
        or recommendation.conservative_net_edge != opportunity.conservative_net_edge
        or not 0 < opportunity.suggested_max_exposure <= recommendation.suggested_max_exposure
    ):
        raise ValueError("delivery does not match the reproducible executable decision")
    recorded = repository.recorded_opportunity(str(forecast.forecast_id))
    if recorded is None:
        raise ValueError("delivery requires the persisted final allocation")
    if opportunity.suggested_max_exposure > recorded.suggested_max_exposure:
        raise ValueError("delivery allocation exceeds the persisted event-capped allocation")
    event_id = opportunity.market.event_id or opportunity.market.market_id
    sibling_exposure = repository.forward_event_entry_exposure(
        event_id=event_id,
        exclude_forecast_id=str(forecast.forecast_id),
    )
    if sibling_exposure + opportunity.suggested_max_exposure > engine.event_exposure_cap:
        raise ValueError("delivery would exceed the aggregate event exposure cap")
    # Revalidate time-sensitive entry controls at the actual delivery boundary.
    # Reproduction above uses the original observation time so the forecast can
    # still match; expiry, observation-end, and averaging are checked against now.
    assert_time_sensitive_entry_controls(config, opportunity.market, contract, as_of=now)
    engine._structural_probability(
        opportunity.market.model_copy(update={"observed_at": now}), crypto, contract
    )
    if not allow_unapproved and not repository.is_calibration_approved(
        profile.profile_id,
        as_of=now,
        deployment_policy_id=deployment_policy_id(config),
    ):
        raise UnapprovedDeliveryError(
            f"held-out backtest criteria have not approved calibration {profile.profile_id}"
        )
