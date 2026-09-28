import copy
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from prediction_market_system.calibration import CalibrationBin, UncertaintyCalibrationProfile
from prediction_market_system.domain import (
    CryptoPriceContract,
    CryptoSnapshot,
    MarketSnapshot,
    TerminalRangeContract,
    ThresholdContract,
    ThresholdDirection,
    ThresholdModelKind,
)
from prediction_market_system.engine import CryptoThresholdEngine
from prediction_market_system.evidence import EVIDENCE_FORWARD_SHADOW, canonical_json, content_id
from prediction_market_system.experiment import (
    compare_observations,
    load_protocol,
    protocol_identity,
)
from prediction_market_system.research import (
    ResearchContext,
    SpotCandle,
    VolatilityObservation,
    calculate_realized_volatility,
)
from prediction_market_system.venues.kalshi import KalshiMarket

AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
DAY = datetime(2026, 8, 10, 16, tzinfo=UTC)
MODEL_NAME = "crypto-terminal-range-market-anchor-arithmetic-60s"
PRODUCTION_WINDOW = 2_592_000
BASE_PROTOCOL: dict[str, Any] = {
    "protocol_version": "pms-phase1-v2",
    "frozen_at": "2026-09-26T08:00:00+00:00",
    "series": "KXBTC",
    "symbol": "BTC",
    "contract_family": "terminal-range",
    "model_name": MODEL_NAME,
    "model_version": "2.1.0",
    "structural_weight": 0.5,
    "expected_annual_return": 0.0,
    "production_features": {
        "semantics": "point-in-time-complete-windows-v2",
        "spot_interval_seconds": 60,
        "realized_interval_seconds": 3600,
        "realized_window_seconds": PRODUCTION_WINDOW,
        "volatility_selection": "matching-interval-dvol-else-realized",
    },
    "market_baseline": "yes_ask",
    "secondary_market_baseline": "recorded_midpoint_anchor",
    "maximum_input_age_seconds": 120,
    "minimum_ask_size": 1,
    "settlement_window_seconds": 60,
    "minimum_seconds_to_entry_horizon": 300,
    "report_as_of": "2026-09-26T00:00:00+00:00",
    "archive_start": "2026-07-01T00:00:00+00:00",
    "archive_end": "2026-09-26T00:00:00+00:00",
    "holdout_start": "2026-09-27T00:00:00+00:00",
    "holdout_end": "2026-11-26T00:00:00+00:00",
    "training_start": "2026-05-29T00:00:00+00:00",
    "training_end": "2026-08-27T00:00:00+00:00",
    "validation_start": "2026-08-27T00:00:00+00:00",
    "validation_end": "2026-09-26T00:00:00+00:00",
    "validation_folds": [
        {
            "name": "validation-1",
            "start": "2026-08-27T00:00:00+00:00",
            "end": "2026-09-11T00:00:00+00:00",
        },
        {
            "name": "validation-2",
            "start": "2026-09-11T00:00:00+00:00",
            "end": "2026-09-26T00:00:00+00:00",
        },
    ],
    "holdout_folds": [
        {
            "name": "holdout-1",
            "start": "2026-09-27T00:00:00+00:00",
            "end": "2026-11-26T00:00:00+00:00",
        }
    ],
    "minimum_calibration_events": 30,
    "minimum_resolved_events": 20,
    "minimum_dates": 20,
    "minimum_validation_folds": 2,
    "minimum_fold_resolved_events": 10,
    "log_loss_epsilon": 1e-12,
    "calibration_bins": 10,
    "bootstrap_replicates": 200,
    "bootstrap_seed": 1729,
    "confidence": 0.95,
    "decision_costs": {"minimum_seconds_to_expiry": 300, "minimum_ask_size": 10},
    "cost_scenarios": [],
}


def make_protocol(**overrides: Any) -> dict[str, Any]:
    body = {**BASE_PROTOCOL, **overrides}
    return {**body, "protocol_id": protocol_identity(body)}


def rehash(item: dict[str, Any], *, kind: str = "synthetic-fixture") -> dict[str, Any]:
    inputs = item["inputs"]
    item["input_id"] = content_id(inputs) if isinstance(inputs, dict) else "missing"
    manifest = {
        "schema": "pms-run-manifest-v2",
        "kind": kind,
        "run_id": item["run_id"],
        "recorded_at": item["observed_at"],
        "configuration": {"recipe_id": item["opportunity"]["forecast"]["recipe_id"]},
        "inputs": {"input_id": item["input_id"]},
    }
    item["run_manifest"] = manifest
    item["run_manifest_sha256"] = content_id(manifest)
    return item


def candle(end_at: datetime, close: Decimal, interval_seconds: int) -> SpotCandle:
    return SpotCandle(
        provider="coinbase",
        product_id="BTC-USD",
        interval_seconds=interval_seconds,
        start_at=end_at - timedelta(seconds=interval_seconds),
        end_at=end_at,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=Decimal("10"),
        retrieved_at=end_at,
        raw_payload={},
    )


def research_context(
    as_of: datetime, *, window_seconds: int = PRODUCTION_WINDOW, implied: bool = False
) -> ResearchContext:
    count = window_seconds // 3600 + 1
    hourly = [
        candle(
            as_of - timedelta(hours=count - 1 - index),
            Decimal(f"{60_000 + 40 * math.sin(index):.2f}"),
            3600,
        )
        for index in range(count)
    ]
    implied_volatility = (
        VolatilityObservation(
            provider="deribit",
            symbol="BTC",
            kind="implied",
            window_seconds=3600,
            source_start_at=as_of - timedelta(hours=1),
            observed_at=as_of,
            annualized_volatility=0.55,
            retrieved_at=as_of,
            raw_payload={"resolution_seconds": 3600},
        )
        if implied
        else None
    )
    return ResearchContext(
        symbol="BTC",
        event_ticker=None,
        as_of=as_of,
        spot=candle(as_of, Decimal("60000.00"), 60),
        realized_volatility=calculate_realized_volatility(
            hourly, symbol="BTC", as_of=as_of, window_seconds=window_seconds
        ),
        implied_volatility=implied_volatility,
    )


def calibration_profile(recipe_id: str, independent_events: int) -> UncertaintyCalibrationProfile:
    return UncertaintyCalibrationProfile(
        generated_at=datetime(2026, 8, 2, tzinfo=UTC),
        symbol="BTC",
        model_name=MODEL_NAME,
        model_version="2.1.0",
        recipe_id=recipe_id,
        training_start=datetime(2026, 6, 1, tzinfo=UTC),
        cutoff_at=datetime(2026, 8, 1, tzinfo=UTC),
        confidence_level=0.9,
        sample_count=40,
        brier_score=0.2,
        bins=(
            CalibrationBin(
                lower_probability=0.0,
                upper_probability=1.0,
                mean_probability=0.5,
                observed_frequency=0.5,
                outcome_interval_lower=0.4,
                outcome_interval_upper=0.6,
                uncertainty_margin=0.06,
                sample_count=40,
                minimum_horizon_seconds=1.0,
                maximum_horizon_seconds=86_400.0,
            ),
        ),
        independent_event_count=independent_events,
    )


def kalshi_rule(end: datetime, lower: float, upper: float, *, averaging: str = "simple") -> str:
    clock = f"{end.hour % 12 or 12} {'AM' if end.hour < 12 else 'PM'} UTC"
    return (
        f"Resolves YES if the {averaging} average of the sixty seconds before {clock} "
        f"is between {lower} and {upper} at {clock} on {end:%b} {end.day}, {end.year}."
    )


def kalshi_market(
    market_id: str, event_id: str, end: datetime, lower: float, upper: float, rule: str
) -> KalshiMarket:
    return KalshiMarket.model_validate(
        {
            "ticker": market_id,
            "event_ticker": event_id,
            "series_ticker": "KXBTC",
            "market_type": "binary",
            "title": "Bitcoin price range",
            "yes_sub_title": f"{lower} to {upper}",
            "no_sub_title": "Outside range",
            "close_time": end.isoformat(),
            "latest_expiration_time": (end + timedelta(days=7)).isoformat(),
            "status": "active",
            "notional_value_dollars": "1",
            "can_close_early": False,
            "strike_type": "between",
            "floor_strike": lower,
            "cap_strike": upper,
            "rules_primary": rule,
            "yes_bid_dollars": "0.28",
            "yes_ask_dollars": "0.30",
            "no_bid_dollars": "0.69",
            "no_ask_dollars": "0.71",
            "yes_bid_size_fp": "25",
            "yes_ask_size_fp": "25",
        }
    )


def observation(
    event_id: str,
    *,
    observation_id: str | None = None,
    end: datetime = DAY,
    observed_at: datetime | None = None,
    lower: float = 59_500.0,
    yes_ask: float | None = 0.30,
    spot: float = 60_000.0,
    window: int = 60,
    contract: CryptoPriceContract | None = None,
    result: str | None = "yes",
    resolved_at: datetime | None = None,
    context: ResearchContext | None = None,
    drift: float = 0.0,
    synthetic_label: bool = True,
    calibration_events: int | None = None,
    kind: str = "synthetic-fixture",
    synthetic_declared: bool = True,
    raw: bool = True,
    venue_rule: str | None = None,
    snapshot_rule: str | None = None,
    context_payload: dict[str, Any] | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    observed_at = observed_at or end - timedelta(hours=1)
    market_id = f"{event_id}-B{int(lower)}"
    if kind == "synthetic-fixture":
        rule = "Synthetic test rule."
        source_metadata: dict[str, Any] = {"synthetic": True} if synthetic_declared else {}
    else:
        venue = kalshi_market(
            market_id,
            event_id,
            end,
            lower,
            lower + 1_000.0,
            venue_rule or kalshi_rule(end, lower, lower + 1_000.0),
        )
        rule = venue.resolution_rule
        source_metadata = venue.model_dump(mode="json") if raw else {}
    market = MarketSnapshot(
        market_id=market_id,
        question="Synthetic BTC range?",
        venue="kalshi",
        observed_at=observed_at,
        expires_at=end,
        observation_end_at=end,
        observation_start_at=end - timedelta(seconds=window) if window else None,
        yes_bid=None if yes_ask is None else max(yes_ask - 0.02, 0.0),
        yes_ask=yes_ask,
        no_bid=0.60 if yes_ask is None else max(round(1.0 - yes_ask - 0.03, 4), 0.0),
        no_ask=0.62 if yes_ask is None else min(round(1.0 - yes_ask + 0.01, 4), 1.0),
        yes_ask_size=None if yes_ask is None else 25.0,
        no_ask_size=25.0,
        resolution_rule=snapshot_rule or rule,
        series_id="KXBTC",
        event_id=event_id,
        source_metadata=source_metadata,
    )
    if context is not None:
        crypto = context.to_crypto_snapshot(strike_price=60_000.0, expected_annual_return=drift)
        if context_id is not None:
            provenance = {**crypto.input_provenance, "research_context_id": context_id}
            crypto = crypto.model_copy(update={"input_provenance": provenance})
    else:
        crypto = CryptoSnapshot(
            symbol="BTC",
            observed_at=observed_at,
            spot_price=spot,
            strike_price=60_000.0,
            annualized_volatility=0.5,
            feature_recipe={
                "source_selection": "synthetic-fixture" if synthetic_label else "manual-explicit"
            },
        )
    contract = contract or TerminalRangeContract(
        lower_bound=lower, upper_bound=lower + 1_000.0, settlement_window_seconds=window
    )
    engine = CryptoThresholdEngine()
    profile = (
        None
        if calibration_events is None
        else calibration_profile(engine.recipe_id(crypto), calibration_events)
    )
    forecast, opportunity = engine.evaluate(market, crypto, contract, profile)
    payload = opportunity.model_dump(mode="json")
    payload["forecast"].pop("input_manifest")
    resolved_at = resolved_at or end + timedelta(minutes=10)
    stored_context = context_payload
    if stored_context is None and context is not None:
        stored_context = context.model_dump(mode="json")
    item = {
        "observation_id": observation_id or f"{market.market_id}@{observed_at.isoformat()}",
        "event_id": event_id,
        "market_id": market.market_id,
        "observed_at": observed_at.isoformat(),
        "opportunity": payload,
        "inputs": json.loads(canonical_json(forecast.input_manifest)),
        "run_id": str(forecast.forecast_id),
        "research_context": stored_context,
        "resolution": (
            None
            if result is None
            else {"result": result, "observed_at": resolved_at.isoformat(), "settlement_ts": None}
        ),
        "execution": [],
    }
    return rehash(item, kind=kind)


def compare(
    observations: list[dict[str, Any]],
    protocol: dict[str, Any] | None = None,
    source_kind: str = "synthetic-fixture",
) -> dict[str, Any]:
    return compare_observations(
        observations, protocol or make_protocol(), source_kind=source_kind, as_of=AS_OF
    )


def reasons(report: dict[str, Any]) -> dict[str | None, str]:
    return {entry["observation_id"]: entry["reason"] for entry in report["excluded"]}


def codes(report: dict[str, Any]) -> dict[str | None, str]:
    return {entry["observation_id"]: entry["reason_code"] for entry in report["excluded"]}


def test_load_protocol_verifies_identity_version_and_pinned_features(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    protocol = make_protocol()
    path.write_text(json.dumps(protocol))
    assert load_protocol(path) == protocol

    path.write_text(json.dumps({**protocol, "structural_weight": 0.6}))
    with pytest.raises(ValueError, match="protocol_id"):
        load_protocol(path)
    path.write_text(json.dumps(make_protocol(model_name="crypto-terminal-range")))
    with pytest.raises(ValueError, match="model_name"):
        load_protocol(path)
    path.write_text(json.dumps(make_protocol(protocol_version="pms-phase1-v1")))
    with pytest.raises(ValueError, match="unsupported protocol version"):
        load_protocol(path)
    features = {**BASE_PROTOCOL["production_features"], "semantics": "manual"}
    path.write_text(json.dumps(make_protocol(production_features=features)))
    with pytest.raises(ValueError, match="production_features.semantics"):
        load_protocol(path)
    incomplete = {key: value for key, value in protocol.items() if key != "confidence"}
    path.write_text(json.dumps({**incomplete, "protocol_id": protocol_identity(incomplete)}))
    with pytest.raises(ValueError, match="missing required keys: confidence"):
        load_protocol(path)


def test_frozen_v2_protocol_loads() -> None:
    path = Path(__file__).resolve().parent.parent / "experiments/phase1/protocol-v2.json"
    protocol = load_protocol(path)
    assert protocol["production_features"]["realized_window_seconds"] == PRODUCTION_WINDOW


def test_repeated_forecasts_share_one_event_weight_and_order_is_irrelevant() -> None:
    first_time = DAY - timedelta(hours=2)
    rows = [
        observation("EVT-A", observation_id="a1", observed_at=first_time, yes_ask=0.25),
        observation("EVT-A", observation_id="a2", observed_at=first_time, yes_ask=0.25),
        observation("EVT-A", observation_id="a3", yes_ask=0.40, spot=60_400.0, result="yes"),
        observation("EVT-B", end=DAY + timedelta(days=1), yes_ask=0.60, result="no"),
    ]
    report = compare(rows)

    assert report["excluded"] == []
    assert report["counts"]["resolved"] == 4
    assert report["counts"]["resolved_events"] == 2
    assert report["counts"]["resolved_dates"] == 2
    weights = {row["observation_id"]: row["event_weight"] for row in report["observations"]}
    assert weights["a1"] == weights["a2"] == weights["a3"] == pytest.approx(1 / 3)
    for arm in ("structural", "blend", "market_yes_ask", "recorded_midpoint_anchor"):
        scores = {
            event: [
                row[f"{arm}_brier"] for row in report["observations"] if row["event_id"] == event
            ]
            for event in ("EVT-A", "EVT-B")
        }
        expected = (sum(scores["EVT-A"]) / 3 + scores["EVT-B"][0]) / 2
        assert report["metrics"]["arms"][arm]["brier"] == pytest.approx(expected)
        bins = report["calibration"]["arms"][arm]
        assert sum(entry["weight"] for entry in bins) == pytest.approx(2.0)
        assert sum(entry["forecasts"] for entry in bins) == 4
    assert {difference["events"] for difference in report["paired_differences"]} == {2}
    assert compare(list(reversed(rows))) == report


def test_missing_future_stale_and_unsupported_inputs_are_excluded_with_reasons() -> None:
    stale = observation("EVT-S", observation_id="stale")
    stale["inputs"]["crypto"]["observed_at"] = (DAY - timedelta(minutes=65)).isoformat()
    future = observation("EVT-F", observation_id="future-input")
    future["inputs"]["crypto"]["observed_at"] = (DAY - timedelta(minutes=59)).isoformat()
    missing = observation("EVT-M", observation_id="missing-inputs")
    missing["inputs"] = None
    tampered = observation("EVT-T", observation_id="tampered")
    tampered["input_id"] = "0" * 64
    threshold = ThresholdContract(
        model_kind=ThresholdModelKind.TERMINAL,
        direction=ThresholdDirection.ABOVE,
        strike_price=60_000.0,
        settlement_window_seconds=60,
    )
    identical = observation("EVT-I", observation_id="same")
    rows = [
        rehash(stale),
        rehash(future),
        rehash(missing),
        tampered,
        observation("EVT-N", observation_id="no-ask", yes_ask=None),
        observation("EVT-R", observation_id="threshold", contract=threshold),
        observation("EVT-W", observation_id="no-averaging", window=0),
        observation("EVT-X", observation_id="near-expiry", observed_at=DAY - timedelta(minutes=3)),
        observation("EVT-U", observation_id="unlabelled", synthetic_label=False),
        observation("EVT-Y", observation_id="undeclared", synthetic_declared=False),
        observation(
            "EVT-L",
            observation_id="after-cutoff",
            end=AS_OF + timedelta(hours=2),
            observed_at=AS_OF + timedelta(hours=1),
        ),
        observation("EVT-D", observation_id="dup"),
        observation("EVT-D", observation_id="dup", lower=60_500.0),
        identical,
        copy.deepcopy(identical),
        observation("EVT-OK", observation_id="valid"),
    ]
    report = compare(rows)

    assert reasons(report) == {
        "stale": "crypto input is stale",
        "future-input": "crypto input is from the future",
        "missing-inputs": "missing forecast inputs",
        "tampered": "input hash mismatch",
        "no-ask": "missing displayed YES ask",
        "threshold": "unsupported contract structure: not a terminal range",
        "no-averaging": "unsupported contract structure: settlement averaging window",
        "near-expiry": "inside minimum time to expiry",
        "unlabelled": "manual features are not explicitly labelled synthetic",
        "undeclared": "synthetic row lacks an explicit synthetic declaration",
        "after-cutoff": "observed after report cutoff",
        "dup": "conflicting duplicate observation_id",
        "same": "identical duplicate observation_id",
    }
    assert codes(report) == {
        "stale": "stale_input",
        "future-input": "future_input",
        "missing-inputs": "malformed",
        "tampered": "identity_mismatch",
        "no-ask": "no_displayed_ask",
        "threshold": "unsupported_contract",
        "no-averaging": "unsupported_contract",
        "near-expiry": "timing_ineligible",
        "unlabelled": "feature_recipe_mismatch",
        "undeclared": "unsupported_contract",
        "after-cutoff": "outside_window",
        "dup": "duplicate_conflict",
        "same": "duplicate_observation",
    }
    assert report["counts"]["excluded_by_reason_code"]["duplicate_conflict"] == 2
    assert report["counts"]["excluded_by_reason_code"]["duplicate_observation"] == 1
    assert [row["observation_id"] for row in report["observations"]] == ["same", "valid"]
    assert report["counts"]["eligible"] + report["counts"]["excluded"] == len(rows)
    assert compare(list(reversed(rows))) == report


def test_unresolved_and_future_outcomes_stay_in_population_but_unscored() -> None:
    late_end = datetime(2026, 9, 25, 23, tzinfo=UTC)
    rows = [
        observation("EVT-R", observation_id="resolved"),
        observation("EVT-U", observation_id="missing", result=None),
        observation(
            "EVT-F",
            observation_id="future-outcome",
            end=late_end,
            result="no",
            resolved_at=AS_OF + timedelta(hours=1),
        ),
        observation("EVT-E", observation_id="early", resolved_at=DAY - timedelta(hours=2)),
    ]
    report = compare(rows)

    counts = report["counts"]
    assert (counts["eligible"], counts["resolved"], counts["unresolved"]) == (3, 1, 2)
    assert counts["unresolved_missing_resolution"] == 1
    assert counts["unresolved_outcome_after_cutoff"] == 1
    assert reasons(report) == {"early": "resolution known before forecast"}
    assert codes(report) == {"early": "outcome_leak"}
    future = next(r for r in report["observations"] if r["observation_id"] == "future-outcome")
    assert future["resolution_status"] == "unresolved"
    assert future["unresolved_reason"] == "outcome-after-cutoff"
    assert future["outcome_yes"] is None and future["resolution_observed_at"] is None
    assert future["blend_brier"] is None and future["event_weight"] is None
    assert report["metrics"]["resolved_events"] == 1
    assert report["coverage"]["resolved_event_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("explicit_null_marker", [False, True])
def test_withheld_outcome_reason_survives_without_label_or_scores(
    explicit_null_marker: bool,
) -> None:
    resolved = observation("EVT-R", observation_id="resolved")
    missing = observation("EVT-U", observation_id="missing", result=None)
    withheld = observation("EVT-W", observation_id="withheld", result=None)
    if explicit_null_marker:
        resolved["resolution_unavailable_reason"] = None
        missing["resolution_unavailable_reason"] = None
    withheld["resolution_unavailable_reason"] = "outcome-after-cutoff"

    report = compare([resolved, missing, withheld])

    counts = report["counts"]
    assert (counts["eligible"], counts["resolved"], counts["unresolved"]) == (3, 1, 2)
    assert counts["unresolved_missing_resolution"] == 1
    assert counts["unresolved_outcome_after_cutoff"] == 1
    assert report["excluded"] == []
    rows = {row["observation_id"]: row for row in report["observations"]}
    assert rows["resolved"]["resolution_status"] == "resolved"
    assert rows["missing"]["unresolved_reason"] == "missing-resolution"
    assert rows["withheld"]["unresolved_reason"] == "outcome-after-cutoff"
    for observation_id in ("missing", "withheld"):
        row = rows[observation_id]
        assert row["resolution_status"] == "unresolved"
        for field in ("outcome_yes", "resolution_observed_at", "settlement_ts", "event_weight"):
            assert row[field] is None
        for arm in ("structural", "blend", "market_yes_ask", "recorded_midpoint_anchor"):
            assert row[f"{arm}_brier"] is None
            assert row[f"{arm}_log_loss"] is None
    assert report["metrics"] == compare([resolved])["metrics"]
    assert report["coverage"]["resolved_event_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("marker", ["missing-resolution", "", False, 1, [], {}])
def test_invalid_resolution_unavailable_reason_is_excluded(marker: Any) -> None:
    row = observation("EVT-U", observation_id="invalid-marker", result=None)
    row["resolution_unavailable_reason"] = marker

    report = compare([row])

    assert codes(report) == {"invalid-marker": "malformed"}
    assert report["counts"]["eligible"] == 0
    assert report["observations"] == []


@pytest.mark.parametrize("resolved_at", [DAY + timedelta(minutes=10), AS_OF + timedelta(hours=1)])
def test_withheld_marker_cannot_hide_a_supplied_resolution(resolved_at: datetime) -> None:
    row = observation("EVT-R", observation_id="contradiction", resolved_at=resolved_at)
    row["resolution_unavailable_reason"] = "outcome-after-cutoff"

    report = compare([row])

    assert codes(report) == {"contradiction": "malformed"}
    assert report["counts"]["eligible"] == 0
    assert report["observations"] == []


def test_log_loss_is_bounded_by_protocol_epsilon() -> None:
    report = compare([observation("EVT-Z", yes_ask=0.0, result="yes")])

    row = report["observations"][0]
    assert row["market_yes_ask_log_loss"] == pytest.approx(-math.log(1e-12))
    assert row["market_yes_ask_brier"] == 1.0
    json.dumps(report, allow_nan=False)


def test_empty_population_reports_nulls_not_certainty() -> None:
    report = compare([])

    assert report["counts"]["input_observations"] == report["counts"]["eligible"] == 0
    assert report["metrics"]["arms"]["blend"] == {"brier": None, "log_loss": None}
    for difference in report["paired_differences"]:
        assert difference["point_estimate"] is None
        assert difference["event_bootstrap"] is None
        assert difference["date_block_bootstrap"] is None
    assert report["coverage"]["eligible_fraction"] is None
    assert report["source_selection_branches"] == []
    assert report["conclusion"]["status"] == "inconclusive"
    json.dumps(report, allow_nan=False)


def test_single_event_or_single_date_has_no_bootstrap_interval() -> None:
    single_event = compare(
        [observation("EVT-A"), observation("EVT-A", lower=60_500.0, yes_ask=0.1, result="no")]
    )
    for difference in single_event["paired_differences"]:
        assert difference["point_estimate"] is not None
        assert difference["event_bootstrap"] is None
        assert difference["date_block_bootstrap"] is None
    assert (
        "Fewer than two resolved events: no event bootstrap interval."
        in (single_event["limitations"])
    )

    same_date = compare(
        [observation("EVT-A"), observation("EVT-B", end=DAY - timedelta(hours=3), yes_ask=0.5)]
    )
    difference = same_date["paired_differences"][0]
    assert difference["event_bootstrap"]["clusters"] == 2
    assert difference["date_block_bootstrap"] is None


def test_date_block_bootstrap_keeps_equal_event_weight_on_shared_dates() -> None:
    rows = [
        observation("EVT-1", end=DAY - timedelta(hours=4), yes_ask=0.2, result="yes"),
        observation("EVT-2", end=DAY, yes_ask=0.5, spot=61_000.0, result="no"),
        observation("EVT-3", end=DAY + timedelta(days=1), yes_ask=0.7, result="yes"),
    ]
    report = compare(rows, make_protocol(bootstrap_replicates=2000, confidence=0.2))

    events = {event["event_id"]: event for event in report["events"]}
    assert events["EVT-1"]["event_date"] == events["EVT-2"]["event_date"]
    difference = report["paired_differences"][0]
    assert difference["comparison"] == "structural_minus_market_yes_ask"
    assert difference["metric"] == "brier"
    deltas = [
        events[name]["structural_brier"] - events[name]["market_yes_ask_brier"]
        for name in ("EVT-1", "EVT-2", "EVT-3")
    ]
    event_weighted = sum(deltas) / 3
    date_weighted = ((deltas[0] + deltas[1]) / 2 + deltas[2]) / 2
    assert not math.isclose(event_weighted, date_weighted)
    assert difference["point_estimate"] == pytest.approx(event_weighted)
    block = difference["date_block_bootstrap"]
    assert block["clusters"] == 2
    # With both dates drawn once (half of resamples), the central replicate must weight
    # every event equally rather than every date equally.
    assert block["lower"] == pytest.approx(event_weighted)
    assert block["upper"] == pytest.approx(event_weighted)


def test_event_with_conflicting_observation_dates_is_excluded() -> None:
    report = compare(
        [
            observation("EVT-C", observation_id="c1"),
            observation("EVT-C", observation_id="c2", end=DAY + timedelta(days=1)),
        ]
    )

    assert codes(report) == {"c1": "event_date_conflict", "c2": "event_date_conflict"}
    assert report["counts"]["eligible_events"] == 0


def test_production_rows_require_pinned_feature_recipe_and_report_branches() -> None:
    forward = EVIDENCE_FORWARD_SHADOW
    second = DAY + timedelta(days=1)
    decision = DAY - timedelta(hours=1)
    missing_context = observation(
        "EVT-M", observation_id="missing-context", context=research_context(decision), kind=forward
    )
    missing_context["research_context"] = None
    tampered = observation(
        "EVT-T", observation_id="tampered-context", context=research_context(decision), kind=forward
    )
    tampered["research_context"]["spot"]["close"] = "59990.00"
    rows = [
        observation(
            "EVT-R",
            observation_id="realized",
            context=research_context(decision),
            kind=forward,
        ),
        observation(
            "EVT-I",
            observation_id="implied",
            end=second,
            context=research_context(second - timedelta(hours=1), implied=True),
            kind=forward,
        ),
        observation(
            "EVT-S",
            observation_id="short-window",
            context=research_context(decision, window_seconds=86_400),
            kind=forward,
        ),
        observation(
            "EVT-D",
            observation_id="drift",
            context=research_context(decision),
            drift=0.05,
            kind=forward,
        ),
        missing_context,
        tampered,
    ]
    report = compare(rows, source_kind="retrospective-local")

    assert codes(report) == {
        "short-window": "feature_recipe_mismatch",
        "drift": "feature_recipe_mismatch",
        "missing-context": "research_context_unavailable",
        "tampered-context": "research_context_invalid",
    }
    assert reasons(report)["short-window"] == (
        "feature recipe realized_window_seconds is not the protocol value"
    )
    assert reasons(report)["drift"] == "expected annual return is not the protocol value"
    assert {row["feature_source"] for row in report["observations"]} == {"research-context"}
    branches = {branch["branch"]: branch for branch in report["source_selection_branches"]}
    assert set(branches) == {"implied", "realized"}
    assert branches["implied"]["selected_volatility_provider"] == "deribit"
    assert branches["implied"]["selected_volatility_window_seconds"] == 3600
    assert branches["realized"]["selected_volatility_provider"] == "coinbase:calculated"
    assert branches["realized"]["selected_volatility_window_seconds"] == PRODUCTION_WINDOW
    assert branches["implied"]["recipe_id"] != branches["realized"]["recipe_id"]
    assert all(branch["resolved_events"] == 1 for branch in branches.values())
    assert report["counts"]["source_selection_branches"] == 2
    assert len(report["identities"]["recipe_ids"]) == 2
    assert report["identities"]["production_features"] == BASE_PROTOCOL["production_features"]
    assert report["conclusion"]["status"] == "inconclusive"


def test_production_rows_revalidate_recorded_kalshi_rules_and_context() -> None:
    forward = EVIDENCE_FORWARD_SHADOW
    decision = DAY - timedelta(hours=1)
    context = research_context(decision)
    stale = decision - timedelta(hours=3)
    widened = context.model_dump(mode="json")
    widened["implied_volatility"] = {
        "provider": "deribit",
        "symbol": "BTC",
        "kind": "implied",
        "window_seconds": 3600,
        "source_start_at": (stale - timedelta(hours=1)).isoformat(),
        "observed_at": stale.isoformat(),
        "annualized_volatility": 0.9,
        "retrieved_at": stale.isoformat(),
        "raw_payload": {"resolution_seconds": 3600},
    }
    rows = [
        observation("EVT-OK", observation_id="valid", context=context, kind=forward),
        observation("EVT-M", observation_id="no-raw", context=context, kind=forward, raw=False),
        observation(
            "EVT-W",
            observation_id="weighted",
            context=context,
            kind=forward,
            venue_rule=kalshi_rule(DAY, 59_500.0, 60_500.0, averaging="weighted"),
        ),
        observation(
            "EVT-X",
            observation_id="rule-text",
            context=context,
            kind=forward,
            snapshot_rule="Resolves YES if BTC finishes inside the range.",
        ),
        observation(
            "EVT-N",
            observation_id="normalized",
            context=context,
            context_payload=widened,
            context_id=content_id(widened),
            kind=forward,
        ),
    ]
    report = compare(rows, source_kind="retrospective-local")

    assert codes(report) == {
        "no-raw": "unsupported_contract",
        "weighted": "unsupported_contract",
        "rule-text": "identity_mismatch",
        "normalized": "research_context_invalid",
    }
    assert reasons(report)["no-raw"] == "missing raw Kalshi market semantics"
    assert reasons(report)["weighted"] == (
        "recorded Kalshi market is unsupported: "
        "only explicit simple arithmetic averaging is supported"
    )
    assert reasons(report)["rule-text"] == "recorded Kalshi rule text does not match"
    assert reasons(report)["normalized"] == "research context changed during validation"
    assert [row["observation_id"] for row in report["observations"]] == ["valid"]


def test_recorded_calibration_profile_must_be_compatible() -> None:
    report = compare(
        [
            observation("EVT-A", observation_id="compatible", calibration_events=40),
            observation("EVT-B", observation_id="too-few-events", calibration_events=5),
        ]
    )

    assert codes(report) == {"too-few-events": "calibration_profile_invalid"}
    assert report["observations"][0]["uncertainty_source"] == "held_out"


def test_fold_coverage_assigns_events_by_earliest_decision_time() -> None:
    validation_day = datetime(2026, 8, 28, 16, tzinfo=UTC)
    report = compare(
        [observation("EVT-V", end=validation_day), observation("EVT-A")],
        make_protocol(minimum_fold_resolved_events=1),
    )

    folds = {fold["name"]: fold for fold in report["coverage"]["folds"]}
    assert folds["validation-1"]["resolved_events"] == 1
    assert folds["validation-1"]["qualifies"] is True
    assert folds["validation-2"]["qualifies"] is False
    assert folds["holdout-1"]["resolved_events"] == 0
    assert report["counts"]["qualifying_validation_folds"] == 1
    assert report["counts"]["qualifying_holdout_folds"] == 0


def test_every_source_kind_is_inconclusive_and_manual_features_are_synthetic_only() -> None:
    synthetic = compare([observation("EVT-A"), observation("EVT-B", end=DAY + timedelta(days=1))])
    assert synthetic["conclusion"]["status"] == "inconclusive"
    assert synthetic["observations"][0]["feature_source"] == "manual-synthetic"
    assert [branch["branch"] for branch in synthetic["source_selection_branches"]] == [
        "manual-synthetic"
    ]

    wrong_class = compare(
        [observation("EVT-A", observation_id="a")], source_kind="retrospective-local"
    )
    assert codes(wrong_class) == {"a": "evidence_class"}
    manual = observation("EVT-A", observation_id="a", kind=EVIDENCE_FORWARD_SHADOW)
    retrospective = compare([manual], source_kind="retrospective-local")
    assert reasons(retrospective) == {"a": "manual features are allowed only for synthetic sources"}
    assert retrospective["conclusion"]["status"] == "inconclusive"

    with pytest.raises(ValueError, match="source kind"):
        compare([], source_kind="forward-holdout")
    with pytest.raises(ValueError, match="timezone"):
        compare_observations(
            [], make_protocol(), source_kind="synthetic-fixture", as_of=datetime(2026, 9, 26)
        )


@pytest.mark.parametrize("offset_seconds", [-1, 1])
def test_report_cutoff_cannot_override_frozen_protocol(offset_seconds: int) -> None:
    row = observation("EVT-F", resolved_at=AS_OF + timedelta(seconds=1))

    with pytest.raises(ValueError, match="report_as_of"):
        compare_observations(
            [row],
            make_protocol(),
            source_kind="synthetic-fixture",
            as_of=AS_OF + timedelta(seconds=offset_seconds),
        )


def test_equivalent_offset_report_cutoff_preserves_frozen_population() -> None:
    rows = [
        observation("EVT-R", observation_id="resolved"),
        observation(
            "EVT-F",
            observation_id="future",
            resolved_at=AS_OF + timedelta(seconds=1),
        ),
    ]

    report = compare_observations(
        rows,
        make_protocol(),
        source_kind="synthetic-fixture",
        as_of=datetime.fromisoformat("2026-09-25T19:00:00-05:00"),
    )

    assert report == compare(rows)
    assert report["as_of"] == AS_OF.isoformat()
    assert report["counts"]["resolved"] == 1
    assert report["counts"]["unresolved_outcome_after_cutoff"] == 1


@pytest.mark.parametrize("offset_seconds", [-1, 1])
def test_manifest_recording_must_match_forecast_time(offset_seconds: int) -> None:
    row = observation("EVT-R", observation_id="mistimed-recording")
    recorded_at = datetime.fromisoformat(row["observed_at"]) + timedelta(seconds=offset_seconds)
    row["run_manifest"]["recorded_at"] = recorded_at.isoformat()
    row["run_manifest_sha256"] = content_id(row["run_manifest"])

    report = compare([row])

    assert codes(report) == {"mistimed-recording": "forecast_time_mismatch"}
    assert report["counts"]["eligible"] == 0
    assert report["observations"] == []


@pytest.mark.parametrize(
    "recording_fields",
    [{}, {"recorded_at": None}, {"recorded_at": "invalid"}, {"recorded_at": "2026-08-10T15:00:00"}],
)
def test_manifest_recording_requires_valid_aware_timestamp(
    recording_fields: dict[str, Any],
) -> None:
    row = observation("EVT-R", observation_id="invalid-recording")
    row["run_manifest"].pop("recorded_at")
    row["run_manifest"].update(recording_fields)
    row["run_manifest_sha256"] = content_id(row["run_manifest"])

    report = compare([row])

    assert codes(report) == {"invalid-recording": "invalid_timestamp"}
    assert report["counts"]["eligible"] == 0
    assert report["observations"] == []


def test_manifest_recording_accepts_equivalent_timezone_offset() -> None:
    row = observation("EVT-R", observation_id="equivalent-recording")
    row["run_manifest"]["recorded_at"] = "2026-08-10T10:00:00-05:00"
    row["run_manifest_sha256"] = content_id(row["run_manifest"])

    report = compare([row])

    assert report["excluded"] == []
    assert report["counts"]["eligible"] == report["counts"]["resolved"] == 1
    assert report["observations"][0]["outcome_yes"] == 1
