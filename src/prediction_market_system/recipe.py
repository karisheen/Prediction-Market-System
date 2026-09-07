"""Canonical identities for probability recipes, independent of execution/risk policy.

Deployment approval uses a separate operational-policy fingerprint so a calibration
profile can remain valid after a risk-policy change, while managed delivery cannot
reuse that approval under incompatible execution, fee, cost, or sizing assumptions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

MODEL_VERSION = "2.1.0"

# Live scans use a completed one-minute Coinbase decision price. Hourly research
# estimates realized volatility and other trailing features. These are independent
# recipe fields; do not collapse them merely because both are "intervals".
DECISION_SPOT_INTERVAL_SECONDS = 60
DEFAULT_RESEARCH_INTERVAL_SECONDS = 3600
MANAGED_KALSHI_PERIOD_MINUTES: Literal[1] = 1
# Fetch only recently completed venue candles. A long first lookback would persist
# late receipts; INSERT OR IGNORE would then pin that late availability forever.
PROSPECTIVE_CANDLE_LOOKBACK_SECONDS = 10 * 60


def recipe_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
