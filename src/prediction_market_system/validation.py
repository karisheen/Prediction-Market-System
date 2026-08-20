from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from prediction_market_system.backtest import BacktestModelValidation


class ValidationCampaignState(StrEnum):
    COLLECTING = "COLLECTING EVIDENCE"
    REJECTED = "NOT YET APPROVED"
    PARTIALLY_APPROVED = "PARTIALLY APPROVED"
    APPROVED = "APPROVED"


@dataclass(frozen=True)
class ValidationCampaignReport:
    series_ticker: str
    symbol: str
    generated_at: datetime
    state: ValidationCampaignState
    coverage_start: datetime
    coverage_end: datetime
    coverage_days: int
    required_days: int
    validations: tuple[BacktestModelValidation, ...] = ()
    run_id: str | None = None

    @property
    def progress(self) -> float:
        if self.required_days <= 0:
            return 1.0
        return min(self.coverage_days / self.required_days, 1.0)
