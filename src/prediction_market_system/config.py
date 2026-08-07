from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PMS_",
        extra="ignore",
        case_sensitive=False,
    )

    database_path: Path = Path("data/prediction_markets.db")
    discord_webhook_url: SecretStr | None = None

    paper_bankroll: float = Field(default=10_000.0, gt=0.0)
    min_conservative_edge: float = Field(default=0.03, ge=0.0, le=1.0)
    uncertainty_margin: float = Field(default=0.05, ge=0.0, le=0.5)
    structural_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    fee_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    binary_fee_coefficient: float = Field(default=0.07, ge=0.0)
    slippage_bps: float = Field(default=25.0, ge=0.0)
    resolution_haircut: float = Field(default=0.01, ge=0.0, le=1.0)
    minimum_ask_size: float = Field(default=10.0, ge=0.0)
    fractional_kelly: float = Field(default=0.25, ge=0.0, le=1.0)
    max_bankroll_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    max_event_bankroll_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    maximum_live_spot_age_seconds: int = Field(default=120, ge=1, le=3600)
    minimum_seconds_to_expiry: int = Field(default=300, ge=0)
