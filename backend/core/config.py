"""
App configuration loaded from environment variables / .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    app_name:    str = "Multicurrency Investment AI"
    app_version: str = "1.0.0"
    debug:       bool = False

    # Auth
    secret_key:         str = "dev-secret-change-in-production"
    algorithm:          str = "HS256"
    access_token_expire_minutes: int = 60

    # Database (optional – not required for ML-only mode)
    database_url: str = "sqlite:///./data/app.db"

    # ML
    processed_data_dir: str = "data/processed"
    checkpoints_dir:    str = "data/checkpoints"
    sequence_length:    int = 60
    default_budget:     float = 10_000.0
    default_risk:       str = "medium"


settings = Settings()
