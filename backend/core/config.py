"""
App configuration loaded from environment variables / .env file.
"""

from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULT_KEY = "dev-secret-change-in-production"


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
    secret_key:         str = _INSECURE_DEFAULT_KEY
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

    @model_validator(mode="after")
    def _enforce_secret_key(self) -> "Settings":
        if self.secret_key == _INSECURE_DEFAULT_KEY:
            if not self.debug:
                raise ValueError(
                    "SECRET_KEY must be set via environment variable in production. "
                    "Add SECRET_KEY=<random-256-bit-string> to your .env file."
                )
            import warnings
            warnings.warn(
                "SECRET_KEY is using the insecure default value. "
                "Set SECRET_KEY in your .env file before deploying.",
                stacklevel=2,
            )
        return self


settings = Settings()
