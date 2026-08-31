from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──
    app_name: str = "ai-data-platform"
    app_version: str = "0.1.0"
    environment: str = "dev"  # dev / staging / prod
    debug: bool = False

    # ── Database ──
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_data_platform"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM ──
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # ── Logging ──
    log_level: str = "INFO"
    log_format: str = "json"  # json / console

    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()