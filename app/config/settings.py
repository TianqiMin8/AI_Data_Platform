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
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 2048
    openai_timeout: float = 30.0

    # ── Logging ──
    log_level: str = "INFO"
    log_format: str = "json"  # json / console

    # ── JWT ──
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 小时

    # ── Encryption ──
    encryption_key: str = ""  # Fernet key，用于加密数据库密码

    # ── SQL Engine ──
    sql_max_rows: int = 100
    sql_timeout_seconds: int = 15

    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()