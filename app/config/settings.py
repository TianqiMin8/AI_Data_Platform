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
    llm_context_window: int | None = None  # 不填则按模型名查表（gpt-4o = 128000）
    prompt_context_ratio: float = 0.7      # Prompt 最多占 context window 的比例，其余留给输出
    max_history_fetch_turns: int = 20      # 从数据库最多读取的历史轮数（再由 token 预算决定保留几轮）

    # ── Logging ──
    log_level: str = "INFO"
    log_format: str = "json"  # json / console

    # ── JWT ──
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 小时

    # ── Encryption ──
    encryption_key: str = ""  # Fernet key，用于加密数据库密码
    encryption_old_keys: str = ""  # 旧的 Fernet key，用于解密数据库密码，多个 key 用逗号分隔

    # ── SQL Engine ──
    sql_max_rows: int = 100
    sql_timeout_seconds: int = 15
    # 全局敏感列关键字：列名包含其中任一关键字（不区分大小写）即视为敏感列
    # .env 中用 JSON 数组配置：SQL_SENSITIVE_COLUMNS=["password","ssn","salary","credit_card"]
    sql_sensitive_columns: list[str] = ["password", "ssn", "salary", "credit_card"]

    # ── Chunking ──
    chunk_strategy: str = "recursive"     # fixed / heading / recursive
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 20

    # ── Re-ranking ──
    rerank_method: str = "cross_encoder"            # cross_encoder / llm
    cross_encoder_model: str = "BAAI/bge-reranker-base"

    # ── Embedding 缓存 ──
    embedding_cache_enabled: bool = True
    # 同一模型下同一段文本的向量不会变，TTL 只用于控制内存；None 表示永不过期
    embedding_cache_ttl_seconds: int | None = 30 * 24 * 3600

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600  # 缓存 1 小时

    # ── 语义缓存 ──
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95  # 余弦相似度高于该值才命中

    # ── 缓存命中率监控 ──
    cache_hit_rate_warn_threshold: float = 0.2  # 命中率低于该值输出 warning
    cache_hit_rate_min_samples: int = 20        # 样本太少时命中率没有意义，不告警；也是告警的间隔

    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()