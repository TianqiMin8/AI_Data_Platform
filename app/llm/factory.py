from app.config.settings import get_settings
from app.llm.base import BaseLLMAdapter
from app.llm.openai_adapter import OpenAIAdapter
from app.llm.stub_adapter import StubAdapter


def get_llm_adapter() -> BaseLLMAdapter:
    """根据配置返回对应的 LLM 适配器。"""
    settings = get_settings()

    if settings.llm_provider == "stub":
        return StubAdapter()

    if settings.llm_provider == "openai":
        return OpenAIAdapter()

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )