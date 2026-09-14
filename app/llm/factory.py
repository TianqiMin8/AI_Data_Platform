from app.config.settings import get_settings
from app.llm.base import BaseLLMAdapter
from app.llm.openai_adapter import OpenAIAdapter


def get_llm_adapter() -> BaseLLMAdapter:
    """根据配置返回对应的 LLM 适配器。"""
    settings = get_settings()

    # 当前只有 OpenAI，后续在这里扩展
    # if settings.llm_provider == "deepseek":
    #     return DeepSeekAdapter()
    # if settings.llm_provider == "azure":
    #     return AzureOpenAIAdapter()

    return OpenAIAdapter()