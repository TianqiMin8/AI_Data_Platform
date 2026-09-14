from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: LLMUsage


class BaseLLMAdapter(ABC):
    """所有 LLM 适配器的抽象基类。"""

    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        """输入 messages，返回纯文本。"""
        ...

    @abstractmethod
    async def generate_with_usage(self, messages: list[dict]) -> LLMResponse:
        """输入 messages，返回文本 + token 用量。"""
        ...