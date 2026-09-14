from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.core.errors import LLMError
from app.llm.base import BaseLLMAdapter, LLMResponse, LLMUsage

logger = get_logger(__name__)


class OpenAIAdapter(BaseLLMAdapter):

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
        )
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens

    async def generate(self, messages: list[dict]) -> str:
        response = await self.generate_with_usage(messages)
        return response.content

    async def generate_with_usage(self, messages: list[dict]) -> LLMResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content or ""
            usage = response.usage

            llm_usage = LLMUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )

            logger.info(
                "llm.generate.completed",
                model=self.model,
                prompt_tokens=llm_usage.prompt_tokens,
                completion_tokens=llm_usage.completion_tokens,
            )

            return LLMResponse(
                content=content,
                model=self.model,
                usage=llm_usage,
            )

        except Exception as e:
            logger.error("llm.generate.failed", model=self.model, error=str(e))
            raise LLMError(f"LLM 调用失败：{str(e)}")