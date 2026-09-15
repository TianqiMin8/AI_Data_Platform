from app.llm.base import BaseLLMAdapter, LLMResponse, LLMUsage


class StubAdapter(BaseLLMAdapter):
    """用于测试和演示的固定返回 LLM Adapter。"""

    async def generate(
        self,
        messages: list[dict],
    ) -> str:
        return "SELECT 1 AS test"

    async def generate_with_usage(
        self,
        messages: list[dict],
    ) -> LLMResponse:
        return LLMResponse(
            content="SELECT 1 AS test",
            model="stub",
            usage=LLMUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )