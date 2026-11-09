"""Prompt 的 token 估算和多轮历史的动态裁剪。"""

from functools import lru_cache

import tiktoken

from app.config.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 各模型的 context window（输入 + 输出的 token 上限）
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
}
DEFAULT_CONTEXT_WINDOW = 8_192  # 不认识的模型按最保守的算

# OpenAI chat 格式的固定开销：每条 message 约 3 个 token（role 等标记），回复引导约 3 个
TOKENS_PER_MESSAGE = 3
TOKENS_REPLY_PRIMING = 3


@lru_cache
def _get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_message_tokens(messages: list[dict], model: str | None = None) -> int:
    """估算一组 messages 发给模型时占用的 prompt token 数。"""
    encoding = _get_encoding(model or get_settings().openai_model)
    total = TOKENS_REPLY_PRIMING
    for message in messages:
        total += TOKENS_PER_MESSAGE + len(encoding.encode(message["content"]))
    return total


def get_context_window(model: str | None = None) -> int:
    settings = get_settings()
    if settings.llm_context_window:
        return settings.llm_context_window
    return MODEL_CONTEXT_WINDOWS.get(model or settings.openai_model, DEFAULT_CONTEXT_WINDOW)


def get_prompt_token_budget(model: str | None = None) -> int:
    """Prompt 允许占用的 token 上限 = context window × 比例（默认 70%），剩下的留给输出。"""
    return int(get_context_window(model) * get_settings().prompt_context_ratio)


def _group_into_turns(history: list[dict]) -> list[list[dict]]:
    """把历史消息按"轮"分组：每个 user 消息开启新的一轮，后面的 assistant 归入同一轮。

    裁剪时按整轮丢弃，避免出现只剩回答、没有问题的半轮。
    """
    turns: list[list[dict]] = []
    for message in history:
        if message["role"] == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def fit_history_to_budget(
    history: list[dict],
    fixed_tokens: int,
    budget: int,
    model: str | None = None,
) -> tuple[list[dict], int]:
    """从最新的一轮往前加，直到再加一轮就会超出预算。

    fixed_tokens: system prompt + Schema + few-shot + 当前问题等必须保留部分的 token 数
    返回 (保留的历史, 保留的轮数)。
    """
    encoding = _get_encoding(model or get_settings().openai_model)
    remaining = budget - fixed_tokens
    kept_turns: list[list[dict]] = []

    for turn in reversed(_group_into_turns(history)):
        turn_tokens = sum(TOKENS_PER_MESSAGE + len(encoding.encode(m["content"])) for m in turn)
        if turn_tokens > remaining:
            break  # 更早的轮次即使更短也不要，保持历史连续
        kept_turns.append(turn)
        remaining -= turn_tokens

    kept_turns.reverse()
    return [m for turn in kept_turns for m in turn], len(kept_turns)
