from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkResult:
    """一个切片的结果。"""
    content: str
    chunk_index: int
    char_count: int
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict | None = None


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> list[ChunkResult]:
        ...


class FixedWindowChunker(BaseChunker):
    """固定窗口 + Overlap 切片。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, **kwargs) -> list[ChunkResult]:
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            # 尽量在句号/换行处切断，而不是从中间切断
            if end < len(text):
                # 从 end 往前找最近的断句点
                for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(ChunkResult(
                    content=chunk_text,
                    chunk_index=index,
                    char_count=len(chunk_text),
                ))
                index += 1

            start = end - self.overlap
            if start >= len(text):
                break

        logger.info("chunker.fixed_window", chunks=len(chunks), chunk_size=self.chunk_size)
        return chunks


class HeadingChunker(BaseChunker):
    """按标题分割——适用于 Markdown / 有结构的文档。"""

    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, **kwargs) -> list[ChunkResult]:
        import re
        # 按 Markdown 标题或数字编号分段
        sections = re.split(r'\n(?=#{1,4}\s|(?:\d+\.)+\s)', text)

        chunks = []
        index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 提取标题
            lines = section.split("\n", 1)
            title = lines[0].lstrip("#").strip() if lines[0].startswith("#") else None

            # 如果段落太长，用 FixedWindowChunker 二次切割
            if len(section) > self.max_chunk_size:
                sub_chunker = FixedWindowChunker(
                    chunk_size=self.max_chunk_size, overlap=50
                )
                sub_chunks = sub_chunker.chunk(section)
                for sc in sub_chunks:
                    sc.chunk_index = index
                    sc.section_title = title
                    chunks.append(sc)
                    index += 1
            else:
                chunks.append(ChunkResult(
                    content=section,
                    chunk_index=index,
                    char_count=len(section),
                    section_title=title,
                ))
                index += 1

        logger.info("chunker.heading", chunks=len(chunks))
        return chunks


class RecursiveChunker(BaseChunker):
    """递归分割——先按大分隔符切，太长再按小分隔符切。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", "。", ".", " "]

    def chunk(self, text: str, **kwargs) -> list[ChunkResult]:
        raw_chunks = self._split_recursive(text, self.separators)

        results = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if chunk_text:
                results.append(ChunkResult(
                    content=chunk_text,
                    chunk_index=i,
                    char_count=len(chunk_text),
                ))

        logger.info("chunker.recursive", chunks=len(results), chunk_size=self.chunk_size)
        return results

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # 没有分隔符了，强制按 chunk_size 切
            return self._force_split(text)

        sep = separators[0]
        parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 如果单个 part 就超长，递归用更小的分隔符
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_recursive(part, separators[1:])
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _force_split(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


# ── 工厂函数 ──

CHUNKER_MAP = {
    "fixed": FixedWindowChunker,
    "heading": HeadingChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    chunker_cls = CHUNKER_MAP.get(strategy)
    if not chunker_cls:
        raise ValueError(f"不支持的 Chunking 策略：{strategy}")
    return chunker_cls(**kwargs)