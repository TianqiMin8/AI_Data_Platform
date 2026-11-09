from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """解析出的一页/一段内容。"""
    content: str
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """解析出的完整文档。"""
    pages: list[ParsedPage]
    total_characters: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.content for p in self.pages if p.content.strip())


class BaseParser(ABC):
    """文档解析器抽象基类。"""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        ...


class TxtParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return ParsedDocument(
            pages=[ParsedPage(content=content, page_number=1)],
            total_characters=len(content),
        )


class MarkdownParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import markdown
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            md_text = f.read()

        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, "html.parser")
        clean_text = soup.get_text(separator="\n")

        # 按标题分段
        pages = []
        current_section = ""
        current_title = None

        for line in clean_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                current_section += "\n"
                continue

            # 检测是否是原始 Markdown 标题
            md_lines = md_text.split("\n")
            is_heading = any(
                l.strip().startswith("#") and stripped in l
                for l in md_lines
            )

            if is_heading and current_section.strip():
                pages.append(ParsedPage(
                    content=current_section.strip(),
                    section_title=current_title,
                ))
                current_section = stripped + "\n"
                current_title = stripped
            else:
                current_section += stripped + "\n"

        if current_section.strip():
            pages.append(ParsedPage(
                content=current_section.strip(),
                section_title=current_title,
            ))

        total_chars = sum(len(p.content) for p in pages)
        return ParsedDocument(pages=pages, total_characters=total_chars)


class DocxParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        pages = []
        current_section = ""
        current_title = None

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # 检测标题样式
            if para.style and para.style.name.startswith("Heading"):
                if current_section.strip():
                    pages.append(ParsedPage(
                        content=current_section.strip(),
                        section_title=current_title,
                    ))
                current_section = text + "\n"
                current_title = text
            else:
                current_section += text + "\n"

        if current_section.strip():
            pages.append(ParsedPage(
                content=current_section.strip(),
                section_title=current_title,
            ))

        total_chars = sum(len(p.content) for p in pages)
        return ParsedDocument(pages=pages, total_characters=total_chars)


class PdfParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(ParsedPage(
                    content=text,
                    page_number=i + 1,
                ))

        total_chars = sum(len(p.content) for p in pages)
        logger.info("pdf.parsed", pages=len(pages), total_characters=total_chars)
        return ParsedDocument(pages=pages, total_characters=total_chars)


# ── 工厂函数 ──

PARSER_MAP = {
    "txt": TxtParser,
    "md": MarkdownParser,
    "markdown": MarkdownParser,
    "docx": DocxParser,
    "pdf": PdfParser,
}


def get_parser(file_type: str) -> BaseParser:
    """根据文件类型返回对应的解析器。"""
    parser_cls = PARSER_MAP.get(file_type.lower())
    if not parser_cls:
        raise ValueError(f"不支持的文件类型：{file_type}")
    return parser_cls()