import re
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


class HtmlParser(BaseParser):
    """HTML 解析：去掉脚本/样式/导航/广告，提取正文，表格转文本，按 h1–h3 分段。

    正文定位顺序：<main> / <article> / role="main" → readability 算法 → 整个 <body>
    """

    # 一定不是正文的标签，整块删除
    NON_CONTENT_TAGS = [
        "script", "style", "noscript", "template", "iframe", "svg", "canvas",
        "nav", "header", "footer", "aside", "form", "button", "select", "input",
    ]
    # class / id 里出现这些词的元素视为广告、侧边栏等（按 - _ 空格切词后整词匹配，"shadow" 不会命中 "ad"）
    # 不含 "header"：<div class="article-header"> 往往装着文章标题；页面顶栏一般用 <header> 标签，已在上面删除
    NOISE_HINTS = {
        "ad", "ads", "advert", "advertisement", "banner", "sidebar", "promo",
        "sponsor", "cookie", "popup", "modal", "share", "social", "comment", "comments",
        "related", "recommend", "breadcrumb", "menu", "navbar", "footer",
    }
    HEADING_TAGS = ["h1", "h2", "h3"]
    # 块级元素：取文本时在后面补换行，避免 <a>、<b> 这类行内元素把一句话拆成多行
    BLOCK_TAGS = [
        "p", "div", "section", "li", "tr", "blockquote", "pre",
        "h4", "h5", "h6", "dt", "dd", "figcaption", "table",
    ]
    HEADING_MARK = "\x00HEADING\x00"
    MIN_READABILITY_CHARS = 50  # readability 提取结果太短，视为失败，退回 <body>

    def parse(self, file_path: str) -> ParsedDocument:
        from bs4 import BeautifulSoup

        # 以字节读入，交给 BeautifulSoup 按 <meta charset> 识别编码（中文网页常见 GBK）
        with open(file_path, "rb") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "lxml")

        page_title = soup.title.get_text(strip=True) if soup.title else None

        self._remove_noise(soup)
        for table in soup.find_all("table"):
            self._table_to_text(soup, table)

        container, extraction = self._locate_content(soup)
        pages = self._split_sections(container, default_title=page_title)

        total_chars = sum(len(p.content) for p in pages)
        logger.info(
            "html.parsed",
            sections=len(pages),
            total_characters=total_chars,
            extraction=extraction,
        )
        return ParsedDocument(
            pages=pages,
            total_characters=total_chars,
            metadata={"title": page_title, "extraction": extraction},
        )

    # ── 去噪 ──

    def _remove_noise(self, soup) -> None:
        for tag in soup.find_all(self.NON_CONTENT_TAGS):
            tag.decompose()

        for tag in soup.find_all(True):
            if tag.decomposed or tag.name in ("html", "body", "main", "article"):
                continue
            hints = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
            words = set(re.split(r"[\s\-_]+", hints.lower()))
            if words & self.NOISE_HINTS:
                tag.decompose()

    # ── 表格 ──

    def _table_to_text(self, soup, table) -> None:
        """把表格转成逐行文本：有表头时每行写成"列名: 值；列名: 值"，没有表头时用 | 连接。

        这样切块后每一行都自带列名，"XX 的价格是多少"这类问题也能检索到。
        """
        rows = [
            [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            for tr in table.find_all("tr")
        ]
        rows = [r for r in rows if any(r)]

        header = None
        first_tr = table.find("tr")
        if first_tr and first_tr.find("th") and not first_tr.find("td"):
            header, rows = rows[0], rows[1:]

        lines = []
        caption = table.find("caption")
        if caption:
            lines.append(f"表格：{caption.get_text(strip=True)}")
        for row in rows:
            if header and len(header) == len(row):
                lines.append("；".join(f"{h}: {v}" for h, v in zip(header, row)))
            else:
                lines.append(" | ".join(row))

        replacement = soup.new_tag("div")
        for line in lines:
            p = soup.new_tag("p")
            p.string = line
            replacement.append(p)
        table.replace_with(replacement)

    # ── 正文定位 ──

    def _locate_content(self, soup):
        from bs4 import BeautifulSoup

        semantic = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
        if semantic:
            return semantic, "semantic"

        body = soup.body or soup
        try:
            from readability import Document as ReadabilityDocument

            summary_html = ReadabilityDocument(str(body)).summary(html_partial=True)
            extracted = BeautifulSoup(summary_html, "lxml")
            if len(extracted.get_text(strip=True)) >= self.MIN_READABILITY_CHARS:
                return extracted, "readability"
        except Exception as e:
            logger.warning("html.readability_failed", error=str(e))

        return body, "body"

    # ── 分段 ──

    def _split_sections(self, container, default_title: str | None) -> list[ParsedPage]:
        for br in container.find_all("br"):
            br.replace_with("\n")
        for tag in container.find_all(self.BLOCK_TAGS):
            tag.append("\n")
        for li in container.find_all("li"):
            li.insert(0, "- ")
        for heading in container.find_all(self.HEADING_TAGS):
            heading.replace_with(f"\n{self.HEADING_MARK}{heading.get_text(' ', strip=True)}\n")

        lines = [line.strip() for line in container.get_text().split("\n")]

        pages: list[ParsedPage] = []
        current_title = default_title
        current_lines: list[str] = []

        def flush():
            content = "\n".join(current_lines).strip()
            if content:
                pages.append(ParsedPage(content=content, section_title=current_title))

        for line in lines:
            if line.startswith(self.HEADING_MARK):
                flush()
                current_title = line.removeprefix(self.HEADING_MARK).strip() or current_title
                current_lines = [current_title]
            elif line:
                current_lines.append(line)

        flush()
        return pages


# ── 工厂函数 ──

PARSER_MAP = {
    "txt": TxtParser,
    "md": MarkdownParser,
    "markdown": MarkdownParser,
    "docx": DocxParser,
    "pdf": PdfParser,
    "html": HtmlParser,
    "htm": HtmlParser,
}


def get_parser(file_type: str) -> BaseParser:
    """根据文件类型返回对应的解析器。"""
    parser_cls = PARSER_MAP.get(file_type.lower())
    if not parser_cls:
        raise ValueError(f"不支持的文件类型：{file_type}")
    return parser_cls()