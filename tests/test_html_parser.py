"""作业 6-1：HTML 解析器。"""

from app.rag.parser import HtmlParser, get_parser, PARSER_MAP
from app.services.document import ALLOWED_TYPES

SEMANTIC_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>2026 年产品手册</title>
  <style>.x { color: red; }</style>
  <script>console.log("tracking");</script>
</head>
<body>
  <header><a href="/">首页</a> <a href="/about">关于我们</a></header>
  <nav><ul><li>导航一</li><li>导航二</li></ul></nav>
  <main>
    <div class="article-header"><h1>产品手册</h1></div>
    <p>本手册介绍公司的 <a href="/p">核心产品</a> 和价格政策。</p>
    <div class="ad-banner">限时广告：全场五折！</div>
    <h2>价格表</h2>
    <table>
      <caption>2026 年价格</caption>
      <tr><th>产品</th><th>价格</th></tr>
      <tr><td>电子产品 A</td><td>100</td></tr>
      <tr><td>服装 B</td><td>50</td></tr>
    </table>
    <h2>售后政策</h2>
    <p>购买后 7 天内可无理由退货。<br>质保期为一年。</p>
    <ul><li>支持上门维修</li><li>支持以旧换新</li></ul>
  </main>
  <aside class="sidebar">热门推荐：其他文章</aside>
  <footer>版权所有 © 2026</footer>
</body>
</html>"""


def _parse(tmp_path, html: str | bytes, name: str = "page.html"):
    path = tmp_path / name
    if isinstance(html, str):
        path.write_text(html, encoding="utf-8")
    else:
        path.write_bytes(html)
    return HtmlParser().parse(str(path))


# ── 注册 ──


def test_registered_for_html_and_htm():
    assert PARSER_MAP["html"] is HtmlParser
    assert PARSER_MAP["htm"] is HtmlParser
    assert isinstance(get_parser("HTML"), HtmlParser)


def test_upload_whitelist_allows_html():
    # 只注册 PARSER_MAP 不够，上传接口的白名单也要放行
    assert {"html", "htm"} <= ALLOWED_TYPES


# ── 去噪 ──


def test_removes_script_style_nav_ads_and_sidebar(tmp_path):
    text = _parse(tmp_path, SEMANTIC_PAGE).full_text
    for noise in ["tracking", "color: red", "导航一", "关于我们", "限时广告", "热门推荐", "版权所有"]:
        assert noise not in text, noise


def test_keeps_article_header_title(tmp_path):
    # class="article-header" 不能被当成页面顶栏删掉
    doc = _parse(tmp_path, SEMANTIC_PAGE)
    assert doc.pages[0].section_title == "产品手册"


def test_uses_main_element(tmp_path):
    doc = _parse(tmp_path, SEMANTIC_PAGE)
    assert doc.metadata["extraction"] == "semantic"
    assert doc.metadata["title"] == "2026 年产品手册"


# ── 分段 ──


def test_splits_sections_by_headings(tmp_path):
    doc = _parse(tmp_path, SEMANTIC_PAGE)
    assert [p.section_title for p in doc.pages] == ["产品手册", "价格表", "售后政策"]
    assert doc.pages[2].content.startswith("售后政策")


def test_inline_links_do_not_break_lines(tmp_path):
    doc = _parse(tmp_path, SEMANTIC_PAGE)
    assert "本手册介绍公司的 核心产品 和价格政策。" in doc.pages[0].content


def test_br_and_list_items(tmp_path):
    content = _parse(tmp_path, SEMANTIC_PAGE).pages[2].content
    assert "购买后 7 天内可无理由退货。\n质保期为一年。" in content
    assert "- 支持上门维修" in content


def test_page_without_headings_uses_title(tmp_path):
    html = "<html><head><title>公告</title></head><body><main><p>系统将于今晚维护。</p></main></body></html>"
    doc = _parse(tmp_path, html)
    assert len(doc.pages) == 1
    assert doc.pages[0].section_title == "公告"


# ── 表格 ──


def test_table_with_header_becomes_key_value_lines(tmp_path):
    content = _parse(tmp_path, SEMANTIC_PAGE).pages[1].content
    assert "表格：2026 年价格" in content
    assert "产品: 电子产品 A；价格: 100" in content
    assert "产品: 服装 B；价格: 50" in content


def test_table_without_header_joins_cells(tmp_path):
    html = "<html><body><main><table><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table></main></body></html>"
    content = _parse(tmp_path, html).full_text
    assert "A | 1" in content
    assert "B | 2" in content


# ── 编码 ──


def test_gbk_encoded_page(tmp_path):
    html = '<html><head><meta charset="gbk"><title>中文</title></head><body><main><p>销售额同比增长百分之二十。</p></main></body></html>'
    doc = _parse(tmp_path, html.encode("gbk"))
    assert "销售额同比增长百分之二十" in doc.full_text


# ── 没有 <main> / <article>：readability ──


def test_readability_drops_link_heavy_block(tmp_path):
    paragraph = (
        "<p>数据平台的核心目标，是让业务人员用自然语言直接查询数据库，"
        "而不需要学习 SQL。系统会根据数据库结构生成查询语句，并在执行前做多层安全校验，"
        "确保只执行只读查询，同时对敏感字段进行过滤，避免泄露用户隐私信息。</p>"
    )
    links = "".join(f'<a href="/p{i}">相关链接{i}</a> ' for i in range(30))
    html = f"""<html><head><title>平台介绍</title></head><body>
        <div class="links-box">{links}</div>
        <div class="content">{paragraph * 6}</div>
    </body></html>"""
    doc = _parse(tmp_path, html)
    assert doc.metadata["extraction"] == "readability"
    assert "让业务人员用自然语言直接查询数据库" in doc.full_text
    assert "相关链接" not in doc.full_text
