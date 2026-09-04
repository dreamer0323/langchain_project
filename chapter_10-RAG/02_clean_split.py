"""
【02 清洗切分层 clean_split.py】把 01 抓回的原始 HTML 清洗成干净文本并按类型切分成 chunk

输入: data/raw/pages/*.json(01 产物, 每页含 html / kind / url 等)
输出:
    data/clean/<kind>.jsonl   清洗后、切分前的整页 Document, 每行一个 {page_content, metadata}
    data/chunks/<kind>.jsonl  切分后的子文档, metadata 里追加 hash(去重/入库幂等判据)

设计要点:
    1) 数据源是 Next.js SSG 站, 词条正文在 <main>/<article> 里直接渲染, 无 JS 依赖。
       精灵图鉴页数值用 flex 卡片而非 <table>, 按行 get_text 即可保留 "字段/值" 顺序。
    2) 图鉴(beastiary)与攻略(world)用同一清洗管线: 取正文容器 -> 行级清理 -> 拼文本。
       文档开头注入类型前缀, 便于检索时区分(如"精灵图鉴"与"攻略")。
    3) 切分器对纯中文语料显式给中文标点分隔符顺序(默认英文分隔符对纯中文会退化成整块)。

用法:
    python chapter_10-RAG/02_clean_split.py [--limit N] [--kind beastiary|world]
    --limit 只处理前 N 页(试跑); 缺省处理全部(本地纯计算, 全量重建, 覆盖旧文件)。
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import data_dir

KIND_LABEL = {"beastiary": "精灵图鉴", "world": "攻略文本"}

# 行级噪音(图鉴页的翻页导航横条等)
NOISE_LINE = re.compile(r"^(上一只|下一只)[:：]")

# 中文语料的分隔符顺序: 段落 -> 换行 -> 句读。默认英文分隔符对纯中文会整块切不开。
SPLITTER_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "……", "，", "、", " ", ""]


def sha1_short(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------
def html_to_text(html: str, kind: str) -> str:
    """从原始 HTML 里抽取正文纯文本(逐元素换行, 保留卡片字段顺序)。"""
    soup = BeautifulSoup(html, "html.parser")
    # 去掉不影响正文的噪音节点
    for t in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "iframe"]):
        t.decompose()
    # 正文容器优先级: <main> > <article> > <body>(都拿不到就空)
    node = soup.find("main") or soup.find("article") or soup.body or soup
    text = node.get_text("\n", strip=True)
    return text


def clean_lines(text: str, kind: str) -> list[str]:
    """行级清理: 去翻页导航噪音、折叠空行与行内空白。"""
    prefix = f"【{KIND_LABEL.get(kind, kind)}】"
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            continue
        if NOISE_LINE.match(line):
            continue
        out.append(line)
    if out and not out[0].startswith("【"):
        out.insert(0, prefix)
    return out


# ----------------------------------------------------------------------
# 切分器
# ----------------------------------------------------------------------
def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=SPLITTER_SEPARATORS,
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def process(pages_dir: Path, kind_allow: set[str], limit: int | None):
    from collections import defaultdict

    splitter = make_splitter()
    cleaned: dict[str, list] = defaultdict(list)   # kind -> 整页清洗文档
    chunked: dict[str, list] = defaultdict(list)   # kind -> 切分后的子文档
    total_pages = total_chunks = 0

    page_files = sorted(pages_dir.glob("*.json"))
    if limit:
        page_files = page_files[:limit]

    for i, f in enumerate(page_files, 1):
        rec = json.loads(f.read_text(encoding="utf-8"))
        kind = rec.get("kind")
        if kind not in kind_allow:
            continue
        text = "\n".join(clean_lines(html_to_text(rec["html"], kind), kind))
        if not text:
            continue
        meta = {
            "title": rec["title"],
            "doc_type": kind,
            "category": KIND_LABEL.get(kind, kind),
            "source_url": rec["url"],
            "fetched_at": rec.get("fetched_at", ""),
        }
        clean_doc = Document(page_content=text, metadata=meta)
        chunks = splitter.split_documents([clean_doc])
        cleaned[kind].append({"page_content": clean_doc.page_content, "metadata": meta})
        for c in chunks:
            m = dict(c.metadata)
            m["hash"] = sha1_short(c.page_content)
            chunked[kind].append({"page_content": c.page_content, "metadata": m})
        total_pages += 1
        total_chunks += len(chunks)
        if i % 100 == 0:
            print(f"    ...已处理 {i} 页, 累计 chunk {total_chunks}")

    # 写盘: clean 与 chunks 各按 kind 一个文件(全量重建, 覆盖旧文件)
    for kind in sorted(cleaned):
        clean_path = data_dir("clean") / f"{kind}.jsonl"
        chunk_path = data_dir("chunks") / f"{kind}.jsonl"
        with clean_path.open("w", encoding="utf-8") as fh:
            for row in cleaned[kind]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with chunk_path.open("w", encoding="utf-8") as fh:
            for row in chunked[kind]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"写入 {clean_path} ({len(cleaned[kind])} 条), "
              f"{chunk_path} ({len(chunked[kind])} 条)")

    print(f"\n完成: 清洗 {total_pages} 页 -> chunk {total_chunks} 个")


def main():
    p = argparse.ArgumentParser(description="HTML 清洗 + 中文文本切分")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 页(试跑)")
    p.add_argument("--kind", default="beastiary,world", help="逗号分隔: beastiary/world")
    args = p.parse_args()
    kind_allow = set(x.strip() for x in args.kind.split(",")) & set(KIND_LABEL)
    process(data_dir("raw", "pages"), kind_allow, args.limit)


if __name__ == "__main__":
    main()
