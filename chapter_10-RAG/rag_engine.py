"""
【统一接口层 rag_engine.py】对上层只暴露 装载/问答 两个动作, 屏蔽内部 RAG 分步概念

生产视角: 使用方不需要懂"采集/清洗切分/向量化/检索"这些步骤, 也不用手动先把数据装上——
统一入口会自动检查数据是否就绪, 未就绪则先装载, 再回答问题。

对外能力(CLI):
    python chapter_10-RAG/rag_engine.py status                 # 就绪情况
    python chapter_10-RAG/rag_engine.py build [--reset]        # 一键装载(增量/全量)
    python chapter_10-RAG/rag_engine.py ask "问题" [--top 4]   # 单次问答(带引用)
    python chapter_10-RAG/rag_engine.py chat                   # 多轮对话

对外能力(Python):
    from rag_engine import RAGEngine
    eng = RAGEngine(auto_build=True)       # auto_build=True: 问答前自动保证数据就绪
    eng.status()                            # -> dict
    eng.build(reset=False)                  # -> dict(各库写入汇总)
    eng.ask("阿布的种族值是多少?")           # -> {"answer": str, "citations": [...]}
    eng.answer(q, history=...) / eng.search(q, top_k=...)

实现说明:
    - 底层步骤脚本是 01_fetch_roco / 02_clean_split / 03_index_milvus / 04_retrieve_qa。
      Python 不允许 import 以数字开头的模块, 故这里用 importlib 按文件路径装载并缓存;
      每个脚本既是可独立运行的 CLI, 又是本引擎的组成部件(职责单一)。
    - 就绪判据: 本地已有 chunk(02 产物) 且 index_state.json 记录各 collection 已入库(03 产物)。
    - 多轮历史由调用方维护 langchain messages 列表传入; 引擎内部自动裁剪最近 N 轮。
"""
import argparse
import importlib.util
import json
from pathlib import Path

import requests

from config import collection_name, data_dir

KINDS = ("beastiary", "world")
_HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# 装载数字前缀脚本模块
# ----------------------------------------------------------------------
def _load_script(name: str):
    """按路径装载 chapter_10-RAG/<name>.py, 返回模块对象(缓存)。name 不带 .py。"""
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_rag_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------
class RAGEngine:
    """《洛克王国:世界》RAG 助手统一接口。"""

    def __init__(self, kinds=KINDS, auto_build: bool = True):
        self.kinds = set(kinds)
        self.auto_build = auto_build          # ask 前若未就绪则自动装载
        self._mods: dict[str, object] = {}

    # ---- 底层脚本模块(懒装载) ----
    def _mod(self, name: str):
        if name not in self._mods:
            self._mods[name] = _load_script(name)
        return self._mods[name]

    # ============================================================
    # 状态
    # ============================================================
    def status(self) -> dict:
        """返回数据与向量库的就绪情况(便于上层展示 /health)。"""
        chunks = {}
        for kind in self.kinds:
            p = data_dir("chunks") / f"{kind}.jsonl"
            chunks[kind] = {
                "exists": p.exists(),
                "rows": sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0,
                "collection": collection_name(kind),
            }
        state_path = data_dir("logs") / "index_state.json"
        indexed = {}
        if state_path.exists():
            indexed = json.loads(state_path.read_text(encoding="utf-8"))
        ready = all(
            chunks[k]["rows"] > 0 and collection_name(k) in indexed
            for k in self.kinds
        )
        return {
            "ready": ready,
            "kinds": sorted(self.kinds),
            "chunks": chunks,
            "indexed": {collection_name(k): indexed.get(collection_name(k)) for k in self.kinds},
        }

    # ============================================================
    # 一键装载(用户无需理解内部步骤)
    # ============================================================
    def build(self, fetch: bool = True, reset: bool = False, verbose: bool = True) -> dict:
        """按 采集 -> 清洗切分 -> 向量化入库 依次执行; 各步均幂等, 可增量重跑。

        fetch=True 且本地没有原始页时, 会联网抓取(首次较慢)。
        reset=True 会重建向量 collection 全量重灌。
        """
        def log(*a):
            if verbose:
                print(*a)

        steps = {}
        # 1) 采集(原始 HTML) —— 若已有原始页则以增量模式补齐
        raw_pages = data_dir("raw", "pages")
        have_raw = any(raw_pages.glob("*.json"))
        if fetch and (not have_raw or reset):
            log("==> [1/3] 采集词条页(增量, 已有未变化则跳过) ...")
            mod01 = self._mod("01_fetch_roco")
            mod01.crawl(requests.Session(), limit=None, resume=True, kind_allow=self.kinds)
            steps["fetch"] = "done"
        else:
            log("==> [1/3] 采集: 已有原始数据, 跳过(需重抓请 fetch=True)")

        # 2) 清洗切分(纯本地, 每次全量重建 chunk, 快)
        log("==> [2/3] 清洗 + 切分 chunk ...")
        mod02 = self._mod("02_clean_split")
        mod02.process(data_dir("raw", "pages"), self.kinds, None)
        steps["clean"] = "done"

        # 3) 向量化入库(幂等; reset=True 重建 collection)
        log("==> [3/3] 向量化入库远程 Milvus ...")
        mod03 = self._mod("03_index_milvus")
        steps["index"] = mod03.run_index(self.kinds, reset=reset, verbose=verbose)
        return {"ready": self.status()["ready"], "steps": steps}

    # ============================================================
    # 问答(自动就绪)
    # ============================================================
    def _ensure(self, verbose: bool = False):
        st = self.status()
        if not st["ready"]:
            if verbose:
                print("数据未就绪, 自动装载中(首次可能需数分钟) ...")
            self.build(fetch=True, reset=False, verbose=verbose)

    def ask(self, question: str, top_k: int = 4, history: list | None = None) -> dict:
        """一句话问答。返回 {"answer", "citations"}。若 auto_build 开启且未就绪则先自动装载。"""
        if self.auto_build:
            self._ensure(verbose=True)
        return self._mod("04_retrieve_qa").answer(question, history=history, top_k=top_k)

    # 语义别名: answer/search 与 ask 等价, 便于库调用方按名取用
    def answer(self, question: str, top_k: int = 4, history: list | None = None) -> dict:
        return self.ask(question, top_k=top_k, history=history)

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        """纯检索(不调 LLM), 返回引用列表。"""
        if self.auto_build:
            self._ensure(verbose=True)
        return self._mod("04_retrieve_qa").search(question, top_k=top_k)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_citations(citations: list[dict]):
    print("=" * 64)
    print("检索来源:")
    for i, c in enumerate(citations, 1):
        print(f"  [{i}] {c['category']} | {c['title']}  (相似度 {c['similarity']})")
        print(f"      {c['source_url']}")
    print("=" * 64)


def _cmd_build(args):
    eng = RAGEngine(auto_build=False)
    result = eng.build(fetch=not args.no_fetch, reset=args.reset, verbose=True)
    print("\n就绪状态:", result["ready"])


def _cmd_status(args):
    print(json.dumps(RAGEngine(auto_build=False).status(), ensure_ascii=False, indent=2))


def _cmd_ask(args):
    eng = RAGEngine(auto_build=True)
    result = eng.ask(args.question, top_k=args.top)
    print("回答:", result["answer"])
    _print_citations(result["citations"])


def _cmd_chat(args):
    eng = RAGEngine(auto_build=True)
    history: list = []  # engine.answer 内部负责追加并只取最近 MAX_HISTORY_PAIRS 轮
    print("《洛克王国:世界》百科助手(统一接口). 输入 exit 退出, 输入 ? 看可用命令")
    while True:
        try:
            q = input("\n你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):  # 非交互/中断时优雅退出
            print("\n再见!")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q == "?":
            print("  exit/quit  退出\n  其余输入作为问题直接提问(自带多轮上下文)")
            continue
        result = eng.ask(q, history=history)
        print("回答:", result["answer"])
        _print_citations(result["citations"])


def main():
    p = argparse.ArgumentParser(
        prog="rag_engine",
        description="《洛克王国:世界》RAG 统一入口(status/build/ask/chat). 不带子命令默认进入 chat",
    )
    sub = p.add_subparsers(dest="cmd")  # 非 required: 无子命令时默认 chat

    sp = sub.add_parser("status", help="查看数据/向量库就绪情况")
    sp.set_defaults(fn=_cmd_status)

    sp = sub.add_parser("build", help="一键装载(采集->清洗切分->入库)")
    sp.add_argument("--reset", action="store_true", help="重建向量 collection 全量重灌")
    sp.add_argument("--no-fetch", action="store_true", help="不联网抓取, 只用已有原始数据")
    sp.set_defaults(fn=_cmd_build)

    sp = sub.add_parser("ask", help="单次问答")
    sp.add_argument("question", help="问题文本")
    sp.add_argument("--top", type=int, default=4, help="引用条数")
    sp.set_defaults(fn=_cmd_ask)

    sp = sub.add_parser("chat", help="多轮对话")
    sp.set_defaults(fn=_cmd_chat)

    args = p.parse_args()
    if hasattr(args, "fn"):
        args.fn(args)
    else:  # 无子命令: 直接进入交互对话(最常用)
        print("(用法: status / build / ask \"问题\" / chat; 无参数默认进入对话)")
        _cmd_chat(args)


if __name__ == "__main__":
    main()
