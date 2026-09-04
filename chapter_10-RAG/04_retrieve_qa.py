"""
【04 检索问答层 retrieve_qa.py】CLI 多轮问答: 检索两库 -> 组装上下文 -> DeepSeek 回答 + 来源引用

数据来源: 远程 Milvus 两个 collection(roco_beastiary 精灵图鉴 / roco_world 攻略文本),
          由 03 写入。Embedding 与入库时必须同一套(SiliconFlow BAAI/bge-m3)。

形态:
    默认 chain: 每次提问同时检索两个库 -> 按相似度归并取 top-k -> 拼进 prompt -> DeepSeek 回答,
                引用块由检索结果在控制台确定性打印(不依赖模型, 防幻觉来源)。
    --agent  : 用 create_agent + 检索工具(复用 chapter_07 惯例), 让 agent 决定何时检索。
    --demo   : 不进入交互, 跑一组预置问题后退出(便于端到端自动验证)。

用法:
    python chapter_10-RAG/04_retrieve_qa.py            # 交互式
    python chapter_10-RAG/04_retrieve_qa.py --demo     # 预置问题自动化
    python chapter_10-RAG/04_retrieve_qa.py --agent    # agent 形态
"""
import argparse

from config import collection_name, get_chat_model, get_embeddings

MAX_HISTORY_PAIRS = 2   # 保留最近 2 轮对话, 控制上下文长度
TOP_K_PER_KIND = 3      # 每个库各取前 3
TOP_K_TOTAL = 4         # 归并后总取 4
KINDS = ["beastiary", "world"]

SYSTEM_PROMPT = (
    "你是《洛克王国:世界》百科助手。请只依据提供的【知识库片段】回答问题, 不要编造知识库外内容;\n"
    "回答中标注信息出处(精灵名/攻略标题); 知识库里没有答案时直接说明'知识库中未找到'。\n"
    "【知识库片段】\n{context}"
)

# 预置示例问题(基于真实抓到的页面: abu 精灵 / 新手指南攻略)
DEMO_QUERIES = [
    "阿布的种族值是多少?它的进化形态是什么?",
    "新玩家应该怎么捕捉精灵?有什么技巧?",
    "前期推荐培养哪些火系精灵?",
]


# ----------------------------------------------------------------------
# 检索
# ----------------------------------------------------------------------
def build_stores(emb):
    """为每个 kind 连接 Milvus collection, 返回 {kind: Milvus}。"""
    from langchain_milvus import Milvus

    stores = {}
    for kind in KINDS:
        stores[kind] = Milvus(
            embedding_function=emb,
            collection_name=collection_name(kind),
            connection_args={"uri": "http://47.96.113.144:19530"},
        )
    return stores


def retrieve_all(stores, query: str):
    """同时检索两库, 归并后按相似度降序返回 [(Document, score), ...]。

    实测确认: langchain-milvus 对 COSINE metric 返回的 score 是余弦相似度, 越大越相关
    (阿布问句: 阿布页 0.67 > 超能阿布 0.64 > 不相关页 <0.56)。故取降序 top-k。
    """
    hits = []
    for kind, store in stores.items():
        for doc, score in store.similarity_search_with_score(query, k=TOP_K_PER_KIND):
            m = doc.metadata
            hits.append((score, doc, kind, m))
    hits.sort(key=lambda x: x[0], reverse=True)  # 相似度高在前
    return hits[:TOP_K_TOTAL]


def format_context(hits) -> str:
    """把命中片段编号拼成可放入 prompt 的上下文文本。"""
    lines = []
    for i, (score, doc, kind, m) in enumerate(hits, 1):
        title = m.get("title") or "(无标题)"
        category = m.get("category") or kind
        text = doc.page_content
        lines.append(f"[{i}] {category} | {title}\n{text}\n")
    return "\n".join(lines)


def print_citations(hits):
    """控制台确定性打印引用块(来源与相似度), 不依赖模型输出。"""
    print("\n" + "=" * 64)
    print("检索来源(确定性命中, 非模型生成):")
    for i, (score, doc, kind, m) in enumerate(hits, 1):
        print(f"  [{i}] {m.get('category')} | {m.get('title')}  "
              f"(相似度 {score:.3f})")
        print(f"      {m.get('source_url')}")
    print("=" * 64)


# ----------------------------------------------------------------------
# 问答
# ----------------------------------------------------------------------
def ask(model, system, history, question: str) -> str:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=system)]
    messages += history[-MAX_HISTORY_PAIRS * 2:]  # 只保留最近几轮
    messages.append(HumanMessage(content=question))
    reply = model.invoke(messages)
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=reply.content))
    return reply.content


def run_demo(model, stores):
    from langchain_core.messages import AIMessage

    history: list = []
    for q in DEMO_QUERIES:
        print("\n" + "#" * 72)
        print("问题:", q)
        hits = retrieve_all(stores, q)
        print_citations(hits)
        context = format_context(hits)
        answer = ask(model, SYSTEM_PROMPT.format(context=context), history, q)
        print("回答:", answer)
        # 演示后清空历史, 各题独立
        history = []


def run_cli(model, stores):
    from langchain_core.messages import HumanMessage

    history: list = []
    print("《洛克王国:世界》百科助手(chain 检索问答), 输入 exit 退出")
    while True:
        q = input("\n你的问题: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        hits = retrieve_all(stores, q)
        print_citations(hits)
        context = format_context(hits)
        answer = ask(model, SYSTEM_PROMPT.format(context=context), history, q)
        print("回答:", answer)


# ----------------------------------------------------------------------
# agent 形态(可选): create_agent + 检索工具
# ----------------------------------------------------------------------
def run_agent():
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    emb = get_embeddings()
    model = get_chat_model()
    stores = build_stores(emb)

    @tool(parse_docstring=True)
    def search_kb(query: str) -> str:
        """在《洛克王国:世界》资料库中检索与问题最相关的片段。
        Args:
            query: 用户问题的关键词
        Returns:
            带 [序号] 来源标题 的文本片段, 回答时须引用其来源
        """
        hits = retrieve_all(stores, query)
        return format_context(hits)

    agent = create_agent(
        model=model,
        tools=[search_kb],
        system_prompt=(
            "你是《洛克王国:世界》百科助手。需要游戏资料时调用 search_kb 检索, "
            "只依据检索结果回答, 末尾给出【参考来源】清单。"
        ),
    )
    print("《洛克王国:世界》百科助手(agent 形态), 输入 exit 退出")
    while True:
        q = input("\n你的问题: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        reply = agent.invoke(q)
        print("回答:", reply)


# ======================================================================
# 可复用 API(供 rag_engine / server 等上层调用, 不依赖 print)
# ======================================================================
_runtime = None  # 模块级懒缓存: {"emb", "model", "stores"}


def get_runtime() -> dict:
    """一次性建立并缓存 Embedding/对话模型/各 collection store(进程内复用)。"""
    global _runtime
    if _runtime is None:
        emb = get_embeddings()
        _runtime = {"emb": emb, "model": get_chat_model(), "stores": build_stores(emb)}
    return _runtime


def _hit_to_citation(score, doc, kind, m) -> dict:
    return {
        "category": m.get("category") or kind,
        "title": m.get("title"),
        "source_url": m.get("source_url"),
        "similarity": round(float(score), 4),
        "content": doc.page_content[:800],  # 引用摘录
    }


def search(question: str, top_k: int = TOP_K_TOTAL) -> list[dict]:
    """纯检索(不调 LLM), 返回引用列表(按相似度降序)。"""
    rt = get_runtime()
    hits = retrieve_all(rt["stores"], question)[:top_k]
    return [_hit_to_citation(s, d, k, m) for s, d, k, m in hits]


def answer(question: str, history: list | None = None,
           top_k: int = TOP_K_TOTAL) -> dict:
    """检索 + DeepSeek 回答, 返回 {"answer": str, "citations": [...], "context_used": [...]}。

    history 为可选的历史消息列表(langchain BaseMessage), 内部只保留最近 MAX_HISTORY_PAIRS 轮。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    rt = get_runtime()
    hits = retrieve_all(rt["stores"], question)[:top_k]
    citations = [_hit_to_citation(s, d, k, m) for s, d, k, m in hits]

    context = format_context(hits)
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    if history:
        messages += history[-MAX_HISTORY_PAIRS * 2:]
    messages.append(HumanMessage(content=question))
    reply = rt["model"].invoke(messages)

    if history is not None:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply.content))

    return {"answer": reply.content, "citations": citations}


def main():
    p = argparse.ArgumentParser(description="RAG 检索问答")
    p.add_argument("--demo", action="store_true", help="跑预置问题后退出")
    p.add_argument("--agent", action="store_true", help="用 agent 形态(create_agent+检索工具)")
    args = p.parse_args()

    if args.agent:
        run_agent()
        return

    emb = get_embeddings()
    model = get_chat_model()
    stores = build_stores(emb)
    if args.demo:
        run_demo(model, stores)
    else:
        run_cli(model, stores)


if __name__ == "__main__":
    main()
