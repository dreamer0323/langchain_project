"""
【03 向量化入库层 index_milvus.py】把 02 切分好的 chunk 用 SiliconFlow embedding 写入远程 Milvus

输入: data/chunks/<kind>.jsonl(02 产物)
输出: 远程 Milvus collection(roco_beastiary / roco_world) + data/logs/index_state.json 幂等记录
前置: .env 已配 SILICONFLOW_API_KEY; 远程 Milvus 47.96.113.144:19530 可达(test.py 可连)

版本核实(装 langchain-milvus==0.3.3 后已确认):
    from langchain_milvus import Milvus   # 0.3.x 主类, 旧名 MilvusVectorStore 已不用
    构造参数是 connection_args={"uri": ...}, 内部走 pymilvus 的 MilvusClient;
    dim 无需手动传(首次写入按 embedding 输出长度 1024 自动建 schema)。

用法:
    python chapter_10-RAG/03_index_milvus.py [--kind beastiary|world] [--reset]
    --reset: 删掉并重建该 collection(全量重灌用); 不加则 collection 已存在且已入库则跳过(幂等)
"""
import argparse
import json

from langchain_core.documents import Document

from config import collection_name, data_dir, get_embeddings

MILVUS_URI = "http://47.96.113.144:19530"
# v2.4 server: HNSW + COSINE 稳妥(也支持 AUTOINDEX, 但显式 HNSW 更可控)
INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}


def load_chunks(kind: str) -> list[Document]:
    """读 data/chunks/<kind>.jsonl 还原成 Document 列表。"""
    path = data_dir("chunks") / f"{kind}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}, 请先运行 02_clean_split.py")
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
    return docs


def get_milvus_store(emb, name: str, reset: bool):
    """连/建 collection。collection 不存在时由 add_documents 自动建 schema(dim=1024)。"""
    from langchain_milvus import Milvus

    # 注意: 不要开 enable_dynamic_field。实测开 True 会把 metadata 平铺成动态字段,
    # langchain 0.4.0 检索时读不到(只能拿到 pk); 保持默认(False)会让 langchain
    # 为每个 metadata 键自动建标准列, 检索能完整读回 title/category/source_url。
    return Milvus(
        embedding_function=emb,
        collection_name=name,
        connection_args={"uri": MILVUS_URI},
        index_params=INDEX_PARAMS,
        auto_id=True,
        drop_old=reset,
    )


def count_entities(name: str) -> int:
    """用独立 MilvusClient 查 collection 行数。

    注意: Milvus 2.4 的 describe/get_collection_stats 对刚重建的 collection 可能暂报 row_count=0
    (数据其实已插入)。故本函数只作展示; 幂等判断以本地 data/logs/index_state.json 为准。
    """
    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(name):
        return 0
    try:
        return client.get_collection_stats(name).get("row_count") or 0
    except Exception:
        return client.describe_collection(name).get("row_count", 0) or 0


def run_index(kinds: set[str], reset: bool = False, verbose: bool = True):
    """把 chunks 向量化写入 Milvus(幂等)。供本文件 CLI 与上层 rag_engine 编排共用。

    幂等判据: 本地 data/logs/index_state.json 记录了 collection 已入库则跳过
    (不依赖 v2.4 server 可能滞后的 row_count)。加 reset=True 强制重建重灌。

    Args:
        kinds: 文档类型集合, 如 {"beastiary", "world"}
        reset: True=drop_old 重建 collection 后全量写入
        verbose: 是否打印进度(库调用时可关)
    Returns:
        汇总 dict: {kind: {"collection", "docs", "server_count"}}
    """
    def log(*a):
        if verbose:
            print(*a)

    emb = get_embeddings()  # 缺 SILICONFLOW_API_KEY 会在此抛清晰错误
    state_path = data_dir("logs") / "index_state.json"
    done = {}
    if state_path.exists():
        done = json.loads(state_path.read_text(encoding="utf-8"))

    summary = {}
    for kind in kinds:
        name = collection_name(kind)
        docs = load_chunks(kind)
        if not docs:
            log(f"[{kind}] 无 chunk, 跳过")
            continue

        if name in done and not reset:
            log(f"[{kind}] index_state 显示 collection '{name}' 已入库 "
                f"({done[name].get('count')} 条), 跳过。需重灌请加 reset=True")
            summary[kind] = {"collection": name, "docs": len(docs), "server_count": done[name].get("count"), "skipped": True}
            continue
        if name not in done and count_entities(name) > 0 and not reset:
            log(f"[{kind}] collection '{name}' 服务端已有数据但无本地记录, "
                f"为保持一致请用 reset=True 全量重建")
            summary[kind] = {"collection": name, "docs": len(docs), "server_count": count_entities(name), "skipped": True}
            continue

        store = get_milvus_store(emb, name, reset=reset)
        log(f"[{kind}] 开始写入 {len(docs)} 个 chunk -> collection '{name}' ...")
        store.add_documents(docs)
        final = count_entities(name)
        done[name] = {"count": final, "chunks": len(docs),
                      "updated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")}
        state_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[{kind}] 完成: collection '{name}' server row_count={final} (入库 {len(docs)} chunks)")
        summary[kind] = {"collection": name, "docs": len(docs), "server_count": final, "skipped": False}

    return summary


def main():
    p = argparse.ArgumentParser(description="chunk 向量化写入远程 Milvus")
    p.add_argument("--kind", default="beastiary,world", help="逗号分隔: beastiary/world")
    p.add_argument("--reset", action="store_true", help="重建 collection 后全量写入")
    args = p.parse_args()
    kinds = set(x.strip() for x in args.kind.split(","))
    run_index(kinds, reset=args.reset, verbose=True)


if __name__ == "__main__":
    main()
