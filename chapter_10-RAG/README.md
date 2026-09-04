# chapter_10-RAG：洛克王国:世界 RAG 向量数据库

以《洛克王国:世界》游戏资料为主题，用 LangChain 1.2.x 搭建的一条**完整可运行的 RAG 链路**：
采集 → 清洗切分 → 向量化入库(远程 Milvus) → 检索 → DeepSeek 问答(带来源引用)。
全部脚本遵循本书其他章节风格：一文件一主题、中文注释、`.env` 集中配置、LangSmith 观测(`LANGSMITH_TRACING=true` 自动生效)。

## 目录与文件职责

```
chapter_10-RAG/
├── config.py           公共配置(唯一读 .env 处): Embedding/对话模型/Milvus/数据目录工厂
├── 01_fetch_roco.py    采集层: 抓 rocokingdomworld.org 词条 HTML 落盘(幂等, 断点续抓)
├── 02_clean_split.py   清洗切分层: HTML→干净文本→中文 Recursive 切分 → clean/ + chunks/
├── 03_index_milvus.py  向量化入库层: SiliconFlow(bge-m3,1024 维) → 远程 Milvus 两个 collection
├── 04_retrieve_qa.py   检索问答层: 双库检索归并 → prompt → DeepSeek; chain/--agent/--demo 三形态
├── rag_engine.py       ★ 统一接口(推荐入口): status/build/ask/chat, 自动保证数据就绪
├── server.py           ★ REST 服务: GET /status, POST /build /ask /search(FastAPI)
├── data/               (已 gitignore) 中间产物: raw/clean/chunks/logs
│     raw/pages/*.json     每页 {title,kind,url,sha1,html}
│     clean/<kind>.jsonl   清洗后整页 Document
│     chunks/<kind>.jsonl  切分后子文档(metadata 含 title/doc_type/source_url/hash)
│     logs/index_state.json  03 幂等记录(collection → 已入库条数)
└── textSplitter.py     空占位(最初的学习文件)
```

## 数据规模(当前快照)

| kind | 内容 | 词条 | chunk |
|---|---|---|---|
| beastiary | 精灵图鉴(种族值/技能/进化/属性/获取) | 612 | 3374 |
| world | 攻略/机制长文(培养/捕捉/进化材料) | 12 | 23 |

来源 [rocokingdomworld.org](https://rocokingdomworld.org/zh/) 粉丝资料站(SSG, HTML 直出)。
Milvus collection: `roco_beastiary` / `roco_world` @ 47.96.113.144:19530(server v2.4.0)。

## 运行顺序

```bash
# 环境: conda 的 langchain1.2(Python 3.13)。.env 需 SILICONFLOW_API_KEY(硅基流动, bge-m3 免费)
cd d:/dev/agent/agent_dev/lang_chain/code
& D:/dev/python/tool/Pytorch/envs/langchain1.2/python.exe chapter_10-RAG/01_fetch_roco.py --resume   # 补抓/更新
& D:/dev/python/tool/Pytorch/envs/langchain1.2/python.exe chapter_10-RAG/02_clean_split.py           # 全量重洗重切
& D:/dev/python/tool/Pytorch/envs/langchain1.2/python.exe chapter_10-RAG/03_index_milvus.py --reset # 全量重灌(幂等, --reset 重建)
& D:/dev/python/tool/Pytorch/envs/langchain1.2/python.exe chapter_10-RAG/04_retrieve_qa.py --demo   # 端到端 demo
& D:/dev/python/tool/Pytorch/envs/langchain1.2/python.exe chapter_10-RAG/04_retrieve_qa.py          # 交互问答
```

## 统一接口(生产使用, 无需理解底层 RAG)

普通使用方**不需要**逐层跑 01~04, 也不需要知道"采集/向量化/检索"这些概念——统一入口会自动检查数据是否就绪, 未就绪先装载再回答:

```bash
# 1) CLI
python rag_engine.py status                 # 就绪情况(数据/向量库)
python rag_engine.py build [--reset]        # 一键装载(未就绪首次会联网抓取, 之后幂等增量)
python rag_engine.py ask "阿布种族值多少?"   # 一句话问答(带引用来源)
python rag_engine.py chat                   # 多轮对话

# 2) Python API
from rag_engine import RAGEngine
eng = RAGEngine(auto_build=True)   # ask 前若数据未就绪会自动装载
eng.status()                       # -> {"ready", "kinds", "chunks", "indexed"}
eng.ask("阿布是什么属性精灵?")      # -> {"answer": str, "citations": [{category,title,source_url,similarity,content}]}
eng.build(reset=False)             # 手动一键装载

# 3) REST 服务(FastAPI, 自带 /docs 交互文档)
python server.py --port 8000
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"新玩家怎么捕捉精灵?"}'
```
> 提示: REST 客户端请以 UTF-8 发送 JSON(Windows 下用 curl 命令行直接敲中文易被 shell 编码搞坏, 用文件或程序发送即可)。

**设计说明**：`rag_engine.py` 内部经 importlib 按路径装载数字前缀脚本(01~04)以复用其函数——Python 语法不允许 `import 01_xxx`, 故保留"每个编号脚本既是可独立运行的单层, 又是统一入口的组成部件"的双形态, 底层函数已抽成可复用 API(`03.run_index` / `04.search / 04.answer`)。生产部署面向使用者的是 `rag_engine`/`server`, 底层概念只对二次开发者开放。

## 已踩坑记录(后续开发务必知道)

1. **数据源为何不是 BWIKI**：原定主源 B站 BWIKI(`wiki.biligame.com/rocom`, MediaWiki) 前置腾讯 EdgeOne
   WAF——requests 连续请求会被返回 `567` JS 挑战页并临时封 IP(冷却 ~75s 才恢复单请求)。
   故改用无 WAF 的 rocokingdomworld.org。若想补 BWIKI 的剧情/活动词条，需走真实浏览器拿 cookie 或极慢速。
2. **langchain-milvus 版本**：`0.3.3 + pymilvus2.6.17` 组合有真实缺陷——0.3.3 内部用 ORM
   `Collection(using=alias)`,而 pymilvus2.6 的 MilvusClient `_using` 是随机伪别名且不注册 ORM,
   写入必报 `ConnectionNotExistException`。**升级到 `0.4.0`(纯 MilvusClient) + pymilvus3 + langchain-core 1.6.1** 解决。
   代价: `langchain-core` 由 1.2.18 → 1.6.1(已同步更新 requirements.txt; 现有各章节代码不受影响)。
3. **metadata 读不回来**：`Milvus(..., enable_dynamic_field=True)` 会把 metadata 平铺成顶层动态字段,
   langchain 检索时只回 `{pk}`。**保持默认 False**,让 langchain 为各 metadata 键建标准列,检索即可完整读回。
4. **score 语义**：langchain-milvus 对 COSINE 返回的是**相似度,越大越相关**(阿布页 0.67 > 无关页 0.55)。
   双库归并要**降序**取 top(04 早期按升序排导致召回了最不相关片段)。
5. **server row_count 假 0**：Milvus v2.4 的 `describe/get_collection_stats` 对刚重建的 collection
   可能暂报 `row_count=0`(数据实际已插入,可检索验证)。03 幂等改用本地 `logs/index_state.json` 判据。
6. **config.data_dir() 会对每段 mkdir**: 只传目录级参数;文件路径请 `data_dir("raw") / "manifest.json"` 写法。

## .env 新增键

```
SILICONFLOW_API_KEY=sk-xxx      # https://cloud.siliconflow.cn (BAAI/bge-m3, 1024维, 单批<=64, 免费)
MILVUS_URI=http://47.96.113.144:19530
MILVUS_TOKEN=                    # 留空=无认证
MILVUS_COLLECTION_BEASTIARY=roco_beastiary
MILVUS_COLLECTION_WORLD=roco_world
```

## 后续可扩展方向

- **世界观剧情文本**：BWIKI(需浏览器/慢速)或增加 items(819 道具, KIND_MAP 放开即可)。
- **检索质量**：引入 rerank(bge-reranker)、Milvus 混合检索(BM25 sparse + dense)、图鉴结构化字段单独入库做精确匹配。
- **Agent 化**：`04 --agent` 已示范 create_agent + 检索工具;可接入长期记忆(chapter-09 PostgresStore)、多轮工具规划。
