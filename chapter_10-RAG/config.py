"""
【RAG 公共配置模块 config.py】

本文件是整个 RAG 流水线(01 采集 -> 02 清洗切分 -> 03 向量化入库 -> 04 检索问答)共享的唯一入口:
- 只在这里读 .env(项目根目录下的 .env,脚本从 chapter_10-RAG/ 内运行也能正确加载);
- 集中定义 Embedding / 对话模型 / Milvus 等外部服务的"唯一来源",避免各脚本各自拼参数导致
  Embedding 模型或向量维度不一致(入库与查询必须用同一套配置)。

用法:
    from config import get_embeddings, get_chat_model, collection_name, data_dir

为什么取名叫 config.py 而不是 00_config.py?
    Python 的 import 语句不允许模块名以数字开头("00_config" 是非法标识符),为了能被 01~05 正常
    import,统一叫 config.py;00 的"最先运行/被依赖"语义由 01~05 顶部的 from config import 表达。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 环境变量加载
# ----------------------------------------------------------------------
# config.py 位于 chapter_10-RAG/ 下,项目根(.env 所在处)就是它的上一级目录。
# 用绝对路径定位 .env,保证无论从哪个工作目录运行脚本都能读到同一份配置。
# override=True: 强制用 .env 里的值覆盖系统里可能已存在的同名环境变量。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ----------------------------------------------------------------------
# 数据目录(所有中间产物统一落盘在 chapter_10-RAG/data/ 下)
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"


def data_dir(*parts) -> Path:
    """返回 data 目录下的子目录路径并确保其存在。

    注意: 本函数会对每个传入段执行 mkdir, 只应传入"目录"级路径, 例如 data_dir("raw", "pages")。
    若是文件路径请先拿目录再拼文件名: data_dir("raw") / "manifest.json", 否则会把文件路径误建成目录。
    """
    p = DATA_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------
# 对话模型(DeepSeek,与全书其他章节同一套配置)
# ----------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = "deepseek:deepseek-v4-flash"


def get_chat_model():
    """初始化 DeepSeek 对话模型。"deepseek:" 前缀让 init_chat_model 自动用 langchain-deepseek。"""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


# ----------------------------------------------------------------------
# Embedding(硅基流动 SiliconFlow,OpenAI 兼容接口; BAAI/bge-m3 输出 1024 维)
# ----------------------------------------------------------------------
# DeepSeek 不提供 embedding 接口,故向量化走 SiliconFlow(免费、大陆可直连)。
# 注意:
#   1) SiliconFlow 单次 /embeddings 请求最多 64 条,超出报 HTTP 413(错误码 20042),
#      因此 OpenAIEmbeddings 的 chunk_size 必须 <=64,这里保守取 32;
#   2) 不要传 dimensions 参数(默认不传),以服务端实际返回 1024 维为准。
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
EMBEDDING_BATCH = 32
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")


def get_embeddings():
    """创建与入库/检索共用的 OpenAIEmbeddings 实例(指向 SiliconFlow)。

    需先在 .env 配好 SILICONFLOW_API_KEY(注册 https://cloud.siliconflow.cn 获取)。
    """
    from langchain_openai import OpenAIEmbeddings

    if not SILICONFLOW_API_KEY:
        raise RuntimeError(
            "缺少 SILICONFLOW_API_KEY: 请在项目根目录 .env 中配置 "
            "(注册 https://cloud.siliconflow.cn 获取,模型 BAAI/bge-m3 免费)。"
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        chunk_size=EMBEDDING_BATCH,
        check_embedding_ctx_length=False,
    )


# ----------------------------------------------------------------------
# Milvus(远程向量库,与 test.py 测试的 47.96.113.144:19530 同一个服务)
# ----------------------------------------------------------------------
MILVUS_URI = os.getenv("MILVUS_URI", "http://47.96.113.144:19530")
# MILVUS_TOKEN 留空 = 无认证(与 test.py 一致); 需要认证时形如 "root:密码"
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN") or None
MILVUS_COLLECTION_BEASTIARY = os.getenv("MILVUS_COLLECTION_BEASTIARY", "roco_beastiary")
MILVUS_COLLECTION_WORLD = os.getenv("MILVUS_COLLECTION_WORLD", "roco_world")


def collection_name(kind: str) -> str:
    """按文档类型返回 Milvus collection 名: kind 为 'beastiary' 或 'world'。"""
    name = {
        "beastiary": MILVUS_COLLECTION_BEASTIARY,
        "world": MILVUS_COLLECTION_WORLD,
    }.get(kind)
    if not name:
        raise ValueError(f"未知的文档类型 kind={kind!r}, 可选 'beastiary' / 'world'")
    return name
