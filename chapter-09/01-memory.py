"""
【LangChain 1.x 长期记忆(Long-term Memory)演示 —— PostgreSQL 持久化版】

核心思路:
- LangGraph 的会话内记忆靠 checkpointer 和 thread 区分,会话结束就丢了;
- 而"长期记忆"是指跨会话、跨进程仍然存在的记忆,靠 BaseStore 实现;
- 这里用 PostgresStore 存进 PostgreSQL,程序关闭、机器重启数据都还在。
"""
import os
from typing import NotRequired  # 重要!NotRequired 来自 typing,不导入会 NameError
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.store.postgres import PostgresStore  # 持久化存储(写进 PostgreSQL)
from langchain.agents import create_agent, AgentState
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ----------------------------------------------------------------------
# 环境变量
# ----------------------------------------------------------------------
# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
# 都会强行用 .env 文件里写的值去覆盖它
load_dotenv(override=True)
# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# PostgreSQL 连接配置
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")

# 拼出连接串。注意:连接串里的密码如果含特殊字符需要做 URL 编码。
PG_CONN_STRING = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# 初始化模型。
# 注意:不要在这里传 store= 或 tools= !
#   - store: ChatDeepSeek 不认识这个参数,传了会直接抛 ValidationError;
#   - tools: 工具应该只传给 create_agent,传给模型会被警告并塞进 model_kwargs 变成摆设。
# "deepseek:" 前缀让 init_chat_model 自动推断用 langchain-deepseek 的 ChatDeepSeek。
model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ----------------------------------------------------------------------
# 长期记忆存储
# ----------------------------------------------------------------------
# PostgresStore.from_conn_string 是一个上下文管理器(contextmanager),
# 连接的存活时间与 with 块相同,所以 store 的建立、agent 的组装和运行
# 都要放进同一个 with 块里。
# setup() 会自动建表(幂等,重复调用不报错),首次运行时执行。
#
# 自定义 Agent 状态:在默认状态(messages 等)基础上,额外加一个 user_id 字段。
# NotRequired 表示这个字段"可以有也可以没有",不会强制每次都必须传。
class CustomState(AgentState):
    user_id: NotRequired[str]

# ----------------------------------------------------------------------
# 定义两个工具:写长期记忆 / 读长期记忆
# ----------------------------------------------------------------------
# parse_docstring=True 表示:从下面的 Google 风格 docstring 自动提取
# 每个参数的作用,转成工具的参数描述,这样大模型才知道该传什么。
@tool(parse_docstring=True)
def save_user_info(name: str, runtime: ToolRuntime) -> str:
    """
    将客户信息保存在长期记忆中

    Args:
        name : 用户名
        runtime : 工具的运行时

    Returns:
        str : 保存状态
    """
    # namespace(命名空间)相当于"目录",key 相当于"主键"。
    # 这里用 ("users",) 当目录、user_id 当主键,
    # 于是不同 user_id 的客户数据在同一个 store 里互不干扰。
    namespace = ("users",)
    key = runtime.state["user_id"]  # 当前会话(线程)状态里的用户ID
    value = {"name": name}

    # runtime.store 就是 create_agent(store=store)注入进来的那个 store,
    # 在这里通过 put 写入长期记忆。
    runtime.store.put(namespace, key, value)
    return "saved"


@tool(parse_docstring=True)
def get_user_info(runtime: ToolRuntime) -> str:
    """
    从长期记忆中读取客户的信息

    Args:
        runtime : 工具的运行时

    Returns:
        str : 用户信息
    """
    namespace = ("users",)
    key = runtime.state["user_id"]

    # 用同样的 namespace + key 把数据读回来。
    # 查不到时 get 返回 None,就回一句 "unknown",模型看到会回答"我不认识你"。
    item = runtime.store.get(namespace, key)
    return str(item.value) if item else "unknown"


# ----------------------------------------------------------------------
# 组装 agent 并运行
# ----------------------------------------------------------------------
# with 块保证整个运行期间 PostgreSQL 连接是活的。
with PostgresStore.from_conn_string(PG_CONN_STRING) as store:
    # 自动建表(表已存在则跳过),首次运行必须执行一次
    store.setup()

    # 参数说明:
    #   model         : 上面初始化好的模型
    #   tools         : 允许 agent 调用的工具列表
    #   store         : 长期记忆存储,会被注入到每个工具的 ToolRuntime.store 中
    #   state_schema  : 自定义的 Agent 状态类型(多了 user_id 字段)
    #   system_prompt : 系统提示词,告诉模型什么时候该用工具
    agent = create_agent(
        model=model,
        tools=[save_user_info, get_user_info],
        store=store,
        state_schema=CustomState,
        system_prompt="用户提及个人信息时，可以使用工具保存用户信息。如果用户询问个人信息时，可以尝试使用工具读取用户信息"
    )

    # 第一个会话:自我介绍"我是小花",模型会调用 save_user_info 存进长期记忆。
    # user_id 是我们在 invoke 时额外传入的状态字段,两个会话都用 user-1。
    print("=" * 30, '-> 第一个会话（线程） <-', "=" * 30)
    response1 = agent.invoke({
        "messages": [HumanMessage("你好，很高兴认识你，我是小花")],
        "user_id": "user-1"
    })
    for msg in response1["messages"]:
        msg.pretty_print()

    # 第二个会话:这是一个全新的会话(没带任何对话历史),
    # 但 user_id 仍是 user-1,所以 get_user_info 能从 store 里把"小花"读回来——
    # 这就是"长期记忆"跨会话生效的地方。
    print("=" * 30, '-> 第二个会话（线程） <-', "=" * 30)
    response2 = agent.invoke({
        "messages": [HumanMessage("我是谁")],
        "user_id": "user-1"
    })
    for msg in response2["messages"]:
        msg.pretty_print()
