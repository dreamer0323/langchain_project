import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from rich import print as rprint
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("你是个非常友好的AI助手"),
    HumanMessage("你好啊，我是老王，你是谁？"),
    AIMessage("你好老王，我是小王"),
    HumanMessage("好的小王，很高兴认识你"),
    AIMessage("你高兴得太早了"),
    HumanMessage("呵呵，你什么意思")
]

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
#都会强行用 .env 文件里写的值去覆盖它

load_dotenv(override=True)
# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")



# 初始化模型
deepseek_llm = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    profile={
        "max_input_tokens": 10000,
    }
)

# 创建智能体
agent = create_agent(
    model = deepseek_llm,
    tools = [],
    # 导入中间件
    middleware = [SummarizationMiddleware(model=deepseek_llm, trigger=
        [
        ("tokens", 100),
        ("messages", 5),
        ("fraction", 0.001)
        ],
        # 保留2条消息
        keep=("messages", 2),
        summary_prompt="将以上内容链式总结起来,\n{messages}"
        )
    ]
)

# 调用智能体
response = agent.invoke({"messages": messages})

for msg in response["messages"]:
    msg.pretty_print()
