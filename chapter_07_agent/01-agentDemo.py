import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from rich import print as rprint
# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
#都会强行用 .env 文件里写的值去覆盖它

load_dotenv(override=True)
# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")



deepseek_llm = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# 创建智能体
agent = create_agent(
    model = deepseek_llm,
    tools = [],
)

# 调用智能体
response = agent.invoke({"messages": [{"role": "system", "content": "你是一个东方project系列琪露诺，你可以帮助用户解决问题。"},
                                       {"role": "user", "content": "你好，请介绍一下自己吧"},
                                       {"role": "assistant", "content": "咱是琪露诺，一个东方project系列的角色。"},
                                       {"role": "user", "content": "你最喜欢的人是谁？"},
                                       ]})

rprint(response)

print("智能体回复：","\n")
rprint(response["messages"][-1].content)
