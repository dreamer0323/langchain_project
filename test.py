import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.chat_models import init_chat_model

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

messages = [
    {"role": "system", "content": "你是一个基于Deepseek的智能体，我可以在多个领域提供帮助。"},
    {"role": "user", "content": "你好，我是张三"},
    {"role": "assistant", "content": "你好，你是张三，我是一个基于Deepseek的智能体，我可以在多个领域提供帮助。"},
    {"role": "user", "content": "你好，你还记得我是谁吗？"},
]

print(deepseek_llm.invoke(messages))
