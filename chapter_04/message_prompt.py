import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from pprint import pprint

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
    # deepseek-v4 默认开启思考模式，要求回传 reasoning_content；
    # langchain 的转换器会丢弃该字段，所以这里显式关闭思考模式
    extra_body={"thinking": {"type": "disabled"}},
)

user_message = HumanMessage(content="你好,介绍一下若叶睦")

# DeepSeek 思考型模型要求：回传的 assistant 消息必须带上 reasoning_content 字段，否则 API 返回 400
ai_message = AIMessage(
    content="若叶睦是Bangdream系列中的角色",
    tool_calls=[
        {"id": "call_123", 
         "name": "get_person_info",
         "args": {"name": "若叶睦"}}
        ]
)

# ToolMessage 必须有 tool_call_id,用于关联 AI 发起的那个工具调用(对应 AIMessage.tool_calls 里的 id)
tool_message = ToolMessage(

    content="若叶睦是一个角色，他是一个有智慧的人，他喜欢和人互动",
    tool_call_id="call_123",
)

messages = [user_message, ai_message, tool_message]

response = deepseek_llm.invoke(messages)

pprint(response)