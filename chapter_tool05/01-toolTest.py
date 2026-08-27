import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 使用标准模型名称
deepseek_llm = init_chat_model(
    model="deepseek-chat",           # 根据实际情况调整
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

@tool(description="获取指定城市的天气情况,我是tool注解上的属性")
def get_weather(city: str):
    """
    获取指定城市的天气情况

    Args:
        city: 城市名称
    """
    return f"{city}的天气是落雨，水系威力技能+75%~"
# 打印工具定义
print(convert_to_openai_tool(get_weather))

model_with_tools = deepseek_llm.bind_tools([get_weather])

messages = [{"role": "user", "content": "北京天气怎么样？请使用工具查询。"}]
response = model_with_tools.invoke(messages)
messages.append(response)

# 提取工具调用
tool_calls = response.additional_kwargs.get("tool_calls")
if tool_calls:
    for tool_call in tool_calls:
        if tool_call["function"]["name"] == "get_weather":
            # 解析参数 JSON
            args = json.loads(tool_call["function"]["arguments"])
            tool_response = get_weather(args["city"])
            messages.append({"role": "tool", "content": tool_response})
else:
    print("模型未调用工具，直接回复如下：")
    response.pretty_print()
    # 可以在这里直接退出或继续对话
    exit()

# 将工具结果发回模型
result_msg = model_with_tools.invoke(messages)
result_msg.pretty_print()