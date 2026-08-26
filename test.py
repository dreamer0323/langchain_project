import os
from dotenv import load_dotenv
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

# deepseek-v4-flash 是带推理链(reasoning)的模型：
# 思考内容通过流式的 reasoning_content 字段返回，正文通过 content 字段返回。
# 而 chunk.text 只读取 content，所以思考阶段屏幕上没有任何输出，
# 复杂任务思考很久时看起来就像"卡住/很慢"。
# 修复：把思考链也实时打印出来，就能看到模型确实在工作。

for chunk in deepseek_llm.stream("写一首七言律诗，总结大模型的发展"):
    reasoning = chunk.additional_kwargs.get("reasoning_content") 
    if reasoning:
        print(f"{reasoning}", end="", flush=True)  # 实时显示思考过程
    if chunk.text:
        print(chunk.text, end="", flush=True)  # 逐token输出正文