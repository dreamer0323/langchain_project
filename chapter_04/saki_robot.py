# 基于对话历史管理与对话历史优化，制作的一个对话机器人
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

###超参数

MODEL_NAME = "deepseek:deepseek-v4-flash"
EXIT_KEYWORD = "exit"
MAX_PAIRS = 2


###优化函数定义
def keep_recent_messages(messages, max_pairs=MAX_PAIRS):
    ### 优化历史记忆

    #获取系统提示词
    system_prompt = []
    for message in messages:
        if message["role"] == "system":
            system_prompt.append(message)
    #获取其他消息
    recent_messages = []
    for message in messages:
        if message["role"] != "system":
            recent_messages.append(message)
    recent_messages = recent_messages[-max_pairs * 2:]# 取最近max_pairs条消息
    #将系统提示词、用户消息、AI消息合并到一个列表中
    recent_messages = system_prompt + recent_messages
    return recent_messages

    



### 初始化模型
deepseek_llm = init_chat_model(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

message = [
    {
        "role": "system",
        "content": "你是Bangdream系列中的丰川祥子,你现在是我的专属客服。请在每句话后增加\"desuwa\"作为你的口癖"
    }
]

print("欢迎来到Bangdream系列的客服机器人,请输入:exit退出对话")

i = 1 # 对话轮数
reply_content = "" # 存储模型回复的内容

while True:
    print("\n","="*10,f"第{i}轮对话开始","="*10,"\n")

    user_input = input("请输入: ")
    #判断用户是否输入了退出关键词
    if user_input == EXIT_KEYWORD:
        print("你是来结束这个乐队的desuwa(▔皿▔)")
        break

    # 对话历史管理，将用户消息添加到消息列表中
    message.append({"role": "user", "content": user_input})
    print("丰川祥子:",end="",flush=True)
    #优化历史记忆
    memory_message = keep_recent_messages(message)

    for chunk in deepseek_llm.stream(memory_message):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply_content += chunk.content

    print("\n","="*10,f"第{i}轮对话结束","="*10,"\n")
    i += 1 # 对话轮数增加

    # 对话历史管理，将AI消息添加到消息列表中
    message.append({
        "role": "assistant",
        "content": reply_content,
        "content_type": "text"
    })
    # 清空reply_content
    reply_content = ""



        