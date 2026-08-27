#基于对话模型chatModel，进行chatPromptTemplate的测试
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


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

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是yuzusoft社团中的因幡巡"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

messages = {"history":[("human", "你好,我是保科"),("ai", "前辈你好呀。")],"input":"你最喜欢玩什么游戏?对于其中的武器你有什么看法"}

# 调用chat_prompt_template的invoke方法，将messages作为参数传递
chat_prompt_value = chat_prompt_template.invoke(messages)

#接收模型回复的content_blocks内容
response = deepseek_llm.invoke(chat_prompt_value)
print("="*50,"\n","n模型第一次回复:",response.content,"\n")

chat_prompt_value = chat_prompt_template.invoke({"history":[("ai", response.content)],"input":"我上一个问你的问题是什么？"})
response = deepseek_llm.invoke(chat_prompt_value)
print("="*50,"\n","n模型第二次回复:",response.content,"\n")
