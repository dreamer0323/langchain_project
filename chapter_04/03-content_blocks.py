import base64
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pprint import pprint

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
#都会强行用 .env 文件里写的值去覆盖它

load_dotenv(override=True)
# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

###超参数

MODEL_NAME = "deepseek:deepseek-v4-flash-vision-exp"
EXIT_KEYWORD = "exit"
MAX_PAIRS = 2

def base64_encode_image(image_path):
    with open(image_path, "rb") as image:
        image_base64 = base64.b64encode(image.read()).decode("utf-8")
    return image_base64

### 初始化模型
deepseek_llm = init_chat_model(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

image_path = r"D:\dev\agent\agent_dev\lang_chain\code\chapter_04\DM_20240906203421_001.jpg"
base64_image = base64_encode_image(image_path)

user_input = HumanMessage(
    content_blocks=[{"type": "text", "text": "这张图是什么"},
                    {"type": "image", "base64": base64_image, "mime_type": "image/jpeg"}]
)
messages = [user_input]

print("="*50,"\n模型回复:")
response = deepseek_llm.invoke(messages)

pprint(response)
print("="*50,"\n模型回复的content:")
print(response.content)
print("="*50,"\n模型回复的content_blocks:")
pprint(response.content_blocks)
