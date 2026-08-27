import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
#都会强行用 .env 文件里写的值去覆盖它

load_dotenv(override=True)
# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
if not DEEPSEEK_API_KEY or not DEEPSEEK_BASE_URL:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")


deepseek_llm = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}} # 禁用思考模式
)

# 艺术家结构体
class Artist(BaseModel):
    """
    艺术家结构体
    """
    name: str = Field(description="艺术家名称")
    genre: str = Field(description="艺术家类型")
    country: str = Field(description="艺术家所属国家")

# 音乐结构体
class MusicStruct(BaseModel):
    """
    音乐结构体
    """
    name: str = Field(description="音乐名称")
    artist: list[Artist] = Field(description="音乐艺术家")
    album: str = Field(description="音乐专辑名称")
    genre: str = Field(description="音乐类型")
    duration: int = Field(description="音乐时长，单位：秒", ge=0)

model_struct = deepseek_llm.with_structured_output(MusicStruct)  # 绑定结构化输出

messages = [{"role": "user", "content": "返回一首音乐Tornado Souls的结构化信息"}]

try:
    response = model_struct.invoke(messages)
except Exception as e:
    raise RuntimeError("结构化输出调用失败，请检查 API Key / Base URL / 模型是否支持 function calling") from e

# model_dump_json 输出更易读：indent 缩进，ensure_ascii=False 保留中文
print(response.model_dump_json(indent=2, ensure_ascii=False))

