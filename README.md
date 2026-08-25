# LangChain 项目

基于 [LangChain](https://www.langchain.com) 的智能体（Agent）学习与开发项目，当前使用 **DeepSeek** 作为底层大模型。

## 技术栈

- **Python 3.x**
- **LangChain** 1.2.x（langchain-core / langchain-community / langchain-classic）
- **LangGraph** 1.1.x（Agent / 多智能体编排）
- **MCP / FastMCP**（模型上下文协议）
- **DeepSeek** API（当前使用的模型）
- 支持多模型扩展：OpenAI、Anthropic、OpenRouter、腾讯云等

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入密钥：

```bash
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

> ⚠️ `.env` 已加入 `.gitignore`，请勿将真实密钥提交到仓库。

### 3. 运行示例

```bash
python test.py
```

[test.py](test.py) 演示了如何通过 `init_chat_model` 初始化 DeepSeek 模型并完成多轮对话（包含记忆能力的验证）。

## 项目结构

```
├── test.py            # 基础模型调用示例
├── requirements.txt   # 依赖清单
└── .env               # 环境变量（不入库）
```

## 说明

- 本项目以学习 LangChain 生态（模型调用、Agent、RAG、MCP）为主，目录会随学习进度持续扩展。
- 依赖清单中未启用（注释掉）的部分（如 Milvus、PyTorch、文档解析等）可按需取消注释安装。
