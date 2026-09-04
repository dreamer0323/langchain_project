"""
【REST 服务 server.py】把统一接口 rag_engine.RAGEngine 暴露成 HTTP API(生产形态)

使用方只需 POST 一个问题即可, 无需理解底层 RAG 步骤; 首次 ask 前若数据未就绪会自动装载。

端点:
    GET  /status         数据/向量库就绪情况
    POST /build          一键装载(可选 reset 全量重建; 首次/联网抓取会较慢)
    POST /ask            问答 {question, top_k?} -> {answer, citations}
    POST /search         纯检索(不调 LLM) {question, top_k?} -> {citations}

启动(默认 127.0.0.1:8000):
    python chapter_10-RAG/server.py
    或: uvicorn chapter_10-RAG.server:app --host 0.0.0.0 --port 8000
"""
import argparse

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

app = FastAPI(title="洛克王国:世界 RAG 助手", version="1.0.0")
# 进程级单例: auto_build=True → 首次 ask 未就绪会自动装载
engine = RAGEngine(auto_build=True)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(4, ge=1, le=10, description="引用条数")


class BuildRequest(BaseModel):
    reset: bool = Field(False, description="重建向量 collection 后全量重灌")


@app.get("/status")
def status():
    return engine.status()


@app.post("/build")
def build(body: BuildRequest):
    result = engine.build(fetch=True, reset=body.reset, verbose=False)
    return result


@app.post("/ask")
def ask(body: AskRequest):
    result = engine.ask(body.question, top_k=body.top_k)
    return {"question": body.question, **result}


@app.post("/search")
def search(body: AskRequest):
    return {"question": body.question, "citations": engine.search(body.question, top_k=body.top_k)}


def main():
    import uvicorn

    p = argparse.ArgumentParser(description="启动 RAG HTTP 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    print(f"RAG 服务启动: http://{args.host}:{args.port}  (文档 /docs)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
