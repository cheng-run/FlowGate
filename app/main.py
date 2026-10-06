"""FlowGate 应用入口：装配 + FastAPI 路由。

对应"五站走位"（docs/architecture-notes.md §一）：/health 是探活旁路；
/v1/chat/completions 目前走"单上游直调"形态（第 3~4 站的 W1 版），
resolve/fallback（第 3 站完整版）是 W3，流式（第 5 站）是 W2。
"""

from fastapi import FastAPI

from app.schemas import ChatCompletionResponse, ChatRequest
from providers.base import Provider
from providers.fake import FakeProvider

# ===== 装配处（composition root）=====
# 全代码库唯一允许点名具体适配器类的地方（ADR-0001）：核心路由只见 Provider 协议，
# 换上游=改这一行。W1 只接 fake；接 DashScope 时改成按 config 里的名字挑实例。
provider: Provider = FakeProvider()

app = FastAPI(
    title="FlowGate",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查：给探活/监控用的最小契约（200 + status=ok）。

    为什么是独立端点而不是复用 /v1/chat/completions：探活要零副作用、零认证、
    响应形状永不变更——它响了就代表进程活着，不掺任何业务语义。
    """
    # 行级：只回固定字典，FastAPI 自动序列化成 JSON 200——不手写 Response，保持直白。
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatRequest) -> ChatCompletionResponse:
    """非流式对话（OpenAI 兼容形状）。

    为什么 async：下游是网络 IO（真实上游），不能堵事件循环；
    为什么形参直接用 ChatRequest：Pydantic 校验就是五站里第 2 站"输入检查"——
    形状不对 FastAPI 在进路由前就自动 422，脏请求永远到不了适配器。
    """
    # 行级：W1 还没有 resolve/fallback（W3），直接把统一请求交给装配好的上游；
    # 路由本身零业务逻辑——这就是 seam 的样子，路由只当"传送带"。
    return await provider.chat(request)
