"""FlowGate 应用入口：装配 + FastAPI 路由。

对应"五站走位"（docs/architecture-notes.md §一）：/health 是探活旁路；
/v1/chat/completions 走"单上游直调"形态（第 3~4 站的 W1 版），并按
request.stream 分派 JSON 整答或 SSE 流（第 5 站，W2 issue 01 最小闭环；
断连/超时语义在后续 issue）。resolve/fallback（第 3 站完整版）是 W3。
"""

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import ChatCompletionResponse, ChatRequest
from providers.base import Provider, UpstreamError
from providers.dashscope import DEFAULT_BASE_URL, DashScopeProvider
from providers.fake import FakeProvider
from streaming.sse import sse_stream


def create_provider() -> Provider:
    """装配处核心：按环境变量选上游实例（spec 故事 5，默认 fake）。

    这是生产装配逻辑本身，不是"为测试加的钩子"（spec 决定：不加工厂函数/不上 DI）——
    选择逻辑无论放哪都得有个名字可调，写成函数只是把装配决策显式化；
    它不接收任何注入参数，测试与生产走同一条入口。
    读环境为什么不引 python-dotenv：uv 的 `--env-file .env` 启动时注入环境，
    代码只读 os.environ——零新增依赖（依赖红线：能零依赖就不引库）。
    """
    # 上游选择：FLOWGATE_PROVIDER 缺省 fake——无 key 也能跑测试与演示
    choice = os.environ.get("FLOWGATE_PROVIDER", "fake")
    if choice == "fake":
        return FakeProvider()
    if choice == "dashscope":
        # 缺 key 必须响亮且指名（故事 10）：静默降级回 fake 会让
        # "我明明配了真上游，怎么答的还是回显"变成难查的悬案。
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "FLOWGATE_PROVIDER=dashscope 但缺少环境变量 DASHSCOPE_API_KEY"
                "（写入 .env，用 uv run --env-file .env 启动）"
            )
        # base_url 可覆盖（默认官方 OpenAI 兼容前缀）：接中转/代理靠这个口子
        base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        return DashScopeProvider(api_key=api_key, base_url=base_url)
    # 未知取值同样响亮报错：拼错的上游名若静默回 fake，配置就形同虚设
    raise RuntimeError(f"未知的 FLOWGATE_PROVIDER={choice!r}（可选：fake / dashscope）")


# ===== 装配处（composition root）=====
# 全代码库唯一允许点名具体适配器类的地方（ADR-0001）：核心路由只见 Provider 协议，
# 换上游=改环境变量 FLOWGATE_PROVIDER，业务代码一行不动。
provider: Provider = create_provider()

app = FastAPI(
    title="FlowGate",
    version="0.1.0",
)


@app.exception_handler(UpstreamError)
async def upstream_error_handler(request: Request, exc: UpstreamError) -> JSONResponse:
    """上游非 2xx → 网关 502，detail 保留上游响应摘要（spec 错误语义）。

    为什么用异常处理器而不是路由里 try/except：路由保持"传送带"零业务逻辑；
    失败翻译集中一处，将来接 fallback/重试（W3）时改动点也在这。
    为什么 502 而不是 500：502（Bad Gateway）= 网关活着、上游答不上——
    客户端据此能区分"网关坏了"和"上游拒了"，且上游原文在 detail 里可直读。
    为什么 async（给初学者的解释）：FastAPI 要求异步路由的异常处理器也能挂进
    事件循环——handler 里没有 await 也不坏事，声明成 async 只是跟它服务的
    异步请求链路同一条调用约定，不占线程、不阻塞循环。
    """
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查：给探活/监控用的最小契约（200 + status=ok）。

    为什么是独立端点而不是复用 /v1/chat/completions：探活要零副作用、零认证、
    响应形状永不变更——它响了就代表进程活着，不掺任何业务语义。
    """
    # 行级：只回固定字典，FastAPI 自动序列化成 JSON 200——不手写 Response，保持直白。
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatRequest) -> ChatCompletionResponse | StreamingResponse:
    """对话端点（OpenAI 兼容形状）：按 request.stream 分派 JSON 整答或 SSE 流。

    为什么 async：下游是网络 IO（真实上游），不能堵事件循环；
    为什么形参直接用 ChatRequest：Pydantic 校验就是五站里第 2 站"输入检查"——
    形状不对 FastAPI 在进路由前就自动 422，脏请求永远到不了适配器
    （流式分支同样先过这道门，见 streaming 测试的 422 护栏）。
    为什么返回类型是联合：同一路径两种响应形状；response_model 仍钉住 JSON 腿的
    契约，而 FastAPI 对 Response 实例（StreamingResponse 是其子类）不做
    response_model 序列化——流式腿直接原样送出，两条腿互不干扰。
    """
    # 行级：流式分支——统一 chunk 流交给 streaming/ 传送带转 SSE 帧。
    # 路由只做"选哪种响应"这一个决定（传送带纪律：序列化、[DONE]、上游收尾
    # 都归 streaming/ 模块，业务逻辑不进 HTTP 层）。
    if request.stream:
        return StreamingResponse(
            sse_stream(provider.chat_stream(request)),
            media_type="text/event-stream",
        )
    # 行级：W1 还没有 resolve/fallback（W3），直接把统一请求交给装配好的上游；
    # 路由本身零业务逻辑——这就是 seam 的样子，路由只当"传送带"。
    return await provider.chat(request)
