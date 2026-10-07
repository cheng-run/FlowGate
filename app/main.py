"""FlowGate 应用入口：装配 + FastAPI 路由。

对应"五站走位"（docs/architecture-notes.md §一）：/health 是探活旁路；
/v1/chat/completions 走"单上游直调"形态（第 3~4 站的 W1 版），并按
request.stream 分派 JSON 整答或 SSE 流（第 5 站，W2 issue 01 最小闭环；
断连清理 02、错误语义 03 已收进 streaming/）。resolve/fallback（第 3 站完整版）是 W3。
"""

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import ChatCompletionResponse, ChatRequest
from providers.base import Provider, UpstreamError
from providers.dashscope import DEFAULT_BASE_URL, DashScopeProvider
from providers.fake import FakeProvider
from routing.chain import AttemptTimeoutError, FallbackChain
from routing.request_id import new_request_id, request_id_var
from streaming.sse import GapTimeoutError, sse_response


def create_provider() -> Provider:
    """装配处核心：按环境变量装上游——逗号表 = 顺序 fallback 链（issue 02，默认 fake）。

    这是生产装配逻辑本身，不是"为测试加的钩子"（spec 决定：不加工厂函数/不上 DI）——
    选择逻辑无论放哪都得有个名字可调，写成函数只是把装配决策显式化；
    它不接收任何注入参数，测试与生产走同一条入口。
    逗号表口径（checklist 2）：FLOWGATE_PROVIDER=a,b = a 先试、a 挂 b 接管；
    **单值即单元素链**——退化回上游本身，历史行为一字不差（向后兼容：既有装配测试的
    isinstance(create_provider(), FakeProvider) 断言不改仍绿）。
    读环境为什么不引 python-dotenv：uv 的 `--env-file .env` 启动时注入环境，
    代码只读 os.environ——零新增依赖（依赖红线：能零依赖就不引库）。
    """
    # 上游选择：FLOWGATE_PROVIDER 缺省 fake——无 key 也能跑测试与演示
    choice = os.environ.get("FLOWGATE_PROVIDER", "fake")
    # 复杂语句（推导式）行上：逗号切表、去空白——顺序即 fallback 优先级
    names = [part.strip() for part in choice.split(",")]
    # 复杂语句（推导式）行上：逐条装成上游实例（未知/空条目在 _build_one 里响亮报错）
    providers = [_build_one(name, choice) for name in names]
    if len(providers) == 1:
        return providers[0]  # 行级：单元素链退化为上游本身（向后兼容的字面兑现）
    return FallbackChain(providers)


def _build_one(name: str, choice: str) -> Provider:
    """装一个上游实例——全代码库唯一点名具体适配器类的函数（ADR-0001）。

    为什么拆出来：逗号表要逐条装配，if-ladder 收在这里，"点名适配器"仍只在
    装配处一处；未知取值/空条目响亮报错（配错喊响），消息带原始 choice 供定位。
    """
    if name == "fake":
        return FakeProvider()
    if name == "dashscope":
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
    # 未知取值（含空条目）同样响亮报错：拼错的上游名若静默回 fake，配置就形同虚设
    raise RuntimeError(
        f"未知的上游 {name!r}（FLOWGATE_PROVIDER={choice!r}；可选：fake / dashscope）"
    )


# ===== 装配处（composition root）=====
# 全代码库唯一允许点名具体适配器类的地方（ADR-0001）：核心路由只见 Provider 协议，
# 换上游=改环境变量 FLOWGATE_PROVIDER，业务代码一行不动。
provider: Provider = create_provider()

app = FastAPI(
    title="FlowGate",
    version="0.1.0",
)


class RequestIdMiddleware:
    """ASGI 中间件：进门发号、响应挂号（X-Request-Id）——每逻辑请求一个对账号。

    给初学者的解释（纯 ASGI 中间件在本代码库首次出现）：ASGI 应用就是一个
    async def __call__(scope, receive, send) 的可调用对象；中间件包一层，在 send
    拦下"响应头那条消息"（http.response.start）补一个头，响应体/断连消息原样放行——
    W2 的流生命周期（逐块搬运、断连清理）因此一字不动。
    为什么不用 FastAPI 的 @app.middleware("http")：那走 BaseHTTPMiddleware，会接管
    响应体的逐块搬运（多一层内存流）——W2"断连不泄漏"建立在体不经第二人之手上，
    为加个响应头去动流的搬运路径得不偿失。
    为什么发号在进门：号要先于路由存在——将来限流/预算门卫（进路由之前的一步）与
    billing 结算都按它对账；fallback 链的 attempts 直接取用，同一请求永不二号。
    """

    def __init__(self, app) -> None:
        """持有被包裹的 ASGI 应用（Starlette 以 app= 关键字注入）。"""
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        """HTTP 请求进门发号并绑进 ContextVar；响应头带号；出门还原背包。"""
        if scope["type"] != "http":
            # 行级：lifespan 等非 HTTP scope 不是"逻辑请求"——不发号、不碰，原样放行
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()
        token = request_id_var.set(request_id)  # 行级：号放进"隐形背包"，链里伸手取

        async def send_with_request_id(message: dict) -> None:
            """替身 send：只改 http.response.start（补 X-Request-Id），其余消息原样放行。"""
            if message["type"] == "http.response.start":
                # 行级：headers 是 [(b"名", b"值"), …] 的列表——追加一项即挂号；
                # 复制出新消息而不是就地改，不惊动上游应用可能复用的消息对象
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", ()),
                        (b"x-request-id", request_id.encode()),
                    ],
                }
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            # 行级：出门还原背包——同一任务若复用（测试里连续请求），号不串门
            request_id_var.reset(token)


app.add_middleware(RequestIdMiddleware)


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


@app.exception_handler(GapTimeoutError)
async def gap_timeout_handler(request: Request, exc: GapTimeoutError) -> JSONResponse:
    """gap 超时（仅首块前能走到这）→ 504，语义=上游没按时说话（spec 错误契约）。

    为什么 504 不是 502：502 是"上游拒了"（有原因可读），504 是"没按时说话"——
    客户端对两者的重试直觉不同（502 该换路/换 key，504 值得等一会再试）。
    为什么只有首块前会走到这：首块后同一条件在 streaming/ 里就地截断（流已承诺、
    200 已发出），异常根本不会穿到 HTTP 层——本 handler 就是"可报错窗口"的出口，
    窗口在首块处关闭（W3"重试窗口限死在首 token 之前"的契约面）。
    """
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.exception_handler(AttemptTimeoutError)
async def attempt_timeout_handler(request: Request, exc: AttemptTimeoutError) -> JSONResponse:
    """尝试预算超时（非流式超时族成员）→ 504，与 gap 超时同一语义出口（issue 02）。

    为什么与 gap 超时同翻 504：超时族的语义都是"上游没按时说话"——504 一个出口，
    客户端重试直觉一致；族成员分开两类只为失败词汇精确（gap=块间预算，attempt=
    非流式尝试预算，一个词只说一件事）。为什么翻译集中在这里：路由保持传送带，
    失败翻译只在这几行 handler（本文件上游错误 handler 的注释早就预告过——
    "将来接 fallback/重试（W3）时改动点也在这"）。
    """
    return JSONResponse(status_code=504, content={"detail": str(exc)})


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
    契约，而 FastAPI 对 Response 实例（StreamingResponse 及其子类）不做
    response_model 序列化——流式腿直接原样送出，两条腿互不干扰。
    """
    # 行级：流式分支——统一 chunk 流交给 streaming/ 装配成 SSE 响应。
    # 路由只做"选哪种响应"这一个决定（传送带纪律：序列化、[DONE]、上游收尾
    # 与断连清理全归 streaming/ 模块，业务逻辑不进 HTTP 层）。
    # await 的理由（issue 03）：装配要先把首块取到手——首块前是可报错窗口，
    # 拒答/超时在这抛出来还能翻 502/504，路由只是把装配结果递出去。
    if request.stream:
        return await sse_response(provider.chat_stream(request))
    # 行级：W1 还没有 resolve/fallback（W3），直接把统一请求交给装配好的上游；
    # 路由本身零业务逻辑——这就是 seam 的样子，路由只当"传送带"。
    return await provider.chat(request)
