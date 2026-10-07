"""FlowGate 应用入口：装配 + FastAPI 路由。

对应"五站走位"（docs/architecture-notes.md §一）：/health 是探活旁路；
/v1/chat/completions 走"单上游直调"形态（第 3~4 站的 W1 版），并按
request.stream 分派 JSON 整答或 SSE 流（第 5 站，W2 issue 01 最小闭环；
断连清理 02、错误语义 03 已收进 streaming/）。resolve/fallback（第 3 站完整版）是 W3；
限流门卫（W3 issue 05）是**进路由之前的一步**（FastAPI 依赖形态），路由本体仍零业务。
"""

import os

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import ChatCompletionResponse, ChatRequest
from billing.identity import current_key, key_var
from billing.ledger import BillingLedger, BudgetExceededError
from billing.settlement import BillingProvider
from providers.base import Provider, UpstreamError
from providers.dashscope import DEFAULT_BASE_URL, DashScopeProvider
from providers.fake import FakeProvider
from providers.kimi import KimiProvider
from ratelimit.bucket import RateLimiter, RateLimitError
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
    拆不动说明（函数物理行数超 40，含 docstring/注释；同 routing/chain.py 先例）：
    if-ladder 就是"唯一点名适配器"的机制本体——拆成每上游一个小函数会把点名面
    摊到多处，抽 _require_env 通用助手则把各档的配置故事（dashscope 的 base_url
    有默认 / kimi 双必填，这个不对称是刻意的）藏进第三处；教学注释是规范硬要求
    删不得，行数超限以本说明豁免。
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
    if name == "kimi":
        # 缺 key 必须响亮且指名（沿 ADR-0002 纪律，与 dashscope 同款）：静默降级回
        # fake 会让"我明明配了 kimi，怎么答的还是回显"变成难查的悬案
        api_key = os.environ.get("KIMI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "FLOWGATE_PROVIDER=kimi 但缺少环境变量 KIMI_API_KEY"
                "（写入 .env，用 uv run --env-file .env 启动）"
            )
        # base_url 同样必填、无内置默认（Kimi 与 DashScope 的不对称是刻意的）：Kimi 走
        # 私有中转，真实地址只活在 .env（issue 08 checklist"真实地址永不进 git"）——
        # 内置默认地址就是猜，猜错（拿中转 key 打官方地址）是运行期 401 的静默悬案
        base_url = os.environ.get("KIMI_BASE_URL", "")
        if not base_url:
            raise RuntimeError(
                "FLOWGATE_PROVIDER=kimi 但缺少环境变量 KIMI_BASE_URL"
                "（真实中转地址写入 .env，不入 git；用 uv run --env-file .env 启动）"
            )
        return KimiProvider(api_key=api_key, base_url=base_url)
    # 未知取值（含空条目）同样响亮报错：拼错的上游名若静默回 fake，配置就形同虚设
    raise RuntimeError(
        f"未知的上游 {name!r}（FLOWGATE_PROVIDER={choice!r}；可选：fake / dashscope / kimi）"
    )


def create_limiter() -> RateLimiter:
    """装配处：按环境变量装限流器（env 口径，与 create_provider 同一纪律）。

    FLOWGATE_RATE_CAPACITY=桶容量（允许的突发），FLOWGATE_RATE_PER_SECOND=稳态吞吐
    （tokens/s）。默认 20/10：LLM 客户端天然突发（一条请求一轮对话），20 的突发余量
    给交互式客户端留了面子，10/s 的稳态又足以拦住打爆型流量——口径随部署调 env 即可，
    不用改代码。配置非法（非数字/越界）响亮报错（配错喊响，同 _build_one 纪律）。
    为什么不加工厂函数/不上 DI：与 create_provider 同一理由——这是生产装配逻辑本身，
    测试与生产走同一条入口，测试要换限流口径时 monkeypatch 装配绑定（app.main.limiter）。
    """
    try:
        capacity = float(os.environ.get("FLOWGATE_RATE_CAPACITY", "20"))
        per_second = float(os.environ.get("FLOWGATE_RATE_PER_SECOND", "10"))
        return RateLimiter(capacity=capacity, per_second=per_second)
    except ValueError as exc:
        # 行级：数字解析失败与桶参数越界都是配置错——统一指名 env 变量，排错不用猜
        raise RuntimeError(
            f"FLOWGATE_RATE_CAPACITY / FLOWGATE_RATE_PER_SECOND 配置非法：{exc}"
        ) from exc


def create_ledger() -> BillingLedger:
    """装配处：按环境变量装账本（env 口径，与 create_provider 同一纪律）。

    FLOWGATE_BILLING_DB=SQLite 单文件路径（生产在 .env 指向落盘文件）；缺省
    ":memory:"=进程内临时库——测试/演示零残留、离线可复跑（checklist 1 的口径：
    测试用 tmp_path 临时文件显式构造账本，不依赖缺省）。
    """
    db_path = os.environ.get("FLOWGATE_BILLING_DB", ":memory:")
    return BillingLedger(db_path)


def create_budget() -> int:
    """装配处：按环境变量装每 key 的 token 预算（FLOWGATE_BUDGET_TOKENS）。

    0=不设限（缺省）：记账先跑、预算口径按部署调 env——"0"是"无预算"不是"零预算"。
    为什么不给个有限缺省：预算一旦有限，账本跨重启累计（生产）就会在跑批场景里
    突然 429；治理口径宁可显式开启（与限流不同：限流天然要挡，预算天然要看部署）。
    """
    try:
        return int(os.environ.get("FLOWGATE_BUDGET_TOKENS", "0"))
    except ValueError as exc:
        # 行级：非数字=配置错，统一指名 env 变量（配错喊响，同 create_limiter 纪律）
        raise RuntimeError(f"FLOWGATE_BUDGET_TOKENS 配置非法：{exc}") from exc


# ===== 装配处（composition root）=====
# 全代码库唯一允许点名具体适配器类的地方（ADR-0001）：核心路由只见 Provider 协议，
# 换上游=改环境变量 FLOWGATE_PROVIDER，业务代码一行不动。
# 账本单例先落座：结算门面与门卫的预算检查共用同一本账（用户账求和=花销唯一出处）
ledger: BillingLedger = create_ledger()
# 每 key 的 token 预算（0=不设限）：门卫进门先查它，超预算在上游调用前就 429
budget: int = create_budget()
# 结算门面包在**装配绑定**上、不进 create_provider：create_provider 的既有契约是
# 装出可 isinstance 的上游/链（装配测试钉住），计费是其外的一层治理（与门卫同理）。
provider: Provider = BillingProvider(create_provider(), ledger=ledger)
# 限流器与上游同在装配处落座：门卫只认 RateLimiter 接口，换限流算法不惊动路由
limiter: RateLimiter = create_limiter()

app = FastAPI(
    title="FlowGate",
    version="0.1.0",
)


class RequestContextMiddleware:
    """ASGI 中间件：进门记身份（发号 + key 进背包）、响应挂号（X-Request-Id）。

    给初学者的解释（纯 ASGI 中间件在本代码库首次出现）：ASGI 应用就是一个
    async def __call__(scope, receive, send) 的可调用对象；中间件包一层，在 send
    拦下"响应头那条消息"（http.response.start）补一个头，响应体/断连消息原样放行——
    W2 的流生命周期（逐块搬运、断连清理）因此一字不动。
    为什么不用 FastAPI 的 @app.middleware("http")：那走 BaseHTTPMiddleware，会接管
    响应体的逐块搬运（多一层内存流）——W2"断连不泄漏"建立在体不经第二人之手上，
    为加个响应头去动流的搬运路径得不偿失。
    为什么发号在进门：号要先于路由存在——限流/预算门卫（进路由之前的一步）与
    billing 结算都按它对账；fallback 链的 attempts 直接取用，同一请求永不二号。
    为什么 key 也在这里进背包（issue 06）：key 在进门才可见，结算却在收尾——
    set/reset 必须罩住**整个**请求（含流式响应的发送）才不串门，能罩全的只有
    中间件这一层；门卫与结算从背包取，不各自再解析一遍 Authorization。
    """

    def __init__(self, app) -> None:
        """持有被包裹的 ASGI 应用（Starlette 以 app= 关键字注入）。"""
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        """HTTP 请求进门发号 + 记 key 进背包；响应头带号；出门还原背包。"""
        if scope["type"] != "http":
            # 行级：lifespan 等非 HTTP scope 不是"逻辑请求"——不发号、不碰，原样放行
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()
        token = request_id_var.set(request_id)  # 行级：号放进"隐形背包"，链里伸手取
        # 行级：bearer 身份进背包（匿名=None）——限流/预算/账本共用这一个身份口径
        key_token = key_var.set(_bearer_key(_scope_header(scope, "authorization")))

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
            # 行级：出门还原背包——同一任务若复用（测试里连续请求），号与 key 都不串门
            request_id_var.reset(token)
            key_var.reset(key_token)


app.add_middleware(RequestContextMiddleware)


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


@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """空桶 429：错误说得清"该退避了"（story 12）——客户端据此退避而不是盲目重试。

    为什么 429（Too Many Requests）：这是"你打得太快"而不是"上游/网关坏了"——
    状态码本身就是重试直觉的分类器（与 502/504 同一翻译纪律，集中在这几行 handler）。
    为什么 detail 保留容量/速率：消息里带上桶口径，客户端调试时能对出"我的节奏超了多少"。
    """
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(BudgetExceededError)
async def budget_exceeded_handler(request: Request, exc: BudgetExceededError) -> JSONResponse:
    """超预算 429：错误说得清"该提额了"（story 14）——与限流的"退避"是两种动作。

    为什么也翻 429 而不是 402：spec 口径就是 429（拒绝进网关，还没到"要钱"的语义）；
    两种 429 靠 detail 文案区分——限流说退避、预算说提额，客户端的下一步动作不同。
    """
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查：给探活/监控用的最小契约（200 + status=ok）。

    为什么是独立端点而不是复用 /v1/chat/completions：探活要零副作用、零认证、
    响应形状永不变更——它响了就代表进程活着，不掺任何业务语义。
    """
    # 行级：只回固定字典，FastAPI 自动序列化成 JSON 200——不手写 Response，保持直白。
    return {"status": "ok"}


def _scope_header(scope: dict, name: str) -> str | None:
    """从 ASGI scope 取请求头（latin-1 解码，HTTP 头的字符集约定）；缺头=None。

    为什么在中间件而不是 Request.headers 取：中间件层只有原始 scope——把 Authorization
    翻成身份串就地完成（_bearer_key），进背包的是结果，不是原始头。
    """
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.decode("latin-1").lower() == name:
            return raw_value.decode("latin-1")
    return None


def _bearer_key(authorization: str | None) -> str | None:
    """从 Authorization 头取 bearer 串当限流 key；无头/空值 = 匿名（返回 None）。

    不透明身份（checklist 3）：不校验格式、不查库、不撤销——凭据的生成/校验/撤销
    是 W4 keys/ 的事，这里只把串当身份用。"Bearer " 前缀剥掉（大小写不敏感）：
    key 是凭据串本身；非 Bearer 形态（如 Basic xxx）整段值当不透明 key——宁可把它
    当陌生 key 限流，也不给"换个鉴权 scheme 就绕过限流"留后门。
    """
    if authorization is None:
        return None
    value = authorization.strip()
    if not value:
        return None
    scheme, _, rest = value.partition(" ")  # 行级：拆"scheme 与凭据串"——只认打头的 Bearer
    if scheme.lower() == "bearer":
        key = rest.strip()
        return key or None  # 行级："Bearer" 后面空空如也=没带凭据，按匿名放行
    return value


async def rate_limit_gate() -> None:
    """门卫（FastAPI 依赖）：进路由之前的一步——按 key 扣令牌，空桶抛 RateLimitError。

    给初学者的解释（FastAPI 依赖在本代码库首现）：Depends(rate_limit_gate) 把本函数
    "钉"在路由前面执行——它跑完返回 None，路由照常；它抛异常，请求到此为止，
    路由本体（乃至上游调用）根本不会发生。门卫与路由分离，路由保持传送带（checklist 6）。
    为什么声明 async（虽然体内没有 await）：同步依赖会被 FastAPI 丢进线程池跑，
    async 依赖直跑事件循环——门卫是纯内存一步（扣令牌），不值得占用线程池，
    也免了线程切换；与路由同为 async 是同一条调用约定（同 502 handler 的理由）。
    为什么拒绝在这里就够：门卫在上游调用之前（checklist 2）——429 是零成本的，
    桶里没令牌的请求连 fake/真上游的面都见不到。
    key 口径（2026-10-07 用户拍板，见 ADR-0007）：key=Authorization bearer 串；
    **匿名（无 Authorization）放行不占桶**——限流治理"每个身份不许打爆"，认证
    （W4）管"有没有身份"；W4 把认证插在本门卫之前，此处的 key 接口一字不改。
    key 的出处（issue 06）：中间件进门时已把身份放进背包——门卫不再自己解析
    Authorization（身份口径单一出处，结算门面取的是同一个）。
    预算为什么排在限流前面（issue 06）："进门先查预算"（spec 口径）——超预算是
    终局性的（再来多少次都没用），先说真话且不给注定被拒的请求消耗限流令牌。
    """
    key = current_key()
    if key is None:
        return  # 行级：匿名放行不占桶也不查预算——口径见 docstring，用例都显式带 bearer
    # 行级：预算前置检查——花销按 key 对用户账求和，超预算在上游调用前 429（story 14）；
    # 不设限（budget=0）连求和都不查，设限时求和只跑一趟（比较与文案共用一份数字）
    if budget > 0:
        spend = ledger.spend(key)
        if spend >= budget:
            raise BudgetExceededError(
                f"预算已超：该 key 累计花销已达 {spend} tokens"
                f"（预算 {budget} tokens），请提额后再来"
            )
    limiter.acquire(key)  # 行级：空桶在此抛 RateLimitError → 异常处理器翻 429


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatRequest,
    _gate: None = Depends(rate_limit_gate),
) -> ChatCompletionResponse | StreamingResponse:
    """对话端点（OpenAI 兼容形状）：按 request.stream 分派 JSON 整答或 SSE 流。

    为什么 async：下游是网络 IO（真实上游），不能堵事件循环；
    为什么形参直接用 ChatRequest：Pydantic 校验就是五站里第 2 站"输入检查"——
    形状不对 FastAPI 在进路由前就自动 422，脏请求永远到不了适配器
    （流式分支同样先过这道门，见 streaming 测试的 422 护栏）。
    为什么有个 _gate 形参：FastAPI 依赖把限流门卫钉在路由之前（checklist 6）——
    它不在路由体里，本体对"限流"二字零知情，仍是传送带；_ 前缀表示只接线不取值。
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
