"""FlowGate 应用入口：从根级 assembly 取装配单例 + FastAPI 路由。

对应"五站走位"（docs/architecture-notes.md §一）：/health 是探活旁路；
/v1/chat/completions 走"单上游直调"形态（第 3~4 站的 W1 版），并按
request.stream 分派 JSON 整答或 SSE 流（第 5 站，W2 issue 01 最小闭环；
断连清理 02、错误语义 03 已收进 streaming/）。resolve/fallback（第 3 站完整版）是 W3；
限流门卫（W3 issue 05）是**进路由之前的一步**（FastAPI 依赖形态），路由本体仍零业务。
"""

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import ChatCompletionResponse, ChatRequest

# 装配处（根级 assembly）的五个单例转绑到本模块命名空间：既有测试/演示
# monkeypatch app.main.provider / limiter / ledger / budget / keystore 的手法一个不破
# （票 05 验收项：名字仍绑在 app.main，门卫与路由读到的是同一绑定）。
# create_provider 一并转出：tests/test_kimi_live.py（live 真网 smoke）以
# app.main 为装配入口直取生产链，该文件不在本票改动面——别名留着它才不破。
# 下一行 `as X` 自别名是显式 re-export 形态：本模块不消费这个函数，写成
# 自别名 ruff F401 才不把它当 unused-import 删掉（不是笔误，是给它的豁免）。
from assembly import budget, keystore, ledger, limiter, provider
from assembly import create_provider as create_provider
from billing.identity import current_key, key_var
from billing.ledger import BudgetExceededError
from keys.scope import AuthorizationError, check_scope
from keys.store import AuthenticationError
from providers.base import UpstreamError
from ratelimit.bucket import RateLimitError
from routing.chain import AttemptTimeoutError
from routing.request_id import new_request_id, request_id_var
from streaming.sse import GapTimeoutError, sse_response

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
    为什么凭据也在这里进背包（issue 06）：凭据在进门才可见，结算却在收尾——
    set/reset 必须罩住**整个**请求（含流式响应的发送）才不串门，能罩全的只有
    中间件这一层；认证门卫与结算从背包取，不各自再解析一遍 Authorization。
    进门放的是凭据串、认证门卫改写成 key_id（W4 身份口径）：本层只管提取与收尾，
    校验与改写归 auth_gate——出门按进门时的还原点回滚背包，中间怎么改写都不用它操心。
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


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    """认证失败 → 401：同一条文案（防枚举），响应体形状与 429/502/504 同款 {"detail"}。

    为什么翻译集中在这（同 502/504/429 纪律）：路由保持传送带，失败翻译只在
    handler 这几行；401/403/429 三分（没身份/不在 scope/太快或超预算），客户端按
    状态码就知道下一步该做什么（带对凭据 / 换 model / 退避提额）。
    为什么 str(exc) 就够、handler 不补信息：AuthenticationError 构造上不收消息
    （keys/store.py：防枚举）——想"顺手丰富一下错误详情"都没有缝。
    """
    return JSONResponse(status_code=401, content={"detail": str(exc)})


@app.exception_handler(AuthorizationError)
async def authorization_error_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
    """授权失败 → 403：detail 点名被拒的 model（story 9），形状同 {"detail"}（同款纪律）。

    为什么翻译集中在这（同 502/504/429/401 纪律）：路由保持传送带，失败翻译只在
    handler 这几行；401（没身份）/ 403（有身份但 model 不在 scope）/ 429（太快或
    超预算）三分，客户端按状态码就知道下一步——带对凭据 / 换 model / 退避提额。
    为什么 detail 可以点名 model 而 401 必须统一文案：403 的话对**已认证**调用方
    要可操作（story 9：别让客户端猜），点名的还是他自己发来的字符串；401 的话对
    陌生人必须含糊（防枚举）——两种"说多少"各有威胁模型，不是双重标准。
    """
    return JSONResponse(status_code=403, content={"detail": str(exc)})


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
    """从 Authorization 头取 bearer 凭据串；无头/空值 = 匿名（返回 None）。

    只提取不校验：校验/撤销是认证门卫（auth_gate → keys/）的事，这里管的是
    "头 → 凭据串"的形状（单一提取处，issue 06）。"Bearer " 前缀剥掉（大小写不
    敏感）：凭据串是 fgk_ 本体；非 Bearer 形态（如 Basic xxx）整段值当凭据串交上去
    ——hash 查无不区分形态，同样一条 401（防枚举），不给"换个 Authorization
    scheme 就绕过认证"留后门。
    """
    if authorization is None:
        return None
    value = authorization.strip()
    if not value:
        return None
    scheme, _, rest = value.partition(" ")  # 行级：拆"scheme 与凭据串"——只认打头的 Bearer
    if scheme.lower() == "bearer":
        credential = rest.strip()
        return credential or None  # 行级："Bearer" 后面空空如也=没带凭据，按匿名算
    return value


async def auth_gate() -> str:
    """认证门卫（FastAPI 依赖）：/v1/* 的第一道门——fail-closed，过门返回 key_id。

    给初学者的解释（依赖**链**在本代码库首现）：rate_limit_gate 的形参挂着
    Depends(auth_gate)——FastAPI 先跑认证、再跑限流，认证抛异常请求就死在这，
    限流门卫与路由本体根本不会执行。这就是"认证依赖接在限流门卫之前"（ADR-0007
    预告）的落地形态，且是结构性保证（写在依赖图里，不靠参数书写顺序）。
    为什么必须 async（比 rate_limit_gate 的理由更重）：同步依赖会被 FastAPI 丢进
    线程池跑，线程里 ContextVar.set 改的是**拷贝**出来的上下文、改不回请求主链——
    身份要写回背包，声明成 async 是正确性的前提，不只是省线程。
    fail-closed：无头/空串/坏串/未注册/已撤销全部同一条 AuthenticationError——
    verify 对不存在/已撤销回同一种"查无"（keys/store.py 的防枚举地基），文案分叉
    在构造上就不可能（AuthenticationError 不收消息）。空库=拒绝一切：verify 查
    任何凭据都是查无，不存在"认证未激活"状态（spec 口径）。
    为什么凭据从背包取：中间件进门时已提取 Authorization（单一提取处，门卫与
    结算都不各自再解析一遍头）——这里只做校验，提取形状不归门卫管。
    身份改写（W4 口径）：过门后把背包里的凭据串换成 key_id——从此账本行、限流桶、
    预算求和全按 key_id 对账，凭据串验过即弃、不落任何库表（story 15）。中间件
    出门按它进门时的还原点回滚背包（ContextVar 乱序 reset 合法，2026-10-08 实测），
    这一刀改写不用自己收尾。
    新增 /v1 路由必须挂门卫链（rate_limit_gate——它嵌套 scope_gate、再嵌套本门卫）：
    现在"全 fail-closed"靠唯一业务路由的依赖链枚举成立，将来谁加路由忘挂 gate
    就是 fail-open——这是 06 边界检查之外的人肉防线（评审前向风险记录，2026-10-08）。
    """
    credential = current_key()  # 行级：背包里是中间件进门放的凭据串（匿名=None）
    key_id = keystore.verify(credential) if credential else None
    if key_id is None:
        raise AuthenticationError()  # 行级：五变体同一条 401（防枚举），不回显凭据
    key_var.set(key_id)  # 行级：身份改写——背包换成公开身份，凭据串就此丢弃
    return key_id


async def scope_gate(request: ChatRequest, _auth: str = Depends(auth_gate)) -> str:
    """授权门卫（FastAPI 依赖）：按 key 的 scope 判本次请求的 model，越权抛
    AuthorizationError → 403。门卫链中段：认证 401 → body 422 → 授权 403。

    给初学者的解释（依赖**吃 body** 在本代码库首现）：本函数的 request 形参是
    ChatRequest（Pydantic 模型），FastAPI 会先校验 body、校验通过才调用我们——
    body 坏了 FastAPI 只把错误汇成 422、**根本不会进本函数**；body 好了才拿得到
    request.model 来判 scope。所以"422 先于 403"不是靠谁先写 if，而是"没有合法
    body 就没有 model 可判"的机械顺序（票面 checklist：授权检查排在 body 校验之后）。
    为什么 401 又排在 422 前：auth_gate 是本门卫的子依赖，FastAPI 解依赖树先跑它、
    抛异常立即中止——认证不看 body（spike 实测口径，tests/test_auth_gate.py 钉死）。
    三者拼起来就是依赖图给出的 401 → 422 → 403，测试在 tests/test_scope_403.py 收口。
    为什么形参挂着 auth_gate（而不是并列挂在路由上）：嵌套依赖=结构性顺序，不靠
    参数书写顺序（票 02 同款理由）；且越权请求死在本门卫、进不了限流桶——被拒的
    请求零成本（不占令牌不进预算），与 401 同待遇。
    身份从背包取（issue 06 单一出处）：auth_gate 过门时已把背包改写成 key_id，
    这里不再解析任何请求头；scope 原文交给 check_scope 解释（keys/ 存取与解释分层）。
    """
    key_id = current_key()  # 行级：背包里是 auth_gate 改写后的 key_id（到这必有身份）
    # 行级：scope_of 取原文（存取），查无=None 进 check_scope 即 fail-closed 拒
    scope_text = keystore.scope_of(key_id) if key_id else None
    check_scope(scope_text, request.model)  # 行级：不在白名单抛 AuthorizationError → 403
    return key_id


async def rate_limit_gate(_key_id: str = Depends(scope_gate)) -> None:
    """门卫（FastAPI 依赖）：授权之后的一步——按 key 扣令牌，空桶抛 RateLimitError。

    给初学者的解释（FastAPI 依赖在本代码库首现）：Depends(rate_limit_gate) 把本函数
    "钉"在路由前面执行——它跑完返回 None，路由照常；它抛异常，请求到此为止，
    路由本体（乃至上游调用）根本不会发生。门卫与路由分离，路由保持传送带（checklist 6）。
    为什么形参挂着 scope_gate（W4 票 03 起；票 02 前是 auth_gate）：门卫链的
    结构性顺序=认证 → 授权 → 限流——嵌套依赖不靠参数书写顺序；_ 前缀=只接线
    不取值（key 仍从背包取，见下）。新 /v1 路由挂本门卫即天然带上整条链。
    为什么声明 async（虽然体内没有 await）：同步依赖会被 FastAPI 丢进线程池跑，
    async 依赖直跑事件循环——门卫是纯内存一步（扣令牌），不值得占用线程池，
    也免了线程切换；与路由同为 async 是同一条调用约定（同 502 handler 的理由）。
    为什么拒绝在这里就够：门卫在上游调用之前（checklist 2）——429 是零成本的，
    桶里没令牌的请求连 fake/真上游的面都见不到。
    key 口径（W4 票 02 兑现 ADR-0007 预告）：认证门卫插在本门卫之前后，匿名/坏
    凭据在认证门就是 401，"匿名放行不占桶"的 W3 口径就此作废——到这的必有身份。
    身份=key_id（认证门卫验出后改写背包）：账本行、限流桶、预算求和全按 key_id
    对账，凭据串不是身份、不落任何库。ADR-0007 的"key=bearer 串"段由 09 的
    ADR-0010 兑现更新。
    key 的出处（issue 06）：中间件进门时已把凭据放进背包、认证门卫改写成 key_id——
    门卫不再自己解析 Authorization（身份口径单一出处，结算门面取的是同一个）。
    预算为什么排在限流前面（issue 06）："进门先查预算"（spec 口径）——超预算是
    终局性的（再来多少次都没用），先说真话且不给注定被拒的请求消耗限流令牌。
    """
    key = current_key()
    if key is None:
        return  # 行级：防御性兜底——fail-closed 后没有 HTTP 路径能进这（认证已拒）
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
