"""sse-streaming 测试（issue 01 流式闭环 / 02 断连不泄漏 / 03 错误语义），接缝由 spec 预先约定。

两条缝：Provider 缝（直接消费 fake.chat_stream 的统一 chunk 流）+ HTTP 契约缝
（TestClient / 手驱 ASGI 打 POST /v1/chat/completions 的 SSE 出口）。全部离线——
fake 上游零外网、事件等待只设上限不真实 sleep，"测试全绿"在任何机器可复跑（spec 故事 29）。
断连用例全在 HTTP 契约缝的 ASGI 协议高度完成（手动喂 http.disconnect），新增缝 0（方案 A）。
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：跑的是真实应用（含装配处），与 chat 端点测试同一高度。
from app.main import app
from app.schemas import ChatCompletionChunk, ChatMessage, ChatRequest, DeltaMessage, StreamChoice
from providers.base import Provider, UpstreamError
from providers.fake import FakeProvider

client = TestClient(app)


def _make_request() -> ChatRequest:
    """最小合法请求——与 providers 测试同款，跨文件读起来是同一套语言。"""
    return ChatRequest(
        model="fake-model",
        messages=[ChatMessage(role="user", content="你好")],
    )


def _payload(stream: bool = False) -> dict:
    """最小合法请求体（dict 直发，模拟真实 HTTP 客户端）；stream 开关可选。"""
    payload: dict = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }
    if stream:
        payload["stream"] = True
    return payload


class _SpyProvider:
    """记账哨兵：chat / chat_stream 被走到就记一笔——"上游零调用"靠空账本证明。"""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(self, request: ChatRequest):
        """走到即记账——422 用例里它必须永不被调。"""
        self.calls.append("chat")
        raise AssertionError("非法请求体不应触达上游 chat()")

    async def chat_stream(self, request: ChatRequest):
        """走到即记账（async 生成器首帧被消费时才执行函数体，见 base.py 的解释）。"""
        self.calls.append("chat_stream")
        yield ChatCompletionChunk(
            id="spy-1",
            model=request.model,
            choices=[StreamChoice(delta=DeltaMessage(content="x"))],
        )


async def test_fake_provider_stream_yields_openai_chunk_sequence() -> None:
    """证明：fake 流式按 OpenAI chat.completion.chunk 形状逐块吐回显句（Provider 缝）。

    怎么证明：收集 chat_stream 的全部 chunk，断言五件外部可见的事实——多于一块
    （单块就退化成整答了）、每块 object=chat.completion.chunk、首块 delta.role=
    assistant（客户端据首个 delta 定角色）、内容拼回等于字面量 "fake-reply: 你好"
    （独立于实现的真源：数据真穿过了 seam，不是 fake 自说自话）、末块
    finish_reason=stop（OpenAI 的结束帧惯例，客户端据此收尾）。
    """
    chunks = [chunk async for chunk in FakeProvider().chat_stream(_make_request())]

    assert len(chunks) >= 3
    assert all(c.object == "chat.completion.chunk" for c in chunks)
    assert all(c.id for c in chunks)
    assert all(c.model == "fake-model" for c in chunks)  # 模型名随请求穿过 seam
    # delta 藏在 choices[0] 里（OpenAI 帧形状）：帧是"对 0 号候选的一次补丁"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "fake-reply: 你好"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_fake_provider_stream_chunks_deterministic_across_runs() -> None:
    """证明：fake 分块确定性——同样请求跑两次，切法一字不差（02/03 断言的地基）。

    怎么证明：两次独立收集各自的 delta.content 序列，断言完全相等；id 不参与比较
    （uuid 每次不同，本来就该不同）。若分块掺了随机或时间，这条会偶发红。
    """
    run_one = [
        c.choices[0].delta.content async for c in FakeProvider().chat_stream(_make_request())
    ]
    run_two = [
        c.choices[0].delta.content async for c in FakeProvider().chat_stream(_make_request())
    ]

    assert run_one == run_two
    assert len(run_one) >= 3


def test_provider_protocol_requires_chat_stream() -> None:
    """证明：Provider 协议已含流式方法且 fake 满足——ADR-0001 的预告在 01 兑现。

    怎么证明：正向 isinstance(FakeProvider(), Provider)；反向造一个只有 name+chat
    的最小实现，断言它不再过协议。反向断言是关键：若协议其实没要求 chat_stream，
    反向也会 True，这条就抓不住"协议忘了加流式方法"的回归。
    """

    class _NonStreamingProvider:
        """反例实现：缺 chat_stream——用来证明协议真的在检查这个方法存在。"""

        name = "non-streaming"

        async def chat(self, request: ChatRequest):
            """不实现（测试不会调用它）：只为凑齐非流式两件套，让反例只差流式方法。"""
            raise NotImplementedError

    assert isinstance(FakeProvider(), Provider)
    assert not isinstance(_NonStreamingProvider(), Provider)


async def test_fake_provider_stream_blocks_when_gate_closed() -> None:
    """证明：fake 带可控闸门——关着时一块也取不出，开一次只放行一块（02/03 复用地基）。

    怎么证明：取块任务挂在关着的闸门上，等一个上限后断言没取到（有上限、无真实
    sleep 排程）；开闸后取到首帧（role=assistant）；再取第二块又被卡住——证明
    "放行即关门"的逐块节奏，测试才能把流稳稳按在第 k 块上（超时/断连用例的前提）。
    """
    gate = asyncio.Event()
    stream = FakeProvider(gate=gate).chat_stream(_make_request())

    # 行级：任务去取块——闸门关着，等待只设上限（0.1s 是兜底，不是节奏），
    # 到点若没取到就证明流被闸门按住了（asyncio.wait 超时不取消任务，流保持存活）
    first_try = asyncio.create_task(anext(stream))
    done, _ = await asyncio.wait({first_try}, timeout=0.1)
    assert not done  # 闸关着 → 一块也取不出

    gate.set()  # 行级：开一次闸 → 放行当前这一块
    first = await asyncio.wait_for(first_try, timeout=1.0)
    assert first.choices[0].delta.role == "assistant"  # 放行的第一块就是首帧

    # 行级：放行后闸门自动关上 → 第二块又被按住（逐块放行，不是一开全放）
    second_try = asyncio.create_task(anext(stream))
    done, _ = await asyncio.wait({second_try}, timeout=0.1)
    assert not done

    gate.set()
    second = await asyncio.wait_for(second_try, timeout=1.0)
    assert second.choices[0].delta.content  # 放行的第二块真的带内容

    await stream.aclose()  # 收摊：显式关流，不把生成器留给 GC


def test_chat_endpoint_streams_sse_frames() -> None:
    """证明：`stream: true` 经 fake 上游在 HTTP 出口流出逐块 SSE（HTTP 契约缝）。

    怎么证明：发带 stream=true 的 POST，断言四件外部可见的事实——200 且
    content-type 是 text/event-stream；body 按 \\n\\n 切出多帧、每帧都是
    "data: {json}" 且 JSON 的 object=chat.completion.chunk；首帧 delta.role=
    assistant、各帧内容拼回等于字面量 "fake-reply: 你好"（数据穿过整条 seam）；
    最后一帧是 data: [DONE]（完成记号，客户端据此分辨"流完了"与"断了"）。
    """
    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # 行级：SSE 帧以空行分隔——切开、丢掉结尾的空串，剩下的就是一帧一帧的 payload
    frames = [f for f in response.text.split("\n\n") if f]
    assert len(frames) >= 4  # 首帧 + 多个内容帧 + [DONE]（单帧就不叫流式了）
    assert all(f.startswith("data: ") for f in frames)
    assert frames[-1] == "data: [DONE]"

    # 行级：剥掉 "data: " 前缀就是纯 JSON——逐帧解析成 chunk，形状在字节层钉住
    chunks = [json.loads(f.removeprefix("data: ")) for f in frames[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["id"] and c["model"] == "fake-model" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assembled = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert assembled == "fake-reply: 你好"  # 字面真源：内容穿过整条链一块不丢
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_endpoint_stream_false_stays_json() -> None:
    """证明：显式 stream=false 与缺省同走 JSON 腿——非流式客户端零改动（checklist 第 3 条）。

    怎么证明：同一请求体显式加 "stream": false，断言仍是 application/json 的
    chat.completion 整答（不是 chunk 流、不发 [DONE]）。缺省（不带 stream 字段）
    那一半由既有 chat 端点测试全绿覆盖，这里钉住 false 这一半——两者必须同路。
    """
    body = _payload()
    body["stream"] = False  # 行级：显式 false 是独立用例——别让它和缺省悄悄分家

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body_json = response.json()
    assert body_json["object"] == "chat.completion"  # 整答形状，不是 chat.completion.chunk
    assert body_json["choices"][0]["message"]["content"] == "fake-reply: 你好"


def test_chat_endpoint_rejects_invalid_body_when_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：非法请求体在流式路径同样 422，且上游零调用——流式不是校验的旁路（故事 6）。

    怎么证明：装一个记账哨兵上游，发"缺 messages + stream=true"的请求体，
    断言 422 + 账本为空。422 来自进路由前的 Pydantic 校验；账本为空则证明
    校验门在上游之前——脏请求永远漏不到适配器（这是守住不变量的回归护栏：
    流式分支落地前它也必须成立）。
    """
    spy = _SpyProvider()
    monkeypatch.setattr("app.main.provider", spy)

    response = client.post("/v1/chat/completions", json={"model": "fake-model", "stream": True})

    assert response.status_code == 422
    assert spy.calls == []  # 空账本 = 上游一腿都没被碰过


def test_chat_endpoint_closes_upstream_stream_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：流正常跑完时上游生成器已收尾——finally 执行，不晚于响应返回（checklist）。

    怎么证明：stub 上游的生成器在 finally 里置旗标；TestClient 消费完整个流返回后
    立即断言旗标为 True——全程不调 gc.collect()、不 sleep。收尾若真靠 GC 时机，
    "响应已返回"这一刻没有任何东西保证 finally 跑过；显式收尾（streaming 模块的
    finally aclose）才让这个时序成为契约。
    """
    closed = {"finished": False}

    class _ClosingProvider:
        """只提供流式腿的桩：finally 旗标就是"上游生成器收尾了"的可观察信号。"""

        name = "closing"

        async def chat(self, request: ChatRequest):
            """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
            raise AssertionError("stream=true 不应走非流式 chat()")

        async def chat_stream(self, request: ChatRequest):
            """吐一块就完的极简流；finally 在生成器结束（跑完或被 aclose）时置旗。"""
            try:
                yield ChatCompletionChunk(
                    id="close-1",
                    model=request.model,
                    choices=[StreamChoice(delta=DeltaMessage(content="x"))],
                )
            finally:
                closed["finished"] = True

    monkeypatch.setattr("app.main.provider", _ClosingProvider())

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    assert closed["finished"] is True  # 紧跟响应断言：收尾在流结束时就发生了


# ===== issue 02：断连不泄漏（断连用例全在 HTTP 契约缝的 ASGI 协议高度，新增缝 0）=====


def _asgi_scope() -> dict:
    """POST /v1/chat/completions 的最小 ASGI scope——手动驱动协议层的"请求"这一半。

    为什么 asgi.spec_version=2.3：uvicorn 0.54 实发 2.3（h11_impl 的 scope 构造），
    StreamingResponse 对 2.4 以下走 listen_for_disconnect + task group 分支——
    测试 scope 与生产同构，断连语义才是生产语义（不是测试自造的旁路）。
    """
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),  # 随便一个地址：路由不看它，凑齐 scope 形状即可
        "server": ("127.0.0.1", 8000),
    }


class _ASGIDriver:
    """手驱 ASGI 的收发两端：请求体一次、断连一个事件、send 记录全部消息。

    为什么手驱 ASGI 而不用 TestClient：断连是协议层事件（uvicorn 生产发的正是
    http.disconnect），TestClient 会替我们把响应一口气消费完——只有自己拿住
    receive/send 才能在"回答半路"精确插进断连（方案 A：同一 HTTP 契约缝的
    ASGI 高度，新增缝 0）。

    给初学者的解释（ASGI 收发回调在本代码库首次出现）：ASGI 服务器不给应用传
    "请求/响应"对象，而是给两个异步回调——receive() 逐条吐请求消息（先是
    http.request 带正文，之后可能来 http.disconnect），send() 逐条收响应消息
    （http.response.start 开头、http.response.body 带正文帧）。谁先动由消息驱动，
    测试接管这两个回调后就能在任意时刻插断连——"断连用例在 ASGI 协议高度"
    说的就是这件事。
    """

    def __init__(
        self,
        payload: dict,
        *,
        release_gate: asyncio.Event | None,
        cancel_after: int,
        hold_send: bool = False,
    ) -> None:
        """三种玩法的旋钮：放行闸门（None=闸门自管）、断连帧数、是否堵在 send 里。

        release_gate 非 None：预放行首帧，此后每收一帧放行下一帧（turnstile 节奏），
        产块进度被测试拿住；None：闸门自管（预放行形态，首帧后自行卡住）——
        并发用例走这条，各路互不干扰。hold_send：断连后不退出 send 而是挂起——
        把断连钉进"发帧窗口"（消费者弃流场景，02 的收尾盖子用例）。
        """
        self._payload = payload
        self._release_gate = release_gate
        self._cancel_after = cancel_after
        self._hold_send = hold_send
        self._hold = asyncio.Event()  # 永不放行：hold_send 模式的挂起点（取消会落在它的 wait 上）
        self.sent: list[dict] = []
        self._disconnect_now = asyncio.Event()
        self._body_sent = False

    async def receive(self) -> dict:
        """首问给请求体（FastAPI 解析 ChatRequest 用），此后只等断连——与 uvicorn 同构。

        为什么 body 只发一次：ASGI 里正文消息一个流一条（more_body=False 收口），
        流生命周期里 receive 之后只会再来 http.disconnect，和生产收发顺序一致。
        """
        if not self._body_sent:
            self._body_sent = True
            body = json.dumps(self._payload).encode()
            return {"type": "http.request", "body": body, "more_body": False}
        await self._disconnect_now.wait()  # 事件等待（无真实 sleep）：等测试决定何时断连
        return {"type": "http.disconnect"}

    async def send(self, message: dict) -> None:
        """记录每条消息；正文帧推进"回答进度"：没到断连点就放行下一帧，到了就断连。"""
        self.sent.append(message)
        if message["type"] != "http.response.body":
            return  # 只关心正文帧：响应头/收尾帧不推进"回答进度"
        # 复杂语句（推导式）行上：只数带正文的帧——空 body 是流收尾帧，不算进度
        frames = sum(1 for m in self.sent if m["type"] == "http.response.body" and m.get("body"))
        if frames < self._cancel_after:
            if self._release_gate is not None:
                self._release_gate.set()  # 放行下一帧：turnstile 关着，不放就卡住
            return
        self._disconnect_now.set()  # 收满即断连——此刻流正停在"回答半路"
        if self._hold_send:
            await self._hold.wait()  # 堵在发帧窗口：这个挂起点就是取消的落点

    async def run(self) -> list[dict]:
        """跑完整次 ASGI 调用（应用返回才回来），返回 send 记下的全部消息。"""
        if self._release_gate is not None:
            self._release_gate.set()  # 预放行首帧：闸门从关着起步，第一块由驱动器放进来
        # 行级：ASGI 应用自己就是可调用对象（服务器的视角）——app(scope, receive, send)
        await app(_asgi_scope(), self.receive, self.send)
        return self.sent


async def test_chat_endpoint_records_cancelled_when_client_disconnects_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：客户端半路断开（ASGI 喂 http.disconnect），fake 记收尾=被取消——
    断连不泄漏（checklist 1）。

    怎么证明：闸门 fake 被驱动器按节奏放行，放过 3 帧（全流 6 帧的一半）后喂
    http.disconnect。app 返回后**立即**断言 fake.stream_endings == ["cancelled"]——
    全程不 gc.collect()、不 sleep：收尾若靠 GC 时机，返回这一刻不会有确定记录；
    显式取消传播才让这条断言稳定为真（02 的"显式传播，不靠 GC 时机"）。
    """
    gate = asyncio.Event()  # 闸门从关着起步：产块节奏全由驱动器放行拿住
    fake = FakeProvider(gate=gate)
    monkeypatch.setattr("app.main.provider", fake)

    driver = _ASGIDriver(_payload(stream=True), release_gate=gate, cancel_after=3)
    sent = await asyncio.wait_for(driver.run(), timeout=2.0)  # 带上限：机制坏了红掉，不挂死 CI

    # 复杂语句（推导式）行上：只数带正文的帧——恰好 3 帧=回答真在半路，不是没开始/已说完
    frames = [m for m in sent if m["type"] == "http.response.body" and m.get("body")]
    assert len(frames) == 3
    assert fake.stream_endings == ["cancelled"]  # 显式传播的证据：无 gc、无 sleep 的断言


def test_chat_endpoint_records_completed_when_stream_runs_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：正常跑完的流 fake 记收尾=正常结束——与"被取消"可区分，防假绿（checklist 2）。

    怎么证明：无闸门 fake（节奏零干扰）走 TestClient 全量消费到 [DONE]，断言
    stream_endings == ["completed"]。与上一条断连用例合看：两种收尾各自可断言——
    若收尾记录只有一种值、或断言恒真，这两条必有一条红。
    """
    fake = FakeProvider()  # 不带闸门：正常节奏跑完整条流
    monkeypatch.setattr("app.main.provider", fake)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")  # 完整流必有完成记号
    assert fake.stream_endings == ["completed"]  # 恰好一条记录且是"正常结束"


async def test_chat_endpoint_records_cancelled_when_disconnect_during_send_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：断连恰逢"发帧窗口"（响应任务悬在 await send）时上游也被显式取消，
    不靠 GC（checklist 1 补强）。

    怎么证明：驱动器把首帧送进 send 后就地堵住再喂 http.disconnect——这是 Starlette
    弃流**不收**生成器的路径（实验证据见 .scratch/sse-streaming/issues/02 评论）：
    取消落在 send 的 await 上、不穿生成器链，上游悬在 yield，收尾若只能等 GC，
    app 返回这一刻账本必是空的。streaming/ 的收尾盖子在 ASGI 调用结束前显式
    aclose，于是同样无 gc、无 sleep 地断言 stream_endings == ["cancelled"]。
    """
    fake = FakeProvider()  # 无闸门：首帧直达 send——本例卡的是发帧窗口，不是产块窗口
    monkeypatch.setattr("app.main.provider", fake)

    # hold_send：首帧进 send 即断连并堵在发帧窗口——模拟"写响应时客户端关了页面"，
    # 取消落在 send 的挂起点上、不穿生成器链（见 docstring 的实验背景）
    driver = _ASGIDriver(_payload(stream=True), release_gate=None, cancel_after=1, hold_send=True)
    sent = await asyncio.wait_for(driver.run(), timeout=2.0)

    # 复杂语句（any+推导式）行上：首帧确实到达过 send——断连在流进行中，场景不是空转
    assert any(m["type"] == "http.response.body" and m.get("body") for m in sent)
    assert fake.stream_endings == ["cancelled"]  # 收尾盖子的证据：弃流路径也显式收


def _pre_opened_gate() -> asyncio.Event:
    """工厂形态闸门：每条流开工时领一把**已放行**的闸——首帧直通、第二帧起卡住。

    为什么并发用例要每流一闸：asyncio.Event 一 set() 唤醒所有等待者，不是计数
    信号量——N 条流共用一把时"放行一块即关门"的 turnstile 会互踩（一条流偷走
    另一条的放行）。为什么预放行：各流不需要外部再点名放行就能稳定停在
    "首帧已发"的半路状态，断连完全由各驱动自己触发，零同步开销。
    """
    gate = asyncio.Event()
    gate.set()
    return gate


async def test_chat_endpoint_leaves_zero_uncancelled_when_concurrent_disconnects_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：并发 N 路半路断连，未被取消的上游 = 0——"断连不泄漏"的实测数字（checklist 3）。

    怎么证明：16 路并发各驱动一次 ASGI 流式请求（每流一把预放行闸门的真 fake，
    首帧发出后即卡住=各路都停在"回答半路"），各自收到首帧后喂断连。结束后数
    fake 的收尾账本两条一起看：不是 cancelled 的条目=记错了收尾（推导式断言），
    总数不足 N=有流根本没记账（计数断言）——"未被取消的上游 = 0"就是这两条
    同时成立，数字写进 issue 02 验收记录供 05/06 引用。
    """
    n = 16
    fake = FakeProvider(gate=_pre_opened_gate)  # 工厂形态：每流一把独立闸门
    monkeypatch.setattr("app.main.provider", fake)

    # 复杂语句（gather+推导式）行上：16 路并发同一个 ASGI 应用（与生产同构——
    # 一个 app 服务所有并发请求），每路带上限：坏一路红一路，不把整套件挂死
    await asyncio.gather(
        *(
            asyncio.wait_for(
                _ASGIDriver(_payload(stream=True), release_gate=None, cancel_after=1).run(),
                timeout=2.0,
            )
            for _ in range(n)
        )
    )

    # 复杂语句（推导式）行上：账本里不是"被取消"的条目=收尾记错的泄漏
    # （没记账的泄漏抓不到账本里，由下面"总数==n"的计数断言兜住）
    uncancelled = [e for e in fake.stream_endings if e != "cancelled"]
    assert uncancelled == []  # 实测：并发 16 路中途断连，未被取消的上游 = 0
    assert fake.stream_endings.count("cancelled") == n  # 每路恰好记了一笔"被取消"


# ===== issue 03：流式错误语义（首块前=可报错窗口 → 502/504；首块后=流已承诺 → 只截断）=====


class _RefusingProvider:
    """首块前必抛 UpstreamError 的流式桩：两条腿同一条失败消息——
    "流式与非流式失败形状一致"的对照组（checklist 1）。"""

    name = "stub-refusing"

    async def chat(self, request: ChatRequest):
        """非流式腿抛协议失败——与 chat_stream 同一消息，供对照两条腿的响应形状。"""
        raise UpstreamError("上游 stub 返回 429: rate limited: quota exceeded")

    async def chat_stream(self, request: ChatRequest):
        """流式腿在**首块之前**抛协议失败——"可报错窗口"内的拒答形态。"""
        raise UpstreamError("上游 stub 返回 429: rate limited: quota exceeded")
        yield  # pragma: no cover —— 凑 async 生成器形状（同 dashscope 占位写法），raise 先于 yield


def test_chat_endpoint_returns_502_when_upstream_refuses_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：首块前上游拒答 → 502 且 detail 保留上游摘要，与非流式失败形状一致（checklist 1）。

    怎么证明：流式桩两条腿抛同一条 UpstreamError，同一请求分别带 stream=true/false
    发出，断言两次响应的 JSON 逐字段相等（同一失败形状的字面证明）且上游摘要原文在
    detail 里。首块前是可报错窗口——错误必须以 HTTP 状态码说话，不许降级成
    "200 + 半截流"（那会把拒答伪装成正常回答）。
    """
    monkeypatch.setattr("app.main.provider", _RefusingProvider())

    streaming = client.post("/v1/chat/completions", json=_payload(stream=True))
    non_streaming = client.post("/v1/chat/completions", json=_payload())

    assert streaming.status_code == 502
    assert non_streaming.status_code == 502
    # 行级：同一失败形状 = 两次响应 body 逐字段一致（状态码已各自断言，这里钉 body）
    assert streaming.json() == non_streaming.json()
    # 行级：上游摘要原文直读——排错不用猜，这是"detail 保留上游摘要"的字面验收
    assert "rate limited: quota exceeded" in streaming.json()["detail"]


class _SilentProvider:
    """一言不发的流式桩：chat_stream 一块不吐就正常结束——沉默版"拒答"（checklist 1 边角）。"""

    name = "stub-silent"

    async def chat(self, request: ChatRequest):
        """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
        raise AssertionError("stream=true 不应走非流式 chat()")

    async def chat_stream(self, request: ChatRequest):
        """空的 async 生成器：函数体一次都不执行，首块等待直接收到"没有下一块"。"""
        return
        yield  # pragma: no cover —— 凑 async 生成器形状，让方法保持异步流协议


def test_chat_endpoint_returns_502_when_upstream_stream_empty_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：首块前上游一条块都不给 → 502，不悬着也不装完整（checklist 1 边角）。

    怎么证明：空流桩走 stream=true，断言 502 且 detail 有可读原因。反例是
    "200 + 只有 [DONE]"——那等于把"上游一言不发"包装成完整空回答，客户端
    分不清是模型真没话说还是链路坏了；或炸成 500 带栈——把上游的沉默变成
    网关自己的故障。两种反例这条都挡。
    """
    monkeypatch.setattr("app.main.provider", _SilentProvider())

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 502
    # 行级：detail 是给人读的失败原因——空流这个根因必须在文案里可直读
    assert response.json()["detail"]


def test_chat_endpoint_returns_504_when_first_chunk_exceeds_gap_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：首块前卡住超 gap 超时 → 504，语义=上游没按时说话（checklist 2）。

    怎么证明：闸门 fake 永不放行（流被稳稳按在"一块都没出"），monkeypatch 把
    gap 常量压到毫秒级，断言 504 且 detail 提到 gap 超时。全程零真实 sleep——
    测试代码一行不睡，等待与超时全在机制内部发生；生产代码零测试钩子——
    超时口径是模块常量（monkeypatch 目标）不是回调/注入。
    """
    # 行级：毫秒级 gap 是"快进键"：真实30 秒常量被压小，超时立刻发生，测试不等真时间
    monkeypatch.setattr("streaming.sse.GAP_TIMEOUT_SECONDS", 0.05)
    fake = FakeProvider(gate=asyncio.Event())  # 闸门永不开：首块永远等不来
    monkeypatch.setattr("app.main.provider", fake)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 504
    # 行级：detail 点名 gap 超时——钉住"这是块间隔预算用尽"，不是随便一个超时
    assert "gap 超时" in response.json()["detail"]


class _SelfTimingOutProvider:
    """自带 TimeoutError 的流式桩：上游自己抛 builtin TimeoutError——gap 误标的反例现场。"""

    name = "stub-self-timeout"

    async def chat(self, request: ChatRequest):
        """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
        raise AssertionError("stream=true 不应走非流式 chat()")

    async def chat_stream(self, request: ChatRequest):
        """首块前抛 builtin TimeoutError——语义是"上游自己超时了"，不是"没按时说话"。"""
        raise TimeoutError("upstream internal timeout")
        yield  # pragma: no cover —— 凑 async 生成器形状，raise 先于 yield


def test_chat_endpoint_returns_502_not_504_when_upstream_raises_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：上游自带的 TimeoutError 不冒充 gap 超时——失败词汇一个词只说一件事。

    怎么证明：自超时桩在首块前抛 builtin TimeoutError，断言 502（"上游答不上"族）
    且 detail 不带"gap 超时"字样。反例是误标成 504"没按时说话"——排错方向整个
    反过来（该查上游的会去查网关的闹钟）。gap 只认自己那个闹钟响，这是评审收严
    钉住的边界。
    """
    monkeypatch.setattr("app.main.provider", _SelfTimingOutProvider())

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 502
    assert "gap 超时" not in response.json()["detail"]  # 不许把上游的超时说成网关的 gap


def test_chat_endpoint_truncates_without_done_when_gap_times_out_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：首块后卡住超 gap 超时 → 流截断、不发 [DONE]（checklist 3）。

    怎么证明：预放行闸门 fake（首帧直通、第二帧起卡死）+ 毫秒级 gap 常量——
    断言响应已 200（流已承诺，状态码这页翻过去了）、首块真发出过（截断在半路）、
    结尾没有 [DONE]（半截可辨），fake 记账"被掐断"（超时处置也显式收上游）。
    与 504 用例合看：同一种超时条件，首块前报错、首块后截断——
    "可报错窗口在首块处关闭"就是这两条的对照。
    """
    monkeypatch.setattr("streaming.sse.GAP_TIMEOUT_SECONDS", 0.05)
    fake = FakeProvider(gate=_pre_opened_gate)  # 每流预放行一把闸：首帧直通后卡死
    monkeypatch.setattr("app.main.provider", fake)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200  # 流已承诺：错误窗口关了，这是截断不是报错
    assert "data: " in response.text  # 首块确实发出过——截断发生在半路，不是没开始
    assert "[DONE]" not in response.text  # 半截回答绝不能带完成记号（故事 10）
    assert fake.stream_endings == ["cancelled"]  # 超时掐流也走显式收尾记账，不悬垂


class _ExplodingProvider:
    """半路爆炸的流式桩：吐一块之后抛错——"首块后上游抛错"的现场（checklist 4）。"""

    name = "stub-exploding"

    async def chat(self, request: ChatRequest):
        """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
        raise AssertionError("stream=true 不应走非流式 chat()")

    async def chat_stream(self, request: ChatRequest):
        """先给一块（承诺已成立），再抛异常——错必须发生在"首块后"才有截断语义。"""
        yield ChatCompletionChunk(
            id="boom-1",
            model=request.model,
            choices=[StreamChoice(delta=DeltaMessage(content="半截"))],
        )
        raise RuntimeError("wire exploded mid-stream")


def test_chat_endpoint_truncates_without_done_when_upstream_errors_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """证明：首块后上游抛错 → 流截断、不发 [DONE]，原因进日志（checklist 4）。

    怎么证明：半路爆炸桩先吐一块再抛 RuntimeError，断言 200 + 首块在正文里 +
    无 [DONE]（与超时截断同一处置），且 caplog 里能查到异常消息——流的死因在
    HTTP 出口看不见（200 + 半截帧），"原因进日志"是排错的唯一线索，必须可断言。
    """
    monkeypatch.setattr("app.main.provider", _ExplodingProvider())

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200  # 流已承诺：抛错也翻不成状态码，只能截断
    assert "半截" in response.text  # 首块内容真的到达过客户端——错发生在半路
    assert "[DONE]" not in response.text  # 半截回答绝不能带完成记号
    # 行级：死因进日志（caplog 收 root 传播的记录）——"原因进日志"的字面验收
    assert "wire exploded mid-stream" in caplog.text
