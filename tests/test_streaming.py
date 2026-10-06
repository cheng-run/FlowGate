"""sse-streaming 测试（issue 01：流式最小闭环），接缝由 spec 预先约定（Testing Decisions）。

两条缝：Provider 缝（直接消费 fake.chat_stream 的统一 chunk 流）+ HTTP 契约缝
（TestClient 打 POST /v1/chat/completions 的 SSE 字节出口）。全部离线——fake 上游
零外网、闸门等待只设上限不排程，"测试全绿"在任何机器可复跑（spec 故事 29）。
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：跑的是真实应用（含装配处），与 chat 端点测试同一高度。
from app.main import app
from app.schemas import ChatCompletionChunk, ChatMessage, ChatRequest, DeltaMessage, StreamChoice
from providers.base import Provider
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
