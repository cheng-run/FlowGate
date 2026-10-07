"""DashScope 流式适配器测试（issue 04）：上游 SSE 方言翻统一 chunk，wire 与端到端两层钉住。

全部离线（httpx.MockTransport 假网络）：不碰真网、不需要 key——"测试全绿"在任何机器
可复跑（spec 故事 29）。两条缝（spec Testing Decisions 预先约定，测试只落在这两条上）：
- Provider 缝：chat_stream 进出统一 chunk + MockTransport 捕获的 wire 字节（URL/鉴权头/请求体）；
- HTTP 契约缝：POST /v1/chat/completions 的 SSE 出口（`"stream": true` + DashScope 适配器端到端）。
真网 smoke 在 tests/test_dashscope_live.py（live 标记、无 key 自动跳过）。
"""

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：端到端跑的是真实应用（含装配处），与 chat 端点测试同一高度。
from app.main import app
from app.schemas import ChatCompletionChunk, ChatMessage, ChatRequest
from providers.base import TransientUpstreamError, UpstreamError
from providers.dashscope import DashScopeProvider

client = TestClient(app)

# 测试专用的假 key / 假地址：只进 mock 断言，永不触网（与 test_dashscope 同款）
TEST_KEY = "sk-test-key"
TEST_BASE_URL = "https://upstream.test/v1"


def _upstream_frame(delta: dict, finish_reason: str | None = None) -> dict:
    """上游 chunk 帧：公共字段（id/created/model）填一次，帧差异只写 delta。

    created 是统一模型没定义的多余字段——翻译时应被静默丢弃而不是炸校验
    （Pydantic 默认行为，这条用例顺带钉住）。
    """
    return {
        "id": "chatcmpl-up",
        "object": "chat.completion.chunk",
        "created": 1728240000,
        "model": "qwen-max",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# 上游分块真源（独立字面量，不依赖实现）：首帧宣告 role、中块带正文、末帧 finish_reason——
# 四帧的 delta.content 拼起来 = UPSTREAM_TEXT，端到端断言全押在这个字面上。
UPSTREAM_FRAMES: tuple[dict, ...] = (
    _upstream_frame({"role": "assistant", "content": ""}),
    _upstream_frame({"content": "你好"}),
    _upstream_frame({"content": "，世界"}),
    _upstream_frame({}, finish_reason="stop"),
)

# 各帧 delta.content 拼回的完整回答——"内容=上游分块拼接"的字面真源
UPSTREAM_TEXT = "你好，世界"


def _upstream_sse_body() -> str:
    """把上游帧渲成 DashScope 的 SSE 方言报文：data: {json} 行 + 上游 data: [DONE] 收尾。"""
    # 复杂语句（推导式）行上：每帧渲成一行 data: + 空行分隔——上游方言的帧格式就这两样
    frames = "".join(f"data: {json.dumps(f, ensure_ascii=False)}\n\n" for f in UPSTREAM_FRAMES)
    return frames + "data: [DONE]\n\n"


def _reply_with_upstream_sse(request: httpx.Request) -> httpx.Response:
    """假上游：回 DashScope 方言的 SSE 报文——供 Provider 缝与 HTTP 缝共用同一份上游真源。"""
    return httpx.Response(200, text=_upstream_sse_body())


def _reject_with_401(request: httpx.Request) -> httpx.Response:
    """假上游拒答：401 + DashScope 风格的错误 JSON——两个拒答用例共用同一现场。"""
    return httpx.Response(401, json={"error": {"message": "Invalid API key provided"}})


def _make_provider(handler: Callable[[httpx.Request], httpx.Response]) -> DashScopeProvider:
    """按给定假上游 handler 配出适配器（同 test_dashscope：假 key/假地址/transport 注入聚一处）。"""
    return DashScopeProvider(
        api_key=TEST_KEY,
        base_url=TEST_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


def _make_request(*, stream: bool = False) -> ChatRequest:
    """最小合法请求；stream 开关供 wire 断言用（缺省 False，见各用例 docstring）。"""
    return ChatRequest(
        model="qwen-max",
        messages=[ChatMessage(role="user", content="你好")],
        stream=stream,
    )


def _payload(stream: bool = False) -> dict:
    """最小合法请求体（dict 直发，模拟真实 HTTP 客户端）；stream 开关可选。"""
    payload: dict = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": "你好"}],
    }
    if stream:
        payload["stream"] = True
    return payload


def test_chat_endpoint_streams_dashscope_sse_through_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：`"stream": true` + DashScope 适配器 → HTTP 缝回多帧 SSE + [DONE]，
    内容=上游分块拼接（checklist 1）。

    怎么证明：装配 monkeypatch 成带 MockTransport 的 DashScopeProvider（假上游回 4 帧
    data: 行 + 上游 [DONE]），发 stream=true 的 POST——断言 200 + text/event-stream；
    正文按 \\n\\n 切出恰好 4 帧 chunk + 末帧 [DONE]（比上游恰好多一帧=网关自己的完成记号）；
    逐帧 JSON 的 object=chat.completion.chunk、model=qwen-max；各帧 delta.content 拼回
    等于字面量 UPSTREAM_TEXT（"内容=上游分块拼接"的字面验收）；首帧 delta.role=
    assistant、末帧 finish_reason=stop（OpenAI 帧惯例跨方言仍成立）。
    """
    monkeypatch.setattr("app.main.provider", _make_provider(_reply_with_upstream_sse))

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # 行级：SSE 帧以空行分隔——切开、丢掉结尾空串，剩下的就是一帧一帧的 payload
    frames = [f for f in response.text.split("\n\n") if f]
    # 行级：上游 4 帧 + 网关的 [DONE]——恰好多一帧，完成记号是网关的不是上游的
    assert len(frames) == len(UPSTREAM_FRAMES) + 1
    assert all(f.startswith("data: ") for f in frames)
    assert frames[-1] == "data: [DONE]"

    # 行级：剥 "data: " 前缀逐帧解析——钉住客户端看到的是 OpenAI chunk 形状，不是上游方言
    chunks = [json.loads(f.removeprefix("data: ")) for f in frames[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["model"] == "qwen-max" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    # 复杂语句（推导式）行上：各帧增量拼回完整回答——半截拼不对就说明翻译吞了块
    assembled = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert assembled == UPSTREAM_TEXT  # 字面真源：内容=上游分块拼接，一块不丢
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_dashscope_stream_sends_wire_request_with_stream_true() -> None:
    """证明：流式 wire 由适配器钉住——URL / Bearer 鉴权头 / 请求体带 stream:true
    （checklist 2 前半）。

    怎么证明：假上游 handler 捕获收到的 Request 供事后断言；请求刻意用 stream=False
    发起——wire 里 stream 仍是 True，证明"流式开关"是适配器的方言知识，不依赖
    调用方恰好传对（传 false 上游就只回整答 JSON，流式整条链会莫名断掉）。
    model/messages 原样上 wire：模型名透传、消息内容不改写（与非流式 wire 同口径）。
    """
    captured: list[httpx.Request] = []

    def capture_wire_and_ack(request: httpx.Request) -> httpx.Response:
        """假上游：存下收到的请求字节供事后断言，回 SSE 报文让流跑完。"""
        captured.append(request)
        return _reply_with_upstream_sse(request)

    # 行级：stream=False 发起是有意的反证——wire 仍必须带 stream:true（见 docstring）
    stream = _make_provider(capture_wire_and_ack).chat_stream(_make_request(stream=False))
    await anext(stream)  # 取一块：HTTP 请求此刻才发出（生成器惰性），wire 捕获才成立
    await stream.aclose()  # 收摊：显式关流，不把生成器留给 GC（test_streaming 同款纪律）

    # 断言全部落在"发出去的字节"上——wire 是适配器的外部输出，不碰它内部任何属性
    sent = captured[0]
    assert str(sent.url) == f"{TEST_BASE_URL}/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {TEST_KEY}"
    body = json.loads(sent.content)
    assert body["stream"] is True  # 流式 wire 开关由适配器自己钉死
    assert body["model"] == "qwen-max"
    assert body["messages"] == [{"role": "user", "content": "你好"}]


def test_chat_endpoint_returns_502_when_dashscope_rejects_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：首块前上游非 2xx → 502，detail 带上游摘要——与非流式同一失败形状（checklist 3）。

    怎么证明：假上游回 401 + 错误 JSON，发 stream=true 的 POST，断言 502 且 detail
    同时含状态码 401 与上游原文 "Invalid API key provided"（摘要直读）。首块前是可
    报错窗口——错误必须用状态码说话，不许降级成"200 + 半截流"（拒答伪装成回答），
    也不许炸 500（把上游的拒绝说成网关的故障）。
    """
    monkeypatch.setattr("app.main.provider", _make_provider(_reject_with_401))

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 502
    detail = response.json()["detail"]
    # 行级：两条断言各钉一半——失败翻译成 502（状态码），上游原文不丢（摘要内容）
    assert "401" in detail
    assert "Invalid API key provided" in detail


async def test_dashscope_stream_wraps_transport_failure_as_upstream_error() -> None:
    """证明：流式腿连不上时也不裸抛 httpx 异常——翻译成 UpstreamError 且带根因（spec 失败形状）。

    怎么证明：假上游 handler 抛 ConnectError（模拟拒绝连接/网络断），断言 chat_stream
    抛的是 UpstreamError 而非 httpx 异常，且消息含根因原文。若不翻译，客户端会吃
    裸 500（网关自己的异常），分不清"网关坏了"还是"够不着上游"——与非流式 chat()
    的同名测试同一口径（spec：失败形状沿用 UpstreamError，两条腿一个词汇）。
    """

    def refuse_connection(request: httpx.Request) -> httpx.Response:
        """假上游连不上：handler 抛异常等价于真网络的连接失败（test_dashscope 同款）。"""
        raise httpx.ConnectError("connection refused")

    with pytest.raises(UpstreamError) as exc_info:
        [c async for c in _make_provider(refuse_connection).chat_stream(_make_request())]

    # 类型断言=失败翻译进协议；内容断言=根因不被吞——"为什么连不上"必须可读
    assert "connection refused" in str(exc_info.value)


async def test_dashscope_stream_transport_failure_carries_transient_shape() -> None:
    """证明：流式腿的连接失败同样携带**可重试**形状——与非流式同一失败分类（issue 04）。

    怎么证明：假上游 handler 抛 ConnectError，断言 chat_stream 抛的是
    TransientUpstreamError（瞬态子类：链在换路窗口里重试 1 次）。与 chat() 的同名
    测试同一口径（两腿有意双写翻译、词汇不分家）：反例是流式腿把毛刺当拒答，
    同一请求两种消费方式的重试行为分叉（故事 2）。
    """

    def refuse_connection(request: httpx.Request) -> httpx.Response:
        """假上游连不上：handler 抛异常等价于真网络的连接失败（test_dashscope 同款）。"""
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TransientUpstreamError) as exc_info:
        # 复杂语句（async 推导式）行上：把整条流消费干净——瞬态失败在开工时抛出，必须吃到它
        [c async for c in _make_provider(refuse_connection).chat_stream(_make_request())]

    assert isinstance(exc_info.value, UpstreamError)  # 行级：仍属答不上族——502 出口不变
    assert "connection refused" in str(exc_info.value)  # 行级：根因原文不丢


async def test_dashscope_stream_raises_upstream_error_on_malformed_data_line() -> None:
    """证明：坏帧不裸抛 pydantic 错误——翻成 UpstreamError（评审发现的失败形状漏网）。

    怎么证明：假上游回一行非法 JSON 的 data: 行，断言 chat_stream 抛 UpstreamError
    且消息里带肇事行残片。反例是 ValidationError 裸漏：首块前会穿成 500（把上游的
    脏数据说成网关自己的故障），违反 spec"失败形状沿用 UpstreamError（首块前适用）"。
    """

    def reply_with_malformed_frame(request: httpx.Request) -> httpx.Response:
        """假上游发脏帧：data: 后跟截断的 JSON——模拟连接半路切出的碎帧。"""
        return httpx.Response(200, text="data: {broken\n\ndata: [DONE]\n\n")

    with pytest.raises(UpstreamError) as exc_info:
        [c async for c in _make_provider(reply_with_malformed_frame).chat_stream(_make_request())]

    # 行级：类型=失败进协议形状；残片=肇事行不被吞（同"上游原文不丢"的摘要纪律）
    assert "{broken" in str(exc_info.value)


async def test_dashscope_stream_raises_upstream_error_with_upstream_message() -> None:
    """证明：流式腿被拒时抛 UpstreamError 且带状态码 + 上游原文——
    协议失败形状与 chat() 同（checklist 3 缝上细查）。

    怎么证明：假上游回 401 + 错误报文，断言异常类型是 UpstreamError（协议的失败形状），
    消息里同时含 401 与原文 "Invalid API key provided"——客户端据此分清 key 错 /
    请求错 / 上游挂了。与 test_dashscope 的非流式同名测试同一口径：两条腿一个失败词汇。
    """
    with pytest.raises(UpstreamError) as exc_info:
        [c async for c in _make_provider(_reject_with_401).chat_stream(_make_request())]

    # 两条断言各钉一半：失败翻译成协议异常（类型），上游原文不丢（消息内容）
    assert "401" in str(exc_info.value)
    assert "Invalid API key provided" in str(exc_info.value)


async def test_dashscope_stream_translates_upstream_sse_into_unified_chunks() -> None:
    """证明：上游 SSE 方言翻成统一 chunk 且解析不出适配器——
    核心永不见 "data:" 字样（checklist 2 后半）。

    怎么证明：假上游回 4 帧 data: 行 + 上游 [DONE]，收集 chat_stream 的产物——
    每件都是 ChatCompletionChunk 实例（不是 bytes/str/半解析的 dict，方言没漏出来）；
    恰好 4 块（[DONE] 正确终止流、不多产假块）；created 多余字段被静默丢弃不炸校验
    （Pydantic 默认行为）；内容拼回=字面量 UPSTREAM_TEXT（数据穿过 seam 一块不丢）。
    """
    chunks = [
        c async for c in _make_provider(_reply_with_upstream_sse).chat_stream(_make_request())
    ]

    # 行级：实例类型断言=方言不外泄的字面证明——出口是统一模型，不是 wire 残渣
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    assert len(chunks) == len(UPSTREAM_FRAMES)  # [DONE] 正好收尾：不多产一块
    assert chunks[0].id == "chatcmpl-up"
    assert all(c.model == "qwen-max" for c in chunks)
    assert chunks[0].choices[0].delta.role == "assistant"
    # 复杂语句（推导式）行上：各块 delta.content 拼回完整回答，与端到端同一字面真源
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == UPSTREAM_TEXT
    assert chunks[-1].choices[0].finish_reason == "stop"
