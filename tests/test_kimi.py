"""Kimi 适配器测试：协议满足 + 双向翻译（spec 主缝 Provider.chat()，issue 08 非流式腿）。

全部离线：httpx.MockTransport 注入假网络——不碰真网、不需要 key，
"测试全绿"在任何机器上可复跑（与 test_dashscope.py 同一纪律）。
wire（发往上游的字节）是适配器的外部输出，经 mock handler 捕获后断言——
不测任何内部私有函数（spec Testing Decisions）。
真网 smoke 在 tests/test_kimi_live.py（live 标记、无 key 自动跳过）。
"""

import json
from collections.abc import Callable

import httpx
import pytest

from app.schemas import ChatMessage, ChatRequest
from providers.base import Provider, TransientUpstreamError, UpstreamError
from providers.kimi import KimiProvider

# 测试专用的假 key / 假地址：只进 mock 断言，永不触网（与 test_dashscope 同款）
TEST_KEY = "sk-test-kimi"
TEST_BASE_URL = "https://kimi-relay.test/v1"

# 带完整字段的假上游报文（独立于任何测试的字面真源）：含 usage 透传（故事 8）、
# model 透传（故事 9），以及中转站可能多给的字段（owned_by）——统一模型没定义它，
# 翻译时应被静默丢弃而不是炸校验（OpenAI 兼容中转的常见富余字段，钉住 Pydantic 默认行为）。
UPSTREAM_SUCCESS_JSON = {
    "id": "chatcmpl-kimi-abc",
    "object": "chat.completion",
    "model": "kimi-k3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "真实回答"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    "owned_by": "moonshot",
}


def _make_request() -> ChatRequest:
    """最小合法请求，与 test_dashscope 同款——让跨文件的测试读起来是同一套语言。"""
    return ChatRequest(
        model="kimi-k3",
        messages=[ChatMessage(role="user", content="你好")],
    )


def _make_provider(handler: Callable[[httpx.Request], httpx.Response]) -> KimiProvider:
    """按给定假上游 handler 配出适配器——假 key/假地址/transport 注入聚成一处（去重复）。

    base_url 由测试显式给定（Kimi 适配器不内置默认地址——真实中转地址只活在 .env，
    见 providers/kimi.py 模块 docstring），假地址只进 wire 断言、永不触网。
    """
    return KimiProvider(
        api_key=TEST_KEY,
        base_url=TEST_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


def test_kimi_provider_satisfies_provider_protocol() -> None:
    """证明：Kimi 与 fake/DashScope 满足同一个 Provider 协议——换上游不动核心（ADR-0001）。

    怎么证明：isinstance(KimiProvider(...), Provider)。它不继承任何基类，仅凭
    name + chat 就过协议——第二真实上游过同一条判据，seam 不是为某一个适配器特供的。
    """
    assert isinstance(KimiProvider(api_key=TEST_KEY, base_url=TEST_BASE_URL), Provider)


async def test_kimi_chat_sends_wire_request_to_upstream() -> None:
    """证明：统一请求被翻成正确的 wire——URL / Bearer 鉴权 / OpenAI 形状 JSON（故事 16 同款）。

    怎么证明：假上游 handler 把收到的 Request 捕获下来，逐项断言 path 以
    /chat/completions 收尾、Authorization 是 Bearer <key>、body 的 model 与
    messages 原样到位。这三条钉住"方言边界"不被改坏——URL、鉴权头、报文布局
    正是适配器独占的知识（故事 13），中转方言不出本文件。
    """
    captured: list[httpx.Request] = []

    def capture_wire_and_ack(request: httpx.Request) -> httpx.Response:
        """假上游：存下收到的请求字节供事后断言，回固定成功报文让 chat() 跑完。"""
        captured.append(request)
        return httpx.Response(200, json=UPSTREAM_SUCCESS_JSON)

    await _make_provider(capture_wire_and_ack).chat(_make_request())

    # 断言全部落在"发出去的字节"上——wire 是适配器的外部输出，不碰它内部任何属性
    sent = captured[0]
    # base_url 精确命中：URL 拼接以装配处注入的地址为准（Kimi 无内置地址，见 docstring）
    assert str(sent.url) == f"{TEST_BASE_URL}/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {TEST_KEY}"
    body = json.loads(sent.content)
    # 模型名透传（故事 9）：客户端写 kimi-k3，上游就得收到 kimi-k3，不改写不加前缀
    assert body["model"] == "kimi-k3"
    assert body["messages"] == [{"role": "user", "content": "你好"}]


async def test_kimi_chat_translates_upstream_response() -> None:
    """证明：上游 JSON 被翻回统一形状，且 id / 模型名 / usage 原样透传（故事 2/8/9）。

    怎么证明：假上游回 UPSTREAM_SUCCESS_JSON（含真实 usage 数字的完整报文），
    断言统一响应逐字段对上——total_tokens=8 原样穿过来（fake 只会回 0，这条测的
    就是"真上游的数据通路"），多余字段 owned_by 不炸校验。
    """

    def reply_with_success_payload(request: httpx.Request) -> httpx.Response:
        """假上游：原样回固定报文，逼适配器做"wire → 统一"的翻译。"""
        return httpx.Response(200, json=UPSTREAM_SUCCESS_JSON)

    response = await _make_provider(reply_with_success_payload).chat(_make_request())

    # 逐字段透传断言：id/模型名/回答内容/usage 全部来自上游，适配器不改写不吞数据
    assert response.id == "chatcmpl-kimi-abc"
    assert response.model == "kimi-k3"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "真实回答"
    # usage 透传是计费的数据地基（故事 8）：3+5=8 原样到达，一个字段都不能丢
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 8


async def test_kimi_chat_raises_upstream_error_with_upstream_message() -> None:
    """证明：上游拒绝时不静默——抛 UpstreamError，且上游原始信息带在身上（故事 7 同款）。

    怎么证明：假上游回 401 + 中转风格的错误 JSON，断言异常类型是 UpstreamError
    （协议的失败形状），消息里同时含状态码 401 与上游原文——错误 JSON 只以摘要
    形态出适配器（ADR-0002：核心永不需要认识任何上游的错误报文）。
    """

    def reject_with_401(request: httpx.Request) -> httpx.Response:
        """假上游拒答：401 + 中转风格的错误 JSON。"""
        return httpx.Response(401, json={"error": {"message": "Invalid API key provided"}})

    with pytest.raises(UpstreamError) as exc_info:
        await _make_provider(reject_with_401).chat(_make_request())

    # 两条断言各钉一半：失败翻译成协议异常（类型），上游原文不丢（消息内容）
    assert "401" in str(exc_info.value)
    assert "Invalid API key provided" in str(exc_info.value)


async def test_kimi_chat_wraps_transport_failure_as_upstream_error() -> None:
    """证明：连接失败/超时不裸抛 httpx 异常——翻译成 UpstreamError 且带根因（故事 6 同款）。

    怎么证明：假上游 handler 直接抛 ConnectError（模拟拒绝连接/网络断），
    断言 chat() 抛的是 UpstreamError 而非 httpx 异常，且消息里含根因原文。
    若不翻译，客户端会吃一个裸 500（网关自己的异常），而不是"网关活着、
    上游够不着"的清晰信号（story 24：Kimi 失败与任何上游同一套语义）。
    """

    def refuse_connection(request: httpx.Request) -> httpx.Response:
        """假上游连不上：handler 抛异常等价于真网络的连接失败。"""
        raise httpx.ConnectError("connection refused")

    with pytest.raises(UpstreamError) as exc_info:
        await _make_provider(refuse_connection).chat(_make_request())

    # 类型断言=失败翻译进协议；内容断言=根因不被吞——排错时"为什么连不上"必须可读
    assert "connection refused" in str(exc_info.value)


async def test_kimi_chat_transport_failure_carries_transient_shape() -> None:
    """证明：连接失败携带**可重试**形状——与 DashScope 同一失败分类（issue 04 词汇）。

    怎么证明：假上游 handler 抛 ConnectError，断言 chat() 抛的是
    TransientUpstreamError（UpstreamError 的瞬态子类：同 502 出口、但链会重试 1 次）。
    反例是传输层失败与拒答共用一个形状：链分不出"够不着"（毛刺，该重试）与
    "被拒了"（判决，该换路）——fallback 链上 Kimi 的重试行为会和拒答分叉。
    """

    def refuse_connection(request: httpx.Request) -> httpx.Response:
        """假上游连不上：handler 抛异常等价于真网络的连接失败。"""
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TransientUpstreamError) as exc_info:
        await _make_provider(refuse_connection).chat(_make_request())

    assert isinstance(exc_info.value, UpstreamError)  # 行级：仍属答不上族——502 出口不变
    assert "connection refused" in str(exc_info.value)  # 行级：根因原文不丢


async def test_kimi_chat_raises_upstream_error_on_malformed_success_body() -> None:
    """证明：2xx 但报文坏也不裸穿 500——翻成 UpstreamError（评审补网：失败只出 UpstreamError）。

    怎么证明：假上游回 200 + 截断的 JSON，断言 chat() 抛 UpstreamError 且消息带
    报文残片。反例是 JSONDecodeError/ValidationError 裸漏——首块前/整答路径会穿成
    500，把上游的脏报文说成网关自己的故障（流式腿的坏帧护栏早有先例，两腿必须同口径）。
    """

    def reply_with_broken_body(request: httpx.Request) -> httpx.Response:
        """假上游 200 但报文是碎的：连接半路切出的截断 JSON。"""
        return httpx.Response(200, text='{"id": "chatcmpl-bro')

    with pytest.raises(UpstreamError) as exc_info:
        await _make_provider(reply_with_broken_body).chat(_make_request())

    # 行级：类型=失败进协议形状；残片=坏报文不被吞（同"上游原文不丢"的摘要纪律）
    assert '{"id": "chatcmpl-bro' in str(exc_info.value)
