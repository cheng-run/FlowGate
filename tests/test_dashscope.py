"""DashScope 适配器测试：协议满足 + 双向翻译（spec 主缝 Provider.chat()）。

全部离线：httpx.MockTransport 注入假网络——不碰真网、不需要 key，
"测试全绿"在任何机器上可复跑（spec 用户故事 15/22）。
wire（发往上游的字节）是适配器的外部输出，经 mock handler 捕获后断言——
不测任何内部私有函数（spec Testing Decisions）。
"""

import json
from collections.abc import Callable

import httpx
import pytest

from app.schemas import ChatMessage, ChatRequest
from providers.base import Provider, TransientUpstreamError, UpstreamError
from providers.dashscope import DashScopeProvider

# 测试专用的假 key / 假地址：只进 mock 断言，永不触网
TEST_KEY = "sk-test-key"
TEST_BASE_URL = "https://upstream.test/v1"

# 带完整字段的假上游报文（独立于任何测试的字面真源）：含故事 8 要透传的 usage、
# 故事 9 要透传的 model，以及 OpenAI 兼容模式可能多给的字段（system_fingerprint）——
# 统一模型没定义它，翻译时应被静默丢弃而不是炸校验。
UPSTREAM_SUCCESS_JSON = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "model": "qwen-max",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "真实回答"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    "system_fingerprint": "fp_abc123",
}


def _make_request() -> ChatRequest:
    """最小合法请求，与 providers 测试同款——让跨文件的测试读起来是同一套语言。"""
    return ChatRequest(
        model="qwen-max",
        messages=[ChatMessage(role="user", content="你好")],
    )


def _make_provider(handler: Callable[[httpx.Request], httpx.Response]) -> DashScopeProvider:
    """按给定假上游 handler 配出适配器——假 key/假地址/transport 注入聚成一处（去重复）。"""
    return DashScopeProvider(
        api_key=TEST_KEY,
        base_url=TEST_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


def test_dashscope_provider_satisfies_provider_protocol() -> None:
    """证明：DashScope 与 fake 满足同一个 Provider 协议——换上游不动核心（ADR-0001）。

    怎么证明：isinstance(DashScopeProvider(...), Provider)。它不继承任何基类，
    仅凭 name + chat 就过协议——与 fake 同一条判据，说明这条 seam 不是为假实现特供的。
    """
    assert isinstance(DashScopeProvider(api_key=TEST_KEY), Provider)


async def test_dashscope_chat_sends_wire_request_to_upstream() -> None:
    """证明：统一请求被翻成正确的 wire——URL / Bearer 鉴权 / OpenAI 形状 JSON（故事 16）。

    怎么证明：假上游 handler 把收到的 Request 捕获下来，逐项断言 path 以
    /chat/completions 收尾、Authorization 是 Bearer <key>、body 的 model 与
    messages 原样到位。这三条钉住"方言边界"不被改坏——URL、鉴权头、报文布局
    正是适配器独占的知识（故事 13）。
    """
    captured: list[httpx.Request] = []

    def capture_wire_and_ack(request: httpx.Request) -> httpx.Response:
        """假上游：存下收到的请求字节供事后断言，回固定成功报文让 chat() 跑完。"""
        captured.append(request)
        return httpx.Response(200, json=UPSTREAM_SUCCESS_JSON)

    await _make_provider(capture_wire_and_ack).chat(_make_request())

    # 断言全部落在"发出去的字节"上——wire 是适配器的外部输出，不碰它内部任何属性
    sent = captured[0]
    # 自定义 base_url 精确命中：既证 URL 拼接，也证 DASHSCOPE_BASE_URL 可覆盖（故事 5 配套）
    assert str(sent.url) == f"{TEST_BASE_URL}/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {TEST_KEY}"
    body = json.loads(sent.content)
    # 模型名透传（故事 9）：客户端写 qwen-max，上游就得收到 qwen-max，不改写不加前缀
    assert body["model"] == "qwen-max"
    assert body["messages"] == [{"role": "user", "content": "你好"}]


async def test_dashscope_chat_translates_upstream_response() -> None:
    """证明：上游 JSON 被翻回统一形状，且 id / 模型名 / usage 原样透传（故事 2/8/9）。

    怎么证明：假上游回 UPSTREAM_SUCCESS_JSON（含真实 usage 数字的完整报文），
    断言统一响应逐字段对上——特别是 total_tokens=8 原样穿过来（fake 上游只会回 0，
    这条测的就是"真上游的数据通路"），多余字段 system_fingerprint 不炸校验。
    """

    def reply_with_success_payload(request: httpx.Request) -> httpx.Response:
        """假上游：原样回固定报文，逼适配器做"wire → 统一"的翻译。"""
        return httpx.Response(200, json=UPSTREAM_SUCCESS_JSON)

    response = await _make_provider(reply_with_success_payload).chat(_make_request())

    # 逐字段透传断言：id/模型名/回答内容/usage 全部来自上游，适配器不改写不吞数据
    assert response.id == "chatcmpl-abc"
    assert response.model == "qwen-max"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "真实回答"
    # usage 透传是 W3 计费的数据地基（故事 8）：3+5=8 原样到达，一个字段都不能丢
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 8


async def test_dashscope_chat_raises_upstream_error_with_upstream_message() -> None:
    """证明：上游拒绝时不静默——抛 UpstreamError，且上游原始信息带在身上（故事 7）。

    怎么证明：假上游回 401 + 错误报文，断言抛出的异常类型是 UpstreamError（协议的
    失败形状），异常消息里同时含状态码 401 与上游原文 "Invalid API key"——
    客户端据此能分清是 key 错、请求错还是上游挂了，而不是吃一个没头没脑的 500。
    """

    def reject_with_401(request: httpx.Request) -> httpx.Response:
        """假上游拒答：401 + DashScope 风格的错误 JSON。"""
        return httpx.Response(401, json={"error": {"message": "Invalid API key provided"}})

    with pytest.raises(UpstreamError) as exc_info:
        await _make_provider(reject_with_401).chat(_make_request())

    # 两条断言各钉一半：失败翻译成协议异常（类型），上游原文不丢（消息内容）
    assert "401" in str(exc_info.value)
    assert "Invalid API key provided" in str(exc_info.value)


async def test_dashscope_chat_wraps_transport_failure_as_upstream_error() -> None:
    """证明：连接失败/超时不裸抛 httpx 异常——翻译成 UpstreamError 且带根因（故事 6）。

    怎么证明：假上游 handler 直接抛 ConnectError（模拟拒绝连接/网络断），
    断言 chat() 抛的是 UpstreamError 而非 httpx 异常，且消息里含根因原文。
    若不翻译，客户端会吃一个裸 500（网关自己的异常），而不是"网关活着、
    上游够不着"的清晰信号——故事 6 的 "clear error" 就落空了。
    """

    def refuse_connection(request: httpx.Request) -> httpx.Response:
        """假上游连不上：handler 抛异常等价于真网络的连接失败。"""
        raise httpx.ConnectError("connection refused")

    with pytest.raises(UpstreamError) as exc_info:
        await _make_provider(refuse_connection).chat(_make_request())

    # 类型断言=失败翻译进协议；内容断言=根因不被吞——排错时"为什么连不上"必须可读
    assert "connection refused" in str(exc_info.value)


async def test_dashscope_chat_transport_failure_carries_transient_shape() -> None:
    """证明：连接失败携带**可重试**形状——"连接失败可重试"在生产路径成立，不只在测试桩上。

    怎么证明：假上游 handler 抛 ConnectError，断言 chat() 抛的是
    TransientUpstreamError（UpstreamError 的瞬态子类：同 502 出口、但链会重试 1 次）。
    反例是传输层失败与拒答共用一个形状：链分不出"够不着"（毛刺，该重试）与
    "被拒了"（判决，该换路），checklist 1 的"连接失败可重试"就只剩桩上成立。
    """

    def refuse_connection(request: httpx.Request) -> httpx.Response:
        """假上游连不上：handler 抛异常等价于真网络的连接失败。"""
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TransientUpstreamError) as exc_info:
        await _make_provider(refuse_connection).chat(_make_request())

    assert isinstance(exc_info.value, UpstreamError)  # 行级：仍属答不上族——502 出口不变
    assert "connection refused" in str(exc_info.value)  # 行级：根因原文不丢
