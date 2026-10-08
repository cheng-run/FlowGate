"""POST /v1/chat/completions 端点测试：端到端接 fake / DashScope 上游（W1 收尾刀）。

测试对应五站走位的几站：第 2 站输入检查（非法体被 422 挡下）、第 3~4 站
（请求穿过 seam 到适配器、响应按 OpenAI 形状回来），以及换上游后契约不变
（spec 故事 17/2：装配切换的证明落在 HTTP 契约缝上）。
"""

import httpx
import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：测试跑的是真实应用（含装配处），不是 mock。
from app.main import app
from providers.base import UpstreamError
from providers.dashscope import DashScopeProvider
from tests.conftest import auth_headers

# 默认头带套件级 TEST_KEY（W4 认证落地后的机械件）：本文件行为断言一字不改
client = TestClient(app, headers=auth_headers())


def _payload() -> dict:
    """最小合法请求体——用 dict 直发，模拟真实 HTTP 客户端（不经 Pydantic 预构造）。"""
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }


class _AlwaysFailingProvider:
    """协议失败的最小假实现：chat() 必抛 UpstreamError——测网关的失败翻译，不测适配器。

    为什么不用 DashScopeProvider+MockTransport 造失败：绑 DashScope 的端到端
    spec 只约定 1 条（故事 17，即下方透传测试）；上游原文进异常消息这件事
    已由 tests/test_dashscope.py 的适配器级测试钉住，这里只证 HTTP 出口的翻译。
    """

    name = "stub-failing"

    async def chat(self, request):
        """抛一个带上游原文的协议失败——模拟适配器从 401 拒答翻译来的消息形态。"""
        raise UpstreamError("上游 stub 返回 401: Invalid API key provided")


def test_chat_endpoint_returns_openai_shape() -> None:
    """证明：POST /v1/chat/completions 全链路通，且按 OpenAI 形状回答。

    怎么证明：发真实形状的 POST（TestClient 内存执行），断言 200 + object 字段 +
    回答里带 fake 回显——请求体→路由→Provider 协议→适配器→响应，每一环都穿过了。
    """
    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-model"
    # 回显断言：内容既带 fake 前缀也带回请求里的话——证明数据流真穿过了 seam
    assert "fake-reply" in body["choices"][0]["message"]["content"]
    assert "你好" in body["choices"][0]["message"]["content"]


def test_chat_endpoint_rejects_invalid_body() -> None:
    """证明：非法请求体被挡在门外（五站走位第 2 站：输入检查）。

    怎么证明：不带 messages 发 POST，断言 422——Pydantic 在进路由前就把形状守住，
    将来接真上游时，脏请求永远不会漏到 DashScope 去。
    """
    response = client.post("/v1/chat/completions", json={"model": "fake-model"})

    assert response.status_code == 422


def test_chat_endpoint_passes_dashscope_response_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：装配换成 DashScope 后 HTTP 契约零变更——客户端代码不用改（故事 17/2/5）。

    怎么证明：进程内把装配绑定 monkeypatch 成带 MockTransport 的 DashScopeProvider
    （spec 指定的做法：不加生产钩子，端到端靠改装配绑定），发同一条 POST，断言
    200 + 上游报文里的模型名/回答/usage 逐字段穿到客户端——请求体→路由→协议→
    适配器→上游→原路返回，整条链换上游前后走的是同一套形状。
    """

    def reply_with_upstream_payload(request: httpx.Request) -> httpx.Response:
        """假上游：回 qwen 风格完整报文——字段值刻意异于请求，证明数据真来自上游。"""
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-e2e",
                "object": "chat.completion",
                "model": "qwen-max",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "来自真实上游的回答"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            },
        )

    # monkeypatch 改的是装配绑定（进程内、测试后自动还原），生产代码为此零改动
    monkeypatch.setattr(
        "app.main.provider",
        DashScopeProvider(
            api_key="sk-test",
            base_url="https://upstream.test/v1",
            transport=httpx.MockTransport(reply_with_upstream_payload),
        ),
    )

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    body = response.json()
    # OpenAI 形状不因上游而变（故事 2）：客户端靠 object 字段吃饭
    assert body["object"] == "chat.completion"
    # 模型名与 usage 来自上游报文（故事 8/9）：透传到 HTTP 出口，一个字段不丢
    assert body["model"] == "qwen-max"
    assert body["choices"][0]["message"]["content"] == "来自真实上游的回答"
    assert body["usage"]["total_tokens"] == 8


def test_chat_endpoint_returns_502_when_upstream_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：上游拒答不穿透给客户端、也不炸成 500——网关回 502 且 detail 带失败信息（故事 7）。

    怎么证明：装配绑定换成必抛 UpstreamError 的假协议实现，发同一条 POST，
    断言 502 + detail 原样含异常消息（含上游原文）。502 语义=网关活着、上游答不上；
    消息里"上游原文不丢"由适配器级测试保证，这里证翻译出口不截断、不改写。
    """
    monkeypatch.setattr("app.main.provider", _AlwaysFailingProvider())

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502
    # detail 一字不差来自 UpstreamError 消息（排错时根因可直读）
    assert "Invalid API key provided" in response.json()["detail"]
