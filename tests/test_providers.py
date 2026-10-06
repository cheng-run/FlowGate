"""providers/ 模块的测试：统一接口 + fake 上游（模块 1/9，考点=抽象 seam 放哪）。

三个测试分别证明三件事：seam 真实存在（同一接口）、接口契约固定（OpenAI 形状）、
请求内容真的穿过了 seam（不是假实现自说自话）。
"""

from app.schemas import ChatMessage, ChatRequest
from providers.base import Provider
from providers.fake import FakeProvider


def _make_request() -> ChatRequest:
    """造一个最小合法请求，避免每个测试重复拼装（测试也要看懂，抽出来一眼明了）。"""
    return ChatRequest(
        model="fake-model",
        messages=[ChatMessage(role="user", content="你好")],
    )


def test_fake_provider_satisfies_provider_protocol() -> None:
    """证明：fake 上游与将来的真���上游走同一个接口，seam 真的存在。

    怎么证明：isinstance(fake, Provider)。Provider 是 @runtime_checkable 的
    Protocol，FakeProvider 不继承任何基类，仅凭"有 name + chat 方法"就满足协议。
    将来 DashScope/Kimi 适配器换进来，这行断言照样成立才叫"换得动"。
    """
    assert isinstance(FakeProvider(), Provider)


async def test_fake_provider_returns_openai_shape() -> None:
    """证明：chat() 的返回是 OpenAI chat.completion 形状（接口契约不漂移）。

    怎么证明：调用后逐字段断言 object/model/choices/usage 齐全，
    choices[0] 是 assistant 消息且 finish_reason=stop——客户端靠这套形状吃饭。
    """
    response = await FakeProvider().chat(_make_request())

    assert response.object == "chat.completion"
    assert response.model == "fake-model"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens >= 0


async def test_fake_provider_echoes_last_user_message() -> None:
    """证明：请求内容真的穿过了 seam——fake 的答复里含请求里的话。

    怎么证明：请求消息说"你好"，断言响应内容含"你好"与回显前缀。
    若 fake 只回固定句，这条会红——防的就是"实现自说自话、接口形同虚设"。
    """
    response = await FakeProvider().chat(_make_request())

    assert "fake-reply" in response.choices[0].message.content
    assert "你好" in response.choices[0].message.content
