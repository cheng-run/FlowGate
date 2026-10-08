"""流式 fallback 链测试（fallback-ratelimit-billing issue 03：重试窗口限死首 token 前）。

三条缝（spec 预先约定）：主缝 POST /v1/chat/completions（HTTP 出口的 SSE 字节序列/
状态码/响应头）+ Provider 副缝（链门面 chat_stream 的外部行为：chunk 序列、attempts
账面）+ fake 记录面（calls / stream_endings——"谁被调用了、谁被取消了"的可断言账本）。
全部离线：fake 双上游零外网；"首块后中途死亡"用流式桩（沿 test_streaming 的桩先例：
fake 的注入形态只管首块前的死法，中途回天是桩的活，fake 保持 issue 01 的词汇不动）。
"""

import json

import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：主缝测试跑的是真实应用（含中间件），不是 mock。
from app.main import app
from app.schemas import ChatCompletionChunk, ChatMessage, ChatRequest, DeltaMessage, StreamChoice
from providers.base import UpstreamError
from providers.fake import FakeProvider
from routing.chain import FallbackChain
from tests.conftest import auth_headers

# 默认头带套件级 TEST_KEY（W4 认证落地后的机械件）：本文件行为断言一字不改
client = TestClient(app, headers=auth_headers())


def _make_request() -> ChatRequest:
    """最小合法请求——与 providers/streaming 测试同款，跨文件读起来是同一套语言。"""
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


# ===== 切片 1：首块前 A 失败 → B 流接管 + attempts（checklist 1/5，Provider 副缝）=====


async def test_stream_takeover_when_first_fails_before_first_chunk() -> None:
    """证明：首块前 A 失败 → B 流接管，attempts 记下两段各自的结果/死因（checklist 1/5）。

    怎么证明：fake-a 注入恒失败、fake-b 正常，走 Provider 副缝消费 chain.chat_stream——
    断言全程只拿到 fake-b 的完整回显流（id 带 fake-b 名、内容拼回字面量真源），
    且 attempts 恰两条：fake-a 记 failed + 形状 UpstreamError + 死因原文，fake-b 记 ok。
    反例是把 A 的失败直接抛给客户端——那客户端看到的是失败，不是"接管"。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b")]
    )

    chunks = [chunk async for chunk in chain.chat_stream(_make_request())]

    assert chunks
    # 复杂语句（推导式+all）行上：全程只有 B 的帧——单流，不是 A/B 各半截的拼接
    assert all(c.id.startswith("fake-b") for c in chunks)
    # 行级：内容拼回字面真源——B 的完整回答一块不丢（数据穿过了换路接管）
    assembled = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert assembled == "fake-reply: 你好"
    # 复杂语句（推导式）行上：attempts 按序记两段——谁、什么结果（otari resolve 的形状）
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-b", "ok"),
    ]
    first = chain.attempts[0]
    assert first.failure_shape == "UpstreamError"  # 行级：失败形状=换路信号的词汇（字面量独立真源）
    assert "fake 注入的恒失败形态" in first.detail  # 行级：死因原文可直读，不被链改写


async def test_stream_takeover_when_first_stream_empty() -> None:
    """证明：首块前 A 一言不发（空流）也归"答不上"——换路信号，B 接管（checklist 1 边角）。

    怎么证明：fake-a 注入空流（零块收场）、fake-b 正常，消费 chain.chat_stream——
    断言拿到的是 fake-b 的完整回显流，attempts 记 a=failed + 形状 UpstreamError
    （与拒答同族，W2 的空流口径延续到链上）。反例是空流装成"接管成功"或直接炸
    RuntimeError——前者把上游的沉默当成回答，后者把上游的沉默说成网关的故障。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="empty"), FakeProvider(name="fake-b")]
    )

    chunks = [chunk async for chunk in chain.chat_stream(_make_request())]

    # 行级：内容拼回 = B 的回显（字面真源）——空流的 A 没有贡献任何一块
    assembled = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert assembled == "fake-reply: 你好"
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-b", "ok"),
    ]
    # 行级：空流的失败形状与拒答同族（UpstreamError → 502 的原料），不另造词汇
    assert chain.attempts[0].failure_shape == "UpstreamError"
    assert "首块前结束" in chain.attempts[0].detail  # 行级：死因说"一言不发"，不是随便一条错


async def test_stream_all_fail_raises_last_shape_with_attempts_summary() -> None:
    """证明：流式全链失败按最后一棒的形状说话，message 附 attempts 摘要（checklist 5 同源）。

    怎么证明：两棒都注入恒失败（消息各带自己的名字），消费 chain.chat_stream 时断言
    抛出的是 UpstreamError（最后一棒的形状，502 的原料），且 str 里含两个上游名与
    死因原文——与非流式同一条收尾函数（_raise_chain_failed），两条腿失败语义不分家。
    反例是只抛最后一棒的错：前一棒的死因从此消失，"每一段尝试为什么死"无处可查。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b", failure="fail")]
    )

    with pytest.raises(UpstreamError) as exc_info:
        # 行级：首块在换路窗口里取——失败就在这一步现形（流还没交出去，错误还能说话）
        await anext(chain.chat_stream(_make_request()))

    text = str(exc_info.value)
    assert "fake-a" in text and "fake-b" in text  # 行级：每段尝试都现身（attempts 摘要）
    assert "fake 注入的恒失败形态" in text  # 行级：死因原文在摘要里可直读


# ===== 切片 4：主缝出口——接管=完整单流（帧格式、[DONE] 与 W2 承诺一致，checklist 1）=====


def test_chat_endpoint_streams_complete_single_stream_after_takeover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：接管在 HTTP 出口是**完整单流**——帧格式、[DONE]、request_id 全与 W2 一致。

    怎么证明：装配绑定换成"fake-a 恒失败 + fake-b 正常"的链（monkeypatch 改装配
    绑定，生产零钩子），POST stream=true——断言 200 + text/event-stream、逐帧
    "data: {json}" 形状、内容拼回 = fake-b 的完整回显（字面真源）、末帧 [DONE]
    （完整流的完成记号），且响应头 x-request-id == attempts 里两条尝试共用的号。
    反例是把接管暴露给客户端（多一段 A 的帧/错误）或 [DONE] 缺席——那"接管"就
    变成了客户端可见的拼接（checklist 1/5）。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b")]
    )
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # 行级：SSE 帧以空行分隔——切开、丢掉结尾空串，剩下的就是一帧一帧的 payload
    frames = [f for f in response.text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"  # 行级：完整流必有完成记号（与 W2 一字不差）
    # 行级：剥掉 "data: " 前缀就是纯 JSON——逐帧解析成 chunk，形状在字节层钉住
    chunks = [json.loads(f.removeprefix("data: ")) for f in frames[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # 复杂语句（推导式+all）行上：全程只有 B 的帧——客户端看到的是单流，不是拼接
    assert all(c["id"].startswith("fake-b") for c in chunks)
    assembled = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert assembled == "fake-reply: 你好"  # 字面真源：B 的完整回答一块不丢
    rid = response.headers.get("x-request-id")
    assert rid  # 行级：供对账的号在响应头上（每逻辑请求一个）
    # 复杂语句（推导式）行上：attempts 两段都挂在头上的号下——对账三处一线（头/链/将来账本）
    assert [a.request_id for a in chain.attempts] == [rid, rid]


# ===== 切片 5：首 token 后绝不再换路——"双半截缝合"反例（checklist 2/6）=====


class _DyingMidStreamProvider:
    """吐一块就抛的流式桩：首块后中途死亡——"双半截缝合"与"绝不二次调用"的现场。

    为什么用桩不用 fake：fake 的注入形态只管首块前的死法（恒失败/卡住/空流，
    issue 01 词汇）；"首块后死"是另一门死法，沿 test_streaming 的 _ExplodingProvider
    桩先例在测试里现做，fake 的词汇与注释一行不动。
    """

    def __init__(self, name: str, content: str) -> None:
        """给桩起名并定死它吐的那一块内容——名字进 chunk id，内容是"半截可辨"的标记。"""
        self.name = name
        self._content = content
        self.calls: list[str] = []  # 行级：调用记录——"B 从未被惊动"靠空账本证明

    async def chat(self, request: ChatRequest):
        """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
        raise AssertionError("stream=true 不应走非流式 chat()")

    async def chat_stream(self, request: ChatRequest):
        """先给一块（首块到手=接管已成立），再抛错——死亡必须发生在"首块后"。"""
        self.calls.append("chat_stream")  # 行级：开工记账——被调用过没有、被调用几次
        yield ChatCompletionChunk(
            id=f"{self.name}-1",
            model=request.model,
            choices=[StreamChoice(delta=DeltaMessage(content=self._content))],
        )
        raise RuntimeError(f"{self.name} 流中途断掉")


def test_chat_endpoint_never_stitches_two_half_answers_when_first_dies_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """证明：A 首块后半路死 → 绝不换 B——客户端只收 A 的半截，绝无两段回答（checklist 6 反例）。

    怎么证明：链 = [吐一块就抛的桩 A, 正常 fake-b]，走主缝 POST stream=true——
    断言 200（流已承诺）+ 正文里有 A 的半截标记、**没有** B 的回显内容（两段缝合的
    字面反证）+ 无 [DONE]（半截可辨），fake-b.calls == []（B 从未被惊动），
    caplog 有 A 的死因。反例实现是"透传回路里死了就换下一棒"——那客户端会收到
    A 的半截 + B 的回答拼成的两段，正是本票要钉死的缝合事故。
    """
    fake_b = FakeProvider(name="fake-b")  # 行级：持有实例引用——"未被惊动"直接查它的账
    chain = FallbackChain([_DyingMidStreamProvider("stub-a", "A的半截"), fake_b])
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200  # 行级：流已承诺——中途死亡翻不成状态码
    assert "A的半截" in response.text  # 行级：A 的首块真到达过客户端——死亡在半路
    assert "fake-reply" not in response.text  # 行级：B 的内容绝无——两段缝合的字面反证
    assert "[DONE]" not in response.text  # 行级：半截回答绝不带完成记号（W2 承诺）
    assert fake_b.calls == []  # 行级：B 从未被调用——换路窗口在首块处关死
    assert "stub-a 流中途断掉" in caplog.text  # 行级：死因进日志（W2 截断语义）


def test_chat_endpoint_truncates_and_never_retries_first_when_second_dies_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """证明：B 首块后半路死 → 只截断不伪造 [DONE]，且 **A 不被二次调用**（checklist 2）。

    怎么证明：链 = [fake-a 恒失败, 吐一块就抛的桩 B]，走主缝 POST stream=true——
    断言 200 + 有 B 的半截标记 + 无 [DONE]（截断语义与 W2 一字不差），fake-a.calls
    恰一条 ["chat_stream"]（若 B 死后链回头再找 A，账本会出现第二条）。
    反例实现是"B 死了回去再问 A"——那是把"绝不再换路"当耳旁风，A 的第二次半截
    会把两段回答缝给客户端。
    """
    fake_a = FakeProvider(name="fake-a", failure="fail")  # 行级：持有实例引用——查"不被二次调用"
    chain = FallbackChain([fake_a, _DyingMidStreamProvider("stub-b", "B的半截")])
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200  # 行级：流已承诺——中途死亡翻不成状态码
    assert "B的半截" in response.text  # 行级：B 的首块真到达过客户端——死亡在半路
    assert "[DONE]" not in response.text  # 行级：半截回答绝不带完成记号（W2 承诺）
    # 行级：A 恰被调用一次（首块前那次失败）——B 死后绝无第二次调用（checklist 2 的字面验收）
    assert fake_a.calls == ["chat_stream"]
    assert "stub-b 流中途断掉" in caplog.text  # 行级：死因进日志（W2 截断语义）


def test_chat_endpoint_all_fail_stream_returns_502_with_attempts_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：流式全链失败在 HTTP 出口是 502 + attempts 摘要，失败响应同样带对账号（checklist 5）。

    怎么证明：两棒都恒失败的链走主缝 POST stream=true——断言 502（可报错窗口内
    错误用状态码说话，W2 契约）、detail 含两个上游名与死因原文、响应头带
    x-request-id 且与 attempts 的号对得上。反例是流式失败降级成"200 + 半截流"——
    那把拒答伪装成了正常回答，客户端会把失败当答案缓存。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b", failure="fail")]
    )
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "fake-a" in detail and "fake-b" in detail  # 行级：attempts 摘要两段都现身
    assert "fake 注入的恒失败形态" in detail  # 行级：死因原文直读（detail 不截断不改写）
    rid = response.headers.get("x-request-id")
    assert rid  # 行级：失败响应同样带号供对账
    # 复杂语句（推导式）行上：失败的尝试也挂在头上的号下——失败请求照样可对账
    assert [a.request_id for a in chain.attempts] == [rid, rid]


# ===== 切片 8：收尾链穿链而不断——弃流时上游显式收尾（W2"断连不泄漏"的延伸）=====


async def test_chain_stream_close_cancels_committed_upstream() -> None:
    """证明：消费方半路弃流（aclose 链的生成器）→ 已接管的上游被显式收尾、记"被取消"。

    怎么证明：无闸门 fake 走单元素链，取到首块后立即 aclose chain 的生成器——
    断言 fake.stream_endings == ["cancelled"]（不 gc、不 sleep：链的 finally 若漏了
    aclose 这一环，fake 会一直悬在自己的 yield 上等 GC，账本这一刻必是空的）。
    反例是链只顾透传忘了收尾——W2"断连不泄漏"会在链部署形态下静默破功。
    """
    fake = FakeProvider()  # 无闸门：首块直达，弃流时刻由测试精确拿住
    chain = FallbackChain([fake])

    stream = chain.chat_stream(_make_request())
    first = await anext(stream)  # 行级：取到首块=已接管，此刻弃流最能考验收尾链
    assert first.choices[0].delta.role == "assistant"
    await stream.aclose()  # 行级：弃流——GeneratorExit 沿链穿进 fake 的挂起点

    # 行级：显式收尾穿链的证据——无 gc、无 sleep，aclose 返回时账本已记"被取消"
    assert fake.stream_endings == ["cancelled"]
