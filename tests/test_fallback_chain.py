"""非流式 fallback 链测试（fallback-ratelimit-billing issue 02）。

三条缝（spec 预先约定）：主缝 POST /v1/chat/completions（HTTP 出口的状态码/
detail/响应头）+ Provider 副缝（链门面的外部行为：isinstance、chat()、attempts
账面）+ fake 记录面（chat_endings，"谁被取消了"的可断言账本）。
全部离线：fake 双上游零外网；卡住用例把尝试预算 monkeypatch 到毫秒级，零真实 sleep。
"""

import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：主缝测试跑的是真实应用（含中间件），不是 mock。
from app.main import app
from app.schemas import ChatMessage, ChatRequest
from providers.base import Provider, UpstreamError
from providers.fake import FakeProvider
from routing.chain import AttemptTimeoutError, FallbackChain

client = TestClient(app)


def _payload() -> dict:
    """最小合法请求体——与 test_chat_endpoint 同款，用 dict 直发模拟真实 HTTP 客户端。"""
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }


def _make_request() -> ChatRequest:
    """最小合法请求——与 providers/streaming 测试同款，跨文件读起来是同一套语言。"""
    return ChatRequest(
        model="fake-model",
        messages=[ChatMessage(role="user", content="你好")],
    )


# ===== 切片 A：链门面满足 Provider 协议（checklist 1）=====


def test_chain_satisfies_provider_protocol() -> None:
    """证明：fallback 链对上层就是一个上游——isinstance 可验（checklist 1）。

    怎么证明：FallbackChain 不继承任何基类，仅凭 name + chat + chat_stream 就过
    isinstance(chain, Provider)——路由继续对着协议说话，链换进去零改动
    （ADR-0001：核心只见协议不见实现，链是"另一个实现"）。
    """
    chain = FallbackChain([FakeProvider(name="fake-a"), FakeProvider(name="fake-b")])

    assert isinstance(chain, Provider)


def test_chain_rejects_empty_provider_list() -> None:
    """证明：空链响亮报错——配置错误不静默变成"永远 502 的链"（配错喊响纪律）。

    怎么证明：传空列表，断言 ValueError。反例是允许空链跑到请求时才炸——
    那错误现场离配置现场十万八千里，悬案难查。
    """
    with pytest.raises(ValueError):
        FallbackChain([])


# ===== 切片 B：A 挂 B 接管 + attempts 记录（checklist 3）=====


async def test_falls_back_to_next_upstream_when_first_fails() -> None:
    """证明：A 挂 B 接管，客户端只看到一次成功；attempts 记下两段各自的结果/死因（checklist 3）。

    怎么证明：fake-a 注入恒失败、fake-b 正常答，走 Provider 副缝调 chain.chat()——
    断言拿到的是 fake-b 的答复（id 带 fake-b 名），且 attempts 恰两条：fake-a 记
    failed + 失败形状 UpstreamError + 原错消息，fake-b 记 ok。反例是把 A 的错误
    直接抛给客户端——那客户端看到的是失败，不是"接管"。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b")]
    )

    response = await chain.chat(_make_request())

    assert response.id.startswith("fake-b")  # 行级：B 接管的证据——答复来自第二棒
    # 复杂语句（推导式）行上：attempts 按序记两段——谁、什么结果（otari resolve 的形状）
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-b", "ok"),
    ]
    first = chain.attempts[0]
    assert first.failure_shape == "UpstreamError"  # 行级：失败形状=换路信号的词汇（字面量独立真源）
    assert "fake 注入的恒失败形态" in first.detail  # 行级：死因原文可直读，不被链改写


# ===== 切片 C：每逻辑请求一个 request_id（checklist 5）=====


async def test_attempts_share_one_request_id_per_logical_request() -> None:
    """证明：一个逻辑请求一个 request_id——一轮内的尝试共用一个号，轮与轮不重号（checklist 5）。

    怎么证明：同一条链跑两轮"接管"请求，断言每轮内部的 attempts 共用同一个非空
    request_id，且两轮的号不同。对账靠它：响应头的号 → attempts 的号 → 将来账本的号，
    三处对得上（响应头那一环在主缝切片验证）。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b")]
    )

    await chain.chat(_make_request())
    await chain.chat(_make_request())

    first_round = chain.attempts[:2]  # 行级：切片取第一轮的尝试（每轮两棒，按序截取）
    second_round = chain.attempts[2:]  # 行级：切片取第二轮——与第一轮对照"轮与轮不重号"
    assert first_round[0].request_id == first_round[1].request_id  # 行级：一轮的尝试共用一号
    assert first_round[0].request_id  # 行级：号非空
    assert second_round[0].request_id == second_round[1].request_id
    # 行级：轮与轮不同号——"每逻辑请求一个"，号才是幂等键（重号=重复扣费的温床）
    assert first_round[0].request_id != second_round[0].request_id


# ===== 切片 D：全链失败——最后一棒的形状说话 + attempts 摘要（checklist 4）=====


async def test_all_fail_raises_last_shape_with_attempts_summary() -> None:
    """证明：全链失败按最后一棒的失败形状说话，message 附 attempts 摘要、死因直读（checklist 4）。

    怎么证明：两棒都注入恒失败（消息各带自己的名字），断言抛出的是 UpstreamError
    （最后一棒的形状，502 的原料），且 str 里含两个上游名与死因原文——detail 就是
    这个 str（502 出口在主缝切片验证）。反例是只抛最后一棒的错：前一棒的死因
    从此消失在日志之外，"每一段尝试为什么死"无处可查。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b", failure="fail")]
    )

    with pytest.raises(UpstreamError) as exc_info:
        await chain.chat(_make_request())

    text = str(exc_info.value)
    assert "fake-a" in text and "fake-b" in text  # 行级：每段尝试都现身（attempts 摘要）
    assert "fake 注入的恒失败形态" in text  # 行级：死因原文在摘要里可直读


# ===== 切片 E：放弃的尝试显式取消 + 尝试预算（checklist 6）=====


async def test_abandons_hanging_attempt_with_explicit_cancel_when_next_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：卡住的 A 在尝试预算内被放弃且**显式取消**（fake 记"被取消"），B 接管（checklist 6）。

    怎么证明：尝试预算 monkeypatch 到毫秒级（零真实 sleep），fake-a 卡住、fake-b
    正常——断言答复来自 b，fake-a 的 chat_endings 记 "cancelled"（取消真落进了
    上游代码，不是干等它烧钱），attempts 里 a 记 timeout、b 记 ok。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.05)
    fake_a = FakeProvider(name="fake-a", failure="hang")
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([fake_a, fake_b])

    response = await chain.chat(_make_request())

    assert response.id.startswith("fake-b")  # 行级：B 接管——被放弃的 A 没拖住整条链
    # 行级：显式取消的证据——取消穿进 fake 的挂起点、收尾账记"被取消"（非干等烧钱）
    assert fake_a.chat_endings == ["cancelled"]
    # 行级：A 记 timeout（被预算放弃）、B 记 ok——attempts 把"放弃"与"失败"分开记
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "timeout"),
        ("fake-b", "ok"),
    ]


# ===== 切片 F：全链失败的形状语义（checklist 4）=====


async def test_all_hang_raises_attempt_timeout_with_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：全链卡死按超时族说话（AttemptTimeoutError→504 的原料），摘要含每段死因（checklist 4）。

    怎么证明：两棒都卡住、尝试预算打到毫秒级，断言抛的是 AttemptTimeoutError 而不是
    UpstreamError——形状说话：超时族 → 504，拒答族 → 502（出口在主缝切片验证）；
    str 含两个上游名，detail 一读全貌。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.05)
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="hang"), FakeProvider(name="fake-b", failure="hang")]
    )

    with pytest.raises(AttemptTimeoutError) as exc_info:
        await chain.chat(_make_request())

    text = str(exc_info.value)
    assert "fake-a" in text and "fake-b" in text  # 行级：两段尝试都现身（attempts 摘要）


async def test_last_attempt_shape_speaks_when_shapes_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：形状混搭时由**最后一棒**说话——不是"沾过超时就算超时族"（checklist 4 的字面语义）。

    怎么证明：两组混搭链，预算打到毫秒级：前超时后拒答 → UpstreamError；前拒答后
    超时 → AttemptTimeoutError。反例实现是"任意一棒超时就抛超时族"——那前超时后拒答
    会被误标 504，客户端按"值得等会再试"重试一条实际被拒答的请求。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.05)

    # 行级：A 卡死（超时族）、B 拒答——最后一棒是拒答，502 的原料
    hang_first = FallbackChain(
        [FakeProvider(name="fake-a", failure="hang"), FakeProvider(name="fake-b", failure="fail")]
    )
    with pytest.raises(UpstreamError):
        await hang_first.chat(_make_request())

    # 行级：A 拒答、B 卡死——最后一棒是超时族，504 的原料
    fail_first = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b", failure="hang")]
    )
    with pytest.raises(AttemptTimeoutError):
        await fail_first.chat(_make_request())


# ===== 切片 G：主缝出口——200/502/504 + X-Request-Id 对账（checklist 3/4/5）=====


def test_takeover_returns_200_with_request_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：接管在 HTTP 出口是"一次成功的 200"，响应头带 request_id 供对账（checklist 3/5）。

    怎么证明：装配绑定换成"fake-a 恒失败 + fake-b 正常"的链（monkeypatch 改装配绑定，
    生产零钩子），POST 同一请求——断言 200、答复来自 fake-b、响应头有 x-request-id，
    且 attempts 的 request_id 与响应头逐字相等：客户端拿头上的号就能对上链的账。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b")]
    )
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    rid = response.headers.get("x-request-id")
    assert rid  # 行级：供对账的号在响应头上（每逻辑请求一个）
    assert response.json()["id"].startswith("fake-b")  # 行级：接管的证据——答复来自 B
    # 复杂语句（推导式）行上：attempts 两段都挂在头上的号下——对账三处一线（头/链/将来账本）
    assert [a.request_id for a in chain.attempts] == [rid, rid]


def test_all_fail_returns_502_with_attempts_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：全链失败在 HTTP 出口是 502 且 detail 附 attempts 摘要——每段死因可直读（checklist 4）。

    怎么证明：两棒都恒失败的链走主缝 POST，断言 502、detail 含两个上游名与死因原文
    （detail 就是异常 str，摘要在链里拼好、出口一字不改），且失败响应头同样带
    x-request-id——失败也要有号可对账。
    """
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b", failure="fail")]
    )
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "fake-a" in detail and "fake-b" in detail  # 行级：attempts 摘要两段都现身
    assert "fake 注入的恒失败形态" in detail  # 行级：死因原文直读（detail 不截断不改写）
    assert response.headers.get("x-request-id")  # 行级：失败响应同样带号供对账


def test_all_hang_returns_504(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：全链卡死在 HTTP 出口是 504——超时族的语义出口（checklist 4）。

    怎么证明：两棒都卡住、预算打到毫秒级的链走主缝 POST，断言 504（不是 502——
    客户端对 504 的直觉是"值得等会再试"，对 502 是"该换路"）且 detail 带 attempts
    摘要。反例是 500：没翻译的异常才会 500，那等于告诉客户端"网关自己坏了"。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.05)
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="hang"), FakeProvider(name="fake-b", failure="hang")]
    )
    monkeypatch.setattr("app.main.provider", chain)

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 504
    detail = response.json()["detail"]
    assert "fake-a" in detail and "fake-b" in detail  # 行级：每段尝试的死因都在 detail 里
