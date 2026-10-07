"""超时预算拆分 + 可重试失败重试 1 次（fallback-ratelimit-billing issue 04）。

三条缝（spec 预先约定）：主缝 POST /v1/chat/completions（HTTP 出口的状态码/字节序列）
+ Provider 副缝（链门面的外部行为：chat()/chat_stream()/attempts 账面）+ fake 记录面
（calls / chat_endings / stream_endings——"谁被重试了、谁被取消了"的可断言账本）。
全部离线：fake/桩零外网；预算常量 monkeypatch 到毫秒级，零真实 sleep。
"""

import asyncio
import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

# 导入装配好的 app：主缝测试跑的是真实应用（含中间件），不是 mock。
from app.main import app
from app.schemas import ChatMessage, ChatRequest
from providers.base import TransientUpstreamError
from providers.fake import FakeProvider
from routing.chain import (
    ATTEMPT_TIMEOUT_SECONDS,
    TOTAL_BUDGET_SECONDS,
    AttemptTimeoutError,
    FallbackChain,
)

client = TestClient(app)


def _make_request() -> ChatRequest:
    """最小合法请求——与既有链测试同款，跨文件读起来是同一套语言。"""
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


class _FlakyProvider:
    """前 fail_times 次调用按注入异常瞬态失败、之后委托 fake 正常答的桩。

    为什么用桩不用 fake 的 failure 旋钮：fake 的注入形态是"恒定"的（恒失败/卡住/
    空流，issue 01 词汇），而重试要的现场是"毛刺——挂一次就好"；沿 test_streaming
    的桩先例（ADR-0005 备选 4）现做，fake 的词汇与注释一行不动。
    为什么委托 FakeProvider 答成功段：成功形状与 fake 一字不差（回显公式同源），
    测试的期望值继续对着字面真源说话，桩自己不新造答复形状。
    """

    def __init__(self, name: str, fail_times: int, fail: Callable[[], Exception]) -> None:
        """定死毛刺次数与毛刺形状工厂；名字经委托的 fake 进答复 id，供"谁答的"断言。

        为什么失败形状是工厂不是实例：一次桩要抛好几次错，复用同一异常实例会让
        traceback 跨尝试纠缠——每次现场新造一个，排错时各次毛刺互不掺杂。
        """
        self._inner = FakeProvider(name=name)
        self._fail_times = fail_times
        self._fail = fail
        self.calls: list[str] = []  # 行级：调用记录——"重试了没有/重试几次"按次数直查

    @property
    def name(self) -> str:
        """链按 name 记账——桩的名字与委托的 fake 同名（attempts/答复 id 断言靠它）。"""
        return self._inner.name

    async def chat(self, request: ChatRequest):
        """前 fail_times 次抛注入异常（瞬态毛刺），之后把请求转交 fake 正常答。"""
        self.calls.append("chat")  # 行级：开工记账——重试次数按 len(calls) 数
        if len(self.calls) <= self._fail_times:
            raise self._fail()
        return await self._inner.chat(request)

    async def chat_stream(self, request: ChatRequest):
        """流式腿同款毛刺：前 fail_times 次在首块前抛，之后透传 fake 的完整流。"""
        self.calls.append("chat_stream")
        if len(self.calls) <= self._fail_times:
            raise self._fail()
        # 行级：透传委托 fake 的流——成功段形状与 fake 一字不差（同 chat 的理由）
        async for chunk in self._inner.chat_stream(request):
            yield chunk


# ===== 切片 1：可重试失败同上游重试 1 次，重试成功即不换路（checklist 1/6）=====


async def test_transient_failure_retried_once_then_succeeds_without_fallback() -> None:
    """证明：瞬态失败（连接失败）同上游重试 1 次即吸收，不惊动下一棒；attempts 记两次尝试（1/6）。

    怎么证明：桩-a 第 1 次抛 TransientUpstreamError、第 2 次起委托 fake 正常答，
    fake-b 在后棒待命——断言答复来自 a（换路没发生）、b.calls == []（B 从未被惊动），
    attempts 恰两条同为 a：第 1 条 failed + 形状 TransientUpstreamError + 死因原文，
    第 2 条 ok（"内部损耗账将来按两次尝试入账"的数据源，checklist 6）。反例是把瞬态
    当拒答直接换路——一次毛刺就放弃上游 A，"重试吸收毛刺"（故事 5）无从谈起。
    """
    flaky_a = _FlakyProvider(
        "fake-a",
        fail_times=1,
        # 行级：连接失败=瞬态形态的字面现场（与 DashScope 传输层翻译同款措辞）
        fail=lambda: TransientUpstreamError("上游 fake-a 连接失败: connection refused"),
    )
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([flaky_a, fake_b])

    response = await chain.chat(_make_request())

    assert response.id.startswith("fake-a")  # 行级：重试成功=A 自己答完——换路根本没发生
    assert fake_b.calls == []  # 行级：B 从未被惊动（换路没发生的字面反证）
    assert flaky_a.calls == ["chat", "chat"]  # 行级：恰两次尝试——首发 + 重试 1 次
    # 复杂语句（推导式）行上：attempts 两次尝试都记账——谁、什么结果（otari resolve 的形状）
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-a", "ok"),
    ]
    first = chain.attempts[0]
    # 行级：失败形状=可重试词汇（字面量独立真源——常量改名测试必须红）
    assert first.failure_shape == "TransientUpstreamError"
    assert "connection refused" in first.detail  # 行级：死因原文可直读，不被链改写


async def test_transient_failure_twice_falls_back_after_two_tries() -> None:
    """证明：瞬态失败重试 1 次仍挂才换路——a 恰试两次，attempts 记两败一接管（checklist 1）。

    怎么证明：桩-a 连续两次抛 TransientUpstreamError（fail_times=2）、fake-b 正常——
    断言答复来自 b（换路发生），a.calls == ["chat", "chat"]（恰两次：首发 + 重试 1 次，
    不是无限重试也不是只试一次），attempts 三条：a 两败 + b ok。反例是重试无上限——
    恒发毛刺的上游会把整条总预算烧光在重试里，B 永远轮不到。
    """
    flaky_a = _FlakyProvider(
        "fake-a",
        fail_times=2,
        fail=lambda: TransientUpstreamError("上游 fake-a 连接失败: connection refused"),
    )
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([flaky_a, fake_b])

    response = await chain.chat(_make_request())

    assert response.id.startswith("fake-b")  # 行级：重试额度用完才换路——B 接管
    assert flaky_a.calls == ["chat", "chat"]  # 行级：恰两次尝试——重试额度=1 次（checklist 1）
    # 复杂语句（推导式）行上：attempts 三段——两次瞬态失败都入账（损耗账按此落库）+ 接管
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-a", "failed"),
        ("fake-b", "ok"),
    ]
    # 复杂语句（推导式+all）行上：两败的形状都是瞬态词汇——重试对的是同一类失败
    assert all(a.failure_shape == "TransientUpstreamError" for a in chain.attempts[:2])


async def test_refusal_is_never_retried() -> None:
    """证明：拒答类失败（请求性失败）不重试、直接换路——a 恰被调一次（checklist 1 后半）。

    怎么证明：fake-a 注入恒失败（拒答形态，光杆 UpstreamError）、fake-b 正常——
    断言 a.calls == ["chat"]（恰一次：若拒答也被重试会出现第二条），b 接管，
    attempts 恰两条（a 一败 + b ok）。反例是"失败就一律重试"——对 4xx/5xx 判决
    重试只是把注定失败的请求再来一遍，白烧一次预算（故事 4："不浪费预算"）。
    """
    fake_a = FakeProvider(name="fake-a", failure="fail")
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([fake_a, fake_b])

    response = await chain.chat(_make_request())

    assert response.id.startswith("fake-b")  # 行级：拒答即换路——B 接管
    assert fake_a.calls == ["chat"]  # 行级：恰一次调用——拒答绝不重试（checklist 1 的字面验收）
    # 复杂语句（推导式）行上：attempts 恰两条——a 一败一换，无第二次尝试
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-b", "ok"),
    ]
    # 行级：拒答形状是光杆 UpstreamError（字面量独立真源）——与瞬态词汇分开数
    assert chain.attempts[0].failure_shape == "UpstreamError"


async def test_upstream_own_timeout_is_retried_as_transient() -> None:
    """证明："预算内超时"（上游自带 TimeoutError）归可重试族，重试 1 次即吸收（checklist 1 前半）。

    怎么证明：桩-a 第 1 次抛 builtin TimeoutError（"上游自己超时了"，预算闹钟没响）、
    第 2 次起正常答——断言答复来自 a（换路没发生）、a.calls == ["chat", "chat"]（重试过），
    attempts 第 1 条的形状是 TransientUpstreamError（_call 的翻译产物：上游超时→瞬态，
    与连接失败同族）。反例是把它当拒答直接换路：上游的自超时多是瞬时毛刺，
    一次就换路等于"放弃 A 过早"（故事 5 原话）。
    """
    flaky_a = _FlakyProvider(
        "fake-a",
        fail_times=1,
        # 行级：上游自带超时的字面现场——预算闹钟没响，是上游自己报了超时
        fail=lambda: TimeoutError("upstream internal timeout"),
    )
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([flaky_a, fake_b])

    response = await chain.chat(_make_request())

    assert response.id.startswith("fake-a")  # 行级：重试成功=A 自己答完——毛刺被吸收
    assert flaky_a.calls == ["chat", "chat"]  # 行级：恰两次尝试——上游超时被当瞬态重试
    assert fake_b.calls == []  # 行级：B 从未被惊动
    # 行级：上游自带超时被 _call 翻成瞬态形状（与连接失败同族）——可重试判定的原料
    assert chain.attempts[0].failure_shape == "TransientUpstreamError"
    assert "upstream internal timeout" in chain.attempts[0].detail  # 行级：根因原文留档


# ===== 切片 5：预算拆分——总 30s / 每尝试 15s，最坏延迟=总预算（checklist 2/5）=====


def test_budget_constants_match_ticket() -> None:
    """证明：预算口径=总 30s、每尝试上限 15s——常量值本身是对外承诺（checklist 2）。

    怎么证明：断言两个常量恰为票面数字（issue 04 / 计划§四.2）。为什么值得钉：
    客户端按"最坏延迟 30s"设自己的超时（故事 9），常量悄悄漂移就是悄悄改契约——
    期望值的真源是票面，不是模块自己抄自己（防tautology：数字来自独立出处）。
    """
    assert ATTEMPT_TIMEOUT_SECONDS == 15.0
    assert TOTAL_BUDGET_SECONDS == 30.0


async def test_total_budget_caps_latency_when_both_upstreams_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：全链失败的最坏延迟=总预算——两段卡死也不超总预算（checklist 5）。

    怎么证明：每尝试尺 0.25s、总预算 0.3s（毫秒级、零真实 sleep），两棒都卡住——
    无总预算时两段各烧满 0.25s（≥0.5s），有总预算时第二棒只剩 ~0.05s 额度、
    总耗时≈0.3s；断言 elapsed < 0.42（居于"总预算+调度余量"与"无总预算下限 0.5s"
    之间），attempts 两段都记 timeout（"两段"都真被试过），抛 AttemptTimeoutError。
    反例是只有限时尝试没有总预算：重试与后棒能把延迟叠到 2 倍尺子以上，
    "最坏延迟一句话讲得清"（故事 9）就破产了。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr("routing.chain.TOTAL_BUDGET_SECONDS", 0.3)
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="hang"), FakeProvider(name="fake-b", failure="hang")]
    )

    started = time.monotonic()
    with pytest.raises(AttemptTimeoutError):
        await chain.chat(_make_request())
    elapsed = time.monotonic() - started

    # 行级：不超总预算+调度余量（0.42）；无总预算的反例下限是 2×0.25=0.5s
    assert elapsed < 0.42
    # 复杂语句（推导式）行上：两段都记 timeout——"两段卡死"是两段都试过的事实
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "timeout"),
        ("fake-b", "timeout"),
    ]


async def test_stream_total_budget_caps_latency_when_both_upstreams_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：流式换路窗口同样受总预算封顶——两段卡死也不超总预算（checklist 5 流式腿）。

    怎么证明：与非流式同款尺子（每发 0.25s、总 0.3s），两棒都卡在首块前——断言
    elapsed < 0.42（无总预算反例 ≥0.5s）、attempts 两段都记 timeout、抛
    AttemptTimeoutError。首 token 前的全部尝试+重试都吃总预算：流式腿若只有
    每发尺没有总尺（或只测了非流式），"最坏延迟=总预算"在流式消费路径上就没有
    证据——本条与非流式同款断言补上这条腿（评审收严补网）。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr("routing.chain.TOTAL_BUDGET_SECONDS", 0.3)
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="hang"), FakeProvider(name="fake-b", failure="hang")]
    )

    started = time.monotonic()
    # 行级：1s 测试护栏——防实现缺失时挂死；绿后 0.3s 的总预算闹钟先响
    with pytest.raises(AttemptTimeoutError):
        await asyncio.wait_for(_collect(chain.chat_stream(_make_request())), timeout=1.0)
    elapsed = time.monotonic() - started

    assert elapsed < 0.42  # 行级：与非流式同一口径——不超总预算+调度余量
    # 复杂语句（推导式）行上：两段都记 timeout——流式窗口竞争里"两段卡死"同样可查
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "timeout"),
        ("fake-b", "timeout"),
    ]


# ===== 切片 6：首 token 前由尝试预算管辖——卡住即放弃+显式取消（checklist 3/4）=====


async def _collect(stream) -> list:
    """把 chunk 流收成列表——流式消费的共用口（多条测试同一语言）。"""
    return [c async for c in stream]


async def test_stream_abandons_hanging_first_chunk_with_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：首 token 前由尝试预算管辖——卡住的 A 被放弃且显式取消，B 的流接管（checklist 3/4）。

    怎么证明：尝试预算打到 0.05s（毫秒级、零真实 sleep），fake-a 卡在首块前（hang）、
    fake-b 正常——消费 chain.chat_stream，断言内容=B 的完整回显、fake_a.stream_endings
    == ["cancelled"]（放弃即显式取消穿进上游、不干等烧钱——checklist 3 的字面验收）、
    attempts 记 a=timeout + b=ok。消费套 1s 测试护栏：实现缺失（首块无预算尺）时
    红在护栏上而不是把测试挂死；绿后护栏永不触发（0.05s 的预算闹钟先响）。
    反例是首块裸取没有尺：A 一言不发地悬着，整条 gap（30s）都归它烧，B 永远轮不到。
    """
    monkeypatch.setattr("routing.chain.ATTEMPT_TIMEOUT_SECONDS", 0.05)
    fake_a = FakeProvider(name="fake-a", failure="hang")
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([fake_a, fake_b])

    # 行级：护栏只防"实现缺失时挂死测试"——wait_for 超时即红，绿后由预算闹钟先响
    chunks = await asyncio.wait_for(_collect(chain.chat_stream(_make_request())), timeout=1.0)

    # 行级：内容拼回字面真源——B 的完整回答一块不丢（接管的数据面）
    assembled = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert assembled == "fake-reply: 你好"
    # 复杂语句（推导式+all）行上：全程只有 B 的帧——单流，不是 A/B 各半截的拼接
    assert all(c.id.startswith("fake-b") for c in chunks)
    # 行级：显式取消的证据——放弃的 A 被 aclose 穿进挂起点，收尾账记"被取消"（非干等）
    assert fake_a.stream_endings == ["cancelled"]
    # 行级：A 记 timeout（被预算放弃）、B 记 ok——attempts 把"放弃"与"失败"分开记
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "timeout"),
        ("fake-b", "ok"),
    ]


async def test_stream_retries_transient_before_first_chunk() -> None:
    """证明：流式换路窗口里的瞬态失败同样重试 1 次——首块前毛刺被吸收，不惊动下一棒（1/4）。

    怎么证明：桩-a 第 1 次 chat_stream 在首块前抛 TransientUpstreamError、第 2 次起
    透传 fake 完整流，fake-b 在后棒待命——断言内容=A 的完整回显（id 带 fake-a 名）、
    b.calls == []（B 从未被惊动）、a.calls 恰两次、attempts 记 a=failed + a=ok
    （窗口竞争里两次尝试都入账，checklist 6）。与非流式同一策略（两腿同族）：反例是
    只有 chat() 会重试、流式腿把毛刺当拒答——同一请求两种消费方式行为分叉（故事 2）。
    """
    flaky_a = _FlakyProvider(
        "fake-a",
        fail_times=1,
        fail=lambda: TransientUpstreamError("上游 fake-a 连接失败: connection refused"),
    )
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([flaky_a, fake_b])

    chunks = await asyncio.wait_for(_collect(chain.chat_stream(_make_request())), timeout=1.0)

    # 行级：内容拼回字面真源且帧全带 fake-a 名——重试成功=A 自己的完整单流
    assembled = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert assembled == "fake-reply: 你好"
    assert all(c.id.startswith("fake-a") for c in chunks)
    assert fake_b.calls == []  # 行级：B 从未被惊动（换路没发生的字面反证）
    assert flaky_a.calls == ["chat_stream", "chat_stream"]  # 行级：恰两次尝试——首发 + 重试 1 次
    # 复杂语句（推导式）行上：attempts 两次尝试都入账——第 1 条瞬态失败、第 2 条 ok
    assert [(a.upstream, a.outcome) for a in chain.attempts] == [
        ("fake-a", "failed"),
        ("fake-a", "ok"),
    ]


# ===== 切片 8：首 token 后归 gap 管辖——口径不变、截断语义不破（checklist 4 后半）=====


def test_gap_still_truncates_when_upstream_hangs_after_first_token(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """证明：首 token 后块间 gap 口径不变——链部署形态下照旧截断、不伪造 [DONE]（checklist 4 后半）。

    怎么证明：fake-b 预置闸门只放第一帧（turnstyle 放行一次即关门=首块后永远悬着），
    链装进主缝、gap 打到 0.05s——断言 200 + 首帧真到达 + 无 [DONE]（截断语义与 W2
    一字不差）、死因 "gap 超时" 进日志、fake 记"被取消"（截断收尾穿链）。首块后归
    gap 管辖而不是尝试预算/换路：反例是首块后还动尝试预算——会提前放弃或惊动别的棒，
    "截断语义不破"当场作废（ADR-0006 的拆分口径：前=尝试预算，后= gap）。
    """
    # 行级：预放行的闸门=turnstyle 只放第一帧就关门——B 首块后永远悬在第二帧前
    gate = asyncio.Event()
    gate.set()
    fake_b = FakeProvider(name="fake-b", gate=gate)
    chain = FallbackChain([fake_b])
    monkeypatch.setattr("app.main.provider", chain)
    monkeypatch.setattr("streaming.sse.GAP_TIMEOUT_SECONDS", 0.05)

    response = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert response.status_code == 200  # 行级：首块后流已承诺——gap 超时翻不成状态码
    # 复杂语句（推导式）行上：SSE 帧以空行分隔——切开、丢掉结尾空串，剩下的是一帧帧 payload
    frames = [f for f in response.text.split("\n\n") if f]
    assert frames  # 行级：首帧真到达过客户端（可报错窗口已关、流确实开了头）
    assert "[DONE]" not in response.text  # 行级：半截绝不带完成记号（W2 承诺一字不动）
    assert "gap 超时" in caplog.text  # 行级：死因=块间 gap——首 token 后归 gap 管辖的字面证据
    assert fake_b.stream_endings == ["cancelled"]  # 行级：截断收尾显式穿进上游（断连不泄漏）
