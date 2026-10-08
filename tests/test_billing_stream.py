"""计费流式腿测试（fallback-ratelimit-billing issue 07）：流末回填 + include_usage + 按已送达结算。

受测面按 spec 测试决定：主缝 POST /v1/chat/completions 的 SSE 出口 × 账本交叉验证
（官方/估算口径、截断按已送达、首 token 前零账、恰一笔）+ Provider 副缝（弃流/双
结算路径的幂等）。全部离线可复跑（fake 上游 + tmp_path 临时 SQLite）；真网"估算 vs
官方 usage 误差百分比"的 live smoke 在 tests/test_dashscope_live.py。
"""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    ChatCompletionChunk,
    ChatRequest,
    DeltaMessage,
    StreamChoice,
    StreamOptions,
    Usage,
)
from billing.ledger import BillingLedger, Charge
from billing.settlement import BillingProvider
from providers.dashscope import DashScopeProvider
from providers.fake import FakeProvider
from routing.chain import FallbackChain
from tests.conftest import auth_headers

# 默认头带套件级 TEST_KEY（W4 认证落地后的机械件）：本文件行为断言一字不改
client = TestClient(app, headers=auth_headers())


def _payload() -> dict:
    """最小合法流式请求体——与既有端点测试同款，dict 直发模拟真实 HTTP 客户端。"""
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
    }


def _plain_request(stream_options: StreamOptions | None = None) -> ChatRequest:
    """Provider 副缝用的最小请求对象——与主缝 payload 同素材（"你好"），口径一致。"""
    return ChatRequest(
        model="fake-model",
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
        stream_options=stream_options,
    )


def _use_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> BillingLedger:
    """装配绑定换成临时文件账本——每用例独立库，离线可复跑（与 test_billing 同纪律）。"""
    ledger = BillingLedger(str(tmp_path / "billing.db"))
    monkeypatch.setattr("app.main.ledger", ledger)
    return ledger


def _sse_frames(response) -> list[str]:
    """SSE 响应体 → 帧文本列表（空行是帧分隔）——帧级断言的公共切分口（既有测试同款切法）。"""
    # 复杂语句（推导式+条件）行上：只留非空段——尾部空行不算帧
    return [frame for frame in response.text.split("\n\n") if frame]


def _sse_chunks(response) -> list[dict]:
    """SSE 帧 → 解析后的 chunk 字典列表（剥掉 "data: " 前缀）——形状断言的公共入口。"""
    # 复杂语句（推导式）行上：逐帧剥前缀解析 JSON——[DONE] 由调用方自理（帧列表里最后一帧）
    return [json.loads(frame.removeprefix("data: ")) for frame in _sse_frames(response)[:-1]]


def _usage_frames(chunks: list[dict]) -> list[dict]:
    """带 usage 字段的帧列表——"恰一帧 usage"的公共计数口（多处用例同一取法）。"""
    # 复杂语句（推导式+条件）行上：只留 usage 非空的帧——null/缺席都不算
    return [c for c in chunks if c.get("usage") is not None]


def _tokens(charge: Charge) -> tuple[int, int, int]:
    """用户账的 (prompt, completion, total) 三元组——手算样例断言的公共取数口。"""
    return (charge.prompt_tokens, charge.completion_tokens, charge.total_tokens)


# ===== 切片 1：流末回填——官方缺失对整段文本估算（checklist 1/2）=====


def test_stream_settles_estimate_over_whole_text_when_official_usage_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：流末回填走自建估算，且对**整段**已送达文本生效、绝不按块加总（checklist 1/2）。

    怎么证明：fake 流（官方 usage 全零占位=缺失）走主缝 stream=true、正常跑完——
    断言账本恰一笔，数字是手算样例：prompt "你好"=2、completion "fake-reply: 你好"=5
    （公式=CJK 每字 1 + 非 CJK 每 4 字符 1 向上取整，独立真源见 billing/estimator）。
    反例是按块加总：fake 把回答切成 4 块（"fak"/"e-re"/"ply"/": 你好"），逐块估算
    1+1+1+3=6≠5——账面恰是 5 就是"整段一次 tokenize"的字面证明。顺带钉住：客户端流
    一字不差（回显拼回字面真源）、include_usage 缺省零 usage 帧（checklist 5 的反面）。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider()
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    assert _sse_frames(response)[-1] == "data: [DONE]"  # 行级：完整流的完成记号不动（W2 契约）
    chunks = _sse_chunks(response)
    # 行级：内容拼回字面真源——结算不惊动透传，客户端收到的仍是 fake 的完整回显
    assert (
        "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "fake-reply: 你好"
    )
    # 行级：include_usage 缺省不发 usage 帧——"恰多发一帧"的另一半是"不请求就零帧"
    assert all(c.get("usage") is None for c in chunks)
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：一条流恰一笔用户账——"至多一笔"的流式腿兑现
    # 手算样例（独立真源=estimator 公式）：prompt "你好"=2；completion 整段 12 非 CJK/4=3
    # + 2 CJK=5。按块加总会得 1+1+1+3=6（每块各自取整），恰不等于账面 5
    assert _tokens(charges[0]) == (2, 5, 7)


# ===== 切片 2：首 token 后截断——按已送达文本结算（checklist 3）=====


class _DyingMidStreamProvider:
    """吐两块就抛的流式桩：首块后中途死亡——"截断按已送达结算"的现场。

    为什么用桩不用 fake：fake 的注入形态只管首块前的死法（恒失败/卡住/空流）；
    "首块后死"是另一门死法，沿既有 _DyingMidStreamProvider 桩先例在测试里现做
    （test_stream_fallback 同款纪律：fake 的词汇一行不动）。
    """

    name = "stub-dying"

    def __init__(self) -> None:
        """固定两块正文再死——已送达文字可手算，结算断言才有独立真源。"""
        self.calls: list[str] = []  # 行级：调用记录——桩被走过的证据

    async def chat(self, request: ChatRequest):
        """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
        raise AssertionError("stream=true 不应走非流式 chat()")

    async def chat_stream(self, request: ChatRequest):
        """先给两块正文（首块到手=流已承诺），再抛错——死亡必须发生在"首块后"。"""
        self.calls.append("chat_stream")  # 行级：开工记账
        for part in ("abcd", "efgh"):  # 行级：两块正文=已送达 "abcdefgh"（8 非 CJK 字符）
            yield ChatCompletionChunk(
                id="stub-dying-1",
                model=request.model,
                choices=[StreamChoice(delta=DeltaMessage(content=part))],
            )
        raise RuntimeError("stub-dying 流中途断掉")


def test_stream_settles_only_delivered_text_when_upstream_dies_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """证明：首 token 后截断只按已送达文本结算——用户不为未收到的文字付钱（checklist 3）。

    怎么证明：吐两块（"abcd"+"efgh"）就抛错的桩走主缝 stream=true——断言 200（流已
    承诺）+ 无 [DONE]（截断语义不动）+ 账本恰一笔，数字=手算已送达 "abcdefgh"：
    prompt "你好"=2、completion 8 非 CJK/4=2（多一个字都不收）。反例是按上游"应该
    会说的话"结算——那用户就要为没收到的字付钱。损耗账为空：死在首 token 后的尝试
    "有结果"（用户账代表它），两本账各说一事（尝试 ≠ 结果的另一半）。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    stub = _DyingMidStreamProvider()
    monkeypatch.setattr("app.main.provider", BillingProvider(stub, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200  # 行级：首块后死亡翻不成状态码（W2 契约一字不动）
    assert "[DONE]" not in response.text  # 行级：半截回答绝不带完成记号——截断可辨
    assert "abcd" in response.text and "efgh" in response.text  # 行级：两块真送达过
    assert stub.calls == ["chat_stream"]  # 行级：桩被走过——死亡发生在流中
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：一条死流也恰一笔——"无论怎么死都不双扣"的截断面
    # 手算样例（独立真源=estimator 公式）：已送达 "abcdefgh"=8 非 CJK/4=2；prompt "你好"=2
    assert _tokens(charges[0]) == (2, 2, 4)
    assert ledger.losses() == []  # 行级：首 token 后的死法不进损耗账——结果由用户账代表


# ===== 切片 3：首 token 前失败——零用户账、尝试进损耗账（checklist 4）=====


def test_stream_failure_before_first_chunk_is_zero_user_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：首 token 前失败零用户账，尝试照进损耗账——答不上就分文不取（checklist 4）。

    怎么证明：双棒全恒失败的链走主缝 stream=true——断言 502（可报错窗口语义不动）、
    用户账**零笔**、损耗账恰两条（两棒各自的失败尝试，号与死因都对得上）。反例是
    "失败也收半价"：story 19 要的是零账；反例之二是失败尝试不记账：损耗账就缺了
    "尝试 ≠ 结果"的另一半证据。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    chain = FallbackChain(
        [
            FakeProvider(name="fake-a", failure="fail"),
            FakeProvider(name="fake-b", failure="fail"),
        ]
    )
    monkeypatch.setattr("app.main.provider", BillingProvider(chain, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502  # 行级：全链失败语义不动——结算不吞错误
    assert ledger.charges() == []  # 行级：零用户账——首 token 前失败分文不取（story 19）
    # 复杂语句（推导式）行上：损耗账=两棒各自的失败尝试（attempt 粒度）——尝试 ≠ 结果
    assert [(loss.upstream, loss.outcome) for loss in ledger.losses()] == [
        ("fake-a", "failed"),
        ("fake-b", "failed"),
    ]


def test_stream_bare_provider_failure_records_one_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：裸上游的流式失败同样入损耗账——补记不因"没有链"缺席（checklist 4 的兜底面）。

    怎么证明：裸 FakeProvider（无链、无 attempts 账本）恒失败走主缝 stream=true——
    断言 502、零用户账、损耗账恰一条（本发=一次尝试，门面就地补记，与非流式腿的
    test_bare_provider_failure_still_records_one_loss 同一口径）。反例只给链记账：
    默认装配（单上游）的流式失败在两本账上查无此事。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider(name="fake", failure="fail")
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502
    assert ledger.charges() == []  # 行级：零用户账（story 19）——与链的失败面同款
    losses = ledger.losses()
    assert len(losses) == 1  # 行级：一次尝试一行——门面补记裸上游的流式尝试
    assert (losses[0].upstream, losses[0].failure_shape) == ("fake", "UpstreamError")


def test_stream_bare_empty_stream_records_one_loss_and_zero_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：裸上游空流（首 token 前一言不发）零用户账 + 尝试进损耗账——checklist 4 的空流形态。

    怎么证明：注入 empty 形态的裸 fake 走主缝 stream=true——空流在 seam 翻 502
    （既有契约不动），断言用户账零笔、损耗账恰一条（门面补记，与链上 _fetch_first
    把空流翻 UpstreamError 记账同一口径）。反例只在异常路径补记：空流是"正常走完
    但什么都没说"，不经异常路径——502 之后损耗账查无此事，"尝试进损耗账"对空流
    形态就缺了席（评审抓出的缺口）。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider(name="fake", failure="empty")
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502  # 行级：空流=答不上——seam 照旧翻 502（W2 契约）
    assert ledger.charges() == []  # 行级：零用户账——空流没有可结算的交付
    losses = ledger.losses()
    assert len(losses) == 1  # 行级：补记一次尝试——"失败尝试入账"不漏空流形态
    assert (losses[0].upstream, losses[0].failure_shape) == ("fake", "UpstreamError")


# ===== 切片 4：include_usage——[DONE] 前恰一帧 usage chunk（checklist 5）=====


def test_stream_include_usage_emits_exactly_one_usage_chunk_before_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：include_usage 置位时流末恰多发一帧 usage chunk，且在 [DONE] 之前（checklist 5）。

    怎么证明：fake 流 + stream_options.include_usage=true 走主缝——逐帧解析 SSE，
    断言带 usage 的帧**恰一帧**、是 [DONE] 前的最后一帧、形状=OpenAI 惯例
    （choices 空列表 + usage）、数字=结算口径的估算（手算 2/5/7，所见即所付）、
    id/model 与流内各帧一致（客户端按 id 归组）。反例是不发/发两帧/发在 [DONE] 后：
    OpenAI SDK 客户端读不到、读重、或错过——兼容形状就是这些细节的总和。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider()
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))
    payload = _payload()
    payload["stream_options"] = {"include_usage": True}  # 行级：OpenAI 兼容的请求形状

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    frames = _sse_frames(response)
    assert frames[-1] == "data: [DONE]"  # 行级：完成记号仍在最后（usage 帧插在它前面）
    chunks = _sse_chunks(response)
    usage_frames = _usage_frames(chunks)  # 行级：带 usage 的帧恰一帧——"恰多发一帧"的字面计数
    assert len(usage_frames) == 1
    assert chunks[-1] is usage_frames[0]  # 行级：usage 帧=[DONE] 前的最后一帧（OpenAI 惯例）
    only = usage_frames[0]
    assert only["choices"] == []  # 行级：choices 空列表——usage 载体帧的形状标记
    assert only["id"] == chunks[0]["id"] and only["model"] == chunks[0]["model"]  # 行级：流内同组
    # 数字=结算口径的估算（所见即所付）：手算 2/5/7（同切片 1 的公式与真源）
    assert only["usage"] == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}
    # 行级：账本与 usage 帧同一份数字——客户端读到的就是被收的钱
    assert _tokens(ledger.charges()[0]) == (2, 5, 7)


# ===== 切片 5：官方 usage 优先 + usage 出口唯一（checklist 1 的"有则用"）=====


def _upstream_sse_with_usage() -> httpx.Response:
    """假上游：正文两块 + 末帧 + usage 载体帧（choices 空 + 官方 3/9/12）——官方 usage 现场。

    帧形状钉死在字面量上（独立真源）：官方数字 3/9/12 与估算口径的手算 2/5/7 明确
    不同——断言等于前者、不等于后者，"官方优先"才有正反两面的证据。
    """
    frames = (
        {
            "id": "chatcmpl-up",
            "object": "chat.completion.chunk",
            "model": "qwen-max",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        },
        {
            "id": "chatcmpl-up",
            "object": "chat.completion.chunk",
            "model": "qwen-max",
            "choices": [{"index": 0, "delta": {"content": "你好"}}],
        },
        {
            "id": "chatcmpl-up",
            "object": "chat.completion.chunk",
            "model": "qwen-max",
            "choices": [{"index": 0, "delta": {"content": "，世界"}}],
        },
        {
            "id": "chatcmpl-up",
            "object": "chat.completion.chunk",
            "model": "qwen-max",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        # usage 载体帧（OpenAI include_usage 惯例）：choices 空列表 + usage——官方计数
        {
            "id": "chatcmpl-up",
            "object": "chat.completion.chunk",
            "model": "qwen-max",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
        },
    )
    # 复杂语句（推导式）行上：每帧渲成 data: 行 + 上游 [DONE] 收尾——DashScope 的 SSE 方言
    body = "".join(f"data: {json.dumps(f, ensure_ascii=False)}\n\n" for f in frames)
    return httpx.Response(200, text=body + "data: [DONE]\n\n")


def test_stream_prefers_official_usage_and_keeps_single_usage_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：上游末帧带官方 usage 则用之，且 usage 只从网关自己的那一帧出口（checklist 1）。

    怎么证明：DashScope MockTransport 回"正文 + usage 载体帧（3/9/12）"的 SSE，
    include_usage 置位走主缝——断言账本与客户端 usage 帧都是官方 3/9/12（不被估算
    2/5/7 改写：数字取向相反即两面证据）、客户端线上带 usage 的帧恰一帧（上游的
    载体帧被吃掉当素材、不透传——否则客户端会读到两帧）。反例是"官方缺失走估算"：
    由切片 1 覆盖；反例之二是两帧 usage：OpenAI 客户端会读重。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    upstream = DashScopeProvider(
        api_key="sk-test",
        base_url="https://upstream.test/v1",
        transport=httpx.MockTransport(lambda request: _upstream_sse_with_usage()),
    )
    monkeypatch.setattr("app.main.provider", BillingProvider(upstream, ledger=ledger))
    payload = _payload()
    payload["stream_options"] = {"include_usage": True}

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    chunks = _sse_chunks(response)
    usage_frames = _usage_frames(chunks)  # 行级：恰一帧——上游载体帧不透传（usage 出口唯一）
    assert len(usage_frames) == 1
    # 行级：数字=官方 3/9/12 原样（有则用）——不是估算的 2/5/7，账本与客户端同一份数字
    assert usage_frames[0]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 9,
        "total_tokens": 12,
    }
    assert _tokens(ledger.charges()[0]) == (3, 9, 12)
    # 行级：正文透传不受影响——内容拼回上游字面真源（结算/洗 usage 不惊动数据）
    assert (
        "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        == "你好，世界"
    )


class _UsageEmbeddedStub:
    """正文帧嵌 usage 的流式桩：有些上游把计数搭在内容/finish 帧上——洗 usage 的现场。"""

    name = "stub-embedded"

    async def chat(self, request: ChatRequest):
        """不该被走到：stream=true 却分派到非流式腿=路由分派写错了，直接炸红。"""
        raise AssertionError("stream=true 不应走非流式 chat()")

    async def chat_stream(self, request: ChatRequest):
        """两块正文都带着 usage（3/9/12）——透传必须把 usage 字段洗掉、数字只进结算。"""
        for part in ("你好", "，世界"):
            yield ChatCompletionChunk(
                id="stub-embedded-1",
                model=request.model,
                choices=[StreamChoice(delta=DeltaMessage(content=part))],
                usage=Usage(prompt_tokens=3, completion_tokens=9, total_tokens=12),
            )


async def test_stream_strips_usage_field_from_forwarded_content_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：正文帧自带的 usage 字段在透传时被洗掉——usage 出口唯一（checklist 5 的护栏）。

    怎么证明：桩上游在正文/末帧上嵌 usage（有些上游把计数搭在 finish 帧上）走
    Provider 副缝——断言每一透传帧 usage 为 None（数字只进结算），账本仍按官方入账。
    反例是原样透传：include_usage 关着客户端也会看到 usage 字段，开关就形同虚设。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    wrapper = BillingProvider(_UsageEmbeddedStub(), ledger=ledger)

    chunks = [chunk async for chunk in wrapper.chat_stream(_plain_request())]

    assert chunks  # 行级：正文帧照常透传——洗 usage 不吞帧
    assert all(c.usage is None for c in chunks)  # 行级：透传帧 usage 全空——出口唯一
    assert _tokens(ledger.charges()[0]) == (3, 9, 12)


# ===== 切片 6/7：断连/弃流路径的结算幂等——一条流无论怎么死都不双扣（checklist 6）=====


class _CallCountingLedger(BillingLedger):
    """结算调用记数账本：settle 被调了几次直接可见——"两条结算路径撞车"的证据面。

    为什么子类化而不是 mock 掉：幂等是被测行为本身，要跑真账本；只把"调用次数"
    这一个观测点亮出来（MockTransport 捕获 wire 字节的同一手法）——撞车没撞、
    幂等吞没吞，一数便知。
    """

    def __init__(self, db_path: str) -> None:
        """真账本照常落 SQLite（幂等行为在真账本上验证），另记结算调用次数。"""
        super().__init__(db_path)
        self.settle_calls = 0  # 行级：settle 每被调用一次 +1——撞车的字面计数

    def settle(self, *args, **kwargs):
        """记一次调用后转真账本——行为与 BillingLedger.settle 一字不差（透传签名从简）。"""
        self.settle_calls += 1  # 行级：先记数再落账——两次调用恰=两条路径都跑过
        return super().settle(*args, **kwargs)


async def test_stream_abandoned_at_usage_chunk_settles_exactly_one_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：两条结算路径撞车也恰一笔——弃流幂等的字面现场（checklist 6）。

    怎么证明：include_usage 的流正常跑完（正常路径已在 usage 帧前结算），消费方恰在
    usage 帧的 yield 上弃流（aclose）——异常路径的二次结算随之触发。用记数账本断言
    settle **被调了两次**（撞车真实发生）而账本仍**恰一笔**（幂等把第二次吞成
    no-op）、数字就是那次正常结算的。反例是结算不幂等：这一发被记两次，"无论怎么
    死都不双扣"当场破功——这正是 ledger.settle 幂等 no-op 要兜的撞车形态。
    """
    ledger = _CallCountingLedger(str(tmp_path / "billing.db"))
    monkeypatch.setattr("app.main.ledger", ledger)
    fake = FakeProvider()
    wrapper = BillingProvider(fake, ledger=ledger)
    request = _plain_request(StreamOptions(include_usage=True))

    stream = wrapper.chat_stream(request)
    usage_chunk = None
    async for chunk in stream:  # 行级：消费到 usage 帧就 break——生成器悬在那个 yield 上
        usage_chunk = chunk
        if chunk.usage is not None:
            break
    await stream.aclose()  # 行级：弃流——异常路径的二次结算在此触发，幂等 no-op 兜住

    assert usage_chunk is not None and usage_chunk.usage is not None  # 行级：usage 帧确已发出
    assert ledger.settle_calls == 2  # 行级：两条结算路径真的都跑了——撞车不是假想
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：撞车后账本仍恰一笔——幂等把第二次吞成 no-op
    assert _tokens(charges[0]) == (2, 5, 7)  # 行级：数字未被改写——仍是那次正常结算的估算


async def test_stream_disconnect_after_delivery_settles_exactly_one_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：断连（弃流）按已送达文本结算、恰一笔——checklist 3/6 的断连面。

    怎么证明：消费到首块+第一块正文（"fak"）就 aclose（模拟客户端断连）——断言
    账本恰一笔，数字=手算已送达 "fak"：prompt "你好"=2、completion 3 非 CJK/4=1
    （后半句 "e-reply: 你好" 没送到，一个字都不收）。与切片 2 的截断口径同一句话：
    首 token 后怎么死都按已送达结算。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider()
    wrapper = BillingProvider(fake, ledger=ledger)

    stream = wrapper.chat_stream(_plain_request())
    await anext(stream)  # 行级：首块（role 宣告帧，空文字）——流已开工
    await anext(stream)  # 行级：第一块正文 "fak"——已送达文字=3 个非 CJK 字符
    await stream.aclose()  # 行级：模拟客户端断连——弃流收尾结算在这触发

    charges = ledger.charges()
    assert len(charges) == 1  # 行级：断连也恰一笔——"无论怎么死都不双扣"的断连面
    # 手算样例（独立真源=estimator 公式）：已送达 "fak"=3 非 CJK/4=1；prompt "你好"=2
    assert _tokens(charges[0]) == (2, 1, 3)
