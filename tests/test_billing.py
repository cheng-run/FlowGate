"""计费主缝测试（fallback-ratelimit-billing issue 06）：结算挂点 + 预算前置检查。

受测面按 spec 测试决定：主缝 POST /v1/chat/completions 的账本交叉验证（恰一笔
用户账、失败尝试入损耗账、预算 429、usage 真数字）+ 账本缝的口径在
tests/test_billing_ledger.py。装配绑定换 BillingProvider + 临时账本（与限流测试
同一 monkeypatch 纪律，不给生产加钩子）；全部离线、可复跑。
"""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ChatMessage, ChatRequest
from billing.ledger import BillingLedger
from billing.settlement import BillingProvider
from providers.dashscope import DashScopeProvider
from providers.fake import FakeProvider
from routing.chain import FallbackChain

client = TestClient(app)


def _payload() -> dict:
    """最小合法请求体——与既有端点测试同款，dict 直发模拟真实 HTTP 客户端。"""
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }


def _use_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> BillingLedger:
    """装配绑定换成临时文件账本——每用例独立库，离线可复跑（checklist 1）。"""
    ledger = BillingLedger(str(tmp_path / "billing.db"))
    monkeypatch.setattr("app.main.ledger", ledger)
    return ledger


def _reply_with_official_usage(request: httpx.Request) -> httpx.Response:
    """假上游：回带官方 usage（3/5/8）的完整报文——真数字的出处是上游，不是我们造的。"""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-usage",
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


def test_fallback_success_settles_exactly_one_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：fallback 请求（A 败 B 成）账本恰一笔用户账——降级不重复扣费的字面证明（checklist 4）。

    怎么证明：装配换成 BillingProvider 包住"A 恒失败、B 正常"的链 + 临时账本，
    发一发非流式请求，断言 200（答复来自 fake-b）、账本恰一笔用户账、损耗账恰
    一条且是 fake-a 的失败尝试——A 尝试了、B 答了，钱只收一笔："尝试 ≠ 结果"。
    反例是按尝试记账：A 的失败也被收钱，用户就被扣两份。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    chain = FallbackChain(
        [FakeProvider(name="fake-a", failure="fail"), FakeProvider(name="fake-b")]
    )
    monkeypatch.setattr("app.main.provider", BillingProvider(chain, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    assert "fake-reply" in response.json()["choices"][0]["message"]["content"]
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：恰一笔用户账——"降级不重复扣费"的字面证明
    losses = ledger.losses()
    assert len(losses) == 1  # 行级：A 的失败尝试进损耗账（attempt 粒度）——尝试 ≠ 结果
    assert losses[0].upstream == "fake-a"
    assert losses[0].outcome == "failed"


def test_n_requests_with_m_failed_attempts_bill_exactly_n_charges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：N 逻辑请求含 M 次失败尝试 → 用户账恰 N 笔——实测数字口径钉死（checklist 5）。

    怎么证明：三棒链（两棒恒失败 + 一棒正常）连发 N=3 发请求，每发都经历 M/N=2
    次失败尝试（拒答不重试、直接换路）——断言用户账恰 N=3 笔、损耗账恰 M=6 条。
    这是"降级不重复扣费"的数字形态：失败尝试再多，钱只按"成功的逻辑请求"收。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    chain = FallbackChain(
        [
            FakeProvider(name="fake-a", failure="fail"),
            FakeProvider(name="fake-mid", failure="fail"),
            FakeProvider(name="fake-b"),
        ]
    )
    monkeypatch.setattr("app.main.provider", BillingProvider(chain, ledger=ledger))

    for _ in range(3):
        response = client.post("/v1/chat/completions", json=_payload())
        assert response.status_code == 200  # 行级：每发都被 fake-b 接住——逻辑请求全部成交

    n, m = 3, 6  # 行级：口径钉死——N 逻辑请求、M 次失败尝试（每发 2 棒拒答 ×3 发）
    assert len(ledger.charges()) == n  # 行级：用户账恰 N 笔——"至多一笔"的实测兑现
    assert len(ledger.losses()) == m  # 行级：失败尝试恰 M 条——尝试 ≠ 结果的另一半


def test_over_budget_key_gets_429_before_any_upstream_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：超预算 key 在上游调用前 429——花销按 key 对用户账求和（checklist 6）。

    怎么证明：预算设 5 tokens（一发请求的估算花销≈7 就能顶穿），第一发正常结算后
    同 key 第二发 429 且 detail 点名预算、上游零调用（fake.calls 只有第一发）；
    换个 key 照常 200——预算按 key 隔离。反例是先生成后拒：钱已经花出去了，
    预算就成了事后统计而不是治理。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider()  # 行级：记账 fake——被拒那发"零上游调用"的可断言面
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))
    monkeypatch.setattr("app.main.budget", 5)
    over_budget = {"Authorization": "Bearer sk-budget"}

    first = client.post("/v1/chat/completions", json=_payload(), headers=over_budget)
    second = client.post("/v1/chat/completions", json=_payload(), headers=over_budget)

    assert first.status_code == 200  # 行级：第一发正常成交并结算——花销由此记上账
    assert second.status_code == 429
    assert "预算" in second.json()["detail"]  # 行级：拒绝理由点名预算——不是限流的"退避"
    assert fake.calls == ["chat"]  # 行级：被拒那发零上游调用——拒绝发生在花任何上游钱之前

    other = client.post(
        "/v1/chat/completions",
        json=_payload(),
        headers={"Authorization": "Bearer sk-other"},
    )
    assert other.status_code == 200  # 行级：别的 key 不受影响——预算按 key 求和的隔离面


def test_nonstream_usage_carries_official_numbers_to_response_and_books(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：非流式响应 usage = 上游官方 usage（有则用，真数字不再占位 0，checklist 7）。

    怎么证明：DashScope MockTransport 回官方 usage 3/5/8（既有透传测试同款现场，
    假上游助手见模块级 _reply_with_official_usage），外面套结算门面——断言 HTTP
    响应 usage 逐字段 3/5/8，且账本记的是**同一份数字**（不是估算值、不是占位 0）：
    "所见即所付"，响应与账本同一出处。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    upstream = DashScopeProvider(
        api_key="sk-test",
        base_url="https://upstream.test/v1",
        transport=httpx.MockTransport(_reply_with_official_usage),
    )
    monkeypatch.setattr("app.main.provider", BillingProvider(upstream, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }  # 行级：响应 usage=官方数字原样（有则用）——不再被估算改写成别的数
    charge = ledger.charges()[0]
    assert (charge.prompt_tokens, charge.completion_tokens, charge.total_tokens) == (
        3,
        5,
        8,
    )  # 行级：账本同一份数字——"所见即所付"的账本侧


def test_failed_request_records_losses_but_no_user_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：全链失败零用户账、失败尝试只入损耗账——失败不收钱（checklist 3/4 的失败面）。

    怎么证明：双棒全恒失败的链 + 结算门面，发一发非流式请求，断言 502（既有
    全链失败语义不动）、用户账**零笔**、损耗账恰两条（两棒各自的失败尝试）。
    反例是失败也记一笔"半价账"：story 19 要的是"答不上就分文不取"。
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
    assert ledger.charges() == []  # 行级：零用户账——没交付的答案不收钱（story 19）
    assert [(loss.upstream, loss.outcome) for loss in ledger.losses()] == [
        ("fake-a", "failed"),
        ("fake-b", "failed"),
    ]  # 行级：两棒的失败尝试都进了损耗账（attempt 粒度）——尝试 ≠ 结果


async def test_stream_close_cancels_inner_upstream_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：结算门面的流收尾穿链——aclose 后内层立即被取消，不等 GC finalizer。

    怎么证明：经 BillingProvider 取出流、消费首块后 aclose 门面生成器，断言 fake 的
    收尾账**当场**记下"被取消"。反例（裸透传）：aclose 停在门面这一层，内层要等
    asyncgen finalizer 才关——W2 实验点名的泄漏形态（sse.py 的弃流盖子正防它），
    清理链必须一环不缺地穿到上游（与 FallbackChain.chat_stream 的 finally 同款）。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider()  # 行级：记账 fake——"被取消"何时落账就是收尾穿没穿链的证据
    wrapper = BillingProvider(fake, ledger=ledger)
    request = ChatRequest(model="fake-model", messages=[ChatMessage(role="user", content="你好")])

    stream = wrapper.chat_stream(request)
    await anext(stream)  # 行级：取到首块——流已开工，fake 悬在产块点上
    await stream.aclose()  # 行级：模拟弃流/断连的显式收尾（SSE 清理链的最后一环）

    # 当场可见=收尾穿链直落 fake；要等 finalizer 的话这里还是空账（泄漏形态的现场）
    assert fake.stream_endings == ["cancelled"]


def test_bare_provider_failure_still_records_one_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：单上游失败同样入损耗账——失败尝试的记账不因"没有链"而缺席（checklist 3）。

    怎么证明：裸 FakeProvider（无链、无 attempts 账本）+ 结算门面，恒失败一发——
    断言 502、零用户账、损耗账恰一条（本发=一次尝试，门面就地补记）。反例只给链记
    账：默认装配（单上游）的失败在两本账上都查无此事，"尝试 ≠ 结果"成了半截承诺。
    """
    ledger = _use_ledger(monkeypatch, tmp_path)
    fake = FakeProvider(name="fake", failure="fail")
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502
    assert ledger.charges() == []  # 行级：失败零用户账（story 19）——与链的失败面同款
    losses = ledger.losses()
    assert len(losses) == 1  # 行级：一次尝试一行（attempt 粒度）——门面补记裸上游的尝试
    assert (losses[0].upstream, losses[0].outcome) == ("fake", "failed")
    assert losses[0].failure_shape == "UpstreamError"  # 行级：失败形状原样——形状说话
