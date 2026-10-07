"""账本缝测试（fallback-ratelimit-billing issue 06）：结算 + 查询 = 唯一读写面。

受测面按 spec 测试决定：账本缝的**结算口径 / 幂等 / 两本账**（离线可复跑——SQLite
走 tmp_path 临时文件）。不测 SQL、不测表结构——只测账本接口对外承诺的行为：
一笔用户账（request_id 唯一）、失败尝试进内部损耗账、二次结算 no-op、花销按 key 求和。
"""

from pathlib import Path

from app.schemas import Usage
from billing.ledger import BillingLedger
from routing.chain import Attempt


def _ledger(tmp_path: Path) -> BillingLedger:
    """临时文件账本——每个用例一只独立库（checklist 1：测试用临时文件，可复跑）。"""
    return BillingLedger(str(tmp_path / "billing.db"))


def test_settle_records_one_charge_with_official_usage(tmp_path: Path) -> None:
    """证明：结算=用户账恰一笔，且官方 usage 有则用（checklist 1/7 的账本侧）。

    怎么证明：临时文件账本 settle 一次（官方 usage 3/5/8），断言返回值与 charges()
    恰一条、逐字段 3/5/8、request_id/key 对得上——"真数字不再占位 0"在账本上的
    第一现场：官方数字原样入账，不被估算器改写。
    """
    ledger = _ledger(tmp_path)

    settled = ledger.settle(
        "req-1",
        "sk-test",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )

    # 官方 usage 原样结算（有则用）——返回口径与账面口径同一份数字
    assert settled == Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：一个逻辑请求恰一笔用户账（"至多一笔"的正向兑现）
    only = charges[0]
    assert only.request_id == "req-1"  # 行级：幂等键落账——客户端对账靠它
    assert only.key == "sk-test"  # 行级：计费身份落账——预算按 key 求和的原料
    assert (only.prompt_tokens, only.completion_tokens, only.total_tokens) == (3, 5, 8)


def test_settle_is_no_op_on_second_call(tmp_path: Path) -> None:
    """证明：二次结算 no-op——同一 request_id 永远只收一笔钱（checklist 2：幂等 + 唯一约束兜底）。

    怎么证明：同一 request_id 结算两次，第二次给不同的用量（1/1/2）来者不善——
    断言第二次返回的是**首次入账**的数字（账本为准，不被后来者改写）、charges()
    仍恰一条且是首次的 3/5/8。反例是没有幂等的账本：fallback 里"尝试被当结果"
    结算两次，用户就被扣两份钱——本测试就是"不重复扣费"在账本缝的字面防线。
    """
    ledger = _ledger(tmp_path)
    ledger.settle(
        "req-1",
        "sk-test",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )

    settled_again = ledger.settle(
        "req-1",
        "sk-test",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    # 幂等 no-op 的返回口径=首次入账的数字——结算结果以账本为准，不被二次调用改写
    assert settled_again == Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：唯一约束兜底后仍恰一笔——"至多一笔"两次调用也成立
    assert charges[0].total_tokens == 8  # 行级：后来的 1/1/2 没能覆盖首次账目


def test_settle_estimates_when_official_usage_missing(tmp_path: Path) -> None:
    """证明：官方 usage 缺失时用自建估算器补真数字——账面永不再躺"占位 0"（checklist 7）。

    怎么证明：两次结算都无官方数字（一次 None、一次全零占位——fake 的现实形态），
    对已知文本断言手算样例：prompt "你好"=2（CJK 每字≈1 token）、completion
    "fake-reply: 你好"=5（12 个非 CJK 字符 12/4=3 + CJK 2）——估算是对**整段文本**
    tokenize 的结果，不是按块加总。反例是记 0：预算按用户账求和会永远看不见花销。
    """
    ledger = _ledger(tmp_path)

    settled = ledger.settle(
        "req-1",
        "sk-test",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=None,
    )

    # 手算样例（独立真源=估算器 docstring 的启发式公式）：2 + (12/4 + 2) = 7
    assert settled == Usage(prompt_tokens=2, completion_tokens=5, total_tokens=7)

    # 全零占位与 None 同义（fake 不装懂的形态）：同样走估算，不留 0 在账上
    settled_zeros = ledger.settle(
        "req-2",
        "sk-test",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(),
    )
    assert settled_zeros == Usage(prompt_tokens=2, completion_tokens=5, total_tokens=7)
    assert [c.total_tokens for c in ledger.charges()] == [7, 7]  # 行级：两笔都是真数字


def test_settle_records_failed_attempts_as_internal_losses(tmp_path: Path) -> None:
    """证明：失败尝试入内部损耗账（attempt 粒度）、永不进用户账（checklist 3）。

    怎么证明：一次"接管"结算（A 失败 + A 超时 + B 成功的 attempts 流水），断言
    损耗账恰两条（failed/timeout 各一条、死因原文保留）、ok 的尝试不进损耗账
    （它是"结果"，由用户账代表），用户账仍恰一笔——两本账各说一事，"尝试 ≠ 结果"
    在这里是记账事实而不是口头承诺。
    """
    ledger = _ledger(tmp_path)
    attempts = [
        Attempt(
            request_id="req-1",
            upstream="fake-a",
            started_at=1.0,
            ended_at=2.0,
            outcome="failed",
            failure_shape="UpstreamError",
            detail="fake 注入的恒失败形态",
        ),
        Attempt(
            request_id="req-1",
            upstream="fake-a",
            started_at=2.0,
            ended_at=17.0,
            outcome="timeout",
            failure_shape="AttemptTimeoutError",
            detail="尝试预算 15 秒内未完成响应",
        ),
        Attempt(
            request_id="req-1",
            upstream="fake-b",
            started_at=17.0,
            ended_at=18.0,
            outcome="ok",
            failure_shape=None,
            detail=None,
        ),
    ]

    ledger.settle(
        "req-1",
        "sk-test",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        attempts=attempts,
    )

    losses = ledger.losses()
    assert len(losses) == 2  # 行级：attempt 粒度——两次失败尝试各一条，不合并不漏记
    assert [(loss.upstream, loss.outcome) for loss in losses] == [
        ("fake-a", "failed"),
        ("fake-a", "timeout"),
    ]
    assert losses[0].failure_shape == "UpstreamError"  # 行级：失败形状进账——"形状说话"的原料
    assert losses[0].detail == "fake 注入的恒失败形态"  # 行级：死因原文保留，排错可直读
    # 失败尝试永不进用户账：账上只有 B 成功带来的那一笔——"降级不重复扣费"的账本侧
    assert len(ledger.charges()) == 1


def test_settle_second_call_does_not_duplicate_losses(tmp_path: Path) -> None:
    """证明：二次结算整笔 no-op——损耗账同样不翻倍（结算幂等覆盖两本账）。

    怎么证明：带失败尝试结算两次（同 request_id、同 attempts 流水），断言损耗账
    仍恰一条、用户账仍恰一笔。反例是只给用户账上幂等：损耗账被重复写，"M 次失败
    尝试"的实测数字口径就被记账噪声污染了。
    """
    ledger = _ledger(tmp_path)
    attempts = [
        Attempt(
            request_id="req-1",
            upstream="fake-a",
            started_at=1.0,
            ended_at=2.0,
            outcome="failed",
            failure_shape="UpstreamError",
            detail="fake 注入的恒失败形态",
        ),
    ]
    settle_kwargs = {
        "prompt_text": "你好",
        "completion_text": "fake-reply: 你好",
        "official_usage": Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        "attempts": attempts,
    }

    ledger.settle("req-1", "sk-test", **settle_kwargs)
    ledger.settle("req-1", "sk-test", **settle_kwargs)

    assert len(ledger.losses()) == 1  # 行级：第二次结算整笔 no-op——损耗账不翻倍
    assert len(ledger.charges()) == 1  # 行级：用户账同一防线（两本账同一套幂等）


def test_spend_sums_user_charges_per_key(tmp_path: Path) -> None:
    """证明：花销按 key 对用户账求和——预算前置检查的数字口径（checklist 6 的查询面）。

    怎么证明：同一 key 结算两笔（7+8 tokens）、另一 key 结算一笔（8 tokens），
    断言 spend() 各自求和、互不串门（key 隔离）；损耗账的记录不计花销——
    失败尝试不收钱，也就不该消耗预算（"尝试 ≠ 结果"在预算上的推论）。
    """
    ledger = _ledger(tmp_path)
    ledger.settle(
        "req-1",
        "sk-a",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )
    ledger.settle(
        "req-2",
        "sk-a",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )
    ledger.settle(
        "req-3",
        "sk-b",
        prompt_text="你好",
        completion_text="fake-reply: 你好",
        official_usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )

    assert ledger.spend("sk-a") == 15  # 行级：同 key 两笔求和——预算上限就卡在这条线上
    assert ledger.spend("sk-b") == 8  # 行级：别的 key 不串门——隔离性即 dict 分键的账本版
    assert ledger.spend("sk-没花过钱") == 0  # 行级：没花过的 key 求和为 0——预算检查的初值


def test_charges_and_losses_filter_by_request_id(tmp_path: Path) -> None:
    """证明：查询口支持按 request_id 对账——每笔账都能单独拎出来核（story 21）。

    怎么证明：两轮结算（各有失败尝试）后按号各查一遍，断言只回那一轮的记录、
    查无此号回空——客户端"拿响应头的号来对账"的入口就在这里。
    """
    ledger = _ledger(tmp_path)
    for rid, upstream in (("req-1", "fake-a"), ("req-2", "fake-b")):
        ledger.settle(
            rid,
            "sk-test",
            prompt_text="你好",
            completion_text="fake-reply: 你好",
            official_usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            attempts=[
                Attempt(
                    request_id=rid,
                    upstream=upstream,
                    started_at=1.0,
                    ended_at=2.0,
                    outcome="failed",
                    failure_shape="UpstreamError",
                    detail="fake 注入的恒失败形态",
                ),
                Attempt(
                    request_id=rid,
                    upstream="fake-ok",
                    started_at=2.0,
                    ended_at=3.0,
                    outcome="ok",
                    failure_shape=None,
                    detail=None,
                ),
            ],
        )

    # 复杂语句（推导式）行上：按号对账——每轮只回自己的用户账与损耗账
    assert [(c.request_id, c.key) for c in ledger.charges("req-1")] == [("req-1", "sk-test")]
    assert [(loss.request_id, loss.upstream) for loss in ledger.losses("req-1")] == [
        ("req-1", "fake-a")
    ]
    assert [c.request_id for c in ledger.charges("req-2")] == ["req-2"]
    assert ledger.charges("req-查无此号") == []  # 行级：查无此号回空——对账的 404 口径
    assert ledger.losses("req-查无此号") == []
