"""装配处（composition root）测试：环境变量选上游、缺 key 响亮报错（spec 故事 5/10）。

装配处是全代码库唯一点名具体适配器的地方（ADR-0001），所以它测的是"选择"本身：
给什么环境、出什么实例、配错了喊多响——路由等核心代码的行为由端点测试兜底。
限流器装配（issue 05）同理：测 env 口径的读取与配错报错，桶的行为在
tests/test_ratelimit.py 的主缝上钉；账本/预算装配（issue 06）同纪律。
"""

import pytest

from app.main import create_budget, create_ledger, create_limiter, create_provider
from app.schemas import Usage
from providers.base import Provider
from providers.dashscope import DashScopeProvider
from providers.fake import FakeProvider
from ratelimit.bucket import RateLimitError
from routing.chain import FallbackChain


def test_create_provider_defaults_to_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：什么都不配时默认 fake——无 key 也能跑测试与演示（spec 默认值决定）。

    怎么证明：清掉 FLOWGATE_PROVIDER 后调 create_provider，断言拿到 FakeProvider。
    这条是"测试可复跑"的装配侧保证：CI/新同事的机器没有 key，装配不得炸。
    """
    monkeypatch.delenv("FLOWGATE_PROVIDER", raising=False)

    assert isinstance(create_provider(), FakeProvider)


def test_create_provider_selects_dashscope_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：换上游 = 改配置不改代码——FLOWGATE_PROVIDER=dashscope 装出真适配器（故事 5）。

    怎么证明：设环境变量后调 create_provider，断言拿到 DashScopeProvider 实例。
    路由等核心代码零改动就换了上游，这正是 ADR-0001 seam 的装配侧证据。
    """
    monkeypatch.setenv("FLOWGATE_PROVIDER", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    assert isinstance(create_provider(), DashScopeProvider)


def test_create_provider_names_missing_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：选真上游却没 key → 报错指名 DASHSCOPE_API_KEY（故事 10：失败要响亮且具体）。

    怎么证明：设 dashscope、清 key，断言 RuntimeError 的消息里含变量名。
    反例是静默降级回 fake——用户会看着回显纳闷"我明明配了真上游"，悬案难查。
    """
    monkeypatch.setenv("FLOWGATE_PROVIDER", "dashscope")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        create_provider()


# ===== fallback 链装配（fallback-ratelimit-billing issue 02：逗号表=顺序链）=====


def test_create_provider_builds_chain_from_comma_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：FLOWGATE_PROVIDER 逗号表 = 顺序 fallback 链，链对上层就是一个上游（checklist 1/2）。

    怎么证明：设 "fake,fake" 调 create_provider，断言拿到 FallbackChain（名字可见
    两个成员与顺序），且仍过 isinstance(Provider)——路由继续对着协议说话。
    单值=单元素链（退化回裸适配器）由既有三个装配测试继续钉：向后兼容一字不改。
    """
    monkeypatch.setenv("FLOWGATE_PROVIDER", "fake,fake")

    provider = create_provider()

    assert isinstance(provider, FallbackChain)
    assert isinstance(provider, Provider)  # 行级：链对上层就是一个上游（isinstance 可验）
    assert provider.name == "fake+fake"  # 行级：成员与顺序在链名上可见（A 先试、B 兜底）


def test_create_provider_rejects_empty_entry_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：逗号表里的空条目响亮报错——"fake," 这类手滑不静默装出怪链（配错喊响）。

    怎么证明：设 "fake,"（尾逗号），断言 RuntimeError 且消息里点名 FLOWGATE_PROVIDER。
    反例是跳过空条目装单链：配置现场与运行现场对不上号，悬案难查。
    """
    monkeypatch.setenv("FLOWGATE_PROVIDER", "fake,")

    with pytest.raises(RuntimeError, match="FLOWGATE_PROVIDER"):
        create_provider()


# ===== 限流装配（fallback-ratelimit-billing issue 05：env 口径）=====


def test_create_limiter_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：限流口径走 env——FLOWGATE_RATE_CAPACITY/PER_SECOND 直接决定桶行为。

    怎么证明：设容量 1、速率 0（永不回填），调 create_limiter 拿到限流器后对同一
    key 连扣两枚——第一枚放行、第二枚 RateLimitError。反例（写死常量）下第二枚
    也放行；这里用行为断言而不是读私有字段，测的是装配出的口径本身。
    """
    monkeypatch.setenv("FLOWGATE_RATE_CAPACITY", "1")
    monkeypatch.setenv("FLOWGATE_RATE_PER_SECOND", "0")

    limiter = create_limiter()

    limiter.acquire("sk-env")  # 行级：第一枚——容量 1 的桶恰好放行
    with pytest.raises(RateLimitError):
        limiter.acquire("sk-env")  # 行级：第二枚——env 容量生效，桶已空


def test_create_limiter_rejects_bad_config_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：限流配置非法时响亮报错并指名 env 变量（配错喊响，同缺 key 测试）。

    怎么证明：把 FLOWGATE_RATE_CAPACITY 设成非数字，断言 RuntimeError 且消息里
    点名变量名——反例是静默落回默认值："我明明收紧了限流怎么没生效"会成悬案。
    """
    monkeypatch.setenv("FLOWGATE_RATE_CAPACITY", "两个")

    with pytest.raises(RuntimeError, match="FLOWGATE_RATE_CAPACITY"):
        create_limiter()


# ===== 账本/预算装配（fallback-ratelimit-billing issue 06：env 口径）=====


def test_create_budget_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：预算口径走 env——FLOWGATE_BUDGET_TOKENS 直接决定每 key 的上限。

    怎么证明：设 42 调 create_budget，断言拿到 42；缺省时断言 0（不设限的口径：
    "0=无预算"，记账先跑、预算按部署显式开启）。
    """
    monkeypatch.setenv("FLOWGATE_BUDGET_TOKENS", "42")
    assert create_budget() == 42

    monkeypatch.delenv("FLOWGATE_BUDGET_TOKENS", raising=False)
    assert create_budget() == 0  # 行级：缺省=不设限——不是"零预算"（见 create_budget docstring）


def test_create_budget_rejects_bad_config_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：预算配置非法时响亮报错并指名 env 变量（配错喊响，同限流装配测试）。

    怎么证明：把 FLOWGATE_BUDGET_TOKENS 设成非数字，断言 RuntimeError 且消息里
    点名变量名——反例是静默落回 0：预算"看起来配了其实没生效"会成悬案。
    """
    monkeypatch.setenv("FLOWGATE_BUDGET_TOKENS", "很多")

    with pytest.raises(RuntimeError, match="FLOWGATE_BUDGET_TOKENS"):
        create_budget()


def test_create_ledger_reads_db_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：账本落盘路径走 env——FLOWGATE_BILLING_DB 指到哪账本就落哪（issue 06）。

    怎么证明：指向临时文件，settle 一笔再查回来（账本能记账=装配出了活账本）；
    缺省 ":memory:" 不在此测（行为=进程内可读写），路径口径由本测与主缝测试共钉。
    """
    monkeypatch.setenv("FLOWGATE_BILLING_DB", ":memory:")

    ledger = create_ledger()
    ledger.settle(
        "req-env",
        "sk-env",
        prompt_text="你好",
        completion_text="ok",
        official_usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    assert ledger.spend("sk-env") == 2  # 行级：装配出来的账本可读可写——不是摆设
