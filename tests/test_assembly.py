"""装配处（composition root）测试：环境变量选上游、缺 key 响亮报错（spec 故事 5/10）。

装配处是全代码库唯一点名具体适配器的地方（ADR-0001），所以它测的是"选择"本身：
给什么环境、出什么实例、配错了喊多响——路由等核心代码的行为由端点测试兜底。
"""

import pytest

from app.main import create_provider
from providers.base import Provider
from providers.dashscope import DashScopeProvider
from providers.fake import FakeProvider
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
