"""Kimi 真网 smoke 测试：显式打标 live，无 key 自动跳过（issue 08 checklist 5/6）。

为什么单独一个文件：它与离线测试的性质完全不同——这是唯一会碰真网络的测试，
独立成文件让"跑离线套件"和"跑真网验证"一眼可分（与 test_dashscope_live.py 同一纪律）；
CI 无 key 时它整文件 skip，永不成为 CI 依赖。
跑法：KIMI_API_KEY / KIMI_BASE_URL（及链用例的 DASHSCOPE_API_KEY）写进 .env 后
`uv run --env-file .env pytest -m live`。
"""

import os

import pytest

from app.main import create_provider
from app.schemas import ChatMessage, ChatRequest
from billing.estimator import estimate_tokens
from providers.kimi import KimiProvider
from routing.chain import FallbackChain

pytestmark = pytest.mark.live  # 模块级打标：整文件都属真网 smoke，可 -m live 单独调度

# 中转侧模型名（2026-09-10 实测默认模型，配置笔记在项目外）：smoke 只证通路不烧贵 token；
# 中转改模型名时这里跟着改——live 用例天然与部署环境耦合，硬编码常量比隐式读配置好排错
KIMI_MODEL = "kimi-k3"


def _kimi_env() -> tuple[str, str] | None:
    """取 Kimi 的两份必需配置；缺任何一份返回 None（调用方 skip）。

    为什么两份都要：KIMI_BASE_URL 无内置默认（真实中转地址只活在 .env）——
    live smoke 与装配处同一纪律，缺了不猜地址。
    """
    api_key = os.environ.get("KIMI_API_KEY")
    base_url = os.environ.get("KIMI_BASE_URL")
    if not api_key or not base_url:
        return None
    return api_key, base_url


async def test_kimi_live_smoke_returns_real_answer() -> None:
    """证明：真网也通——真 key、真中转、真回答，适配器不是只在 mock 里能跑。

    怎么证明：缺配置时 pytest.skip（不是 xfail、不是空过——跳过原因打印在报告里）；
    有配置则发一条最小请求，断言 OpenAI 形状 + 至少一档候选 + 内容非空。
    只测"答得上"——usage 是否回传由下一条计费误差用例专门验，避免一条用例两个失败面。
    """
    env = _kimi_env()
    if env is None:
        # 跳过而非失败：本地/CI 没 key 是常态，smoke 要"可跑"而不是"必跑"（故事 21）
        pytest.skip("未配置 KIMI_API_KEY / KIMI_BASE_URL，跳过真网 smoke")
    api_key, base_url = env

    # 不传 transport → 走真网络；base_url 由 .env 注入（适配器不内置地址）
    provider = KimiProvider(api_key=api_key, base_url=base_url)
    response = await provider.chat(
        ChatRequest(
            model=KIMI_MODEL,
            messages=[ChatMessage(role="user", content="只回复两个字：pong")],
        )
    )

    assert response.object == "chat.completion"
    assert response.choices, "真上游至少给一档候选——空 choices 说明翻译层吞了数据"
    assert response.choices[0].message.content, "真回答非空——空内容说明翻译层吞了正文"


async def test_kimi_live_stream_yields_real_chunks() -> None:
    """证明：真网也流得动——真 Kimi 流式一条，chat_stream 真逐块吐统一 chunk（checklist 1）。

    怎么证明：缺配置时 pytest.skip（同模块约定）；有配置则发 stream=true 的最小请求，
    断言多于一块（单块=其实没流起来）、内容拼接非空（数据真穿过方言翻译）。只测
    "流得动"——帧惯例已由离线 mock 钉住，真网再断言就是对上游报文细节下注（评审收窄口径）。
    """
    env = _kimi_env()
    if env is None:
        pytest.skip("未配置 KIMI_API_KEY / KIMI_BASE_URL，跳过真网流式 smoke")
    api_key, base_url = env

    provider = KimiProvider(api_key=api_key, base_url=base_url)
    # 复杂语句（async 推导式）行上：把整条统一 chunk 流收集成列表——逐块到货是流式的核心
    chunks = [
        c
        async for c in provider.chat_stream(
            ChatRequest(
                model=KIMI_MODEL,
                messages=[ChatMessage(role="user", content="只回复两个字：pong")],
                stream=True,
            )
        )
    ]

    assert len(chunks) >= 2, "真流式至少两块（首帧 + 内容）——单块说明上游没按流式回"
    # 复杂语句（推导式）行上：拼回完整回答——内容非空=方言翻译没把正文吞掉
    assert "".join(c.choices[0].delta.content or "" for c in chunks)


async def test_kimi_live_estimate_error_percentage_vs_official_usage() -> None:
    """证明：自建估算器对 Kimi 的误差可量化——live 对照官方 usage 报误差百分比
    （spec"计费误差可量化"数字出处的 Kimi 一份，与 DashScope 的同名用例成对）。

    怎么证明：缺配置时 pytest.skip；有配置则发一条最小非流式请求，拿真上游的官方
    usage 当真值，对同一段 prompt+completion 文本跑自建估算器，算误差百分比并打印
    （`pytest -m live -s` 的输出就是考点卡引用的数字）；断言误差落在宽上界内（估算
    本就是粗粒度启发式，上界防的是"错到离谱"，不是钉精度）。为什么走非流式：非流式
    响应的 usage 恒有（真值拿得最稳）；若中转不回 usage，>0 断言会红——那是中转
    数据面的事实暴露，不是估算器的错（账本侧有估算兜底，billing/estimator.py 口径）。
    """
    env = _kimi_env()
    if env is None:
        pytest.skip("未配置 KIMI_API_KEY / KIMI_BASE_URL，跳过真网计费误差 smoke")
    api_key, base_url = env

    prompt = "用一句话介绍令牌桶限流"  # 行级：全中文更能压出 CJK 口径的误差（与 DashScope 同款）
    provider = KimiProvider(api_key=api_key, base_url=base_url)
    response = await provider.chat(
        ChatRequest(
            model=KIMI_MODEL,
            messages=[ChatMessage(role="user", content=prompt)],
        )
    )

    official_total = response.usage.total_tokens
    assert official_total > 0, "真上游必回真实计数——0 说明 usage 没穿过翻译层"
    completion = response.choices[0].message.content
    # 行级：估算=对整段 prompt 与 completion 各 tokenize 一次再加总（结算口径同款）
    estimated_total = estimate_tokens(prompt) + estimate_tokens(completion)
    error_pct = abs(estimated_total - official_total) / official_total * 100
    # 行级：数字出处——这一行的输出进考点卡（"计费误差可量化"的实测百分比）
    print(
        f"\n[计费误差·Kimi] 估算 {estimated_total} vs 官方 {official_total}，误差 {error_pct:.1f}%"
    )
    assert error_pct < 100, (
        f"估算 {estimated_total} vs 官方 {official_total}，误差 {error_pct:.1f}%"
    )


async def test_kimi_live_chain_takeover_when_dashscope_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：真实双上游链可跑通且**接管是真的**——DashScope 拒答、Kimi 兜底（checklist 6）。

    怎么证明：缺任何一份配置就 skip；有配置则把 FLOWGATE_PROVIDER 钉成 dashscope,kimi
    走装配处 create_provider()——链装配与真网行为在**生产同一条入口**上验（装出的就是
    部署时那条链，不是测试手搓的替身）；发 model=kimi-k3 的请求——DashScope 不认识这个
    模型（拒答=换路信号），链按序落到 Kimi；断言最终回答非空，且链的 attempts 账本可见
    "dashscope 失败 + kimi ok"的接管现场（谁挂了谁接管按名字对账）。为什么用模型名逼
    拒答：这是真实上游间天然的分工边界，不需要给生产塞故障注入钩子（fake 的失败形态
    注入只属测试设施）。
    """
    env = _kimi_env()
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    if env is None or not dashscope_key:
        pytest.skip("未配置 DASHSCOPE_API_KEY + KIMI_API_KEY / KIMI_BASE_URL，跳过真链 smoke")

    # 行级：钉成真实双上游顺序链——装配处读的就是这份环境，与生产同一条装配路径
    monkeypatch.setenv("FLOWGATE_PROVIDER", "dashscope,kimi")
    chain = create_provider()
    assert isinstance(chain, FallbackChain)  # 行级：双元素链的装配形状护栏（离线已钉）
    response = await chain.chat(
        ChatRequest(
            model=KIMI_MODEL,  # 行级：DashScope 不认识 kimi 系模型——拒答逼出换路，Kimi 答
            messages=[ChatMessage(role="user", content="只回复两个字：pong")],
        )
    )

    assert response.choices[0].message.content, "接管棒（Kimi）给出了真回答——空内容=没答上"
    # 行级：attempts 是链的唯一观测面——接管现场按名字对账：前棒失败、后棒 ok
    assert chain.attempts[0].upstream == "dashscope"
    assert chain.attempts[0].outcome == "failed"
    assert chain.attempts[-1].upstream == "kimi"
    assert chain.attempts[-1].outcome == "ok"
