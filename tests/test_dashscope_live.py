"""DashScope 真网 smoke 测试：显式打标 live，无 key 自动跳过（spec 故事 21/22）。

为什么单独一个文件：它与离线测试的性质完全不同——这是唯一会碰真网络的测试，
独立成文件让"跑离线套件"和"跑真网验证"一眼可分；CI 无 key 时它整文件 skip，
永不成为 CI 依赖（故事 22：CI 不碰真网、不见密钥）。
跑法：DASHSCOPE_API_KEY 写进 .env 后 `uv run --env-file .env pytest -m live`。
"""

import os

import pytest

from app.schemas import ChatMessage, ChatRequest
from billing.estimator import estimate_tokens
from providers.dashscope import DashScopeProvider

pytestmark = pytest.mark.live  # 模块级打标：整文件都属真网 smoke，可 -m live 单独调度


async def test_dashscope_live_smoke_returns_real_answer() -> None:
    """证明：真网也通——真 key、真上游、真回答，适配器不是只在 mock 里能跑。

    怎么证明：无 DASHSCOPE_API_KEY 时 pytest.skip（不是 xfail、不是空过——
    跳过原因打印在报告里）；有 key 则发一条最小请求，断言 OpenAI 形状 +
    至少一档候选 + usage.total_tokens > 0（真上游回真实计数，fake 只会回 0，
    这个 >0 就是"数据真的来自百炼"的凭据）。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        # 跳过而非失败：本地/CI 没 key 是常态，smoke 要"可跑"而不是"必跑"（故事 21）
        pytest.skip("未配置 DASHSCOPE_API_KEY，跳过真网 smoke")

    # 不传 transport → 走真网络；base_url 用内置默认（官方 OpenAI 兼容前缀）
    provider = DashScopeProvider(api_key=api_key)
    response = await provider.chat(
        ChatRequest(
            model="qwen-turbo",  # 便宜快模型：smoke 只证通路，不烧贵 token
            messages=[ChatMessage(role="user", content="只回复两个字：pong")],
        )
    )

    assert response.object == "chat.completion"
    assert response.choices, "真上游至少给一档候选——空 choices 说明翻译层吞了数据"
    # usage > 0 是"真网凭据"：mock 上游给的是固定值，真上游给的是真实计数
    assert response.usage.total_tokens > 0


async def test_dashscope_live_stream_yields_real_chunks() -> None:
    """证明：真网也流得动——真 DashScope 流式一条，chat_stream 真逐块吐统一 chunk
    （issue 04 checklist 4）。

    怎么证明：无 DASHSCOPE_API_KEY 时 pytest.skip（同模块既有约定，跳过原因进报告）；
    有 key 则发 stream=true 的最小请求，断言多于一块（单块=其实没流起来）、内容拼接
    非空（数据真穿过方言翻译）。只测"流得动"——帧惯例（finish_reason 等）已由离线
    mock 钉住，真网 smoke 再断言就是对上游报文细节下注，稍改就脆断（评审收窄口径）。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        # 跳过而非失败：本地/CI 没 key 是常态，smoke 要"可跑"而不是"必跑"（故事 21）
        pytest.skip("未配置 DASHSCOPE_API_KEY，跳过真网流式 smoke")

    # 不传 transport → 走真网络；base_url 用内置默认（官方 OpenAI 兼容前缀）
    provider = DashScopeProvider(api_key=api_key)
    # 复杂语句（async 推导式）行上：把整条统一 chunk 流收集成列表——逐块到货是流式的核心
    chunks = [
        c
        async for c in provider.chat_stream(
            ChatRequest(
                model="qwen-turbo",  # 便宜快模型：smoke 只证通路，不烧贵 token
                messages=[ChatMessage(role="user", content="只回复两个字：pong")],
                stream=True,
            )
        )
    ]

    assert len(chunks) >= 2, "真流式至少两块（首帧 + 内容）——单块说明上游没按流式回"
    # 复杂语句（推导式）行上：拼回完整回答——内容非空=方言翻译没把正文吞掉
    assert "".join(c.choices[0].delta.content or "" for c in chunks)


async def test_dashscope_live_estimate_error_percentage_vs_official_usage() -> None:
    """证明：自建估算器的误差可量化——live 对照官方 usage 报误差百分比（issue 07 数字出处）。

    怎么证明：无 DASHSCOPE_API_KEY 时 pytest.skip（同模块约定）；有 key 则发一条
    最小非流式请求，拿真上游的官方 usage 当真值，对同一段 prompt+completion 文本
    跑自建估算器，算误差百分比并打印（`pytest -m live -s` 的输出就是考点卡引用的
    数字）；断言误差落在宽上界内（估算本就是粗粒度启发式，上界防的是"错到离谱"，
    不是钉精度）。为什么走非流式：非流式响应的 usage 恒有（真值拿得最稳）——
    流式 usage 依赖上游支持 include_usage，拿它当真值会把"上游没回 usage"误判成
    估算器的错（估算器口径见 billing/estimator.py：CJK 每字 1 + 非 CJK 4 字符 1）。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        # 跳过而非失败：本地/CI 没 key 是常态，smoke 要"可跑"而不是"必跑"（故事 21）
        pytest.skip("未配置 DASHSCOPE_API_KEY，跳过真网计费误差 smoke")

    prompt = "用一句话介绍令牌桶限流"  # 行级：中英混杂不够真——全中文更能压出 CJK 口径的误差
    provider = DashScopeProvider(api_key=api_key)
    response = await provider.chat(
        ChatRequest(
            model="qwen-turbo",  # 便宜快模型：误差对照只测 tokenizer 口径，不烧贵 token
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
    print(f"\n[计费误差] 估算 {estimated_total} vs 官方 {official_total}，误差 {error_pct:.1f}%")
    assert error_pct < 100, (
        f"估算 {estimated_total} vs 官方 {official_total}，误差 {error_pct:.1f}%"
    )
