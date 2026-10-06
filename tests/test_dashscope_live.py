"""DashScope 真网 smoke 测试：显式打标 live，无 key 自动跳过（spec 故事 21/22）。

为什么单独一个文件：它与离线测试的性质完全不同——这是唯一会碰真网络的测试，
独立成文件让"跑离线套件"和"跑真网验证"一眼可分；CI 无 key 时它整文件 skip，
永不成为 CI 依赖（故事 22：CI 不碰真网、不见密钥）。
跑法：DASHSCOPE_API_KEY 写进 .env 后 `uv run --env-file .env pytest -m live`。
"""

import os

import pytest

from app.schemas import ChatMessage, ChatRequest
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
