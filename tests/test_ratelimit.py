"""令牌桶限流测试（fallback-ratelimit-billing issue 05）。

受测面按 spec 测试决定：**主缝**（POST /v1/chat/completions 的 429 出口、fake 零调用、
并发计数、key 隔离性）+ **装配缝**（env 口径，用例在 tests/test_assembly.py）。
不测令牌桶内部算法——只测网关对调用方承诺的外部行为。
时钟一律 monkeypatch ratelimit.bucket.clock（W2 纪律：改模块属性，不加构造参数钩子），
零真实 sleep；限流器/上游装配绑定也走 monkeypatch（app.main.provider 先例）。
"""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from providers.fake import FakeProvider
from ratelimit.bucket import RateLimiter
from tests.conftest import auth_headers, mint_key

# 默认头带套件级 TEST_KEY（W4 认证落地后的机械件）：同 key 用例直接用默认身份
client = TestClient(app, headers=auth_headers())


def _payload(stream: bool | None = None) -> dict:
    """最小合法请求体——与端点/流式测试同款；stream 传 None 即非流式（字段不出现）。"""
    body: dict = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }
    if stream is not None:
        body["stream"] = stream  # 行级：显式控制流式开关——None 时字段不进请求体
    return body


class _FakeClock:
    """可控时钟：测试拨动 now 推进/冻结时间——限流用例零真实 sleep 的地基。

    为什么是可调用对象：生产时钟引用是 clock = time.monotonic（函数），
    鸭子类型冒充它只要"可调用且返回 float"——不给生产加 clock= 构造钩子。
    """

    def __init__(self, now: float = 1000.0) -> None:
        """起点取 1000 而不是 0：桶只关心时间差，顺手避开"从零开始"的边界误读。"""
        self.now = now

    def __call__(self) -> float:
        """像 time.monotonic 一样被调用——返回当前（可控）时刻。"""
        return self.now


def test_rate_limit_rejects_before_upstream_when_bucket_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：同一 key 打空桶后第二发 429，且拒绝发生在上游调用之前（checklist 2）。

    怎么证明：装配绑定换成容量 1 的限流器 + 记账 fake，冻结时钟（无回填）后连发
    两发同 key 非流式请求——断言第一发 200、第二发 429 且 detail 说清"退避"，
    fake.calls 恰一条（被拒那发连上游都没碰到：拒绝是零成本的）。
    """
    # 时钟先冻结再装限流器：桶的回填只看 ratelimit.bucket.clock，全程不动真时间
    monkeypatch.setattr("ratelimit.bucket.clock", _FakeClock())
    monkeypatch.setattr("app.main.limiter", RateLimiter(capacity=1, per_second=1.0))
    fake = FakeProvider()  # 行级：记账 fake——被拒请求"零上游调用"的可断言面
    monkeypatch.setattr("app.main.provider", fake)

    first = client.post("/v1/chat/completions", json=_payload())
    second = client.post("/v1/chat/completions", json=_payload())

    assert first.status_code == 200
    assert second.status_code == 429
    # 文案要说清"该退避了"（story 12）——客户端据此知道要退避而不是盲目重试
    assert "退避" in second.json()["detail"]
    # 被拒那发零上游调用（fake.calls 只有第一发的 chat）——"拒绝在上游调用之前"的证据
    assert fake.calls == ["chat"]


def test_rate_limit_rejects_stream_before_upstream_when_bucket_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：流式腿同在门卫之内——第二发 429 且上游零调用（checklist 2 的流式面）。

    怎么证明：与非流式同款布置，但请求体带 stream=true——断言第一发 200、
    第二发 429，fake.calls == ["chat_stream"]（只走了第一发；被拒那发连流都没建）。
    门卫是路由级依赖，两条腿共享同一道门——本测试钉住"流式不漏限"这件事本身。
    """
    monkeypatch.setattr("ratelimit.bucket.clock", _FakeClock())
    monkeypatch.setattr("app.main.limiter", RateLimiter(capacity=1, per_second=1.0))
    fake = FakeProvider()
    monkeypatch.setattr("app.main.provider", fake)

    first = client.post("/v1/chat/completions", json=_payload(stream=True))
    second = client.post("/v1/chat/completions", json=_payload(stream=True))

    assert first.status_code == 200
    assert second.status_code == 429
    assert "退避" in second.json()["detail"]
    # 行级：流式开工记账是"chat_stream"——被拒那发连生成器都没建，零上游调用
    assert fake.calls == ["chat_stream"]


async def test_concurrent_requests_pass_exactly_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：并发硬保证——N 路同 key 并发，恰 C 路通过、其余 429（checklist 4）。

    怎么证明：容量 3 的限流器 + 冻结时钟（全程零回填），httpx.AsyncClient 走 ASGI
    传输 gather 8 路同 key 并发请求——断言 200 恰 3 个、429 恰 5 个、fake.calls 恰 3 条。
    "恰 C"的结构原因（take 是无 await 的同步临界区，事件循环插不进手）写在
    TokenBucket.take 的"给初学者的解释"里；冻结时钟让结果与调度顺序无关，确定可复跑。
    """
    monkeypatch.setattr("ratelimit.bucket.clock", _FakeClock())
    monkeypatch.setattr("app.main.limiter", RateLimiter(capacity=3, per_second=1.0))
    fake = FakeProvider()
    monkeypatch.setattr("app.main.provider", fake)

    # async with 形态：ASGI 传输的客户端要走完 lifespan 交接（W2 断连测试同款高度）
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        # 复杂语句（推导式+gather）行上：8 路同时打同一 key——调度交错是真实的
        # （AsyncClient 没有 client 默认头，显式带套件级凭据）
        responses = await asyncio.gather(
            *[
                async_client.post(
                    "/v1/chat/completions",
                    json=_payload(),
                    headers=auth_headers(),
                )
                for _ in range(8)
            ]
        )

    # 状态码排序后逐位对账：恰 3 个 200（桶容量）、恰 5 个 429——多一个少一个都红
    assert sorted(r.status_code for r in responses) == [200, 200, 200, 429, 429, 429, 429, 429]
    assert len(fake.calls) == 3  # 行级：上游只被 3 发放行的请求碰过（拒绝零成本）


def test_burst_up_to_capacity_then_refill_at_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：容量=允许突发、速率=稳态吞吐（checklist 1）——突发被吸收，超速被速率拦。

    怎么证明：容量 3、速率 1/s + 可拨时钟，同一 key 三段推进：① 连发 4 发——恰 3 过
    1 拒（突发被桶容量吸收）；② 拨快 1s 再发 2 发——恰 1 过 1 拒（稳态速率 1/s 恰好
    回填出 1 枚，半枚/多枚都红）；③ 拨快 10s 连发 4 发——仍恰 3 过 1 拒（回填封顶
    容量，溢出的令牌不存在）。全程拨假时钟，零真实 sleep。
    """
    clock = _FakeClock()
    monkeypatch.setattr("ratelimit.bucket.clock", clock)
    monkeypatch.setattr("app.main.limiter", RateLimiter(capacity=3, per_second=1.0))
    monkeypatch.setattr("app.main.provider", FakeProvider())

    def fire(n: int) -> list[int]:
        """连发 n 发同 key 请求，返回状态码列表——三段推进共用的"打一梭子"手法。"""
        # 复杂语句（推导式）行上：默认头=套件级身份，连发 n 发共享同一只桶
        return [client.post("/v1/chat/completions", json=_payload()).status_code for _ in range(n)]

    # ① 突发段：桶出生即满（3 枚）——第 4 发见空桶
    assert fire(4) == [200, 200, 200, 429]
    # ② 稳态段：拨快恰 1s（速率 1/s）→ 恰回填 1 枚——第 2 发又见空桶
    clock.now += 1.0
    assert fire(2) == [200, 429]
    # ③ 封顶段：拨快 10s 也只回填到容量 3——第 4 发照旧 429（桶深=突发上限）
    clock.now += 10.0
    assert fire(4) == [200, 200, 200, 429]


def test_keys_have_independent_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：不同 key 的桶互不影响（checklist 5）——一个 key 被限不连累别的 key。

    怎么证明：容量 1 的限流器 + 冻结时钟，key-a 连发两发打空桶后，key-b 再发一发
    ——断言 a 的第二发 429、b 的那一发 200。反例（全局一只桶）下 b 也会 429，
    隔离性即结构：每 key 一只桶（dict 分键），互不为邻。桶的分键是 key_id——
    两把独立注册 key（不同凭据串 → 不同 key_id）就是对照组。
    """
    monkeypatch.setattr("ratelimit.bucket.clock", _FakeClock())
    monkeypatch.setattr("app.main.limiter", RateLimiter(capacity=1, per_second=1.0))
    monkeypatch.setattr("app.main.provider", FakeProvider())
    # 两把独立 key：认证后身份=key_id，隔离组必须是两个真实注册身份
    credential_a = mint_key(name="ratelimit-a")["credential"]
    credential_b = mint_key(name="ratelimit-b")["credential"]

    def fire(credential: str) -> int:
        """以指定凭据串发一发，返回状态码——同款请求只换敲门砖。"""
        return client.post(
            "/v1/chat/completions", json=_payload(), headers=auth_headers(credential)
        ).status_code

    assert fire(credential_a) == 200
    assert fire(credential_a) == 429  # 行级：key-a 的桶已空
    assert fire(credential_b) == 200  # 行级：key-b 有自己的桶——a 的空桶碍不着它


def test_anonymous_request_gets_401_without_touching_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证明：匿名请求 401 且不占桶——fail-closed 取代了 W3 的"匿名放行不占桶"。

    W3 时代本测试钉"匿名放行不占桶"（2026-10-07 用户拍板旧口径），并预告
    "W4 认证插进门卫之前后应改写成匿名 401，改写点就在这"——本测试就是那场改写：
    三发匿名全部 401，且**一令牌都没占**（容量 1 的桶还满着，带凭据的第一发仍 200）。
    证明手法：裸 client（无默认头）发匿名三发，再用带凭据 client 发第四发——
    若 401 发生在占桶之前，桶必然毫发无损；反例（先占桶再拒）第四发就会 429。
    """
    monkeypatch.setattr("ratelimit.bucket.clock", _FakeClock())
    monkeypatch.setattr("app.main.limiter", RateLimiter(capacity=1, per_second=1.0))
    monkeypatch.setattr("app.main.provider", FakeProvider())

    # 三发匿名（裸 client，请求里没有 Authorization）——fail-closed 全数 401
    bare = TestClient(app)
    statuses = [bare.post("/v1/chat/completions", json=_payload()).status_code for _ in range(3)]

    assert statuses == [401, 401, 401]
    # 带凭据的一发仍 200：匿名三发没占桶（容量 1 的令牌还在）——401 先于占桶
    assert client.post("/v1/chat/completions", json=_payload()).status_code == 200
