"""令牌桶限流核心：per-key 一只桶，容量=允许突发、速率=稳态吞吐（W3 issue 05）。

选令牌桶不选漏桶的取舍（容量=吸收突发 vs 漏桶=整形输出速率）见 ADR-0007。
时钟走模块级引用 clock——测试 monkeypatch 这个名字即可冻结/拨动时间（W2 纪律：
改模块属性，不加 clock= 构造参数钩子），限流用例零真实 sleep。
"""

import time

# 时钟引用（W2 纪律）：桶只问"过了多久"，测试 monkeypatch 本名字即可接管时间。
# 为什么不用构造参数注入时钟：那是给生产开测试钩子；monkeypatch 模块属性是
# W2 已验证的手法（streaming.sse.GAP_TIMEOUT_SECONDS 同款），生产代码零痕迹。
clock = time.monotonic


class RateLimitError(Exception):
    """空桶拒绝的失败形状：该 key 的令牌桶没令牌了——"请退避"（app/ 翻成 429）。

    为什么是异常不是返回 bool：与 UpstreamError → 502 同一失败词汇纪律——
    深模块只抛失败形状，HTTP 状态码翻译集中在 app/ 的异常处理器，路由保持传送带。
    """


class TokenBucket:
    """单只令牌桶：容量=突发上限（桶深），速率=每秒回填令牌数（稳态吞吐）。"""

    def __init__(self, capacity: float, per_second: float) -> None:
        """出生即满桶（先允许一轮突发）、以 per_second 持续回填、封顶 capacity。"""
        self._capacity = capacity
        self._per_second = per_second
        self._tokens = float(capacity)  # 行级：当前存量——每次请求扣 1，按时间回填
        self._last = clock()  # 行级：上次回填结算时刻（懒结算的锚点）

    def take(self) -> bool:
        """扣一枚令牌：有则扣了放行（True），空桶拒绝（False）。

        给初学者的解释（"同步临界区=原子"在本代码库首现）：asyncio 单线程跑事件
        循环，两个协程只在 await 处互相切换——本方法**一个 await 都没有**，从
        "看存量"到"扣存量"是一段不会被打断的同步代码，N 路并发也各自完整走完它。
        所以"N 路同 key 并发恰 C 路通过"是结构事实，不需要锁（上锁反而多一处
        死锁/泄漏现场）。懒结算（先回填后扣减）：不维护后台回填任务，时间只在
        有人来取令牌时被问一次——少一个后台任务，就少一处取消传播要照顾的角落
        （W2"断连不泄漏"教训的同款口味）。
        """
        now = clock()
        # 行级：max 钳住负值——时钟被拨回（测试重置/校时）不倒扣存量，只当没回填
        elapsed = max(0.0, now - self._last)
        # 行级：先按流逝时间回填、封顶容量——桶深=突发上限，溢出的令牌不存在
        self._tokens = min(self._capacity, self._tokens + elapsed * self._per_second)
        self._last = now  # 行级：结算时刻前移——回填只按上次以来的增量算，不重复计
        if self._tokens < 1.0:
            return False  # 行级：不足一枚=空桶（半枚令牌不能当一枚花）
        self._tokens -= 1.0  # 行级：扣一枚放行——take 之名的兑现
        return True


class RateLimiter:
    """per-key 令牌桶注册表：每 key 一只桶，互不影响（隔离性=结构，dict 分键）。"""

    def __init__(self, capacity: float, per_second: float) -> None:
        """容量与速率是全局限流口径（装配处从 env 读进来）；桶按 key 惰性创建。

        为什么惰性创建：key 是运行时才见到的 bearer 串——预先造桶得先认识所有
        key，那是 W4 凭据管理（keys/）的地盘；这里只在第一发到来时为它开桶，
        W4 把认证插进门卫之前时本接口一字不改（story 40 的兑现口）。
        """
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0（桶容量=允许的突发）")
        if per_second < 0:
            raise ValueError("per_second 必须 >= 0（稳态吞吐；0=永不回填）")
        self._capacity = capacity
        self._per_second = per_second
        self._buckets: dict[str, TokenBucket] = {}

    def acquire(self, key: str) -> None:
        """扣一枚该 key 的令牌；空桶抛 RateLimitError（消息说清"该退避了"）。

        为什么一次请求扣 1：LLM 客户端天然突发（一条请求一轮对话），按"进门次数"
        限流就是按请求数限流——容量=一轮对话的突发余量，取舍见 ADR-0007。
        """
        # 行级：get→建→放回 三步无 await（原子性理由同 take）——并发同 key 也只建一只桶
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self._capacity, self._per_second)
            self._buckets[key] = bucket
        if not bucket.take():
            # 消息不回显 key 本身：bearer 串是凭据，detail 会回到客户端与日志——不泄密
            raise RateLimitError(
                "请求过于频繁：令牌桶已空，请退避后重试"
                f"（容量 {self._capacity:g} 允许突发，速率 {self._per_second:g} tokens/s）"
            )
