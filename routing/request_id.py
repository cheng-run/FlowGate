"""逻辑请求号 request_id：幂等键与对账号（W3 issue 02）。

给初学者的解释（contextvars 在本代码库首次出现）：ContextVar 是"随调用链走的隐形
背包"——HTTP 进门处（app/ 的 ASGI 中间件）把号放进去，深处的 fallback 链伸手取，
不用把 request_id 缝进每一个函数签名；asyncio 里每个请求跑在自己的任务上下文里，
背包不串门。为什么不用函数参数硬传：请求号要一路传到 attempts 记账（将来还有
billing 结算），逐层加形参会把"对账"这个横切关注点缝进每个签名——接口面被污染
（与 ADR-0001 同一纪律：横切的东西收在一个词汇里）。
为什么要有它：一个逻辑请求内部可能有 N 次尝试（A 挂 B 接），对账要把"一次请求"
与"多次尝试"缝在一起——号挂响应头给客户端，挂 attempts 给自己；billing 的用户账
按它去重（幂等键），一个逻辑请求至多一笔钱。
"""

import uuid
from contextvars import ContextVar

# 号包本体：默认 None = "还没发号"（Provider 副缝直调链、不经 HTTP 时的现场——
# 链自己补发一个，保证 attempts 永远有号可归组）
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """发号：每个逻辑请求一个（uuid4 hex，幂等键的原料——不猜、不复用）。"""
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    """取当前请求号；None = 不在 HTTP 请求里（直调 Provider 的场景调用方自行发号）。"""
    return request_id_var.get()
