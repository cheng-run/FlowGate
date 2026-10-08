"""请求的计费身份（key_id）：进门记进背包，认证改写，结算取出记账（issue 06）。

为什么与 request_id 同款"背包"手法（ContextVar，见 routing/request_id.py 的
给初学者解释）：身份在 HTTP 进门处才可见，结算却发生在请求收尾——逐层传参会把
身份缝进每个签名（Provider 协议、链、适配器全都得加形参），接口面被污染。
身份口径（W4 票 02）：中间件进门放的是凭据串，认证门卫（auth_gate）验过后把
背包**改写成 key_id**——本模块只是那个背包，改写逻辑不在这（story 40 的兑现）。
账本/限流/预算全按 key_id 对账，凭据串验过即弃、不落任何库表（story 15）。
"""

from contextvars import ContextVar

# 身份背包：None = 匿名请求（无 Authorization）——W4 起匿名在认证门就 401
# （fail-closed），走到结算的必是 key_id；None 只可能出现在 /health 这类
# 不挂认证的旁路（那里也不读背包）
key_var: ContextVar[str | None] = ContextVar("billing_key", default=None)


def current_key() -> str | None:
    """取当前请求的计费身份（key_id）；None = 匿名。结算与预算检查共用这一个口径。"""
    return key_var.get()
