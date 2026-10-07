"""请求的计费身份（bearer key）：进门记进背包，结算取出记账（issue 06）。

为什么与 request_id 同款"背包"手法（ContextVar，见 routing/request_id.py 的
给初学者解释）：key 在 HTTP 进门处才可见，结算却发生在请求收尾——逐层传参会把
身份缝进每个签名（Provider 协议、链、适配器全都得加形参），接口面被污染。
为什么是不透明串：W3 不做凭据的生成/校验/撤销（那是 W4 keys/ 的事）——这里只是
"限流/预算/账本共用的那一个身份串"，W4 换真凭据时只动提取处，本文件一字不改
（story 40 的插口就在这）。
"""

from contextvars import ContextVar

# 身份背包：None = 匿名请求（无 Authorization）——限流放行不占桶、预算不设限、
# 用户账记 key=NULL（对账时"匿名"仍可见，但不并入任何 key 的花销）
key_var: ContextVar[str | None] = ContextVar("billing_key", default=None)


def current_key() -> str | None:
    """取当前请求的计费身份；None = 匿名。结算与预算检查共用这一个口径。"""
    return key_var.get()
