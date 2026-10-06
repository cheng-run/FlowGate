"""上游适配器统一接口（模块 1/9 考点=抽象 seam 放哪，答案见 ADR-0001）。

seam 的位置就在这里：网关核心只认下面的 Provider 协议，具体上游
（DashScope / Kimi / fake）各自实现它。学的是 otari 的端口命名纪律——
接口按能力命名（Provider、chat），实现按技术命名（DashScopeProvider）。
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.schemas import ChatCompletionChunk, ChatCompletionResponse, ChatRequest


class UpstreamError(Exception):
    """上游非 2xx 时由适配器抛出的错误——Provider 协议的"失败形状"。

    为什么放 base.py：协议不只规定怎么成功，还规定怎么失败。核心代码只认
    这个异常类型，翻译成 HTTP 502 时永远不需要认识 DashScope 的错误 JSON——
    这与"核心只见协议不见实现"（ADR-0001）是同一条纪律的两面。
    消息内容约定：状态码 + 上游响应摘要，让客户端能分辨 key 错 / 请求错 / 上游挂。
    """


# @runtime_checkable 让 isinstance 可用：测试能断言"满足协议"这件事本身。
# 代价：isinstance 只查方法/属性存在，不查签名——签名错误靠测试兜（ADR-0001 取舍）。
@runtime_checkable
class Provider(Protocol):
    """上游统一接口：结构化协议，适配器不需要继承任何东西。"""

    name: str

    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """非流式对话：把统一请求翻成上游方言，调用后把结果翻回统一响应。

        为什么 async：上游调用是网络 IO，不能堵事件循环（W2 流式会更依赖这点）。
        """
        ...  # Protocol 方法体永远是省略号：它只声明形状，不做实事

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """流式对话：返回统一 chunk 的异步流——进出都是 chunk，**不是 SSE 字节**。

        为什么不是 SSE：SSE 是网关**面向客户端**的 HTTP 方言（归 streaming/ 模块），
        上游各自的 SSE 方言归各适配器，统一 chunk 居中——两头方言互不泄漏
        （ADR-0001：核心只见协议，W2 按预告在此扩方法）。

        给初学者的解释（async 生成器在本代码库首次出现）：`async def` 里带 `yield`
        的函数叫异步生成器——调用它不执行函数体，只拿到一个"惰性流"；`async for`
        每问一句，函数体才往下跑到下一个 `yield` 吐出一块。这让"上游逐块到达、
        网关逐块转发"天然对齐：内存里永远只有当前这一块，不必等整答攒完——
        打字机效果的机制本体就在这里。
        """
        ...  # 与 chat 同理：Protocol 只声明形状，实现见各适配器
