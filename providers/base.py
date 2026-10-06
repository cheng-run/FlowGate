"""上游适配器统一接口（模块 1/9 考点=抽象 seam 放哪，答案见 ADR-0001）。

seam 的位置就在这里：网关核心只认下面的 Provider 协议，具体上游
（DashScope / Kimi / fake）各自实现它。学的是 otari 的端口命名纪律——
接口按能力命名（Provider、chat），实现按技术命名（DashScopeProvider）。
"""

from typing import Protocol, runtime_checkable

from app.schemas import ChatCompletionResponse, ChatRequest


# @runtime_checkable 让 isinstance 可用：测试能断言"满足协议"这件事本身。
# 代价：isinstance 只查方法/属性存在，不查签名——签名错误靠测试兜（ADR-0001 取舍）。
@runtime_checkable
class Provider(Protocol):
    """上游统一接口：结构化协议，适配器不需要继承任何东西。"""

    name: str

    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """非流式对话：把统一请求翻成上游方言，调用后把结果翻回统一响应。

        为什么 async：上游调用是网络 IO，不能堵事件循环（W2 流式会更依赖这点）。
        流式 chat_stream 是 W2 的事，到时协议加方法、各适配器补实现。
        """
        ...  # Protocol 方法体永远是省略号：它只声明形状，不做实事
