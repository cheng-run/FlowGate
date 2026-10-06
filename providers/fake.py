"""fake 上游：测试专用的"诚实假实现"（学 otari 的 Null Object 纪律）。

为什么需要它：规范要求测试可复跑、不碰真网络，但又要走完整调用路径——
fake 就是那个确定性、零延迟、永不赖床的上游。它不继承 Provider，
仅凭方法签名就满足协议（鸭子类型），这本身就是 seam 的演示。
"""

import uuid

from app.schemas import ChatCompletionResponse, ChatMessage, ChatRequest, Choice, Usage


class FakeProvider:
    """确定性 fake 上游：把最后一条用户消息回显进固定前缀的答复。"""

    name = "fake"

    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """回显式回答：证明请求内容穿过了 seam，而不是 fake 自说自话。

        为什么回显而不是固定句：固定句测不出"请求真的传进来了"——
        回显让"数据流穿过适配器"这件事变得可断言。
        """
        # 行级：取最后一条消息做回显素材——多轮请求里最新输入最能代表"内容传进来了"。
        last_user = request.messages[-1].content

        return ChatCompletionResponse(
            id=f"fake-{uuid.uuid4().hex[:8]}",  # id 唯一即可，形状对齐 OpenAI
            model=request.model,  # 回显请求里的模型名：证明请求字段流进了适配器
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",  # 固定 assistant：OpenAI 形状的硬约定
                        content=f"fake-reply: {last_user}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(),  # 计数归 billing/（W3），fake 回 0，不装懂
        )
