"""fake 上游：测试专用的"诚实假实现"（学 otari 的 Null Object 纪律）。

为什么需要它：规范要求测试可复跑、不碰真网络，但又要走完整调用路径——
fake 就是那个确定性、零延迟、永不赖床的上游。它不继承 Provider，
仅凭方法签名就满足协议（鸭子类型），这本身就是 seam 的演示。
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable

from app.schemas import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatRequest,
    Choice,
    DeltaMessage,
    StreamChoice,
    Usage,
)

# 流式回显句切成固定 4 块：确定性分块是 02/03 断言的地基——切法掺随机，
# "拼回等于完整回显""卡在第几块"这类断言就没法写了。
STREAM_CHUNK_COUNT = 4

# 流的两种收尾方式（fake 的记账词汇）：正常耗尽 vs 被掐断。
# 被取消=GeneratorExit（aclose）或取消异常穿出生成器；断连/超时/异常都归"被掐断"——
# 只有"消费方取空了我们"才是正常跑完，二分干净，两种收尾各自可断言。
# 为什么测试断言用字面量不用这两个常量：字面量是独立真源——常量改名或两值撞车
# 时测试必须红（防"断言永远为真"的假绿），这是刻意的词汇分家，不是漏用。
STREAM_COMPLETED = "completed"
STREAM_CANCELLED = "cancelled"


def _chunk(
    chunk_id: str,
    model: str,
    *,
    role: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    """把三种帧（首块 role / 中块 content / 末块 finish_reason）的拼装收成一处。

    为什么抽这层：chat_stream 要造多个只差字段的同构 chunk，逐个手写
    ChatCompletionChunk(...) 会把"帧形状"重复多遍——形状将来一改（W3 加 usage）
    要改多处。参数化成唯一构造点，形状真源仍是一处。
    """
    return ChatCompletionChunk(
        id=chunk_id,
        model=model,
        choices=[
            StreamChoice(
                delta=DeltaMessage(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )


class FakeProvider:
    """确定性 fake 上游：把最后一条用户消息回显进固定前缀的答复（非流式 + 流式）。"""

    name = "fake"

    def __init__(self, gate: asyncio.Event | Callable[[], asyncio.Event] | None = None) -> None:
        """可选闸门：控制 chat_stream 的产块节奏（spec 决定：fake 本来就是测试设施）。

        为什么闸门放在 fake 而不是生产代码：W1 纪律"生产零测试钩子"——
        超时/断连测试要能把流稳稳卡住，这个能力属于测试设施，装在测试设施身上。
        为什么是 asyncio.Event：测试能显式 set 放行、可控可断言，不需要真实 sleep。
        两种形态（鸭子类型，靠 callable 区分）：传 Event 是单流共用一把；
        传工厂（返回 Event 的可调用）是**每流各拿一把**——并发用例要互不干扰，
        一把共享闸会被多流互踩（Event 一 set 唤醒所有等待者，不是计数信号量）。
        """
        self._gate = gate
        # 收尾记录：每条流结束（正常跑完/被取消）追加一条——02"断连不泄漏"断言的地基。
        # 为什么记在实例上：断言要在流生死之外事后查账，实例账本最直白（fake 就是测试设施）。
        self.stream_endings: list[str] = []

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

    # 拆不动说明（chat_stream 整块超 40 行，含注释）：帧序列+闸门+收尾记账是同一条
    # 取消传播链的走读现场，抽子生成器会把链条拆成两截（内层还得再显式 aclose 一次——
    # 正是本票在防的泄漏形态），得不偿失，故保持单函数。
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """回显句的确定性流式版：首块宣告 role、中块按固定 N 块切、末块给 finish_reason。

        与 chat() 共用同一条回显公式（fake-reply: <最后一条用户消息>）——
        非流式与流式两条路径拼回的是同一句话，客户端怎么切着看都对得上。
        为什么 async 生成器（协议形状，见 providers/base.py 的给初学者解释）：
        每次 yield 一块，消费方（streaming/ 传送带）拿到一块转发一块。
        """
        # 行级：开工取闸——工厂形态下每流一把，整个流的产块节奏都用这一把
        gate = self._stream_gate()
        # 默认记"被取消"：只有完整吐完才改判——收尾账本宁冤勿纵，
        # 防"断言永远为真"的假绿（02 checklist 2：两种收尾必须可区分）
        ending = STREAM_CANCELLED
        try:
            # 行级：回显素材与 chat() 同源——证明 stream 请求也把内容送进了 seam
            text = f"fake-reply: {request.messages[-1].content}"
            # 行级：同一次流内各帧共用 id（OpenAI 惯例，客户端靠它把帧归成一组）
            chunk_id = f"fake-{uuid.uuid4().hex[:8]}"

            # 行级：首帧也过闸——03 的"首块前卡住"就卡在这道闸上
            await self._pass_gate(gate)
            # 行级：首帧宣告角色（OpenAI 惯例：先说"我是 assistant"，content 先给空串）
            yield _chunk(chunk_id, request.model, role="assistant", content="")

            # 行级：确定性切块——第 i 块 = text[i*L//N : (i+1)*L//N]，
            # 按长度等比分、无随机源：同样输入永远切出同样的边界
            for i in range(STREAM_CHUNK_COUNT):
                part = text[
                    i * len(text) // STREAM_CHUNK_COUNT : (i + 1) * len(text) // STREAM_CHUNK_COUNT
                ]
                if not part:  # 短句切出空段就跳过——空 content 帧没有信息量
                    continue
                await self._pass_gate(gate)
                yield _chunk(chunk_id, request.model, content=part)

            # 行级：末帧给结束信号（delta 空补丁 + finish_reason=stop），客户端据此收尾
            await self._pass_gate(gate)
            yield _chunk(chunk_id, request.model, finish_reason="stop")
            # 走到这里 = 消费方又取了一次而我们没有了——生成器正常耗尽，改判"正常跑完"
            ending = STREAM_COMPLETED
        finally:
            # 收尾落账（02 断言的地基）：正常跑完/被取消都从这记一笔。
            # 被取消 = GeneratorExit（aclose）或取消异常穿出本生成器——记账是纯同步代码，
            # 取消传播途中也一步跑完，不给 GC 留时机（"断连不泄漏"的可观察面就在这）。
            self.stream_endings.append(ending)

    def _stream_gate(self) -> asyncio.Event | None:
        """开工取闸：传 Event 就共用一把（单流），传工厂就每流新建一把（并发互不干扰）。

        为什么工厂形态返回的是调用者给的函数的结果：闸门的初始状态（开/关）由
        测试决定——预放行=首帧直通后卡住，关着=等测试逐帧放行，fake 不猜。
        """
        # 行级：Event 实例不可调用，callable() 就是两种形态的分界（鸭子类型，零注册表）
        return self._gate() if callable(self._gate) else self._gate

    async def _pass_gate(self, gate: asyncio.Event | None) -> None:
        """产一块之前过一次闸：闸门关着就等，放行一块后立刻关门（turnstile 节奏）。

        为什么放行即关：普通 Event 一 set 就永远开，测试没法"只放一块"——
        02/03 要的是"把流稳稳按在第 k 块"的确定性节奏，每一帧都得重新开闸。
        为什么闸门为 None 时直接放行：生产装配（create_provider）不传闸门，
        fake 在生产路径上零开销、零等待——闸门纯粹是测试可选件。
        """
        if gate is not None:
            await gate.wait()  # 等待点：卡住的流就停在这一行，取消也落在这一行
            gate.clear()  # 行级：放行一块立即关门——下次产块要重新 set
