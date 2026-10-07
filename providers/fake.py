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
from providers.base import UpstreamError

# 流式回显句切成固定 4 块：确定性分块是 02/03 断言的地基——切法掺随机，
# "拼回等于完整回显""卡在第几块"这类断言就没法写了。
STREAM_CHUNK_COUNT = 4

# 收尾账的词汇（两腿同族）：正常跑完 / 被掐断 / 自己抛错——stream_endings 记流式
# 生成器的收场，chat_endings 记非流式协程的收场（issue 02 扩账到非流式腿，词汇不变）。
# 被取消=GeneratorExit（aclose）或取消异常穿出；断连/超时都归"被掐断"——
# 只有"消费方取空了我们 / 正常返回"才是正常跑完。"自己抛错"是 issue 01 失败注入
# 加的第三种：恒失败抛 UpstreamError 收场，与"被取消"必须可区分——W3 断言
# "谁被取消了"若把失败也算进被取消，就是假绿。
# 为什么测试断言用字面量不用这些常量：字面量是独立真源——常量改名或取值撞车
# 时测试必须红（防"断言永远为真"的假绿），这是刻意的词汇分家，不是漏用。
ENDING_COMPLETED = "completed"
ENDING_CANCELLED = "cancelled"
ENDING_FAILED = "failed"

# 失败形态词汇（issue 01 注入面）：fake 按注入的形态**按需失败**。
# 三种形态对应 W3 fallback 要仿真的三类上游死法：恒失败（拒答/5xx，换路信号）、
# 卡住（一言不发地悬着，等尝试预算放弃 + 显式取消）、空流（说完了但一句没有）。
# 为什么注入在 fake 而不是生产代码：W2 闸门先例——fake 本来就是测试设施，
# "让上游按需挂掉"是测试能力，不给生产留钩子。
FAILURE_FAIL = "fail"
FAILURE_HANG = "hang"
FAILURE_EMPTY = "empty"


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
    """确定性 fake 上游：把最后一条用户消息回显进固定前缀的答复（非流式 + 流式）。

    issue 01 起支持两个测试旋钮（纯测试设施）：`failure` 注入按需失败形态
    （恒失败/卡住/空流），`name` 给实例起名（fake-a / fake-b）——W3 的 fallback
    测试靠它们断言"谁挂了、谁接管了、谁被取消了"（见模块头的失败形态词汇）。
    """

    name = "fake"

    def __init__(
        self,
        gate: asyncio.Event | Callable[[], asyncio.Event] | None = None,
        *,
        name: str = "fake",
        failure: str | None = None,
    ) -> None:
        """可选闸门：控制 chat_stream 的产块节奏；name/failure 是 issue 01 的注入旋钮。

        为什么闸门放在 fake 而不是生产代码：W1 纪律"生产零测试钩子"——
        超时/断连测试要能把流稳稳卡住，这个能力属于测试设施，装在测试设施身上。
        为什么是 asyncio.Event：测试能显式 set 放行、可控可断言，不需要真实 sleep。
        两种形态（鸭子类型，靠 callable 区分）：传 Event 是单流共用一把；
        传工厂（返回 Event 的可调用）是**每流各拿一把**——并发用例要互不干扰，
        一把共享闸会被多流互踩（Event 一 set 唤醒所有等待者，不是计数信号量）。
        name/failure（issue 01）：实例命名（fake-a / fake-b）与按需失败形态注入
        （fail / hang / empty，词汇见模块头），W3 fallback 测试断言"谁挂了、谁接管了"。
        """
        self._gate = gate
        # 行级：非法失败形态响亮报错（与装配处"未知上游名报错"同一纪律）——
        # 静默忽略会让"本想让 a 挂掉"的 fallback 用例假绿
        if failure not in (None, FAILURE_FAIL, FAILURE_HANG, FAILURE_EMPTY):
            raise ValueError(f"未知的 failure={failure!r}（可选：fail / hang / empty 或 None）")
        # 行级：实例名覆盖类默认——fake-a / fake-b 可区分（issue 01 命名注入）；
        # 默认 "fake" 与历史行为一字不差（id 前缀、错误消息里的名字都不变）
        self.name = name
        self._failure = failure
        # 行级：恒失败消息在构造时定死一条，两腿共用——"非流式与流式失败行为一致"
        # 由构造保证，不靠两条腿各自拼消息时"恰好拼得一样"（checklist 5 的地基）
        self._fail_message = f"上游 {self.name} 返回 500: fake 注入的恒失败形态"
        # 收尾记录：每条流结束（正常跑完/被取消/自己抛错）追加一条——W2-02"断连不泄漏"
        # 断言的地基。为什么记在实例上：断言要在流生死之外事后查账，实例账本最直白
        # （fake 就是测试设施，"谁被取消了"按实例查账，W3 断言靠这个）。
        # 口径（评审裁决点）：账本记**收场方式**，不记语义上的成败——
        # 空流"一言不发地走到头"记 completed，它的"失败身份"由 seam 承担
        # （streaming/sse 把空流翻成 UpstreamError → 502）。
        self.stream_endings: list[str] = []
        # 非流式收尾账（issue 02 扩账）：与 stream_endings 同一词汇、同一"宁冤勿纵"
        # 口径。issue 01 曾定"非流式无收尾账"（失败/完成当场可见）——W3 fallback 要断言
        # "放弃的尝试被显式取消、非干等烧钱"，而"被取消"恰是 chat() 唯一看不见的收场，
        # 必须由 fake 记账作证（现实赢：此口径取代 issue 01 的裁决点，词汇不变）。
        self.chat_endings: list[str] = []
        # 调用记录：每条腿开工记一笔方法名（"chat" / "chat_stream"）——"谁被调用了
        # 几次"按实例数 len(calls)。流式腿在生成器**开工**（首块被消费）才记：
        # async 生成器创建时不执行函数体（语义见 base.py），没被消费的流不算调用
        self.calls: list[str] = []

    # 拆不动说明（chat 整块超 40 行，含 docstring/注释；同 chat_stream 的先例）：
    # 收尾账的 try/finally 必须罩住整段收场判定——响应构造、失败注入、卡住等待是
    # 同一条"收场链"的走读现场，抽小函数会把"ending 何时改判"拆到两处对照着读，
    # 收尾记账的现场反而讲不清，故保持单函数，超限以本说明豁免。
    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """回显式回答：证明请求内容穿过了 seam，而不是 fake 自说自话。

        为什么回显而不是固定句：固定句测不出"请求真的传进来了"——
        回显让"数据流穿过适配器"这件事变得可断言。
        收尾账（issue 02）：与 chat_stream 同款"默认记被取消、正常答完才改判"——
        记账在 finally 里纯同步一步跑完，取消传播途中也执行、不悬垂。
        """
        self.calls.append("chat")  # 行级：开工记账——失败/成功都先记"来过"（checklist 4）
        # 行级：默认记"被取消"（宁冤勿纵）：只有完整答完/自己抛错才改判——
        # 卡住被 cancel 时 ending 恰好停在这个默认值，就是现场的正确答案
        ending = ENDING_CANCELLED
        try:
            # 行级：恒失败注入（issue 01）——先于任何正常路径产物抛出；失败消息与流式腿
            # 同一条（构造时定死），两腿失败行为一致（checklist 5）
            if self._failure == FAILURE_FAIL:
                ending = ENDING_FAILED  # 行级：失败 ≠ 被取消，账本要能分开数
                raise UpstreamError(self._fail_message)
            # 行级：卡住注入（issue 01）——悬在产出之前；取消落在 _hang 的等待点上
            if self._failure == FAILURE_HANG:
                await self._hang()
            # 行级：取最后一条消息做回显素材——多轮请求里最新输入最能代表"内容传进来了"。
            last_user = request.messages[-1].content
            # 行级：空流注入在非流式腿的讲法是"空答复"——一言不发（checklist 5）；
            # 其余形态走到这里就是正常回显（恒失败/卡住在上面就已拦掉）
            content = "" if self._failure == FAILURE_EMPTY else f"fake-reply: {last_user}"

            response = ChatCompletionResponse(
                id=f"{self.name}-{uuid.uuid4().hex[:8]}",  # id 唯一且带实例名；形状对齐 OpenAI
                model=request.model,  # 回显请求里的模型名：证明请求字段流进了适配器
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",  # 固定 assistant：OpenAI 形状的硬约定
                            content=content,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(),  # 计数归 billing/（W3），fake 回 0，不装懂
            )
            ending = ENDING_COMPLETED  # 行级：走到返回 = 正常答完，改判收场方式
            return response
        finally:
            # 收尾落账（issue 02"放弃的尝试被显式取消"断言的地基）：正常答完/被取消/
            # 自己抛错都从这记一笔。纯同步一步跑完，取消传播途中也执行，不给 GC 留时机。
            self.chat_endings.append(ending)

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
        # 行级：开工记账（checklist 4）——本函数体在首块被消费时才执行（async 生成器
        # 语义），所以"记在开工"与"流真的动了"是同一时刻，没被消费的流不算调用
        self.calls.append("chat_stream")
        # 行级：开工取闸——工厂形态下每流一把，整个流的产块节奏都用这一把
        gate = self._stream_gate()
        # 默认记"被取消"：只有完整吐完/自己抛错才改判——收尾账本宁冤勿纵，
        # 防"断言永远为真"的假绿（02 checklist 2：三种收尾必须可区分）
        ending = ENDING_CANCELLED
        try:
            # 行级：恒失败注入（issue 01）——首块前抛，正是 W3"可报错窗口/重试窗口"
            # 依赖的失败时机；收尾记 "failed"：失败 ≠ 被取消，账本要能分开数
            if self._failure == FAILURE_FAIL:
                ending = ENDING_FAILED
                raise UpstreamError(self._fail_message)
            # 行级：空流注入（issue 01）——零块直接收场；收尾记"正常跑完"（账本记
            # 收场方式、不记语义成败，裁决口径见 stream_endings 注释）——它的
            # "失败身份"在 seam 上兑现：sse_response 把空流翻成 UpstreamError → 502
            if self._failure == FAILURE_EMPTY:
                ending = ENDING_COMPLETED
                return
            # 行级：卡住注入（issue 01）——悬在首块之前；取消穿进来时 ending 停留在
            # 默认的"被取消"（宁冤勿纵的默认值恰好就是这里的正确答案）
            if self._failure == FAILURE_HANG:
                await self._hang()
            # 行级：回显素材与 chat() 同源——证明 stream 请求也把内容送进了 seam
            text = f"fake-reply: {request.messages[-1].content}"
            # 行级：同一次流内各帧共用 id（OpenAI 惯例，客户端靠它把帧归成一组）；
            # id 带实例名（与 chat() 同款命名可见面），默认名下与历史形状一字不差
            chunk_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

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
            ending = ENDING_COMPLETED
        finally:
            # 收尾落账（02 断言的地基）：正常跑完/被取消/自己抛错都从这记一笔。
            # 被取消 = GeneratorExit（aclose）或取消异常穿出本生成器——记账是纯同步代码，
            # 取消传播途中也一步跑完，不给 GC 留时机（"断连不泄漏"的可观察面就在这）。
            self.stream_endings.append(ending)

    async def _hang(self) -> None:
        """永远等不到的等待点：卡住形态的本体，取消传播就落在这一行上。

        为什么走既有闸门通道（票面"配合既有闸门"）：给它一把**永不开闸**的闸，
        等待/取消落点与闸门卡住一字不差（_pass_gate 里同一行 wait）——不另造
        第二种悬法，W3"放弃尝试时显式取消"只用面对一种卡住现场。
        为什么不 sleep 大数：sleep 是假时间，Event.wait 是纯等待——测试零真实 sleep
        的纪律不破，取消响应也即时（取消只等当前 await，不等睡完）。
        """
        await self._pass_gate(asyncio.Event())  # 行级：一把永不开闸的闸=永远等不到放行

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
