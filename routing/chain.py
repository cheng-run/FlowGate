"""fallback 链门面：把一串上游装成**一个** Provider（W3 issue 02 非流式 / 03 流式）。

为什么链要满足 Provider 协议：链对上层就是一个上游——路由本体零改动仍是传送带
（ADR-0001 的 deep module：新增"接治理"不新增接口面）。attempts 列表是链的观测面：
每次尝试记上游名/起止时刻/结果/失败形状（otari resolve 的形状），全链失败的 detail
摘要与将来 billing 的内部损耗账都从这一份记录走（spec：观测故事只有一个源）。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.schemas import ChatCompletionChunk, ChatCompletionResponse, ChatRequest
from providers.base import Provider, UpstreamError
from routing.request_id import current_request_id, new_request_id

# 每次尝试的时间上限（issue 02 的一把尺）：卡住的尝试到点被放弃并显式取消，不拖整条链。
# 为什么先只有一把尺：预算拆分（总 30s / 每尝试 15s + 可重试失败重试 1 次）是 issue 04
# 的票面，那张票会重开 ADR-0003 的单一 gap 决策；这里先把"放弃要可取消"的机制落稳。
ATTEMPT_TIMEOUT_SECONDS = 30.0


class AttemptTimeoutError(Exception):
    """尝试预算超时（超时族的非流式成员）：上游没在预算内答完——"没按时说话"。

    为什么另立一类不复用 UpstreamError：那是"上游拒了"（502），这是"没按时说话"
    （504）——失败词汇一个词只说一件事（同 streaming.GapTimeoutError 的拆分理由）。
    为什么不继承 GapTimeoutError：那是 streaming/ 的块间预算词汇，本模块只 import
    Provider 协议与统一模型（ticket checklist / ADR-0001）——"超时族"的族关系写在
    语义与 504 出口上（app/ 异常处理器把两者翻成同一个 504），不写在类继承上。
    """


async def _call(provider: Provider, request: ChatRequest) -> ChatCompletionResponse:
    """裸调上游 chat()，把上游漏出的 TimeoutError 换成 UpstreamError——预算闹钟不许被冒充。

    为什么需要这层隔离（同 streaming._fetch 的词汇纪律）：wait_for 抛的 TimeoutError
    说的是"我们的预算到了"；若适配器自己漏出 builtin TimeoutError，两者会同款到达
    except，会被误标成"预算放弃"（换路信号），而真话是"上游自己超时了"（失败）——
    先在这换掉上游的，外面再看到 TimeoutError 就只可能是我们的闹钟。
    """
    try:
        return await provider.chat(request)
    except TimeoutError as exc:
        # 行级：归"上游答不上"族（与拒答同族）——不是"预算到了"（尝试超时）
        raise UpstreamError(f"上游 {provider.name} 调用时抛出 TimeoutError: {exc}") from exc


async def _fetch_first(
    stream: AsyncIterator[ChatCompletionChunk], upstream: str
) -> ChatCompletionChunk:
    """裸取上游流的首块：上游自带的 TimeoutError 换成 UpstreamError——首块前可现形失败的统一入口。

    为什么说"可现形"（评审收严）：拒答/空流/自带超时在这里变成异常现形；挂起类
    失败（一言不发地悬着）不会现形——放弃卡住的尝试要预算尺（尝试预算 + 显式
    取消），那是 issue 04 的票面，本票不预支半套预算。
    为什么需要这层隔离（同 _call / streaming._fetch 的词汇纪律）：将来尝试预算的
    闹钟（issue 04）抛的 TimeoutError 说的是"预算到了"（换路信号）；若适配器自己
    漏出 builtin TimeoutError，两款会同款到达 except，会被误标成"预算放弃"——
    先在这换掉上游的，外面的失败词汇就只认 UpstreamError 一种形状（ADR-0004 决定 3）。
    为什么空流也在这翻成 UpstreamError（W2 空流口径延续到链上）：上游一言不发地
    走完，对客户端没有任何可答复的内容——单上游路径 sse_response 翻 UpstreamError
    → 502，链上这是**换路信号**（A 空流，B 还能答）；词汇仍是一族，不另造异常。
    """
    try:
        # 行级：异步生成器第一次被消费才执行函数体——"开工记账/失败注入"都发生在这一行之后
        return await anext(stream)
    except StopAsyncIteration:
        # 行级：一块都没有就走完了——归"答不上"族，死因直说"一言不发"
        raise UpstreamError(f"上游 {upstream} 流在首块前结束，未产出任何 chunk") from None
    except TimeoutError as exc:
        # 行级：归"上游答不上"族（与拒答同族）——不是"没按时说话"（超时族）
        raise UpstreamError(f"上游 {upstream} 取块时抛出 TimeoutError: {exc}") from exc


@dataclass
class Attempt:
    """一次尝试的记录（otari resolve 的形状）：谁、何时、结果、失败形状。

    为什么是 dataclass 不是 Pydantic：这是**内部**观测面（attempts 账本），不进 wire——
    wire 形状归 app/schemas（单一真源），两边各管一事，不混用一个基类。
    """

    request_id: str  # 逻辑请求号：同一请求的多次尝试共用一个（对账/幂等的缝线）
    upstream: str  # 上游名（fake-a / dashscope …）——"谁挂了谁接管"按名字对账
    started_at: float  # 起算时刻（time.monotonic()：墙钟会被校时跳变，单调钟才能量耗时）
    ended_at: float  # 结束时刻；与 started_at 相减即尝试耗时
    outcome: str  # 结果词汇：ok / failed / timeout
    failure_shape: str | None  # 失败形状（异常类名）；成功为 None——"形状说话"的原料
    detail: str | None  # 失败消息原文（进全链失败的 attempts 摘要）；成功为 None


def _format_attempts(attempts: list[Attempt]) -> str:
    """把一轮尝试渲染成一行摘要：每段"上游 → 形状: 死因"——排错不用回源码。

    为什么成功段只标 ok 不注水：摘要是给"全链失败"看的死因清单，成功的段落只
    需要"它答上了"这一个事实，把答复内容塞进错误 detail 反而淹没死因。
    """
    # 复杂语句（推导式+条件表达式）行上：失败段带形状与死因原文，成功段只报 ok
    parts = [
        f"{a.upstream} → {a.failure_shape}: {a.detail}"
        if a.outcome != "ok"
        else f"{a.upstream} → ok"
        for a in attempts
    ]
    return "；".join(parts)


def _raise_chain_failed(last_error: Exception | None, attempts: list[Attempt]) -> None:
    """全链失败收尾：最后一棒的失败形状说话，message 附 attempts 摘要——永远抛，不返回。

    为什么形状要说话、不统一翻 502：拒答族（UpstreamError）→ 502 与超时族
    （AttemptTimeoutError）→ 504 的客户端重试直觉相反（502 该换路，504 值得等）——
    出口状态码由最后一棒定（checklist 4）。摘要为什么拼进 message：detail 就是
    异常 str，"每一段尝试的死因可直读"因此在 HTTP 出口一字不改地成立。
    """
    summary = f"fallback 链全链失败：{_format_attempts(attempts)}"
    if isinstance(last_error, AttemptTimeoutError):
        raise AttemptTimeoutError(summary) from last_error
    raise UpstreamError(summary) from last_error


class FallbackChain:
    """顺序 fallback 链门面：A 挂 B 接管；对上层就是一个普通的 Provider。"""

    def __init__(self, providers: list[Provider]) -> None:
        """按给定顺序持有上游（顺序即 fallback 优先级——A 先试，A 挂了才轮到 B）。

        为什么空链响亮报错：空链跑到请求时才炸会离配置现场十万八千里（配错喊响纪律）。
        为什么拷贝一份列表：调用方随后改自己的列表不该悄悄改掉链的顺序。
        """
        if not providers:
            raise ValueError("fallback 链至少要有一个上游（空链是配置错误）")
        self._providers = list(providers)
        # 行级：链名=各上游名相连——日志/排错里一眼看清链的构成；协议要求 name 属性
        self.name = "+".join(p.name for p in providers)
        # attempts 账本：只追加、跨请求累积（每条带 request_id 归组）——链的唯一观测面，
        # 将来 billing 的内部损耗账从这里落库；并发请求各自 append 不互相覆盖
        self.attempts: list[Attempt] = []

    # 拆不动说明（chat 整块超 40 行，含 docstring/注释；同 providers/fake.py 先例）：
    # 尝试回路 + 三种收场判定（ok/拒答/预算放弃）+ 取消传播讲解是**同一条**失败处理链
    # 的走读现场——拆出"单次尝试"子函数会把失败词汇的判定（谁是换路信号、谁是预算
    # 闹钟）拆到两处对照着读；全链失败收尾已提成 _raise_chain_failed（微拆先行），
    # 教学注释是规范硬要求删不得，余量超限以本说明豁免。
    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """非流式对话：按序尝试，A 挂 B 接管——全链失败按最后一棒的失败形状说话。

        为什么换路信号只认 UpstreamError：Provider 协议规定适配器把一切上游失败翻成
        这一种形状（ADR-0002：新上游错误语义零新代码）——别的异常=自己的 bug，响亮
        穿出不吞掉，否则 fallback 会把编程错误藏成"换路成功"。
        给初学者的解释（asyncio.wait_for 的首现段落在 streaming/sse.py，这里补本票
        用法）：wait_for(协程, 秒数) 给这次尝试上闹钟——预算内答完就原样返回；到点
        还没答完，它先把取消穿进上游协程（"显式取消"的机制本体，fake 收尾记
        "被取消"就是这一步的证据），再抛 TimeoutError。
        """
        # 行级：取号——HTTP 路径上中间件已发号（ContextVar 里），直调链时这里补发一个；
        # 一轮尝试共用一号，attempts 才能按"逻辑请求"归组对账（checklist 5）
        request_id = current_request_id() or new_request_id()
        # 行级：本轮尝试的流水——全链失败时摘要只报本轮，不掺账本里的历史请求
        round_attempts: list[Attempt] = []
        last_error: Exception | None = None
        for provider in self._providers:  # 行级：顺序即优先级——A 先试，A 挂才轮到 B
            started = time.monotonic()
            try:
                # 行级：给每次尝试上预算闹钟——卡住的尝试到点被放弃；取消传播见 docstring
                response = await asyncio.wait_for(
                    _call(provider, request), timeout=ATTEMPT_TIMEOUT_SECONDS
                )
            except TimeoutError:
                # 行级：只可能是预算闹钟（_call 已把上游漏的 TimeoutError 换成 UpstreamError）——
                # 记 timeout 收场（被放弃 ≠ 被拒答），换下一棒
                err = AttemptTimeoutError(f"尝试预算 {ATTEMPT_TIMEOUT_SECONDS:g} 秒内未完成响应")
                round_attempts.append(
                    self._record(request_id, provider.name, started, "timeout", err)
                )
                last_error = err
                continue
            except UpstreamError as exc:
                # 行级：拒答/够不着上游都归这类——记账后换下一棒，客户端此时还一无所知
                round_attempts.append(
                    self._record(request_id, provider.name, started, "failed", exc)
                )
                last_error = exc
                continue
            # 行级：成功即收工——后棒不再尝试（A 答了就绝不惊动 B）
            round_attempts.append(self._record(request_id, provider.name, started, "ok", None))
            return response
        # 行级：走到这=全链失败；下一行永远抛（函数名即语义），收尾逻辑见它自己的 docstring
        _raise_chain_failed(last_error, round_attempts)

    def _record(
        self,
        request_id: str,
        upstream: str,
        started: float,
        outcome: str,
        error: Exception | None,
    ) -> Attempt:
        """记一次尝试入账（attempts 只追加）并返回本条：失败形状取异常类名，消息留原文。"""
        attempt = Attempt(
            request_id=request_id,
            upstream=upstream,
            started_at=started,
            ended_at=time.monotonic(),
            outcome=outcome,
            failure_shape=type(error).__name__ if error else None,
            detail=str(error) if error else None,
        )
        self.attempts.append(attempt)
        return attempt

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """流式对话：换路只发生在首块之前，首块一到手就永远单流走到底（issue 03）。

        为什么换路只活在 _acquire_stream 里（换路窗口由**代码结构**保证）：它 return
        = 首块到手 = 窗口关死；本函数剩下的只有一条透传回路，回路里没有第二个上游
        可引用——"首块后绝不换路"不是注释约定，是"想换也没有代码可走"。双半截缝合
        （A 的半截 + B 的半截拼成两段回答）因此在结构上不可能。
        首块后的失败怎么处置（W2 错误契约原样成立）：异常沿透传回路穿出本生成器，
        streaming/sse 的收尾分支只截断、不伪造 [DONE]、死因进日志——链不吞、不翻译、
        不补帧，"半截/完整可辨"的客户端承诺一字不动。
        给初学者的解释（透传回路在 routing/ 首现，async 生成器本体见 providers/base.py）：
        `async for chunk in stream` 是"上游吐一块、我转一块"的搬运工循环——与
        streaming/sse 里消费本生成器的是同一形态，链条上每一环都这么接。
        """
        # 行级：换路全在这一步——返回即"首块到手、流已承诺给唯一一个上游"
        stream, first = await self._acquire_stream(request)
        try:
            yield first  # 行级：首块先交出去（与 sse_response"取到手才建响应"同一节奏）
            async for chunk in stream:  # 行级：单流透传——这段代码里没有第二个上游
                yield chunk
        finally:
            # 行级：收尾纪律（W2 显式收尾穿链到这里）——正常耗尽是空操作；弃流/截断时
            # 立刻关掉上游生成器，fake 的收尾账因此照常记"被取消"，不等 GC 时机
            await stream.aclose()

    # 拆不动说明（_acquire_stream 整块超 40 行，含 docstring/注释；同 chat 的先例）：
    # 尝试回路 + 两种收场判定（失败换棒/首块关窗）+ 取消传播讲解是**同一条**换路处理链
    # 的走读现场——拆出"单次尝试"子函数会把"什么算换路信号"的判定拆到两处对照着读。
    # 与 chat 的记账骨架同形是**刻意**的词汇复用（两腿同族，同 fake 两腿收尾账的先例），
    # 抽共享簿记会把它变成回调迷宫，得不偿失；教学注释是规范硬要求删不得，
    # 余量超限以本说明豁免。
    async def _acquire_stream(
        self, request: ChatRequest
    ) -> tuple[AsyncIterator[ChatCompletionChunk], ChatCompletionChunk]:
        """流式换路窗口的本体：逐棒取首块，谁先交出首块谁接管——交出即 return 关窗。

        为什么"首块到手"是关窗点：那是 W2"可报错窗口"的边界——首块前失败还能用
        状态码说话（502/504）、还能换路；首块后 200 已在路上，再换路就是把两个半截
        缝成两段回答（checklist 的反例现场）。为什么用 return 关窗不用标志位：函数
        返回后循环自然消亡——窗口关闭是控制流的事实，不是要人记得检查的布尔。
        attempts 口径：记的是**窗口竞争**的结果（谁在首块前死了、谁拿到了首块）；
        首块后的截断不回改这里——那是流的收尾账（fake.stream_endings）与日志的活，
        两本账各说一事，billing 按已送达结算（issue 07）不需要这里撒谎。
        """
        # 行级：取号——HTTP 路径上中间件已发号（ContextVar 里），直调链时这里补发一个；
        # 一轮尝试共用一号，流式 attempts 与响应头/将来账本对得上（checklist 5）
        request_id = current_request_id() or new_request_id()
        # 行级：本轮尝试的流水——全链失败时摘要只报本轮，不掺账本里的历史请求
        round_attempts: list[Attempt] = []
        last_error: Exception | None = None
        for provider in self._providers:  # 行级：顺序即优先级——A 先试，A 挂才轮到 B
            started = time.monotonic()
            # 行级：异步生成器懒执行——到这里还什么都没发生，首块在下一行才真的去取
            stream = provider.chat_stream(request)
            try:
                # 行级：首块前可现形的失败（拒答/空流/自带超时）都在这一行现形——
                # 取到手才算这一棒接管成立；挂起不现形，归 issue 04 的预算票面
                first = await _fetch_first(stream, provider.name)
            except UpstreamError as exc:
                # 行级：拒答/空流/上游自带超时（都已翻成 UpstreamError）——显式收尾
                # 放弃的尝试（不悬垂），记账后换下一棒，客户端此时还一无所知
                await stream.aclose()
                round_attempts.append(
                    self._record(request_id, provider.name, started, "failed", exc)
                )
                last_error = exc
                continue
            except BaseException:
                # 行级：取消等非失败异常（断连/外层预算）响亮穿出，但收尾纪律不变——
                # 生成器通常已终止，这里的 aclose 多是空操作，兜"还活着"的漏网形态
                await stream.aclose()
                raise
            # 行级：首块到手=接管成立——记 ok、return 关窗；此行之后本函数再无"下一棒"
            round_attempts.append(self._record(request_id, provider.name, started, "ok", None))
            return stream, first
        # 行级：走到这=全链失败；下一行永远抛（函数名即语义），形状说话见它自己的 docstring
        _raise_chain_failed(last_error, round_attempts)
