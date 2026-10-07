"""fallback 链门面：把一串上游装成**一个** Provider（W3 issue 02，链本体）。

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
        """流式 fallback 是 issue 03 的票面——这里响亮报错，不装死直通第一棒。

        为什么不直通第一棒凑合：静默降级会让"配了双上游，流式怎么不换路"变成悬案
        （与装配处"配错喊响"同一纪律）；门面形状先做齐（协议三件套），行为长在 03。
        """
        raise NotImplementedError(
            "fallback 链的流式 fallback 尚未实现（issue 03）；非流式 chat() 已可用"
        )
        yield  # pragma: no cover —— 凑 async 生成器形状（同 test_streaming 桩写法）
