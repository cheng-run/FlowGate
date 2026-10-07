"""结算挂点：Provider 门面，非流式收尾结算（issue 06；流式回填是 issue 07）。

为什么是包在 Provider 外的一层而不是路由里 try/except：路由保持传送带（零业务），
结算对上层透明——链对上层就是一个上游，本门面亦然（ADR-0001 同一纪律）。
为什么结算挂在这里（ADR-0003 清理链警告的复核结论，详见 ADR-0008）：非流式腿的
"收尾结算点"= chat() 返回/抛出的那一刻，**不在任何取消清理链（aclose/finally）里**；
落账走同步 sqlite3 短写（无 await 挂起点），"取消中不被打断"的前提一字不动。
流式腿的收尾结算（收尾时也要同步短写）是 issue 07 的挂点，口径沿用本结论。
"""

import time
from collections.abc import AsyncIterator

from app.schemas import ChatCompletionChunk, ChatCompletionResponse, ChatRequest, Usage
from billing.identity import current_key
from billing.ledger import BillingLedger
from providers.base import Provider
from routing.chain import Attempt
from routing.request_id import current_request_id, new_request_id, request_id_var


def _prompt_text(request: ChatRequest) -> str:
    """估算器的 prompt 素材：各条消息正文按行拼接——粗粒度估算只看文本量。"""
    return "\n".join(message.content for message in request.messages)


def _completion_text(response: ChatCompletionResponse) -> str:
    """估算器的 completion 素材：答复正文（W1 单选，取第一档）。"""
    return response.choices[0].message.content if response.choices else ""


class BillingProvider:
    """结算门面：包住一个 Provider，收尾时把这一轮记进账本——对上层就是一个上游。"""

    def __init__(self, inner: Provider, *, ledger: BillingLedger) -> None:
        """持住内层上游与账本；name 透传——日志/链名里看到的仍是内层的名字。"""
        self._inner = inner
        self._ledger = ledger
        self.name = inner.name

    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """非流式对话：内层答完/失败都在收尾点结算——一个逻辑请求至多一笔用户账。

        为什么结算在返回前：响应要带上结算口径的 usage（"所见即所付"，checklist 7）——
        官方数字原样、估算数字补位，客户端看到的就是账本记的。
        """
        # 行级：取号——HTTP 路径上中间件已发号；直调时这里补发，且写回背包让链同号
        request_id = current_request_id() or new_request_id()
        token = request_id_var.set(request_id)
        started = time.monotonic()  # 行级：本发尝试的起算时刻——裸上游补记要用（链自己记）
        try:
            try:
                response = await self._inner.chat(request)
            except BaseException as exc:
                # 失败/中断同样在收尾点结算：无交付=零用户账，失败尝试入损耗账——
                # 结算是同步短写（无 await），取消传播中也不被打断（ADR-0008）
                attempts = self._round_attempts(request_id)
                if not attempts:
                    # 行级：裸上游没有 attempts 账本——本发就是"一次尝试"，就地补记
                    # （checklist 3 无条件版：失败尝试的记账不因"没有链"而缺席）
                    attempts = [self._lone_attempt(request_id, exc, started)]
                self._settle_round(
                    request_id,
                    request,
                    completion_text=None,
                    official_usage=None,
                    attempts=attempts,
                )
                raise
            # 行级：收尾结算点（ADR-0008）——同步短写，不在任何取消清理链里
            settled = self._settle_round(
                request_id,
                request,
                completion_text=_completion_text(response),
                official_usage=response.usage,
                attempts=self._round_attempts(request_id),
            )
            # 行级：响应 usage=结算口径——官方/估算都是真数字，账面与所见同一份数
            return response.model_copy(update={"usage": settled})
        finally:
            request_id_var.reset(token)  # 行级：出门还原背包——号不串门（同中间件纪律）

    def _settle_round(
        self,
        request_id: str,
        request: ChatRequest,
        *,
        completion_text: str | None,
        official_usage: Usage | None,
        attempts: list[Attempt],
    ) -> Usage | None:
        """收尾结算的共用入参：号、身份、prompt 素材、本轮尝试——两条路径同一张脸。

        为什么抽这层：成功/失败只差"交没交付"两个参数（completion_text/usage），
        号、身份、prompt 素材、尝试流水的取法必须一字不差——共用一份，结算入参
        不会在两条路径上悄悄漂移（评审 Duplicated Code 的处置）。
        """
        return self._ledger.settle(
            request_id,
            current_key(),
            prompt_text=_prompt_text(request),
            completion_text=completion_text,
            official_usage=official_usage,
            attempts=attempts,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """流式对话（issue 06 不结算流式腿）：透传 + 收尾显式穿链——回填结算在 issue 07。

        为什么 finally 里必须 await stream.aclose()（评审实验证实的泄漏形态）：裸
        async for 透传时，aclose 只停在本生成器——内层生成器被弃置，要等 asyncgen
        GC finalizer 才关（W2 实验点名的形态，sse.py 的弃流盖子正防它）。显式
        aclose 把"该关了"穿到上游，与 FallbackChain.chat_stream 的 finally 同款——
        SSE 清理链（sse.py → 本门面 → 链 → 适配器）一环不缺（ADR-0008 的复核结论）。
        """
        stream = self._inner.chat_stream(request)
        try:
            async for chunk in stream:  # 行级：单流透传——06 不记账，07 在此接回填
                yield chunk
        finally:
            # 行级：收尾纪律穿链——正常耗尽是空操作；弃流时立即关掉内层，不等 GC 时机
            await stream.aclose()

    def _round_attempts(self, request_id: str) -> list[Attempt]:
        """取本轮的尝试流水入损耗账；单上游没有 attempts 账本=空（见 _lone_attempt）。

        为什么从内层账本取：attempts 流水是链的唯一观测面（ADR-0004）——损耗账
        从同一份落库，不另造第二个"失败记录"来源（spec：观测故事只有一个源）。
        """
        # 行级：鸭子类型取账本——链有 attempts 属性，裸适配器没有（getattr 兜住）
        book = getattr(self._inner, "attempts", None) or []
        # 复杂语句（推导式+条件）行上：只取本轮 request_id 的流水——不掺历史请求
        return [attempt for attempt in book if attempt.request_id == request_id]

    def _lone_attempt(self, request_id: str, exc: BaseException, started: float) -> Attempt:
        """裸上游失败时补记的那一次尝试：一发调用即一次尝试（attempt 粒度的兜底）。

        为什么需要它：链逐次尝试自记流水，裸适配器（单值装配）没有账本——不补记，
        默认装配的失败在损耗账上查无此事，"失败尝试入账"就成了只对链生效的半截承诺。
        upstream 取内层名（fake / dashscope）：补记的是"对这个上游的一次尝试"，
        与链流水同一词汇（failure_shape 取异常类名，同 routing.chain._record 口径）。
        """
        return Attempt(
            request_id=request_id,
            upstream=self._inner.name,
            started_at=started,
            ended_at=time.monotonic(),
            outcome="failed",
            failure_shape=type(exc).__name__,
            detail=str(exc),
        )
