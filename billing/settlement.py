"""结算挂点：Provider 门面，收尾结算（issue 06 非流式 + issue 07 流式末回填）。

为什么是包在 Provider 外的一层而不是路由里 try/except：路由保持传送带（零业务），
结算对上层透明——链对上层就是一个上游，本门面亦然（ADR-0001 同一纪律）。
为什么结算挂在这里（ADR-0003 清理链警告的复核结论，详见 ADR-0008）：非流式腿的
"收尾结算点"= chat() 返回/抛出的那一刻，**不在任何取消清理链（aclose/finally）里**；
落账走同步 sqlite3 短写（无 await 挂起点），"取消中不被打断"的前提一字不动。
流式腿（issue 07）沿用同一结论：回填结算在流耗尽/失败传播的收尾点做同步短写，
断连/截断路径同样结算且靠账本幂等兜底——两腿的挂点纪律一字不差。
"""

import time
from collections.abc import AsyncIterator

from app.schemas import ChatCompletionChunk, ChatCompletionResponse, ChatRequest, Usage
from billing.identity import current_key
from billing.ledger import BillingLedger
from providers.base import Provider, UpstreamError
from routing.chain import Attempt
from routing.request_id import current_request_id, new_request_id, request_id_var

# anext 的"流尽"哨兵（取号窗的空流判定）：object() 唯一实例，`is` 判身份、永不与
# 真实 chunk 撞车——不用 None 是因为 None 在别的口径里已有"无"的语义，混用会串话
_NO_CHUNK: object = object()


def _prompt_text(request: ChatRequest) -> str:
    """估算器的 prompt 素材：各条消息正文按行拼接——粗粒度估算只看文本量。"""
    return "\n".join(message.content for message in request.messages)


def _completion_text(response: ChatCompletionResponse) -> str:
    """估算器的 completion 素材：答复正文（W1 单选，取第一档）。"""
    return response.choices[0].message.content if response.choices else ""


def _chunk_text(chunk: ChatCompletionChunk) -> str:
    """一块 chunk 的正文增量（各档 delta.content 拼接）——流式"已送达文字"的唯一记账口。

    为什么逐块记增量、结算时才拼整段：估算器对整段 tokenize（不按块加总，estimator
    口径）——但流是逐块到货的，"已送达"只能边走边收，收尾把增量拼回整段再结算。
    """
    # 复杂语句（推导式）行上：W1 单选=一档；多档形状先按顺序拼齐不丢字
    return "".join(choice.delta.content or "" for choice in chunk.choices)


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

    # 拆不动说明（chat_stream 整块超 40 行、try/while/if/if 嵌套超 3 层，含 docstring/
    # 注释；同 routing.chain 的先例）：取号窗 + 透传记账回路 + 正常收尾结算 +
    # include_usage 发帧 + 异常收尾结算是**同一条**"流怎么死、账怎么结"的走读现场——
    # 拆子生成器会把"已送达文字/官方 usage"的状态在两层间倒手，结算三档的判定
    # （正常耗尽/截断/首 token 前）也得拆到两处对照着读；教学注释是规范硬要求删不得，
    # 余量与嵌套超限以本说明豁免。
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """流式对话：透传期间不记账，收尾回填结算——官方 usage 优先、缺失对整段估算。

        结算三档（checklist 1/3/4 的口径；"首 token"=已送达文字的第一个字，role/finish
        空帧不算交付文字）：**正常耗尽**按已送达结算（空答复只结 prompt，与非流式空
        答复同口径）；**首 token 后死**（截断/断连/弃流）按已送达结算——用户不为未
        收到的文字付钱，官方计数（若有）是整段生成口径，截断时**不用**（用了就是为
        没送到的字付钱）；**首 token 前死**零用户账、尝试进损耗账（story 19）。
        为什么流末才算（spec 计划§四.1）：SSE 块默认不带计数，按块加总会把每块的
        取整误差累加放大——收尾对**整段**已送达文本 tokenize 一次（estimator 口径），
        上游末帧带官方 usage 则用之。断连/截断路径同样在收尾结算：两条路径撞车
        （如弃流恰好弃在 usage 帧上）靠 ledger 的幂等 no-op——一条流无论怎么死都不
        双扣（checklist 6）。
        include_usage（OpenAI 兼容，story 22）：请求置位时流末在 [DONE] 之前恰多发
        一帧 usage chunk；透传帧的 usage 字段被洗掉、上游的 usage 载体帧（choices
        空）整帧不透传——**usage 出口唯一**，客户端线上的 usage 恰 0 帧（不请求）或
        1 帧（请求），数字=结算口径（所见即所付）。
        给初学者的解释（取号窗——async 生成器 + ContextVar 的坑，2026-10-07 实测）：
        ContextVar 的 set/reset 像借书/还书，**必须在同一个"房间"（Context）里**。
        async 生成器的每次恢复都可能跑在不同 Task 的 context 副本里（sse.py 的
        wait_for 就给每次取块开了新 Task）——在开头 set、在结尾 reset，还原时早换了
        房间，Python 直接 ValueError。所以 set/reset 只罩住"内层开工那一下"（链在
        内层第一次被消费时取号记 attempts），借还在同一个恢复帧内完成。
        为什么收尾结算是同步短写（ADR-0008 沿用）：正常路径在流耗尽那一刻结算；
        异常/弃流路径在失败传播（含 GeneratorExit）里结算——两条都是**同步**账本
        写入、零 await 挂起点，ADR-0003 清理链"全同步收尾"的前提一字不动。
        为什么 finally 里必须 await stream.aclose()（评审实验证实的泄漏形态）：裸
        async for 透传时，aclose 只停在本生成器——内层生成器被弃置，要等 asyncgen
        GC finalizer 才关（W2 实验点名的形态，sse.py 的弃流盖子正防它）。显式
        aclose 把"该关了"穿到上游，与 FallbackChain.chat_stream 的 finally 同款——
        SSE 清理链（sse.py → 本门面 → 链 → 适配器）一环不缺（ADR-0008 的复核结论）。
        """
        # 行级：取号——HTTP 路径上中间件已发号；直调时这里补发，且写回背包让链同号
        request_id = current_request_id() or new_request_id()
        started = time.monotonic()  # 行级：本发尝试的起算时刻——裸上游补记要用（同 chat）
        stream = self._inner.chat_stream(request)
        delivered: list[str] = []  # 行级：已送达文字按块收——结算整段拼接，不按块加总
        official: Usage | None = None  # 行级：上游帧的官方 usage——后见覆盖先见（末帧优先）
        head: ChatCompletionChunk | None = None  # 行级：首块——usage 帧的 id/model 素材
        try:
            # 取号窗（机制本体见 docstring 给初学者的解释）：借号→开工→还号一气呵成，
            # 本恢复帧内完成——内层（链）恰在此刻取号记 attempts，两边同号对得上账
            token = request_id_var.set(request_id)
            try:
                chunk = await anext(stream, _NO_CHUNK)  # 行级：开工取首块（空流=哨兵值）
            finally:
                request_id_var.reset(token)  # 行级：还号——同一帧内，永不再踩 context 坑
            while chunk is not _NO_CHUNK:  # 行级：逐块记账 + 透传——哨兵=上游正常说完了
                if chunk.usage is not None:
                    official = chunk.usage  # 行级：官方数字素材——结算"有则用"就取它
                if chunk.choices:  # 行级：usage 载体帧（choices 空）整帧不透传——usage 出口唯一
                    if head is None:
                        head = chunk  # 行级：记住首块——末尾 usage 帧沿用流内 id/model（惯例）
                    delivered.append(_chunk_text(chunk))
                    # 行级：透传帧洗掉 usage 字段——usage 只从结算出口走，别处不漏给客户端
                    yield (
                        chunk.model_copy(update={"usage": None})
                        if chunk.usage is not None
                        else chunk
                    )
                chunk = await anext(stream, _NO_CHUNK)  # 行级：取下一块——空流/耗尽落哨兵
            # 行级：正常耗尽=收尾结算点（ADR-0008）——同步短写；从未交付帧（空流）=零用户账
            attempts = self._round_attempts(request_id)
            if not delivered and not attempts:
                # 行级：裸上游空流（一言不发地走完）=首 token 前的"答不上"——不经异常
                # 路径也补记一次尝试（与链上 _fetch_first 把空流翻 UpstreamError 记账
                # 同一口径，死因措辞一字不差）；seam 随后照样 502（评审补网的缺口）
                attempts = [
                    self._lone_attempt(
                        request_id,
                        UpstreamError(f"上游 {self._inner.name} 流在首块前结束，未产出任何 chunk"),
                        started,
                    )
                ]
            settled = self._settle_round(
                request_id,
                request,
                completion_text="".join(delivered) if delivered else None,
                official_usage=official,
                attempts=attempts,
            )
            # 复杂语句（多条件与对象判空）行上：include_usage 置位且确有交付才发 usage 帧——
            # 空流零交付时发它会伪造"有内容"、把 seam 的空流 502 护栏骗过去
            if (
                request.stream_options
                and request.stream_options.include_usage
                and head is not None
                and settled is not None
            ):
                # 行级：usage 帧=choices 空列表 + usage（OpenAI 兼容形状）——发在这=流末，
                # sse_stream 只会在流尽后接 [DONE]，"恰一帧在 [DONE] 前"由结构保证
                yield ChatCompletionChunk(id=head.id, model=head.model, choices=[], usage=settled)
        except BaseException as exc:
            # 失败/截断/断连/弃流的收尾结算（同步短写）：口径见 docstring 三档——
            # 与正常路径撞车（弃在 usage 帧上）时二次结算=幂等 no-op，绝不双扣（checklist 6）
            text = "".join(delivered)  # 行级：已送达整段——截断口径的结算素材
            attempts = self._round_attempts(request_id)
            if not attempts and not text:
                # 行级：裸上游首 token 前失败——补记一次尝试入损耗账（chat() 的同款兜底）；
                # 已交付文字的死法不补记：那次尝试"有结果"，由用户账代表（尝试 ≠ 结果）
                attempts = [self._lone_attempt(request_id, exc, started)]
            self._settle_round(
                request_id,
                request,
                completion_text=text if text else None,
                # 行级：截断丢官方计数——官方是整段生成口径，而已送达可能只有一半
                # （usage 搭在内容帧上的上游尤其如此）；两数冲突时 story 18（不为未收到
                # 的文字付钱）赢过"官方优先"，宁可按已送达估算（ADR-0008 落地口径①）
                official_usage=None,
                attempts=attempts,
            )
            raise
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
