"""SSE 传送带：统一 chunk 流 → 客户端 SSE 字节帧（issue 01 的核心机制本体）。

为什么 SSE 帧手工拼不引 sse-starlette（spec 决定）：格式就两行
（"data: {json}\\n\\n"），引库撞依赖红线；更关键的是这个模块的全部考点就是
流生命周期（序列化、[DONE]、超时、上游收尾）——交给库，生命周期就讲不清了。
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from app.schemas import ChatCompletionChunk
from providers.base import UpstreamError

# 截断原因的落点（spec："原因进日志"）：流的死因在 HTTP 出口看不见（200 + 半截帧），
# 排错全靠这条日志——用模块 logger 不建日志设施，宿主怎么配 handler 是部署的事。
logger = logging.getLogger(__name__)

# 单一 gap 超时（spec 决定）：一个机制同时覆盖"首块等待"与"块间空闲"——每次取块
# 重新起算，30 秒量级。为什么不为 W3 预埋"首块预算/总预算"拆分参数：预算拆分是
# W3 的决策，现在拆就是给不存在的需求交设计税；口径一个，最坏延迟一句话讲得清
# （故事 9：不读源码也能推理 worst-case——相邻两块最多隔一个 gap）。
GAP_TIMEOUT_SECONDS = 30.0


class GapTimeoutError(Exception):
    """gap 超时的失败形状：上游没在预算内吐出下一块（"上游没按时说话"）。

    为什么它只在首块前翻成 504、首块后只截断（spec 错误契约）：首块之前响应还没
    开始，错误还能用状态码说话；首块之后 200 已经承诺，同一条件只能截断流。
    异常只有一种，两个窗口各自决定处置——"可报错窗口在首块处关闭"因此可测。
    为什么不复用 UpstreamError：那是"上游拒了"（有摘要可带）→ 502，这是
    "上游没说话"→ 504——语义不同，客户端的重试直觉也不同（502 该换路，504 值得等）。
    """


def _sse_frame(chunk: ChatCompletionChunk) -> bytes:
    """一块统一 chunk → 一帧 SSE 字节（"data: {json}\\n\\n"）。

    为什么抽这一行：首块与后续块各有一个发送点，帧格式只该有一个出处——
    将来动帧格式（W4 若加心跳注释帧）只改这里，不会两处漂移。
    """
    # 行级：SSE 帧= "data: " 前缀 + JSON + 空行分隔；整块序列化（含 null 字段）
    # ——OpenAI SDK 的 Optional 字段对 null/缺席两可，不为洁癖加排除规则
    return f"data: {chunk.model_dump_json()}\n\n".encode()


async def _fetch(chunks: AsyncIterator[ChatCompletionChunk]) -> ChatCompletionChunk:
    """裸取下一块：上游自带的 TimeoutError 换成 UpstreamError，不许冒充 gap 超时。

    为什么需要这层隔离（评审收严）：_next_chunk 靠 wait_for 抛的 TimeoutError 认
    "闹钟响了"——若适配器自己也漏出 builtin TimeoutError，两款异常会同款到达
    except，会被误标成"gap 超时"（504 说"没按时说话"，真话是"上游自己超时了"，
    排错方向整个相反）。先在这换掉上游的，外面再看到 TimeoutError 就只可能是
    我们的闹钟——失败词汇一个词只说一件事。
    """
    try:
        return await anext(chunks)
    except TimeoutError as exc:
        # 行级：归"上游答不上"族（502 语义，与拒答同族）——不是"没按时说话"（504）
        raise UpstreamError(f"上游取块时抛出 TimeoutError: {exc}") from exc


async def _next_chunk(chunks: AsyncIterator[ChatCompletionChunk]) -> ChatCompletionChunk:
    """取下一块，超 gap 预算没等到就抛 GapTimeoutError——首块与块间共用的唯一取块口。

    给初学者的解释（asyncio.wait_for 在生产代码里首次出现；测试 02 等事件用过它）：
    wait_for(协程, 秒数) 像给这次取块上闹钟——预算内取到就原样返回；到点还没取到，
    它先把取块任务取消（上游生成器在等待点收到取消、走自己的 finally 收尾，
    02 的记账链因此不断），再抛 TimeoutError。我们把它翻译成 GapTimeoutError：
    失败形状只说"上游没按时说话"这一件事，不让裸 TimeoutError 满天飞。
    """
    try:
        # 行级：每次取块重新起算 gap——"首块等待"与"块间空闲"是同一把尺子（spec 口径）
        return await asyncio.wait_for(_fetch(chunks), timeout=GAP_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        # 行级：上游的 TimeoutError 已被 _fetch 换掉，这里的只可能是 gap 闹钟；
        # 消息里带预算值，客户端从 detail 直读"等了多久没等到"，不用回源码翻常量
        raise GapTimeoutError(f"gap 超时：上游 {GAP_TIMEOUT_SECONDS:g} 秒内未产出下一块") from exc


# 拆不动说明（sse_stream 整块超 40 行，含 docstring/注释；同 providers/fake.py 的先例）：
# 首帧 + 取块回路 + 截断分支 + 收尾 aclose 是**同一条**流生死链的走读现场——
# 抽子生成器会把链条拆成两截，内层还得再显式 aclose 一次（正是 02 在防的泄漏形态），
# 截断分支也得跨函数传"别发 [DONE]"的暗号；教学注释是规范硬要求，删不得。
# 故保持单函数，超限以本说明豁免。
async def sse_stream(
    first: ChatCompletionChunk, chunks: AsyncIterator[ChatCompletionChunk]
) -> AsyncIterator[bytes]:
    """把上游 chunk 流逐块序列化成 SSE 字节帧，流末发 [DONE]，并显式收尾上游。

    给初学者的解释（这个函数本身也是 async 生成器，形状同 providers/base.py
    chat_stream 的解释）：每个 yield 出去的 bytes 会经 StreamingResponse 直接写进
    HTTP 连接——客户端一块一块收到，我们从头到尾不攒整答。

    为什么首块是参数、不是从 chunks 里现取（issue 03 错误契约）：首块要在
    **可报错窗口**里取到手——sse_response 装配时先取一块，失败还来得及翻 502/504；
    这里接手的是"已承诺的流"，一旦开始发帧就不再有状态码可言。
    为什么 [DONE] 必须有：OpenAI 流用它做"完成记号"——没有它，客户端分不清
    "流正常说完了"和"流半路断了"（spec 故事 3，截断语义的地基）。
    首块后的失败怎么处置（issue 03 错误契约）：卡住或抛错都只截断流、不伪造
    [DONE]、原因进日志——半截与完整的可辨性全押在"[DONE] 有没有"上。
    """

    try:
        # 行级：首块在装配时已取到手——先发它，后续块再按原节奏续流
        yield _sse_frame(first)
        while True:
            try:
                # 行级：后续块走同一个 gap 取块口——"块间空闲"与"首块等待"一把尺
                chunk = await _next_chunk(chunks)
            except StopAsyncIteration:
                break  # 行级：上游正常说完了——落到下面的 [DONE]
            except Exception as exc:
                # 首块后卡住或抛错（spec 错误契约）：流已承诺，只截断、不发 [DONE]、
                # 原因进日志——客户端靠"没等到 [DONE]"识别半截回答，排错靠这条日志找死因。
                # 为什么 return 不 break：break 会掉进 [DONE] 发送点，等于伪造完成记号。
                # 为什么 except Exception 一把抓：契约说首块后**任何**失败都只截断，
                # 逐个列异常反而漏（CancelledError 是 BaseException 不在此列，
                # 断连取消照常穿出——02 的显式收尾链不受影响）
                logger.warning("SSE 流首块后截断（未发 [DONE]），原因：%s", exc, exc_info=True)
                return
            yield _sse_frame(chunk)
        # 行级：上游正常吐完才发 [DONE]——发了它就等于承诺"这是完整回答"
        yield b"data: [DONE]\n\n"
    finally:
        # 给初学者的解释（取消传播在本代码库首次出现）：无论流是正常跑完、
        # 客户端中途断开还是序列化抛错，都会走到这个 finally。在这里显式
        # await 上游的 aclose()，把"该关了"用我们自己写的代码传下去——
        # 不等垃圾回收器的 finalizer 时机（时机不可控，泄漏风险恰恰藏在
        # "等它自己被收"里）。aclose() 会往上游生成器的挂起点抛 GeneratorExit，
        # 上游的 finally 因此立即执行——fake 在那里记"被取消"，02 的
        # "断连不泄漏"断言就落在这条传播链上。
        await chunks.aclose()


class SSEStreamResponse(StreamingResponse):
    """SSE 流式响应：把"流的收尾"钉死在 ASGI 调用返回之前（"断连不泄漏"的兜底盖子）。

    为什么光有 sse_stream 的 finally 还不够（实验证据 2026-10-06，见 issue 02 评论）：
    Starlette 的 StreamingResponse 弃流时**不收** body_iterator——客户端断连恰逢
    "发帧窗口"（响应任务悬在 await send、生成器悬在 yield）时，取消不穿生成器链，
    sse_stream 的 finally 没人触发，上游收尾只能等 GC finalizer（实测：ASGI 调用
    返回后收尾旗标仍为假，gc.collect() 之后才变真）。卡在上游产块窗口的断连靠
    取消穿链能显式收，但两个窗口合起来才是"断连"的全部——所以这里在 __call__
    收尾处显式 aclose，把弃流路径也变成确定性收尾。
    02 留下的前提复核（gap 超时落地时核过）："取消中不被打断"靠清理链全同步收尾——
    gap 超时（issue 03）挂在**取块回路**（_next_chunk 的 wait_for），不进清理链；
    清理链仍只有"aclose 上游 + fake 同步记账"，前提成立。W3 若往清理链里加
    await 挂起点，须再核一次（记入 ADR-0003）。
    """

    async def __call__(self, scope, receive, send) -> None:
        """ASGI 入口：流跑完、被取消还是 send 抛错，返回前都把流显式收掉。"""
        try:
            await super().__call__(scope, receive, send)
        finally:
            # 给初学者的解释（弃流收尾形态的 aclose）：aclose() 会往生成器的挂起点抛
            # GeneratorExit——sse_stream 的 finally 于是立刻执行、再显式 aclose 上游。
            # 整条清理链全是同步收尾（fake 记账是纯同步），一步跑完不产生挂起点，
            # 即使正处于取消传播中也不会被打断。正常跑完时生成器已耗尽，aclose 是空操作。
            await self.body_iterator.aclose()


async def sse_response(chunks: AsyncIterator[ChatCompletionChunk]) -> SSEStreamResponse:
    """路由装配点：统一 chunk 流 → 带确定性收尾的 SSE 响应（首块前=可报错窗口）。

    为什么先取首块再装配响应（issue 03 错误契约）：StreamingResponse 一开跑就发
    200 响应头，错误若发生在那之后就只剩截断一种处置。把"取到首块"挪进装配时刻，
    首块之前的失败还来得及用状态码说话（拒答 502 / 超时 504），首块之后流已承诺、
    只截断——同一段代码自然长出"可报错窗口在首块处关闭"这条边界，W3 的
    "重试窗口限死在首 token 之前"就落在这条边界上。
    为什么抽这层：路由只该做"选哪种响应"（传送带纪律）——媒体类型、帧序列化、
    [DONE]、超时、上游收尾、弃流盖子全在 streaming/ 一家，流生命周期的家只有一个
    （spec 故事 20），换实现路由一行不动。
    """
    try:
        # 行级：可报错窗口——首块经 gap 取块口取到手；UpstreamError 原样穿出翻 502，
        # GapTimeoutError 原样穿出翻 504（两个翻译都收在 app/ 的异常处理器）
        first = await _next_chunk(chunks)
    except StopAsyncIteration:
        # 行级：上游一言不发（空流）也归"答不上"——复用 UpstreamError 而不是另造
        # 异常：spec 决定首块前失败形状沿用 UpstreamError → 502，空流是其中一种
        raise UpstreamError("上游流在首块前结束，未产出任何 chunk") from None
    except BaseException:
        # 行级：失败路径也显式收上游（02 的收尾纪律）——生成器通常已终止，
        # 这里的 aclose 多是空操作，兜的是"生成器还活着"的漏网形态
        await chunks.aclose()
        raise
    return SSEStreamResponse(sse_stream(first, chunks), media_type="text/event-stream")
