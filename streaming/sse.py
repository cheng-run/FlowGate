"""SSE 传送带：统一 chunk 流 → 客户端 SSE 字节帧（issue 01 的核心机制本体）。

为什么 SSE 帧手工拼不引 sse-starlette（spec 决定）：格式就两行
（"data: {json}\\n\\n"），引库撞依赖红线；更关键的是这个模块的全部考点就是
流生命周期（序列化、[DONE]、超时、上游收尾）——交给库，生命周期就讲不清了。
"""

from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from app.schemas import ChatCompletionChunk


async def sse_stream(chunks: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[bytes]:
    """把上游 chunk 流逐块序列化成 SSE 字节帧，流末发 [DONE]，并显式收尾上游。

    给初学者的解释（这个函数本身也是 async 生成器，形状同 providers/base.py
    chat_stream 的解释）：每个 yield 出去的 bytes 会经 StreamingResponse 直接写进
    HTTP 连接——客户端一块一块收到，我们从头到尾不攒整答。

    为什么 [DONE] 必须有：OpenAI 流用它做"完成记号"——没有它，客户端分不清
    "流正常说完了"和"流半路断了"（spec 故事 3，截断语义的地基）。
    """

    try:
        async for chunk in chunks:
            # 行级：SSE 帧= "data: " 前缀 + JSON + 空行分隔；整块序列化（含 null 字段）
            # ——OpenAI SDK 的 Optional 字段对 null/缺席两可，不为洁癖加排除规则
            yield f"data: {chunk.model_dump_json()}\n\n".encode()
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
    注意："取消中不被打断"的前提是清理链全同步收尾（当前成立）；W3 的 gap 超时
    若在清理链里引入 await 挂起点，需重新核对此前提（记入 ADR-0003）。
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


def sse_response(chunks: AsyncIterator[ChatCompletionChunk]) -> SSEStreamResponse:
    """路由装配点：统一 chunk 流 → 带确定性收尾的 SSE 响应。

    为什么抽这层：路由只该做"选哪种响应"（传送带纪律）——媒体类型、帧序列化、
    [DONE]、上游收尾、弃流盖子全在 streaming/ 一家，流生命周期的家只有一个
    （spec 故事 20），换实现（如 W3 加 gap 超时）路由一行不动。
    """
    return SSEStreamResponse(sse_stream(chunks), media_type="text/event-stream")
