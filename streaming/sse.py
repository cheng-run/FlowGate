"""SSE 传送带：统一 chunk 流 → 客户端 SSE 字节帧（issue 01 的核心机制本体）。

为什么 SSE 帧手工拼不引 sse-starlette（spec 决定）：格式就两行
（"data: {json}\\n\\n"），引库撞依赖红线；更关键的是这个模块的全部考点就是
流生命周期（序列化、[DONE]、超时、上游收尾）——交给库，生命周期就讲不清了。
"""

from collections.abc import AsyncIterator

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
