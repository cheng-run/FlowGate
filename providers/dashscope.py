"""DashScope 真实上游适配器：httpx 裸调 OpenAI 兼容端点（W1 收尾，spec dashscope-upstream）。

为什么裸调不用官方 SDK（依赖红线，AGENTS.md "核心机制一律自建"）：
dashscope-sdk 带的是全家桶（自动重试、埋点、自有消息类型），而我们只需要一次
POST；统一形状已经由 app/schemas 定死，方言翻译必须收在本文件内可见、可测——
交给库的黑盒，翻译边界就讲不清了（面试口径：调库谁都会，边界自己守）。
"""

from collections.abc import AsyncIterator

import httpx

from app.schemas import ChatCompletionChunk, ChatCompletionResponse, ChatRequest
from providers.base import UpstreamError

# DashScope 的 OpenAI 兼容前缀（官方文档口径）：/chat/completions 挂在它下面。
# 做成常量而非硬编码进方法：DASHSCOPE_BASE_URL 可覆盖（接中转、测试都靠这个口子）。
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 显式超时（spec 故事 20）：httpx 默认 5s 对生成式模型偏紧，30s 量级留足生成时间；
# 不显式写死，行为就随 httpx 版本默认值漂移，出了"挂住"的悬案没法复盘。
TIMEOUT_SECONDS = 30.0


class DashScopeProvider:
    """百炼 DashScope 适配器：统一请求 ⇄ OpenAI 兼容 wire 的双向翻译（ADR-0001 第二实现）。"""

    name = "dashscope"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """持有凭证与 HTTP 客户端；key/base_url 由装配处从环境注入，适配器不读 os.environ。

        为什么 transport 是形参：httpx 自带的测试注入点——生产不传（走真网络），
        测试传 MockTransport（假网络、可捕获 wire）。这不是代码库自己的新缝，
        是上游客户端本来就开着的口子，spec 明确允许。
        """
        self._api_key = api_key
        # rstrip("/")：base_url 手滑带尾斜杠时不产生 //——wire 断言与上游路由都更稳
        self._base_url = base_url.rstrip("/")
        # timeout 显式传入：见模块级 TIMEOUT_SECONDS 的注释（故事 20）
        self._client = httpx.AsyncClient(transport=transport, timeout=TIMEOUT_SECONDS)

    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """把统一请求翻成 DashScope 方言发出去，再把上游 JSON 翻回统一形状。

        为什么双向翻译都收在这一个方法里：适配器的全部理由就是"守住方言边界"
        （ADR-0001）——拆成私有函数反而把外部行为（wire 出、形状回）藏进内部细节。
        """
        # 行级：请求翻译点（统一 → wire）。model_dump() 恰好就是 OpenAI 形状——
        # app/schemas 本来按 OpenAI 最小子集建模，序列化即翻译；将来加 tools 等
        # 字段时要回到这里显式控制报文，不能无脑 dump。
        payload = request.model_dump()
        # 行级：wire 三要素（URL / Bearer 鉴权头 / JSON 体）全在这一句——
        # 这些 DashScope 方言知识被隔离在适配器内，核心代码永不需要知道（故事 13）。
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except httpx.TransportError as exc:
            # 行级：传输层失败（超时/拒连/DNS 失败）同样翻译进协议异常（故事 6）——
            # 裸 httpx 异常会变成网关自己的 500，客户端分不清"网关坏了"还是"够不着上游"。
            # TransportError 是超时(TimeoutException)与连接错误的共同父类，一次接住。
            raise UpstreamError(f"上游 {self.name} 请求失败: {exc}") from exc
        # 行级：失败翻译点（wire → 协议异常）。非 2xx 绝不静默、也不当成功解析——
        # 上游的错误报文长得像 JSON 但没有 choices，硬解析会炸出误导性的 500。
        # 摘要截断 500 字符：错误页可能巨长，detail 要能一眼读完（spec：保留上游响应摘要）。
        if response.is_error:
            raise UpstreamError(
                f"上游 {self.name} 返回 {response.status_code}: {response.text[:500]}"
            )
        # 行级：响应翻译点（wire → 统一）。上游跑的就是 OpenAI 兼容模式，校验即翻译——
        # id/model/choices/usage 原样透传进统一模型；多余字段（system_fingerprint 之类）
        # Pydantic 默认丢弃而非报错，上游悄悄加字段不会打爆网关。
        return ChatCompletionResponse.model_validate(response.json())

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """流式对话：wire 带 stream:true 发出，把上游 SSE 方言逐块翻成统一 chunk。

        为什么方言解析收进 _iter_unified_chunks（issue 04 的核心）：data: 行、上游
        [DONE]、流式 wire 开关都是 DashScope 侧知识，收在适配器内可见可测（ADR-0001
        方言边界）——核心只见 ChatCompletionChunk，永不见 "data:" 字样。
        为什么 async with 持有 HTTP 流（生命周期考点）：client.stream() 打开的是同一条
        HTTP 连接，连接生命周期就挂在这个生成器的栈帧上——生成器被 aclose（断连/截断）
        时 GeneratorExit 在 yield 点抛入，async with 的 __aexit__ 立刻关连接。spec 写
        "生成器 finally 里显式收"，async with 就是那层显式收口（__aexit__ 与 finally
        同一时机跑，都不等 GC）；不再补一层空 finally——收尾代码一处即可，写两遍是假动作。
        给初学者的解释：本方法是 async 生成器（形状见 providers/base.py）——方法体在
        首次被消费时才执行，每 yield 一块，消费方（streaming/ 传送带）拿一块转一块。
        """
        # 行级：请求翻译点（统一 → wire，同 chat()）——model_dump 恰是 OpenAI 形状
        payload = request.model_dump()
        # 行级：wire 的 stream:true 由适配器自己钉死——"流式"是本方法的方言知识，
        # 不依赖调用方恰好传对（传 false 上游只回整答 JSON，流式整条链会莫名断掉）
        payload["stream"] = True
        try:
            async with self._client.stream(
                # 行级：wire 三要素（URL / Bearer 鉴权头 / JSON 体）与非流式同一方言（故事 13）
                "POST",
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                # 行级：失败翻译点（wire → 协议异常，同 chat()）——非 2xx 绝不静默、
                # 也不当成功解析：错误页长得像 JSON 但没有 chunk，硬解析会炸出误导性 500。
                # 首块前抛 UpstreamError → 502（issue 03 的可报错窗口）；摘要截断 500 字符，
                # 与 chat() 同口径（错误页可能巨长，detail 要一眼读完）。
                if response.is_error:
                    # 行级：先把错误页读出来——流式响应不读就取不到 text，摘要就是空的
                    body = await response.aread()
                    raise UpstreamError(
                        f"上游 {self.name} 返回 {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}"
                    )
                # 行级：逐行喂给方言解析入口——本函数只管流的生命周期，不认 data: 字样
                async for chunk in self._iter_unified_chunks(response.aiter_lines()):
                    yield chunk
        except httpx.TransportError as exc:
            # 行级：传输层失败（超时/拒连/中途断线）同翻进协议异常——裸 httpx 异常会
            # 变成网关 500，客户端分不清"网关坏了"还是"够不着上游"。与 chat() 的同名
            # 翻译**有意双写**（措辞一字不差）：两条腿各自的失败翻译都留在各自现场走读，
            # 抽公共小函数省 3 行却把"失败形状在哪翻的"藏进第三处（chat() docstring 同理）
            raise UpstreamError(f"上游 {self.name} 请求失败: {exc}") from exc

    async def _iter_unified_chunks(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """上游 SSE 行流 → 统一 chunk 流：方言解析的唯一入口（data: 行 / 上游 [DONE] / 坏帧）。

        为什么单独一层（两重理由）：①这里是"wire → 统一"的翻译点，独立成函数走读时
        一眼可指"方言止步于此"；②AGENTS.md 嵌套 ≤3——解析回路若嵌在 chat_stream 的
        try/async with 里会到 4 层，搬出来后两个函数各自收在 3 层内。
        """
        async for line in lines:
            # 行级：SSE 只认 data: 行——空行是帧分隔，event:/注释行是方言里不消费的部分
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break  # 行级：上游完成记号——翻译到此为止，不产出假 chunk
            try:
                # 行级：翻译点（wire → 统一）；上游多给的字段（created 等）被 Pydantic
                # 默认丢弃——方言垃圾不进统一模型，与非流式 chat() 同一策略
                chunk = ChatCompletionChunk.model_validate_json(data)
            except ValueError as exc:
                # 行级：坏帧也翻进协议失败形状（评审补网）——ValidationError 裸漏会在
                # 首块前穿成 500，把上游的脏数据说成网关自己的故障；消息带肇事行残片排错
                # （ValidationError 继承 ValueError，json 解析失败同样在此接住）
                raise UpstreamError(
                    f"上游 {self.name} 发来无法解析的 SSE 数据行: {data[:200]}"
                ) from exc
            yield chunk
