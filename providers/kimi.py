"""Kimi 中转真实上游适配器：httpx 裸调 OpenAI 兼容端点（W3 issue 08，第二真实上游）。

为什么裸调不用官方 SDK（依赖红线，AGENTS.md "核心机制一律自建"）：同 dashscope.py
的理由——我们只需要一次 POST；统一形状已经由 app/schemas 定死，方言翻译必须收在
本文件内可见、可测（ADR-0001：每个上游一个适配器文件，方言止步于此）。
上游方言口径：Kimi 中转只支持 OpenAI 协议（2026-09-10 实测，配置笔记在项目外）——
SSE 的 data: 行 / [DONE] 与错误 JSON 全部是本文件的知识，核心只见统一 chunk 与
UpstreamError（issue 08 checklist：上游方言不出适配器）。

为什么与 dashscope.py 的翻译逻辑**有意双写**而不抽"OpenAI 兼容公共层"（面试预备
答案）："OpenAI 兼容"是松散惯例不是标准——错误壳、SSE 细节、富余字段各家随时不同，
中转站尤其爱改；方言知识归属各自的上游文件，谁的上游变了只改谁的文件。抽公共层
会把"失败形状在哪翻的"摊到第三个文件（dashscope.py 的 chat/chat_stream 有意双写
先例的跨适配器延伸），两个适配器也就没法各自独立走读到底。

为什么 base_url **必填、无内置默认**（与 DashScope 的不对称是刻意的）：DashScope
有公开官方地址可当默认，而本项目的 Kimi 部署走私有中转——真实地址只活在 .env
（issue 08 checklist"真实地址/密钥永不进 git"）。内置默认地址就是猜：猜错（拿中转
key 打官方地址）会变成运行期 401 的静默悬案；缺了就该在装配处响亮报错指名变量。
"""

from collections.abc import AsyncIterator

import httpx

from app.schemas import ChatCompletionChunk, ChatCompletionResponse, ChatRequest
from providers.base import TransientUpstreamError, UpstreamError

# 显式超时（同 dashscope.py 的口径）：httpx 默认 5s 对生成式模型偏紧，30s 量级留足
# 生成时间；不显式写死，行为就随 httpx 版本默认值漂移，出了"挂住"的悬案没法复盘。
TIMEOUT_SECONDS = 30.0


class KimiProvider:
    """Kimi 中转适配器：统一请求 ⇄ OpenAI 兼容 wire 的双向翻译（ADR-0001 第三实现）。"""

    name = "kimi"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """持有凭证与 HTTP 客户端；key/base_url 由装配处从环境注入，适配器不读 os.environ。

        为什么 base_url 没有默认值：见模块 docstring——真实中转地址只活在 .env，
        适配器内不内置任何地址可猜。
        为什么 transport 是形参：httpx 自带的测试注入点——生产不传（走真网络），
        测试传 MockTransport（假网络、可捕获 wire）。这不是代码库自己的新缝，
        是上游客户端本来就开着的口子，spec 明确允许（DashScope 先例一字不差）。
        """
        self._api_key = api_key
        # rstrip("/")：base_url 手滑带尾斜杠时不产生 //——wire 断言与上游路由都更稳
        self._base_url = base_url.rstrip("/")
        # timeout 显式传入：见模块级 TIMEOUT_SECONDS 的注释
        self._client = httpx.AsyncClient(transport=transport, timeout=TIMEOUT_SECONDS)

    async def chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """把统一请求翻成 Kimi 方言发出去，再把上游 JSON 翻回统一形状。

        为什么双向翻译都收在这一个方法里：适配器的全部理由就是"守住方言边界"
        （ADR-0001）——拆成私有函数反而把外部行为（wire 出、形状回）藏进内部细节。
        """
        # 行级：请求翻译点（统一 → wire）。model_dump() 恰好就是 OpenAI 形状——
        # app/schemas 本来按 OpenAI 最小子集建模，序列化即翻译；将来加 tools 等
        # 字段时要回到这里显式控制报文，不能无脑 dump。
        payload = request.model_dump()
        # 行级：wire 三要素（URL / Bearer 鉴权头 / JSON 体）全在这一句——
        # Kimi 中转的方言知识被隔离在适配器内，核心代码永不需要知道（故事 13）。
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except httpx.TransportError as exc:
            # 行级：传输层失败（超时/拒连/DNS 失败）同样翻译进协议异常——
            # 裸 httpx 异常会变成网关自己的 500，客户端分不清"网关坏了"还是"够不着上游"。
            # TransportError 是超时(TimeoutException)与连接错误的共同父类，一次接住。
            # 瞬态形状（issue 04 词汇）：够不着上游=毛刺不是判决——链重试 1 次再换路；
            # 仍属 UpstreamError 族（子类），502 出口与既有语义一字不动。
            raise TransientUpstreamError(f"上游 {self.name} 请求失败: {exc}") from exc
        # 行级：失败翻译点（wire → 协议异常）。非 2xx 绝不静默、也不当成功解析——
        # 中转的错误 JSON 长什么样都不出本文件，出的只是文本摘要（ADR-0002 语义）。
        # 摘要截断 500 字符：错误页可能巨长，detail 要能一眼读完。
        if response.is_error:
            raise UpstreamError(
                f"上游 {self.name} 返回 {response.status_code}: {response.text[:500]}"
            )
        # 行级：响应翻译点（wire → 统一）。上游跑的就是 OpenAI 兼容模式，校验即翻译——
        # id/model/choices/usage 原样透传进统一模型；多余字段（owned_by 之类）
        # Pydantic 默认丢弃而非报错，上游悄悄加字段不会打爆网关。
        try:
            return ChatCompletionResponse.model_validate(response.json())
        except ValueError as exc:
            # 行级：2xx 但报文坏（非 JSON / 形状不符）也翻进协议失败形状（评审补网）——
            # JSONDecodeError 与 ValidationError（继承 ValueError）裸漏会穿成 500，
            # 把上游的脏报文说成网关自己的故障；流式腿的坏帧护栏就在这，两腿必须同口径
            raise UpstreamError(
                f"上游 {self.name} 返回 200 但报文无法解析: {response.text[:500]}"
            ) from exc

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """流式对话：wire 带 stream:true 发出，把上游 SSE 方言逐块翻成统一 chunk。

        为什么方言解析收进 _iter_unified_chunks：data: 行、上游 [DONE]、流式 wire
        开关都是 Kimi 侧知识，收在适配器内可见可测（ADR-0001 方言边界）——核心
        只见 ChatCompletionChunk，永不见 "data:" 字样。
        为什么 async with 持有 HTTP 流（生命周期同 dashscope.py 先例）：client.stream()
        打开的是同一条 HTTP 连接，连接生命周期挂在这个生成器的栈帧上——生成器被
        aclose（断连/截断）时 GeneratorExit 在 yield 点抛入，async with 的 __aexit__
        立刻关连接，不等 GC。
        """
        # 行级：请求翻译点（统一 → wire，同 chat()）——model_dump 恰是 OpenAI 形状
        payload = request.model_dump()
        # 行级：wire 的 stream:true 由适配器自己钉死——"流式"是本方法的方言知识，
        # 不依赖调用方恰好传对（传 false 上游只回整答 JSON，流式整条链会莫名断掉）
        payload["stream"] = True
        try:
            async with self._client.stream(
                # 行级：wire 三要素（URL / Bearer 鉴权头 / JSON 体）与非流式同一方言
                "POST",
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                # 行级：失败翻译点（wire → 协议异常，同 chat()）——非 2xx 绝不静默、
                # 也不当成功解析：错误页长得像 JSON 但没有 chunk，硬解析会炸出误导性 500。
                # 首块前抛 UpstreamError → 502（W2 可报错窗口）；摘要截断 500 字符。
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
            # 翻译**有意双写**（措辞一字不差，理由见 dashscope.py 同处注释）：
            # 两条腿各自的失败翻译都留在各自现场走读，抽公共小函数省 3 行却把
            # "失败形状在哪翻的"藏进第三处。瞬态形状与 chat() 同款：毛刺可重试
            raise TransientUpstreamError(f"上游 {self.name} 请求失败: {exc}") from exc

    async def _iter_unified_chunks(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """上游 SSE 行流 → 统一 chunk 流：方言解析的唯一入口（data: 行 / 上游 [DONE] / 坏帧）。

        为什么单独一层（同 dashscope.py 的两重理由）：①这里是"wire → 统一"的翻译点，
        独立成函数走读时一眼可指"方言止步于此"；②AGENTS.md 嵌套 ≤3——解析回路若嵌在
        chat_stream 的 try/async with 里会到 4 层，搬出来后两个函数各自收在 3 层内。
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
                # 行级：坏帧也翻进协议失败形状——ValidationError 裸漏会在首块前穿成
                # 500，把上游的脏数据说成网关自己的故障；消息带肇事行残片排错
                # （ValidationError 继承 ValueError，json 解析失败同样在此接住）
                raise UpstreamError(
                    f"上游 {self.name} 发来无法解析的 SSE 数据行: {data[:200]}"
                ) from exc
            yield chunk
