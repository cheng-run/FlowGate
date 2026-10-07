"""OpenAI chat completions 兼容形状（app/ 模块考点=接口契约设计）。

为什么集中在这里：wire 形状是网关的对外契约，app 路由与 providers 适配器
都对着这份模型说话——单一真源，改形状只改一处。
只放当前版本实际用到的字段（非流式整答 + 流式 chunk），字段取舍随 W3 需求走，
不为未来预埋（usage 回填、tools 等到了再加）。
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """一条对话消息（role + 纯文本 content）。"""

    role: str  # "system" | "user" | "assistant"——W1 不加枚举校验，契约稳定后再说
    content: str


class StreamOptions(BaseModel):
    """流式选项（OpenAI 兼容 stream_options）：目前只认 include_usage（issue 07）。

    为什么单独一个模型不摊进 ChatRequest：stream_options 是嵌套对象形状
    （{"include_usage": true}），摊平成顶层布尔就不是 OpenAI 兼容的报文了——
    客户端按 OpenAI SDK 的字段名发请求，形状本身就是契约（story 22）。
    """

    # 置位时流末在 [DONE] 之前恰多发一帧 usage chunk（OpenAI 惯例）；缺省 False=零帧
    include_usage: bool = False


class ChatRequest(BaseModel):
    """客户端请求体：OpenAI 形状的最小子集（model + messages + stream + stream_options）。"""

    model: str
    messages: list[ChatMessage]
    # 缺省 False：既有非流式客户端零改动（spec 故事 5）；True 时路由分派到 SSE 流
    stream: bool = False
    # 流式选项（issue 07）：缺省 None=既有客户端零改动（spec 故事 25 的"纯加法"）
    stream_options: StreamOptions | None = None


class Usage(BaseModel):
    """token 用量。fake 上游先回 0——真实计数是 billing/（W3 考点）的事，不在这里装懂。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    """一档候选回答（非流式先只有单选）。"""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """服务端响应体：OpenAI chat.completion 形状。"""

    id: str
    object: str = "chat.completion"  # 固定字面量，客户端靠它识别响应形状
    model: str
    choices: list[Choice]
    usage: Usage


# ===== 流式形状（W2 issue 01：chat.completion.chunk）=====


class DeltaMessage(BaseModel):
    """流式增量（chunk 的 choices[].delta）：对整条消息的一次"补丁"。

    为什么字段可空：OpenAI 的 delta 是逐块累积——首块带 role 宣告角色、
    中块只带 content、末块是空对象，没有的字段就是 None（不是另一种形状）。
    """

    role: str | None = None  # 只在首块出现："我是 assistant"
    content: str | None = None  # 中块的正文增量，各块拼起来=完整回答


class StreamChoice(BaseModel):
    """一档候选的流式增量（与非流式 Choice 同 index 语义，message 换成 delta）。"""

    index: int = 0  # 非流式先只有单选，恒 0
    delta: DeltaMessage  # 本帧的增量内容
    finish_reason: str | None = None  # 只在末块为 "stop"，其余帧为 None


class ChatCompletionChunk(BaseModel):
    """服务端流式响应体：OpenAI chat.completion.chunk 形状（每帧一个）。

    为什么不复用 ChatCompletionResponse：整答的 choices[].message 是"完整消息"，
    流的 choices[].delta 是"增量补丁"——两种消费方式，硬塞进一个模型会让两侧
    字段全变成可空的"四不像"，客户端契约反而更难讲清。
    """

    id: str  # 同一次流内各帧共用一个 id（OpenAI 惯例：客户端靠它归组）
    object: str = "chat.completion.chunk"  # 固定字面量，客户端靠它区分"帧"与"整答"
    model: str
    choices: list[StreamChoice]
    # 末帧可带官方 usage（OpenAI 的 include_usage 载体，choices 为空列表）；其余帧 None。
    # 两个用途（issue 07）：适配器把上游的官方计数带进来（流末回填"有则用"的素材）；
    # 网关自己在 [DONE] 前发的那帧也用这个形状——统一 chunk 一处定义，两头方言都对齐
    usage: Usage | None = None
