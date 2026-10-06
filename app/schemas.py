"""OpenAI chat completions 兼容形状（app/ 模块考点=接口契约设计）。

为什么集中在这里：wire 形状是网关的对外契约，app 路由与 providers 适配器
都对着这份模型说话——单一真源，改形状只改一处。
只放 v1 非流式必需的字段，字段取舍随 W2~W3 需求走，不为未来预埋。
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """一条对话消息（role + 纯文本 content）。"""

    role: str  # "system" | "user" | "assistant"——W1 不加枚举校验，契约稳定后再说
    content: str


class ChatRequest(BaseModel):
    """客户端请求体：OpenAI 形状的最小子集（model + messages）。"""

    model: str
    messages: list[ChatMessage]


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
