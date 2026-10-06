"""FlowGate 应用入口：组装 FastAPI 应用。

这是"五站走位"（见 docs/architecture-notes.md §一）里第 1~2 站的落点：
请求进来先过认证与输入检查，后面的 resolve/执行/流式是 W2~W3 的事。
"""

from fastapi import FastAPI

# 用 create_app 工厂而不是模块级直接建 app 吗？——暂时不用。
# 现在只有一个端点、零可变状态，工厂是"为架构而架构"（规范：简单优先）；
# 将来要挂中间件/生命周期状态时再拆工厂，拆的动因写在那个 commit 里。
app = FastAPI(
    title="FlowGate",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查：给探活/监控用的最小契约（200 + status=ok）。

    为什么是独立端点而不是复用 /v1/chat/completions：探活要零副作用、零认证、
    响应形状永不变更——它响了就代表进程活着，不掺任何业务语义。
    """
    # 行级：只回固定字典，FastAPI 自动序列化成 JSON 200——不手写 Response，保持直白。
    return {"status": "ok"}
