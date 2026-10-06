"""POST /v1/chat/completions 端点测试：端到端接 fake 上游（W1 收尾刀）。

两个测试对应五站走位的两站：第 2 站输入检查（非法体被 422 挡下）、
第 3~4 站（请求穿过 seam 到适配器、响应按 OpenAI 形状回来）。
"""

from fastapi.testclient import TestClient

# 导入装配好的 app：测试跑的是真实应用（含装配处），不是 mock。
from app.main import app

client = TestClient(app)


def _payload() -> dict:
    """最小合法请求体——用 dict 直发，模拟真实 HTTP 客户端（不经 Pydantic 预构造）。"""
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }


def test_chat_endpoint_returns_openai_shape() -> None:
    """证明：POST /v1/chat/completions 全链路通，且按 OpenAI 形状回答。

    怎么证明：发真实形状的 POST（TestClient 内存执行），断言 200 + object 字段 +
    回答里带 fake 回显——请求体→路由→Provider 协议→适配器→响应，每一环都穿过了。
    """
    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-model"
    # 回显断言：内容既带 fake 前缀也带回请求里的话——证明数据流真穿过了 seam
    assert "fake-reply" in body["choices"][0]["message"]["content"]
    assert "你好" in body["choices"][0]["message"]["content"]


def test_chat_endpoint_rejects_invalid_body() -> None:
    """证明：非法请求体被挡在门外（五站走位第 2 站：输入检查）。

    怎么证明：不带 messages 发 POST，断言 422——Pydantic 在进路由前就把形状守住，
    将来接真上游时，脏请求永远不会漏到 DashScope 去。
    """
    response = client.post("/v1/chat/completions", json={"model": "fake-model"})

    assert response.status_code == 422
