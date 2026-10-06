"""GET /health 健康检查端点的测试（W1 第一个 TDD 闭环）。"""

from fastapi.testclient import TestClient

# 从 app.main 导入应用对象：测试跑真实 FastAPI 应用，不是 mock。
# 此刻 app.main 尚不存在——TDD 红灯阶段的预期失败点就在这行。
from app.main import app

# TestClient 把 ASGI 应用跑在内存里，不开端口、不占网络 → 测试可复跑（规范：测试不依赖外网）。
client = TestClient(app)


def test_health_returns_200_and_ok_status() -> None:
    """证明：健康检查端点活着，且契约固定为 200 + {"status": "ok"}。

    怎么证明：真实 GET /health，断言状态码与 JSON 体。探活/监控将来靠这个契约吃饭，
    形状变了就是破坏性变更，必须先红在这里。
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
