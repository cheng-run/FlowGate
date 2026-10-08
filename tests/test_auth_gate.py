"""认证门卫测试（W4 票 02）：fail-closed 401、错误优先级、身份改写、防枚举。

受测面按 spec 测试决定：**主缝**（POST /v1/chat/completions 的 401/422 出口、
响应体形状、/health 豁免、key_id 身份落账）——好测试=只测外部行为：HTTP 状态码
与响应体形状，不测门卫内部调用链。全部离线、确定性（conftest 种 TEST_KEY、
tmp_path 临时 SQLite）。
错误优先级全链是 401 → 422 → 403：403（授权/scope）腿随票 03 落地（scope 语义
在那票才存在），本文件只钉"401 先于 422"（依赖先于 body 校验，spike 实测口径）。
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as gateway
from app.main import app
from billing.ledger import BillingLedger
from billing.settlement import BillingProvider
from keys.store import KeyStore
from providers.fake import FakeProvider
from tests.conftest import TEST_KEY, TEST_KEY_ID, auth_headers, mint_key

# 无默认头的 client：认证变体用例要发"真·匿名"（请求里没有 Authorization 头）
bare_client = TestClient(app)
# 带默认头的 client：合法注册 key 的正常路径（身份改写等用例走它）
client = TestClient(app, headers=auth_headers())


def _payload() -> dict:
    """最小合法请求体——与端点测试同款，dict 直发模拟真实 HTTP 客户端。"""
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "你好"}],
    }


def test_business_endpoint_rejects_anonymous_with_401() -> None:
    """证明：/v1/* fail-closed——匿名请求 401（W4 前"匿名放行"的旧口径已死）。

    怎么证明：裸 client（无默认头）发一发匿名 POST，断言 401 + 响应体只有
    detail 一个键——"空库/无凭据 = 拒绝一切，不存在认证未激活状态"（spec 口径）。
    反例是 W3 的匿名放行：任何人不带凭据就能用网关，治理从第一步就漏了。
    """
    response = bare_client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 401
    # 响应形状与 429/502/504 同款 {"detail": ...}——客户端一套错误处理走天下
    assert set(response.json().keys()) == {"detail"}


def test_health_stays_open_without_credential() -> None:
    """证明：/health 不挂认证——探活零副作用零认证语义，监控无需管理凭据（story 6）。

    怎么证明：裸 client GET /health，断言 200 + 固定形状——认证门只钉在业务端点上，
    这是"fail-closed 不等于把进程藏起来"的对照面。
    """
    response = bare_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_authentication_precedes_body_validation() -> None:
    """证明：错误优先级 401 → 422——认证不看 body，子依赖先于 body 校验（spike 口径）。

    怎么证明：两发坏 body（缺 messages）对照——匿名那发 401（撞认证门）、
    合法凭据那发 422（过了认证才轮到 body 校验）。两段拼出"401 在 422 前"：
    未认证的请求连请求形状都探不到（防探 schema，story 5）。
    """
    # 坏 body + 匿名：先撞认证门 → 401 而不是 422
    response = bare_client.post("/v1/chat/completions", json={"model": "fake-model"})

    assert response.status_code == 401

    # 坏 body + 合法凭据：认证过了 body 校验照旧 → 422（优先级不是"永远 401"）
    ok = client.post("/v1/chat/completions", json={"model": "fake-model"})
    assert ok.status_code == 422


def test_401_detail_uniform_across_five_credential_shapes() -> None:
    """证明：无头/空串/坏串/未注册/已撤销 → 同一条 401 文案，且不回显凭据（防枚举，story 3）。

    怎么证明：五种变体各发一发，断言全 401、响应体只有 detail 一个键、凭据串一字
    不回显；五条 detail 收进集合恰 1 个元素——"未注册与已撤销不可区分"，枚举攻击
    读不到任何差别（story 3 的字面证据）。已撤销变体先 mint 再 revoke（软删）。
    """
    # 已撤销变体：先签发再软删——"撤销后与从未存在同文案"（CONTEXT.md 撤销词条）
    revoked = mint_key(name="auth-revoked")
    assert gateway.keystore.revoke(revoked["key_id"]) is True
    cases = {
        "无头": None,  # 行级：请求里根本没有 Authorization 头
        "空串": "",  # 行级：Bearer 后面空空如也——带了头但凭据串是空
        "坏串": "not-a-credential",  # 行级：随手垃圾串，连 fgk_ 形态都不像
        "未注册": "fgk_" + "z" * 43,  # 行级：形态合法但库里查无——与坏串必须同文案
        "已撤销": revoked["credential"],  # 行级：库里有行但已软删——与从未存在同文案
    }

    details = set()
    for label, credential in cases.items():
        headers = {}  # 无头变体：连 Authorization 都不带（bare client 也无默认头）
        if credential is not None:
            headers = auth_headers(credential)  # 其余变体：Bearer + 该变体的凭据串
        response = bare_client.post("/v1/chat/completions", json=_payload(), headers=headers)

        assert response.status_code == 401, f"{label} 该 401，实得 {response.status_code}"
        body = response.json()
        assert set(body.keys()) == {"detail"}, label
        if credential:
            assert credential not in response.text, f"{label} 回显了凭据串"
        details.add(body["detail"])

    assert len(details) == 1  # 行级：五变体同一条文案——防枚举的字面证据


def test_revocation_takes_effect_on_next_request() -> None:
    """证明：撤销即时生效——同一凭据先 200、revoke 后下一发 401（story 4）。

    怎么证明：mint 一把 key 走一发正常请求（200），软删后再发同一凭据（401）——
    每请求查库（WHERE revoked_at IS NULL）就是"即时"的机制；反例是缓存凭据→撤销
    迟迟不生效，撤销就不是即时控制而是迟钝通知。
    """
    minted = mint_key(name="auth-revoke-immediate")
    headers = auth_headers(minted["credential"])

    first = client.post("/v1/chat/completions", json=_payload(), headers=headers)
    assert first.status_code == 200  # 行级：撤销前正常成交

    assert gateway.keystore.revoke(minted["key_id"]) is True

    second = client.post("/v1/chat/completions", json=_payload(), headers=headers)
    assert second.status_code == 401  # 行级：撤销后的下一发就被挡——不是"下下次"


def test_ledger_row_keyed_by_key_id_not_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """证明：账本/预算的对账身份是 key_id，凭据串不落任何库（story 15）。

    怎么证明：临时账本 + 套件级 TEST_KEY（默认头）发一发非流式请求，断言恰一笔
    用户账且 charge.key == TEST_KEY_ID、≠ TEST_KEY；再看预算求和口径——
    spend(key_id) 有花销、spend(凭据串) 恒 0。库泄露换不回凭据（keys 只存 hash），
    账本泄露同样换不回。
    """
    ledger = BillingLedger(str(tmp_path / "billing.db"))
    monkeypatch.setattr("app.main.ledger", ledger)
    fake = FakeProvider()
    monkeypatch.setattr("app.main.provider", BillingProvider(fake, ledger=ledger))

    response = client.post("/v1/chat/completions", json=_payload())  # 默认头 = TEST_KEY

    assert response.status_code == 200
    charges = ledger.charges()
    assert len(charges) == 1  # 行级：恰一笔用户账——身份断言只看这一笔
    assert charges[0].key == TEST_KEY_ID  # 行级：对账身份 = 公开 key_id
    assert charges[0].key != TEST_KEY  # 行级：凭据串不是身份、不落账
    assert ledger.spend(TEST_KEY_ID) > 0  # 行级：预算求和按 key_id——门卫口径同源
    assert ledger.spend(TEST_KEY) == 0  # 行级：凭据串查不到花销（不落账的另一面）


def test_empty_key_store_rejects_registered_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """证明：空库 = 拒绝一切（fail-closed）——丢库/新库不会变成开门迎客（story 2）。

    怎么证明：把认证门卫的 key 库换成全新的空库，再拿套件级 TEST_KEY（在原库里
    明明有效）发请求——断言 401。"认证未激活"状态不存在：库里没行=全拒。
    """
    monkeypatch.setattr("app.main.keystore", KeyStore(":memory:"))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 401
