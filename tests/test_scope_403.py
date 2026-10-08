"""scope 授权测试（w4 票 03）：主缝上的越权 403、放行口径与 401→422→403 全链。

受测面按 spec 测试决定（接缝 1，主缝）：POST /v1/chat/completions 的 HTTP 状态码
与响应体——越权 403（detail 点名 model、不回显凭据）/ `*` 与空 scope 放行 / 合规
200 / 流式腿同口径；好测试=只测外部行为，不测门卫内部调用链。全部离线、确定性
（conftest 种 TEST_KEY、mint_key 现铸各用例的独立 scope key）。
错误优先级全链（401 → 422 → 403）在本文件收口——403 腿是票 02 评审裁定移交
本票的条件项，401/422 腿的单腿证据在 tests/test_auth_gate.py。
"""

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import auth_headers, mint_key

# 带默认头的 client：套件级 TEST_KEY（scope=`*`）的正常路径
client = TestClient(app, headers=auth_headers())
# 无默认头的 client：匿名变体（全链用例的 401 腿）走它
bare_client = TestClient(app)


def _payload(model: str = "fake-model", stream: bool = False) -> dict:
    """最小合法请求体——model 可指名（scope 判定读的就是它），stream 开流式腿。"""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "你好"}],
        "stream": stream,
    }


def test_scope_violation_returns_403_naming_model() -> None:
    """证明：请求的 model 不在白名单 → 403，detail 点名被拒 model、不回显凭据（story 7/9）。

    怎么证明：铸一把 scope="fake-model" 的 key，拿它请求 model="qwen-max"——断言
    403 而不是 401（有身份但 model 不在 scope，与"没身份"三分立）、响应体同 {"detail"} 形状、
    文案含 "qwen-max"（客户端不必猜）、凭据串一字不回显。反例：403 一个空 detail
    让客户端猜自己缺什么权（story 9 明文反对）。
    """
    minted = mint_key(name="scope-violation", scope="fake-model")

    response = client.post(
        "/v1/chat/completions",
        json=_payload(model="qwen-max"),
        headers=auth_headers(minted["credential"]),
    )

    assert response.status_code == 403
    body = response.json()
    assert set(body.keys()) == {"detail"}  # 行级：与 401/429/502/504 同款形状
    assert "qwen-max" in body["detail"]  # 行级：story 9——detail 说清无权使用的 model
    assert minted["credential"] not in response.text  # 行级：不回显凭据（同 401 纪律）


def test_compliant_scope_returns_200() -> None:
    """证明：model 在白名单内 → 200——scope 是白名单不是摆设，命中即放行。

    怎么证明：铸一把 scope="fake-model" 的 key 请求 fake-model，断言 200。与越权
    用例同构对照（同 key 同请求形状，只差 model 在不在名单），403/200 之别只由
    scope 判定决定。
    """
    minted = mint_key(name="scope-compliant", scope="fake-model")

    response = client.post(
        "/v1/chat/completions",
        json=_payload(model="fake-model"),
        headers=auth_headers(minted["credential"]),
    )

    assert response.status_code == 200


def test_wildcard_scope_allows_any_model() -> None:
    """证明：scope=`*` 的 key 请求任何 model 都放行——通配就是"不再枚举"（story 8）。

    怎么证明：铸 scope="*" 的 key 请求一个不在任何列表里的怪名字，断言 200——
    与越权用例对照，`*` 是白名单的宽侧极值（契约钉：解释层直调版在
    tests/test_keys_scope.py，这里钉 HTTP 面）。born-green：实现同轮落地（票面
    checklist "`*` 通配放行"的字面证据）。
    """
    minted = mint_key(name="scope-wildcard", scope="*")

    response = client.post(
        "/v1/chat/completions",
        json=_payload(model="whatever-model"),
        headers=auth_headers(minted["credential"]),
    )

    assert response.status_code == 200


def test_empty_scope_allows_any_model() -> None:
    """证明：空 scope = 不限，与 `*` 同义——票面定死语义的 HTTP 面字面证据。

    怎么证明：铸 scope="" 的 key 请求任意 model，断言 200。这是防"建 key 忘填
    scope 即全拒"哑弹的宽侧缺省（03 票面建议语义，docstring 已定死）——再想
    翻成全拒，本用例与 check_scope 直调用例必须先红。
    """
    minted = mint_key(name="scope-empty", scope="")

    response = client.post(
        "/v1/chat/completions",
        json=_payload(model="whatever-model"),
        headers=auth_headers(minted["credential"]),
    )

    assert response.status_code == 200


def test_stream_scope_violation_returns_403() -> None:
    """证明：流式腿同口径——stream=true 的越权请求同样 403，不是"流式绕过授权"。

    怎么证明：铸 scope="fake-model" 的 key 发 stream=true + model="qwen-max"，
    断言 403（流式分支在路由里排在门卫之后，越权请求根本走不到 SSE 装配）。
    反例：授权只拦非流式——流式是常开的旁路。
    """
    minted = mint_key(name="scope-stream-violation", scope="fake-model")

    response = client.post(
        "/v1/chat/completions",
        json=_payload(model="qwen-max", stream=True),
        headers=auth_headers(minted["credential"]),
    )

    assert response.status_code == 403
    assert "qwen-max" in response.json()["detail"]  # 行级：流式腿的 detail 同样点名 model


def test_stream_compliant_scope_returns_200_sse() -> None:
    """证明：流式腿的合规放行也同口径——scope 命中 → 200 且照常流出 SSE。

    怎么证明：铸 scope="fake-model" 的 key 发 stream=true + fake-model，断言 200
    且 content-type 是 text/event-stream——授权门卫对流式腿零副作用（放行后
    W2 的流生命周期一字不动）。
    """
    minted = mint_key(name="scope-stream-ok", scope="fake-model")

    response = client.post(
        "/v1/chat/completions",
        json=_payload(model="fake-model", stream=True),
        headers=auth_headers(minted["credential"]),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_error_priority_chain_401_422_403() -> None:
    """证明：错误优先级全链 401 → 422 → 403——多个错并存时链头的先报（票 02 评审
    裁定移交本票的收口条件：02 只钉 401→422，403 腿在此补齐）。

    怎么证明：三发"剥洋葱"变体——每发只去掉上一发的第一个缺陷，下一个错才轮到
    被报：① 匿名 + 合法 body + 越权 model → 401（认证压过授权：没身份时连"你缺
    哪个 model 的权"都不告诉）；② 合法 key + 坏 body（缺 messages）+ 越权 model
    → 422（body 校验压过授权：整体 body 不合法时没有 model 可判）；③ 合法 key +
    合法 body + 越权 model → 403（前两关都过了才轮到授权拒绝）。再补一发三错
    并存（匿名 + 坏 body + 越权 model）→ 401，钉死链头最优先。
    """
    limited = mint_key(name="chain-limited", scope="fake-model")
    headers = auth_headers(limited["credential"])

    # ① 匿名 + 合法 body + 越权 model：认证最先咬下 → 401（压过 403）
    anon_ok_body = bare_client.post("/v1/chat/completions", json=_payload(model="qwen-max"))
    assert anon_ok_body.status_code == 401

    # ② 合法 key + 坏 body + 越权 model：认证过了，body 校验先于授权判定 → 422
    bad_body = client.post("/v1/chat/completions", json={"model": "qwen-max"}, headers=headers)
    assert bad_body.status_code == 422

    # ③ 合法 key + 合法 body + 越权 model：前两关都过，才轮到授权拒绝 → 403
    violation = client.post(
        "/v1/chat/completions", json=_payload(model="qwen-max"), headers=headers
    )
    assert violation.status_code == 403

    # 补钉：三错并存（匿名 + 坏 body + 越权 model）→ 仍是 401——链头最优先
    all_bad = bare_client.post("/v1/chat/completions", json={"model": "qwen-max"})
    assert all_bad.status_code == 401
