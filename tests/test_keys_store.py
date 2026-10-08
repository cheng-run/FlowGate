"""keys 缝测试（w4 issue 01）：生成 / 校验 / 撤销 / list 的公共 API 直调。

受测面按 spec 测试决定（接缝 2）：只测 keys store 公共 API 的对外承诺——凭据格式、
库内容无明文、校验的"查无"口径、软删与审计视图；不测 SQL、不测私有函数。全程走
tmp_path 临时库，离线可复跑。
"""

import hashlib
from pathlib import Path

from keys.store import KeyStore


def _store(tmp_path: Path) -> KeyStore:
    """临时文件 key 库——每个用例一只独立库（与账本测试同口径：可复跑零残留）。"""
    return KeyStore(str(tmp_path / "keys.db"))


def test_create_returns_key_id_and_credential_with_prefixes(tmp_path: Path) -> None:
    """证明：生成一次返回 key_id 与凭据串两个字段，且格式符合 spec（key_ / fgk_ 前缀）。

    怎么证明：对临时库 create 一把 key，断言返回字典恰含 key_id、credential 两键，
    key_id 以 key_ 开头（公开标识）、credential 以 fgk_ 开头且全长 47——凭据串的
    随机主体由 secrets 生成，测试只钉 spec 指定的格式与长度。
    """
    store = _store(tmp_path)

    created = store.create(name="demo", scope="*")

    # 返回形状：恰两个字段——凭据串与公开标识各归其位（spec：{key_id, credential}）
    assert set(created) == {"key_id", "credential"}
    assert created["key_id"].startswith("key_")  # 行级：公开标识前缀（与凭据串一眼分清）
    # 行级：fgk_(4 字符) + token_urlsafe(32) 恰 43 字符 = 47
    assert created["credential"].startswith("fgk_")
    assert len(created["credential"]) == 47


def test_create_stores_sha256_and_never_plaintext_credential(tmp_path: Path) -> None:
    """证明：库里只有凭据的 sha256，明文交出后物理上不再存在（story 12：库泄露≠凭据泄露）。

    怎么证明：create 后直接读 SQLite 文件字节——明文凭据的字节序列**不在**文件里，
    而其 sha256 十六进制摘要**在**；独立真源 = spec 指定的 sha256 算法（手算复现）。
    这是刻意绕开公共接口的存储面断言：被测对象就是"库文件内容"这个泄露场景本身。
    """
    store = _store(tmp_path)

    created = store.create(name="demo", scope="*")
    db_bytes = (tmp_path / "keys.db").read_bytes()

    # 行级：明文逐字节不在库里——泄漏库文件拿不到可用凭据
    assert created["credential"].encode() not in db_bytes
    # 行级：库里躺着的确实是指定算法的摘要（防止"存了别的形态"蒙混过关）
    expected_hash = hashlib.sha256(created["credential"].encode()).hexdigest()
    assert expected_hash.encode() in db_bytes


def test_verify_returns_key_id_for_registered_credential(tmp_path: Path) -> None:
    """证明：有效凭据校验命中，回的是公开 key_id（身份口径就是它，不是凭据串）。

    怎么证明：create 后用刚交出的凭据 verify，断言返回值 == create 给出的 key_id——
    校验的对外契约是"凭据换身份"，为限流/计费按 key_id 对账接线（story 15）。
    """
    store = _store(tmp_path)
    created = store.create(name="demo", scope="*")

    assert store.verify(created["credential"]) == created["key_id"]


def test_verify_returns_none_for_unknown_credential(tmp_path: Path) -> None:
    """证明：未注册凭据一律"查无"（None）——防枚举 401 的底层依据（story 3）。

    怎么证明：空库上对随机串与空串各 verify 一次，断言都是 None——store 层不给出
    任何可以区分"不存在"细节的返回，HTTP 层才能发同一条 401 文案。
    """
    store = _store(tmp_path)

    assert store.verify("fgk_never-registered") is None
    assert store.verify("") is None  # 空串同样是"查无"，fail-closed 无特殊通道


def test_list_reports_key_fields_with_scope_verbatim(tmp_path: Path) -> None:
    """证明：list 给出审计所需五字段，scope 原样存取不解释（解释归 03 授权层）。

    怎么证明：两把 key 分别以 "*" 与 "model-a,model-b" 建档，断言 list 逐条回出
    key_id/name/scope/created_at/status——逗号串原样不被拆开、通配符不被翻译，
    created_at 非空、新 key 状态 active。
    """
    store = _store(tmp_path)
    star = store.create(name="all", scope="*")
    pair = store.create(name="pair", scope="model-a,model-b")

    # 行级：推导式——审计结果按公开标识建索引，后文按 key_id 取行免得数位置
    rows = {row["key_id"]: row for row in store.list()}

    # 两把 key 都在审计视图里（按公开标识取行）
    assert set(rows) == {star["key_id"], pair["key_id"]}
    # 行级：解包两行——分开命名后断言才读得出"谁是通配、谁是逗号串"
    star_row, pair_row = rows[star["key_id"]], rows[pair["key_id"]]
    # 行级：字段原样回出——"*" 与逗号串一字不改，store 不做任何 scope 解释
    assert (star_row["name"], star_row["scope"], star_row["status"]) == ("all", "*", "active")
    assert pair_row["scope"] == "model-a,model-b"
    assert pair_row["name"] == "pair"
    assert star_row["created_at"] != ""  # 行级：签发时间非空（list 五字段齐）


def test_revoke_keeps_row_and_fails_verification(tmp_path: Path) -> None:
    """证明：撤销=软删——校验即时失败，行仍留在审计视图（story 4 + story 14）。

    怎么证明：create → revoke → 断言 verify 立刻从 key_id 变 None（每请求查库、
    即时生效），且 list 里该行**仍在**、状态已是 revoked——删掉行是反例（审计痕迹
    消失），校验仍放行是反例（撤销变延迟控制）。
    """
    store = _store(tmp_path)
    created = store.create(name="demo", scope="*")

    assert store.revoke(created["key_id"]) is True

    assert store.verify(created["credential"]) is None  # 即时生效：下一次校验就挡
    rows = store.list()
    assert [row["key_id"] for row in rows] == [created["key_id"]]  # 行级：软删行保留
    assert rows[0]["status"] == "revoked"  # 行级：审计视图标出已撤销


def test_revoke_unknown_key_returns_false(tmp_path: Path) -> None:
    """证明：对不存在的 key_id 撤销返回 False——CLI 报错的依据（keyctl 04）。

    怎么证明：空库上 revoke 一个凭空捏造的 id，断言 False——管理员面不防枚举，
    存储层给出"查无"与"已撤销"可区分的返回（与 verify 的防枚举口径相反，因为
    调用方不同：CLI 是自己人）。
    """
    store = _store(tmp_path)

    assert store.revoke("key_nope") is False


def test_revoke_twice_keeps_returning_true(tmp_path: Path) -> None:
    """证明：对已撤销的 key 再撤销仍回 True（行在即 True）——幂等契约，CLI 不误报查无。

    怎么证明：同一 key 连撤两次，断言都回 True——第二次若回 False，keyctl 会把
    "已经撤销过"错报成"查无此 id"。评审补充钉住（spec 未定义此行为，语义自洽）。
    """
    store = _store(tmp_path)
    created = store.create(name="demo", scope="*")
    store.revoke(created["key_id"])

    assert store.revoke(created["key_id"]) is True
