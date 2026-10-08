"""scope 解释层测试（w4 票 03）：白名单判定的公共 API 直调 + store 的 scope 存取口。

受测面按 spec 测试决定（接缝 2）：keys 模块函数直调——check_scope 的放行/拒绝
口径（含"空=不限"定死语义）与 KeyStore.scope_of 的存取承诺；不测解析内部、
不测 SQL。全程离线、确定性（tmp_path 临时库）。
401 → 422 → 403 的 HTTP 全链归 tests/test_scope_403.py（主缝），这里只钉判定本体。
"""

from pathlib import Path

import pytest

from keys.scope import AuthorizationError, check_scope
from keys.store import KeyStore


def test_check_scope_rejects_model_outside_whitelist() -> None:
    """证明：白名单里没有的 model 被拒，且 AuthorizationError 点名被拒 model（story 9）。

    怎么证明：给 scope="gpt-4o,claude" 判 qwen-max——断言抛 AuthorizationError、
    文案里含 "qwen-max"。反例是"403 一个空 detail 让客户端猜"（story 9 明文反对）；
    点名的是调用方自己发来的字符串，不泄任何库内信息。
    """
    with pytest.raises(AuthorizationError) as exc_info:
        check_scope("gpt-4o,claude", "qwen-max")

    assert "qwen-max" in str(exc_info.value)  # 行级：detail 必须说得清被拒的是哪个 model


def test_check_scope_allows_model_listed_in_scope() -> None:
    """证明：逗号分隔白名单命中即放行（不抛即放行）。

    怎么证明：给 scope="gpt-4o,claude" 判 claude——直接调用，不抛异常即通过。
    """
    check_scope("gpt-4o,claude", "claude")  # 不抛 = 放行（该函数的对外承诺就是"抛或不抛"）


def test_check_scope_allows_any_model_for_wildcard() -> None:
    """证明：scope=`*` = 不限 model（CONTEXT.md scope 词条：`*` 表示全部）。

    怎么证明：给 `*` 判一个不在任何列表里的怪名字——不抛即放行。
    """
    check_scope("*", "whatever-model")  # 不抛 = 通配放行


def test_check_scope_allows_any_model_for_empty_scope() -> None:
    """证明：空 scope = 不限（票面定死语义：与 `*` 同义，防"忘填 scope 即全拒"哑弹）。

    怎么证明：给空串判任意 model 不抛——"建 key 忘填 scope"落在宽侧而不是全拒侧；
    语义由本用例钉死，再想改口径必须先改红本测试。
    """
    check_scope("", "whatever-model")  # 不抛 = 空 scope 与 `*` 同义
    check_scope("   ", "whatever-model")  # 不抛 = 纯空白按空处理（strip 后无实体）


def test_check_scope_parses_comma_list_with_whitespace_and_empty_segments() -> None:
    """证明：列表解析容差——逗号旁空白与空段不参与判定（存取原样、解释在这一层）。

    怎么证明：scope=" gpt-4o , ,claude "（带空白与空段）判 gpt-4o / claude 均不抛；
    判别的名字仍拒。这钉住"store 原样存、解释层负责读得懂"的分层承诺。
    """
    check_scope(" gpt-4o , ,claude ", "gpt-4o")  # 不抛 = 左空白剥掉
    check_scope(" gpt-4o , ,claude ", "claude")  # 不抛 = 空段直接跳过

    with pytest.raises(AuthorizationError):
        check_scope(" gpt-4o , ,claude ", "qwen-max")  # 仍拒 = 容差不放大白名单


def test_check_scope_rejects_when_scope_missing() -> None:
    """证明：scope 查无（None）= fail-closed 拒绝——与"空 scope=不限"严格分家。

    怎么证明：check_scope(None, ...) 必抛 AuthorizationError。None 只会来自
    scope_of 查无 key_id（身份凭空消失的防御位）；fail-closed 传统（story 2：
    空库=拒绝一切）下查无绝不能落进"空=不限"的宽侧——两种"没有"必须可区分。
    """
    with pytest.raises(AuthorizationError):
        check_scope(None, "whatever-model")


# ===== store 的 scope 存取口（w4 票 03 新增读口，票 01 契约的窄扩展）=====


def test_scope_of_returns_stored_scope_text_verbatim(tmp_path: Path) -> None:
    """证明：scope_of 按 key_id 原样取回 scope 文本——只存取不解释（票 01 纪律）。

    怎么证明：临时库 create 一把 scope="gpt-4o,claude" 的 key，scope_of 取回
    断言逐字符相等——"读得懂"（拆逗号、剥空白）是 check_scope 的活，这里不许
    夹带任何解释（原样返回就是分层的字面证据）。
    """
    store = KeyStore(str(tmp_path / "keys.db"))

    created = store.create(name="demo", scope="gpt-4o,claude")

    assert store.scope_of(created["key_id"]) == "gpt-4o,claude"


def test_scope_of_returns_none_for_unknown_key_id(tmp_path: Path) -> None:
    """证明：查无 key_id 回 None——与 check_scope 的 fail-closed 位衔接（身份消失=拒）。

    怎么证明：对空库查一个瞎编的 key_id，断言 None。这是"空库=拒绝一切"链条的
    读口一环：None 进 check_scope 必拒，fail-closed 闭环（story 2）。
    """
    store = KeyStore(str(tmp_path / "keys.db"))

    assert store.scope_of("key_nope") is None


def test_scope_of_still_returns_scope_after_revocation(tmp_path: Path) -> None:
    """证明：撤销不挡 scope 存取——存取层不解释撤销语义（归认证门卫的 verify）。

    怎么证明：create 后 revoke，scope_of 仍原样取回。撤销的拦截点在 verify
    （WHERE revoked_at IS NULL → 401），scope_of 是审计读口；"谁能进来"与
    "进来后能用什么"两问分属两层，互不偷跑。
    """
    store = KeyStore(str(tmp_path / "keys.db"))
    created = store.create(name="demo", scope="gpt-4o")

    assert store.revoke(created["key_id"]) is True

    assert store.scope_of(created["key_id"]) == "gpt-4o"
