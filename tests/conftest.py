"""测试套件共享装配：套件级 TEST_KEY 的铸造 + 默认认证头 + 限流器放宽（W4 票 02）。

为什么种子走生产 create() 而不是插桩开后门：测试与生产同一条写入路径——库里
永远只有 credential_hash（"明文不落库"由 tests/test_keys_store.py 钉住，这里只
复用它的保证），KeyStore 的公共面不必为测试开第二个入口。
"固定 TEST_KEY"的口径：**一把**套件级凭据——import 时铸一次、整个测试进程不变，
与"每用例各自铸 key"相对；明文值是 create() 返回的随机凭据串（写死字面量反而
要给生产 API 开"指定凭据"后门，面试故事不如"同一条路"干净）。
要独立身份的用例（限流/预算隔离）再各自 mint_key()——铸几把都是生产同款。
限流器放宽的口径与证据见下方注释（7 红实录）——它是回归的机械件，不是行为断言。
"""

import app.main as gateway
from app.main import keystore
from ratelimit.bucket import RateLimiter

# import 时铸造：pytest 先载入 conftest 再收测试模块，各文件的 client 默认头
# （模块级一行）拿到的一定是同一把 TEST_KEY——顺序上不可能踩空
_minted = keystore.create(name="test-suite", scope="*")
TEST_KEY: str = _minted["credential"]  # 套件级测试凭据串（明文仅存于本常量）
TEST_KEY_ID: str = _minted["key_id"]  # 它的公开身份 key_id——账本/限流/预算断言用

# 套件级限流器放宽（2026-10-08 实测踩坑）：W3 时代的 HTTP 用例匿名不占桶，W4 带上
# 默认 TEST_KEY 后共享同一把桶，test_streaming 连发 20+ 发被生产默认口径（20/10s）
# 挤成 429（7 红实录）——与限流无关的用例不该被默认桶口径误伤；限流行为由
# tests/test_ratelimit.py 自备限流器（monkeypatch）专门钉，那里一行不靠这个放宽。
gateway.limiter = RateLimiter(capacity=10_000, per_second=10_000)


def mint_key(name: str = "test", scope: str = "*") -> dict[str, str]:
    """再铸一把独立测试虚拟 key（走生产同一条 create 路），返回 {key_id, credential}。

    需要"独立身份"的用例用它（限流桶隔离、预算隔离）：不同凭据串 → 不同 key_id
    → 各自的桶/预算互不为邻——与生产签发一字不差。
    """
    return keystore.create(name=name, scope=scope)


def auth_headers(credential: str = TEST_KEY) -> dict[str, str]:
    """Bearer + 凭据串的请求头（缺省=套件级 TEST_KEY）——认证头的单点构造处。

    为什么凭据串不再是身份（W4 口径）：认证门卫验出 key_id 才是账本/限流/预算的
    对账口径，凭据串只是敲门砖（验过即弃）——头里装的永远是敲门砖，别处不再
    手拼 "Bearer …"（评审 Duplicated Code 的处置：三处并一）。
    """
    return {"Authorization": f"Bearer {credential}"}
