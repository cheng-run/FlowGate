"""虚拟 key 表的唯一读写面（w4 issue 01）。

为什么是单表 + 四个操作（create/verify/revoke/list）：spec 与票面把 update/rotate/
父子 key 全部否决（Speculative Generality）——管理面只需要"签发、查验、吊销、审计"
四件事，多一个入口就多一条绕过审计的路。
为什么 sqlite3 不引 ORM：依赖红线——stdlib 就是单文件库（与 billing 账本同纪律），
ORM 带来的迁移/关系设施本项目用不到。

为什么库里只存 hash 不存明文：凭据串=泄露即可用的机密（story 12），明文落库等于
把"库文件被读"直接兑换成"冒充调用方"；hash 单向、校验时现算现比，库泄露换不回凭据。
为什么 sha256 够、不用 bcrypt 慢哈希：bcrypt 的价值是拖慢对**低熵口令**的字典暴力
（口令只有几千种常见组合）；本凭据是 32 字节密码学随机 = 256-bit 熵，暴力空间
2^256 根本走不动，慢哈希拖慢的只有自己每次校验的延迟、换不来任何安全收益——这
就是面试要讲的 trade-off。
"""

import hashlib
import secrets
import sqlite3
from datetime import UTC, datetime

# 凭据串前缀：机密串与公开标识（key_）在日志、报错里一眼分清——回显事故少一层遮掩
_CREDENTIAL_PREFIX = "fgk_"
# key_id 前缀：spec 规定的公开标识形态（CONTEXT.md：账本/限流/预算统一身份）
_KEY_ID_PREFIX = "key_"


def _now() -> str:
    """UTC ISO-8601 时间戳（created_at / revoked_at 共用）。

    为什么带时区偏移：跨进程（keyctl 与 serve 同库）可比、可排序，纯本地时间在
    换时区的机器上会把审计时间读错。
    """
    return datetime.now(UTC).isoformat()


def _hash_credential(credential: str) -> str:
    """凭据串 → sha256 十六进制摘要——库里唯一存在的凭据形态（详见模块 docstring）。

    为什么是 hex 文本而非裸字节：列类型 TEXT，十六进制 64 字符可读可直接比对，
    省掉 BLOB 的编解码又不损失任何东西（摘要本就不是机密，无需再编码隐藏）。
    """
    return hashlib.sha256(credential.encode()).hexdigest()


class KeyStore:
    """虚拟 key 表的唯一读写面：create / verify / revoke / list。"""

    def __init__(self, db_path: str) -> None:
        """key 表落在 SQLite 单文件（":memory:"=进程内临时库，测试走 tmp_path 文件）。

        为什么 check_same_thread=False：与 BillingLedger 同一口径——TestClient/uvicorn
        会在别的线程跑 ASGI，sqlite3 默认"谁建谁用"会在在线程边界上炸。
        为什么 CREATE TABLE IF NOT EXISTS：与 billing 各管各表、自治建表——同库两个
        写面互不依赖对方的构造顺序（spec：一个 SQLite 文件的完工承诺）。
        """
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS keys (
                key_id TEXT PRIMARY KEY,
                credential_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                scope TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            )
            """
        )
        self._db.commit()

    def create(self, name: str, scope: str) -> dict[str, str]:
        """签发一把新虚拟 key：生成凭据串，返回 {key_id, credential}——明文仅此一次。

        name/scope 由调用方给定、原样落库：store 层不解释 scope（解释归 03 授权层），
        name 是人读标签也就不做校验（keyctl/serve 各定各的默认值）。
        """
        # 行级：凭据串=fgk_ + 32 字节密码学随机——token_urlsafe 输出 URL 安全字符，
        # 可直接进 Authorization 头与命令行，不会撞上需要转义的符号
        credential = _CREDENTIAL_PREFIX + secrets.token_urlsafe(32)
        # 行级：公开标识同样带随机后缀——PK 撞车在 2^-96 量级，不值得重试循环
        key_id = _KEY_ID_PREFIX + secrets.token_urlsafe(12)
        with self._db:  # 行级：with 连接=事务提交，写入失败自动回滚（与账本同纪律）
            # 行级：入库的是 sha256 摘要，明文只随返回值离开（为什么见模块 docstring）
            self._db.execute(
                "INSERT INTO keys (key_id, credential_hash, name, scope, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (key_id, _hash_credential(credential), name, scope, _now()),
            )
        return {"key_id": key_id, "credential": credential}

    def verify(self, credential: str) -> str | None:
        """凭据串 → key_id；查无（不存在 / 已撤销）返回 None——两者刻意不可区分。

        为什么同一返回：HTTP 层据此发**同一条** 401（防枚举，story 3）——store 若
        分开两种失败，文案分叉只是时间问题；每请求现查库 = 撤销即时生效（软删当请求
        就被 WHERE revoked_at IS NULL 挡住）。
        为什么校验现算 hash 而不反查明文：库里本来就没有明文可查（见模块 docstring）。
        """
        # 行级：凭据先 sha256 再按 credential_hash 精确匹配，未撤销是同一条 SQL 的一部分
        row = self._db.execute(
            "SELECT key_id FROM keys WHERE credential_hash = ? AND revoked_at IS NULL",
            (_hash_credential(credential),),
        ).fetchone()
        return None if row is None else row[0]

    def list(self) -> list[dict[str, str]]:
        """全量审计视图（含已撤销项）：key_id / name / scope / created_at / status。

        为什么含已撤销项：软删的意义就是审计痕迹（story 14）——撤销过的 key 必须
        仍可见"存在过、何时没的"，物理删除会让签发历史失去对账面。
        为什么 status 是机械映射而非解释：由 revoked_at 是否为 NULL 推出
        active/revoked，属存取；scope 这种"判断语义"才归 03 授权层。
        为什么按 created_at 排序：keyctl list 与人读审计要稳定顺序（新库新签发在后）。
        """
        rows = self._db.execute(
            "SELECT key_id, name, scope, created_at, revoked_at FROM keys"
            " ORDER BY created_at, rowid"  # 行级：同刻签发按落库顺序兜底，输出确定
        ).fetchall()
        # 行级：sqlite 行 → 对外字典；状态由 revoked_at 机械推出（是存取不是解释）
        return [
            {
                "key_id": key_id,
                "name": name,
                "scope": scope,
                "created_at": created_at,
                "status": "revoked" if revoked_at is not None else "active",
            }
            for key_id, name, scope, created_at, revoked_at in rows
        ]

    def revoke(self, key_id: str) -> bool:
        """置 revoked_at 软删一把 key：返回 True=已吊销，False=key_id 查无。

        为什么是软删不是 DELETE：行保留供 list 审计（story 14），账本与限流桶也
        按"撤销只断今后、不改历史"的口径不动旧账（spec：Out of Scope 明确不清理）。
        为什么返回 bool 给 CLI：keyctl 是管理员面不必防枚举（票 04 要对不存在的 id
        响亮报错）——与 verify 统一回"查无"的防枚举口径分工不同，因为调用方不同。
        为什么 COALESCE：首撤时刻是审计事实，重复撤销不改写第一次的时间。
        """
        with self._db:  # 行级：事务提交（与 create 同纪律），失败自动回滚
            # 行级：rowcount>0 ⇔ 行存在——COALESCE 保证已撤销的行也计入，一次 UPDATE
            # 同时给出"吊销"与"查无"两个答案
            cursor = self._db.execute(
                "UPDATE keys SET revoked_at = COALESCE(revoked_at, ?) WHERE key_id = ?",
                (_now(), key_id),
            )
        return cursor.rowcount > 0
