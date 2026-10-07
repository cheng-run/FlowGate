"""账本：结算 + 查询，两本账藏在 SQLite 后面（issue 06 的唯一读写面）。

为什么接口只有"结算 + 查询"：幂等与不重复扣费要有一个**可观测面**——用户账按
request_id 唯一（一个逻辑请求至多一笔）、失败尝试只进内部损耗账（尝试 ≠ 结果），
全部经本接口兑现；谁绕过它直写库，"至多一笔"就没有了唯一防线（spec 决定）。
为什么 SQLite 不引 ORM：依赖红线——stdlib sqlite3 就是单文件账本（"SQLite 单文件
落账"），ORM 换来的是本项目用不到的迁移/关系设施。
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas import Usage
from billing.estimator import estimate_tokens
from routing.chain import Attempt


class BudgetExceededError(Exception):
    """超预算拒绝的失败形状：该 key 的累计花销已顶穿预算——app/ 翻成 429（story 14）。

    为什么是异常不是返回 bool：与 RateLimitError / UpstreamError 同一失败词汇纪律——
    深模块只抛失败形状，HTTP 状态码翻译集中在 app/ 的异常处理器，路由保持传送带。
    为什么与 RateLimitError 分开两类：客户端动作不同——限流该"退避后再来"，
    超预算是"再来多少次都没用，得提额"（两个 429 的 detail 文案各说各的）。
    """


@dataclass
class Charge:
    """用户账一行：一个逻辑请求至多一笔（request_id 唯一约束兜底）。

    为什么是 dataclass 不是 Pydantic：内部观测面（对账/查账），不进 wire——
    wire 形状归 app/schemas（与 routing.chain.Attempt 同一纪律）。
    """

    request_id: str  # 幂等键：二次结算按它判 no-op
    key: str | None  # 计费身份（匿名=None）——预算按 key 对用户账求和
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Loss:
    """内部损耗账一行：一次失败尝试（attempt 粒度）——入账不收钱（尝试 ≠ 结果）。

    字段从链的 attempts 流水落库（spec：观测故事只有一个源）：账本留"谁、什么
    收场、怎么死的"（身份 + 死因原文），起止时刻留在 attempts 观测面（耗时分析的
    材料）——两本账各取所需，死因不改写、不摘要。
    """

    request_id: str
    upstream: str  # 谁挂了——"fake-a 失败、fake-b 接管"按名字对账
    outcome: str  # 失败收场词汇（failed / timeout）——ok 的尝试不进损耗账
    failure_shape: str | None  # 失败形状（异常类名）
    detail: str | None  # 死因原文


class BillingLedger:
    """两本账的唯一读写面：settle（结算）+ charges/spend（查询）。"""

    def __init__(self, db_path: str) -> None:
        """账本落 SQLite 单文件（":memory:"=进程内临时库，测试走 tmp_path 文件）。

        为什么连接持一只、check_same_thread=False：写都是毫秒级短写（ADR-0003
        清理链警告的落地口径之一），TestClient/uvicorn 会在别的线程跑 ASGI——
        sqlite3 默认"谁建谁用"，不放开线程限制会在线程边界上炸。
        """
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_charges (
                request_id TEXT PRIMARY KEY,
                key TEXT,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL
            )
            """
        )
        # 内部损耗账：attempt 粒度一行一尝试——与用户账同库两表（spec：账本两本）
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS internal_losses (
                request_id TEXT NOT NULL,
                upstream TEXT NOT NULL,
                outcome TEXT NOT NULL,
                failure_shape TEXT,
                detail TEXT
            )
            """
        )
        self._db.commit()

    def settle(
        self,
        request_id: str,
        key: str | None,
        *,
        prompt_text: str,
        completion_text: str | None,
        official_usage: Usage | None,
        attempts: Sequence[Attempt] = (),
    ) -> Usage | None:
        """结算一个逻辑请求：官方 usage 优先、缺失则估算入账恰一笔；二次结算 no-op。

        completion_text=None 表示无交付（失败/中断）——零用户账、失败尝试照入损耗账
        （story 19：答不上就分文不取）。prompt_text/completion_text 是估算器的原料
        （官方缺失时对整段文本 tokenize，不按块加总），估算口径见 billing/estimator.py。
        attempts=本轮尝试流水，失败的（outcome≠ok）入内部损耗账、永不进用户账
        （checklist 3）。幂等两道防线（checklist 2）：先判"已结算过"整笔 no-op（覆盖
        两本账）；request_id PRIMARY KEY 唯一约束是兜底——绕过判重硬写也不会出第二笔。
        """
        with self._db:
            # 行级：幂等判定——用户账有记录=已结算过，整笔 no-op、返回以账本为准
            existing = self._find_charge(request_id)
            if existing is not None:
                return Usage(
                    prompt_tokens=existing.prompt_tokens,
                    completion_tokens=existing.completion_tokens,
                    total_tokens=existing.total_tokens,
                )
            if self._has_losses(request_id):
                return None  # 行级：先前按失败结算过（零用户账）——幂等口径两本账一致
            # 行级：无交付=零用户账、口径返回 None；有交付才走官方/估算的结算口径
            settled = None
            if completion_text is not None:
                # 行级：结算口径=官方 usage 优先（有则用=不重算不改写）、缺失（None
                # 或全零占位）则对整段文本估算——账面永不再躺"占位 0"（checklist 7）
                settled = self._usage_or_estimate(prompt_text, completion_text, official_usage)
                # 行级：OR IGNORE=唯一约束兜底——判重被绕过硬写时写入 0 行，账本仍恰一笔
                self._db.execute(
                    "INSERT OR IGNORE INTO user_charges VALUES (?, ?, ?, ?, ?)",
                    (
                        request_id,
                        key,
                        settled.prompt_tokens,
                        settled.completion_tokens,
                        settled.total_tokens,
                    ),
                )
            self._record_losses(attempts)  # 行级：失败尝试入损耗账（attempt 粒度）
            return settled

    def _record_losses(self, attempts: Sequence[Attempt]) -> None:
        """失败尝试入内部损耗账（attempt 粒度一行一条）；ok 的尝试不入账。

        为什么 ok 不入：那次是"结果"、由用户账代表——再进损耗账就成了重复记账；
        为什么拆出来：settle 的三段（幂等判定/用户账/损耗账）各管一事，损耗账的
        过滤规则（outcome≠ok）单独可讲，settle 也因此收进 40 行红线（AGENTS.md）。
        """
        # 复杂语句（推导式+条件）行上：只收失败尝试——过滤口径就是"尝试 ≠ 结果"
        for attempt in attempts:
            if attempt.outcome != "ok":
                self._db.execute(
                    "INSERT INTO internal_losses VALUES (?, ?, ?, ?, ?)",
                    (
                        attempt.request_id,
                        attempt.upstream,
                        attempt.outcome,
                        attempt.failure_shape,
                        attempt.detail,
                    ),
                )

    def _usage_or_estimate(
        self, prompt_text: str, completion_text: str | None, official_usage: Usage | None
    ) -> Usage:
        """结算口径：官方 usage 优先；缺失（None 或全零占位）则对整段文本估算。

        为什么全零也算缺失：fake 的 Usage() 就是全零占位（"不装懂"，见 app/schemas）——
        把占位 0 当官方数字入账，预算求和就永远看不见花销（spec"真数字不再占位 0"）。
        """
        # 行级：三字段任一 > 0 即视为官方数字（全零占位走估算）——直白写三个比较，
        # 不用 getattr 字符串循环（走读优先：一眼看清"看哪三个数"）
        has_official = official_usage is not None and (
            official_usage.prompt_tokens > 0
            or official_usage.completion_tokens > 0
            or official_usage.total_tokens > 0
        )
        if has_official:
            return official_usage  # 行级：有则用——官方数字原样入账，不重算
        # 行级：缺失走估算——prompt 与 completion 各对整段文本 tokenize，total=两者之和
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(completion_text or "")
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _find_charge(self, request_id: str) -> Charge | None:
        """取一笔用户账（幂等判定/读回用）；None=还没结算过。"""
        row = self._db.execute(
            "SELECT request_id, key, prompt_tokens, completion_tokens, total_tokens"
            " FROM user_charges WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return None if row is None else Charge(*row)

    def _has_losses(self, request_id: str) -> bool:
        """该 request_id 是否已有损耗记录——失败路径（零用户账）的幂等判定面。"""
        row = self._db.execute(
            "SELECT 1 FROM internal_losses WHERE request_id = ? LIMIT 1",
            (request_id,),
        ).fetchone()
        return row is not None

    def charges(self, request_id: str | None = None) -> list[Charge]:
        """查用户账：缺省全量，带 request_id 则只查那一笔——"账本恰 N 笔"的断言口。"""
        query = (
            "SELECT request_id, key, prompt_tokens, completion_tokens, total_tokens"
            " FROM user_charges"
        )
        params: tuple = ()
        if request_id is not None:
            query += " WHERE request_id = ?"
            params = (request_id,)
        # 复杂语句（推导式）行上：SQL 行 → Charge 对外形状——查询口不漏 sqlite3.Row
        return [Charge(*row) for row in self._db.execute(query, params)]

    def losses(self, request_id: str | None = None) -> list[Loss]:
        """查内部损耗账：缺省全量，带 request_id 则只查那一轮——"M 次失败尝试"的断言口。"""
        query = "SELECT request_id, upstream, outcome, failure_shape, detail FROM internal_losses"
        params: tuple = ()
        if request_id is not None:
            query += " WHERE request_id = ?"
            params = (request_id,)
        # 复杂语句（推导式）行上：SQL 行 → Loss 对外形状（与 charges 同一查询口纪律）
        return [Loss(*row) for row in self._db.execute(query, params)]

    def spend(self, key: str) -> int:
        """某 key 的累计花销（用户账 total_tokens 求和）——预算前置检查的数字口径。

        为什么只对用户账求和：失败尝试不收钱（内部损耗账），也就不该消耗预算——
        "尝试 ≠ 结果"在预算上的直接推论；为什么按 total_tokens：记账不论价
        （spec 明确出界：单价表/币种不做），token 数就是 v1 的"花销"单位。
        """
        row = self._db.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM user_charges WHERE key = ?",
            (key,),
        ).fetchone()
        return int(row[0])
