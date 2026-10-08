"""装配处（composition root）：全代码库唯一点名具体适配器、装出生产单例的地方。

为什么独立成根级模块（W4 票 05，把 ADR-0001 落成物理事实）：五个装配函数与四个
单例原本住在 app/main.py，"app/ 不 import 具体适配器"就只能靠豁免名单加自觉；
整体迁出后 app/ 只留路由、中间件、异常 handler、门卫，边界检查（票 06 的 ast 扫
import）才能零例外钉死——"唯一点名处"从一句自律变成一个物理事实。

四个单例（ledger/budget/provider/limiter）在这里落座、由 app/main.py 导入引用：
名字在 app.main 命名空间重新绑定，既有测试 monkeypatch app.main.provider / limiter
的手法（含 demo 脚本的 gateway.provider = …）一个不破。
"""

import os

from billing.ledger import BillingLedger
from billing.settlement import BillingProvider
from keys.store import KeyStore
from providers.base import Provider
from providers.dashscope import DEFAULT_BASE_URL, DashScopeProvider
from providers.fake import FakeProvider
from providers.kimi import KimiProvider
from ratelimit.bucket import RateLimiter
from routing.chain import FallbackChain


def create_provider() -> Provider:
    """装配处核心：按环境变量装上游——逗号表 = 顺序 fallback 链（issue 02，默认 fake）。

    这是生产装配逻辑本身，不是"为测试加的钩子"（spec 决定：不加工厂函数/不上 DI）——
    选择逻辑无论放哪都得有个名字可调，写成函数只是把装配决策显式化；
    它不接收任何注入参数，测试与生产走同一条入口。
    逗号表口径（checklist 2）：FLOWGATE_PROVIDER=a,b = a 先试、a 挂 b 接管；
    **单值即单元素链**——退化回上游本身，历史行为一字不差（向后兼容：既有装配测试的
    isinstance(create_provider(), FakeProvider) 断言不改仍绿）。
    读环境为什么不引 python-dotenv：uv 的 `--env-file .env` 启动时注入环境，
    代码只读 os.environ——零新增依赖（依赖红线：能零依赖就不引库）。
    """
    # 上游选择：FLOWGATE_PROVIDER 缺省 fake——无 key 也能跑测试与演示
    choice = os.environ.get("FLOWGATE_PROVIDER", "fake")
    # 复杂语句（推导式）行上：逗号切表、去空白——顺序即 fallback 优先级
    names = [part.strip() for part in choice.split(",")]
    # 复杂语句（推导式）行上：逐条装成上游实例（未知/空条目在 _build_one 里响亮报错）
    providers = [_build_one(name, choice) for name in names]
    if len(providers) == 1:
        return providers[0]  # 行级：单元素链退化为上游本身（向后兼容的字面兑现）
    return FallbackChain(providers)


def _build_one(name: str, choice: str) -> Provider:
    """装一个上游实例——全代码库唯一点名具体适配器类的函数（ADR-0001）。

    为什么拆出来：逗号表要逐条装配，if-ladder 收在这里，"点名适配器"仍只在
    装配处一处；未知取值/空条目响亮报错（配错喊响），消息带原始 choice 供定位。
    拆不动说明（函数物理行数超 40，含 docstring/注释；同 routing/chain.py 先例）：
    if-ladder 就是"唯一点名适配器"的机制本体——拆成每上游一个小函数会把点名面
    摊到多处，抽 _require_env 通用助手则把各档的配置故事（dashscope 的 base_url
    有默认 / kimi 双必填，这个不对称是刻意的）藏进第三处；教学注释是规范硬要求
    删不得，行数超限以本说明豁免。
    """
    if name == "fake":
        return FakeProvider()
    if name == "dashscope":
        # 缺 key 必须响亮且指名（故事 10）：静默降级回 fake 会让
        # "我明明配了真上游，怎么答的还是回显"变成难查的悬案。
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "FLOWGATE_PROVIDER=dashscope 但缺少环境变量 DASHSCOPE_API_KEY"
                "（写入 .env，用 uv run --env-file .env 启动）"
            )
        # base_url 可覆盖（默认官方 OpenAI 兼容前缀）：接中转/代理靠这个口子
        base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        return DashScopeProvider(api_key=api_key, base_url=base_url)
    if name == "kimi":
        # 缺 key 必须响亮且指名（沿 ADR-0002 纪律，与 dashscope 同款）：静默降级回
        # fake 会让"我明明配了 kimi，怎么答的还是回显"变成难查的悬案
        api_key = os.environ.get("KIMI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "FLOWGATE_PROVIDER=kimi 但缺少环境变量 KIMI_API_KEY"
                "（写入 .env，用 uv run --env-file .env 启动）"
            )
        # base_url 同样必填、无内置默认（Kimi 与 DashScope 的不对称是刻意的）：Kimi 走
        # 私有中转，真实地址只活在 .env（issue 08 checklist"真实地址永不进 git"）——
        # 内置默认地址就是猜，猜错（拿中转 key 打官方地址）是运行期 401 的静默悬案
        base_url = os.environ.get("KIMI_BASE_URL", "")
        if not base_url:
            raise RuntimeError(
                "FLOWGATE_PROVIDER=kimi 但缺少环境变量 KIMI_BASE_URL"
                "（真实中转地址写入 .env，不入 git；用 uv run --env-file .env 启动）"
            )
        return KimiProvider(api_key=api_key, base_url=base_url)
    # 未知取值（含空条目）同样响亮报错：拼错的上游名若静默回 fake，配置就形同虚设
    raise RuntimeError(
        f"未知的上游 {name!r}（FLOWGATE_PROVIDER={choice!r}；可选：fake / dashscope / kimi）"
    )


def create_limiter() -> RateLimiter:
    """装配处：按环境变量装限流器（env 口径，与 create_provider 同一纪律）。

    FLOWGATE_RATE_CAPACITY=桶容量（允许的突发），FLOWGATE_RATE_PER_SECOND=稳态吞吐
    （tokens/s）。默认 20/10：LLM 客户端天然突发（一条请求一轮对话），20 的突发余量
    给交互式客户端留了面子，10/s 的稳态又足以拦住打爆型流量——口径随部署调 env 即可，
    不用改代码。配置非法（非数字/越界）响亮报错（配错喊响，同 _build_one 纪律）。
    为什么不加工厂函数/不上 DI：与 create_provider 同一理由——这是生产装配逻辑本身，
    测试与生产走同一条入口，测试要换限流口径时 monkeypatch 装配绑定（app.main.limiter）。
    """
    try:
        capacity = float(os.environ.get("FLOWGATE_RATE_CAPACITY", "20"))
        per_second = float(os.environ.get("FLOWGATE_RATE_PER_SECOND", "10"))
        return RateLimiter(capacity=capacity, per_second=per_second)
    except ValueError as exc:
        # 行级：数字解析失败与桶参数越界都是配置错——统一指名 env 变量，排错不用猜
        raise RuntimeError(
            f"FLOWGATE_RATE_CAPACITY / FLOWGATE_RATE_PER_SECOND 配置非法：{exc}"
        ) from exc


def create_ledger() -> BillingLedger:
    """装配处：按环境变量装账本（env 口径，与 create_provider 同一纪律）。

    FLOWGATE_BILLING_DB=SQLite 单文件路径（生产在 .env 指向落盘文件）；缺省
    ":memory:"=进程内临时库——测试/演示零残留、离线可复跑（checklist 1 的口径：
    测试用 tmp_path 临时文件显式构造账本，不依赖缺省）。
    """
    db_path = os.environ.get("FLOWGATE_BILLING_DB", ":memory:")
    return BillingLedger(db_path)


def create_keystore() -> KeyStore:
    """装配处：按环境变量装虚拟 key 库（env 口径与账本**同一个** FLOWGATE_BILLING_DB）。

    为什么和账本共用 env：spec 的"一个 SQLite 文件"完工承诺——key 表与账本表落
    同一个文件、各管各表（CREATE TABLE IF NOT EXISTS 自治建表，keys/store.py 同款
    纪律），部署只用管一个库路径。
    为什么缺省也是 ":memory:"：与 create_ledger 同款分层缺省——测试/演示零残留；
    注意 :memory: 下两个连接是两座互不相见的临时库（key 表与账本表各活各的），
    测试要的正是这个隔离（conftest 种子只进 key 库），落盘时才合成同一个文件。
    """
    db_path = os.environ.get("FLOWGATE_BILLING_DB", ":memory:")
    return KeyStore(db_path)


def create_budget() -> int:
    """装配处：按环境变量装每 key 的 token 预算（FLOWGATE_BUDGET_TOKENS）。

    0=不设限（缺省）：记账先跑、预算口径按部署调 env——"0"是"无预算"不是"零预算"。
    为什么不给个有限缺省：预算一旦有限，账本跨重启累计（生产）就会在跑批场景里
    突然 429；治理口径宁可显式开启（与限流不同：限流天然要挡，预算天然要看部署）。
    """
    try:
        return int(os.environ.get("FLOWGATE_BUDGET_TOKENS", "0"))
    except ValueError as exc:
        # 行级：非数字=配置错，统一指名 env 变量（配错喊响，同 create_limiter 纪律）
        raise RuntimeError(f"FLOWGATE_BUDGET_TOKENS 配置非法：{exc}") from exc


# ===== 装配处（composition root）=====
# 全代码库唯一允许点名具体适配器类的地方（ADR-0001）：核心路由只见 Provider 协议，
# 换上游=改环境变量 FLOWGATE_PROVIDER，业务代码一行不动。
# 虚拟 key 库先落座：认证门卫每请求查它（撤销即时生效的前提）；与账本同库文件
keystore: KeyStore = create_keystore()
# 账本单例先落座：结算门面与门卫的预算检查共用同一本账（用户账求和=花销唯一出处）
ledger: BillingLedger = create_ledger()
# 每 key 的 token 预算（0=不设限）：门卫进门先查它，超预算在上游调用前就 429
budget: int = create_budget()
# 结算门面包在**装配绑定**上、不进 create_provider：create_provider 的既有契约是
# 装出可 isinstance 的上游/链（装配测试钉住），计费是其外的一层治理（与门卫同理）。
provider: Provider = BillingProvider(create_provider(), ledger=ledger)
# 限流器与上游同在装配处落座：门卫只认 RateLimiter 接口，换限流算法不惊动路由
limiter: RateLimiter = create_limiter()
