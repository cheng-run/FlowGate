"""W3 可复跑 demo：一条命令演完治理三幕（fake 上游、零外网、真 uvicorn）。

对应 .scratch/fallback-ratelimit-billing/issues/09。W3 周交付物 = 测试全绿 + 可演示——
本脚本就是"可演示"那半场，三幕各对应一个工程点（口径与 tests/ 同源，见各幕注释）：
① fallback 接管且账本恰 1 笔（降级不重复扣费，plan §四.2）；
② 并发压出 429、通过数=桶容量（限流硬保证）；
③ 流末回填 + 二次结算幂等去重（计费两段式，plan §四.1）。
可选第四幕（live）：估算 vs 官方 usage 的误差百分比——无 key 自动跳过，是"计费误差
可量化"（工程点之三）的数字出处。

为什么是 demo 不是测试（沿 W2 分工："测试求确定、demo 求真实"）：三幕的机制证明在
tests/（test_billing.py / test_ratelimit.py / test_billing_stream.py——fake + 临时
SQLite + 可拨时钟，零外网零真实 sleep）；本脚本把同一套数字在**真 uvicorn + 真 HTTP
并发**的高度再演一遍（8 路并发是真连接、流式帧是真 SSE 字节），三幕自检全过退出码 0。
live 幕的语义与 pytest live 标记一致：无 key 跳过归跳过，跑了就得过（不过=退出码 1）。

跑法（一条命令、不用改代码、不用配 key，任何机器可复跑）::

    uv run python -m scripts.demo_w3

live 幕（可选加演，需要真 key，只有这一幕碰真网）::

    uv run --env-file .env python -m scripts.demo_w3
"""

import asyncio
import json
import logging
import os
import socket
import sys

import httpx
import uvicorn

# 行级：装配处（app.main 的 create_provider）在 import 时读环境——先钉死 fake 再 import，
# 保证零外网、零 key 不受宿主环境影响（用户就算配了 DASHSCOPE_API_KEY，三幕也绝不触网）
os.environ["FLOWGATE_PROVIDER"] = "fake"

# 行级：输出编码钉死 UTF-8——沿 demo_w2 的坑（2026-10-06）：GBK 控制台 print 特殊字符
# 直接 UnicodeEncodeError 崩；errors="replace" 兜底，演示宁可个别字形缺角也不中途死
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 以下项目内 import 必须垫在环境钉子之后（app.main 的装配处在 import 时读环境）
import app.main as gateway  # noqa: E402
import ratelimit.bucket as bucket_module  # noqa: E402
from app.schemas import ChatMessage, ChatRequest, Usage  # noqa: E402
from billing.estimator import estimate_tokens  # noqa: E402
from billing.ledger import BillingLedger  # noqa: E402
from billing.settlement import BillingProvider  # noqa: E402
from providers.dashscope import DashScopeProvider  # noqa: E402
from providers.fake import FakeProvider  # noqa: E402
from ratelimit.bucket import RateLimiter  # noqa: E402
from routing.chain import FallbackChain  # noqa: E402

# 行级：uvicorn/ASGI 内部日志全静音——演示输出要干净可念（沿 demo_w2 纪律）
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.lifespan"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# 限流口径=tests/test_ratelimit.py 并发用例同源：gather 8 路同 key、桶容量 3 → 恰 3 过
RATE_CAPACITY = 3
RATE_PER_SECOND = 1.0
CONCURRENT_CLIENTS = 8
# 单次 HTTP 的总闹钟：机制坏了就地红掉，不把演示挂死在台上（沿 demo_w2 的 READ_TIMEOUT）
HTTP_TIMEOUT = 5.0


def _check(ok: bool, message: str) -> None:
    """自检断言：不过就带着可念的原因红掉——"退出码 0"是三幕全过的承诺，不是默认值。"""
    if not ok:
        raise RuntimeError(message)


def _payload(content: str, *, stream: bool = False) -> dict:
    """最小合法请求体；stream=True 时带 include_usage（OpenAI 兼容形状，issue 07）。

    为什么 include_usage 只在流式腿置位：第三幕要演"流末 usage 帧"，这正是 OpenAI
    兼容客户端索取 usage 的标准开关——不置位则流末零 usage 帧（缺省口径），一幕一开关。
    """
    body: dict = {"model": "demo", "messages": [{"role": "user", "content": content}]}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _auth(key: str) -> dict:
    """bearer 头——限流/计费的身份口径（ADR-0007：key=Authorization bearer 串）。

    为什么每幕各用一个 key：桶是 per key 的（不同 key 互不影响），各幕自带身份
    就互不抢令牌——第二幕把桶压干也不会波及第一/三幕的节奏。
    """
    return {"Authorization": f"Bearer {key}"}


async def _start_server() -> tuple[uvicorn.Server, asyncio.Task, int]:
    """起真 uvicorn（真 socket；端口 0 交给 OS 分配，任何机器不撞端口）。

    与 demo_w2 同款，刻意不抽公共模块：两个 demo 各自一文件讲完一个故事，
    走读不用在两个脚本之间跳来跳去（18 行机械装配的复制好过一层跨文件依赖）。
    """
    # 先绑后传：端口在 serve() 之前就知道，不用翻 uvicorn 内部状态
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    # log_config=None：uvicorn 日志全静音——演示输出要干净可念，访问日志不是剧情
    server = uvicorn.Server(
        uvicorn.Config(app=gateway.app, host="127.0.0.1", port=port, log_config=None)
    )
    # 为什么 create_task（限用并发原语，用则配注释）：uvicorn 的 serve() 是长循环，
    # 必须当后台任务跑，主线才走得动三幕；退出时 _amain 里 should_exit + await 回收
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)  # 行级：serve() 的绑定是异步的，等 socket 真正开听
    return server, task, port


async def act1(port: int) -> None:
    """第一幕：fake-a 拒答、fake-b 接管——账本恰 1 笔用户账（降级不重复扣费）。"""
    # 行级：每幕独立内存账本——数字互不污染、演示零残留（缺省装配同款 ":memory:"）
    ledger = BillingLedger(":memory:")
    fake_a = FakeProvider(name="fake-a", failure="fail")
    fake_b = FakeProvider(name="fake-b")
    chain = FallbackChain([fake_a, fake_b])
    # 行级：装配绑定=测试同款 monkeypatch 点（生产零钩子）——链包上结算门面，
    # "链对上层就是一个上游"（ADR-0004）在生产高度同样成立
    gateway.provider = BillingProvider(chain, ledger=ledger)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=HTTP_TIMEOUT
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload("降级不重复扣费"),
            headers=_auth("demo-key-a"),
        )
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    charges = ledger.charges()
    losses = ledger.losses()
    _check(response.status_code == 200, f"接管后该 200，实得 {response.status_code}：{body}")
    _check("fake-reply" in content, f"答复不是 fake-b 的回显：{content!r}")
    _check(fake_a.chat_endings == ["failed"], f"fake-a 不该算正常收场：{fake_a.chat_endings}")
    _check(fake_b.chat_endings == ["completed"], f"fake-b 没有正常答完：{fake_b.chat_endings}")
    _check(len(charges) == 1, f"用户账该恰 1 笔，实得 {len(charges)} 笔")
    _check(len(losses) == 1 and losses[0].upstream == "fake-a", f"损耗账该恰 1 条 fake-a：{losses}")
    # 行级：所见即所付（ADR-0008）——响应 usage 与账本是同一份数字
    _check(body["usage"]["total_tokens"] == charges[0].total_tokens, "响应 usage 与账本数字不一致")
    # 复杂语句（推导式）行上：attempts 一本账（ADR-0004）——谁先试、谁接管一眼可读
    trace = " → ".join(f"{a.upstream} {a.outcome}" for a in chain.attempts)
    print(f"fake-a 恒失败注入 → FallbackChain 自动换路 fake-b 作答，attempts 流水：{trace}")
    print(
        f"用户账恰 {len(charges)} 笔（key=demo-key-a，total={charges[0].total_tokens} tokens），"
        f"内部损耗账恰 {len(losses)} 条（{losses[0].upstream} 的失败尝试，不收钱）"
    )
    print("✓ 第一幕通过——「降级不重复扣费」式子：N 逻辑请求含 M 次失败尝试 → 用户账恰 N 笔")
    print("  （本幕 N=1、M=1；更大口径 N=3、M=6 同式子见 tests/test_billing.py，两处互证）")


# 拆不动说明（act2 整块超 40 行，含 docstring/注释）：冻结时钟→压桶→8 路并发→逐位
# 对账是同一条"限流硬保证"故事的走读现场，抽子函数会把"尺子怎么钉死、压出什么数"
# 的因果链切成两截，故保持单函数（同 act3 的先例）。
async def act2(port: int) -> None:
    """第二幕：8 路同 key 并发硬压——通过恰 C 路（C=桶容量 3），其余 429。"""
    # 冻结桶的时钟（测试同款手法：monkeypatch 时钟引用，不给生产加钩子）——
    # "恰 3 过"绝不掺墙钟：真实时钟下这波请求若摊过 1s，回填就凑出第 4 枚令牌，
    # 慢机器上 4 过 4 拒直接红掉演示；冻结后 elapsed=0 永不回填，与
    # tests/test_ratelimit.py 并发用例同一确定性口径（它也冻结时钟，零真实 sleep）
    old_clock = bucket_module.clock
    bucket_module.clock = lambda: 0.0
    try:
        # 行级：把限流口径换成测试同款（容量 3、速率 1/s）——同 monkeypatch 装配绑定的
        # 手法，生产不加钩子；缺省 20/10 要压出 429 得发 21 发，演示数字反而不如 8 选 3 直白
        gateway.limiter = RateLimiter(capacity=RATE_CAPACITY, per_second=RATE_PER_SECOND)
        fake = FakeProvider()
        ledger = BillingLedger(":memory:")
        gateway.provider = BillingProvider(fake, ledger=ledger)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=HTTP_TIMEOUT
        ) as client:
            # 为什么 gather（限用并发原语，用则配注释）：8 路必须**同时在飞**才叫并发压测——
            # 串行 await 就成了连发 8 发，中间隔了多少帧调度谁也说不清，"恰 C 通过"没了现场
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json=_payload("限流硬保证"),
                        headers=_auth("demo-key-b"),
                    )
                    for _ in range(CONCURRENT_CLIENTS)
                )
            )
    finally:
        bucket_module.clock = old_clock  # 行级：尺子用完放回去，不污染后续演示/测试
    # 复杂语句（推导式）行上：状态码排序逐位对账——恰 3 个 200、恰 5 个 429，
    # 多一个少一个都红（与 tests/test_ratelimit.py 并发用例同一条断言口径）
    statuses = sorted(r.status_code for r in responses)
    expected = [200] * RATE_CAPACITY + [429] * (CONCURRENT_CLIENTS - RATE_CAPACITY)
    _check(statuses == expected, f"通过/拒绝数目不对：{statuses}")
    _check(len(fake.calls) == RATE_CAPACITY, f"上游该只见 {RATE_CAPACITY} 次调用：{fake.calls}")
    rejected = next(r for r in responses if r.status_code == 429)  # 行级：取一发 429 读文案
    print(
        f"{CONCURRENT_CLIENTS} 路并发同 key（Bearer demo-key-b）同时打网关……"
        f"状态码逐位对账：200 恰 {statuses.count(200)} 个、429 恰 {statuses.count(429)} 个"
    )
    print(
        f"上游调用恰 {len(fake.calls)} 次——被拒的 {statuses.count(429)} 发连上游都没碰到"
        f"（拒绝发生在上游调用之前）；429 文案：{rejected.json()['detail']}"
    )
    print("✓ 第二幕通过——「限流硬保证」：并发 8 路同 key，通过数=桶容量 C=3，不多不少")


# 拆不动说明（act3 整块超 40 行，含 docstring/注释）：收帧→验 usage 帧位置→账本对数→
# 二次结算幂等是同一条"流怎么收尾、账怎么结"故事的走读现场，故保持单函数（同 act1）。
async def act3(port: int) -> None:
    """第三幕：流末回填结算恰 1 笔 + usage 帧恰一帧在 [DONE] 前 + 二次结算幂等去重。"""
    ledger = BillingLedger(":memory:")
    fake = FakeProvider()
    gateway.provider = BillingProvider(fake, ledger=ledger)
    prompt = "流末回填"
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=HTTP_TIMEOUT
    ) as client:
        # async with + stream：HTTP 响应体是逐块到货的字节流，要用 async for 边收边看——
        # 这就是"流式"在客户端侧的形状（服务端侧的 async 生成器在 streaming/sse.py）
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=_payload(prompt, stream=True),
            headers=_auth("demo-key-c"),
        ) as response:
            _check(response.status_code == 200, f"流式请求该 200，实得 {response.status_code}")
            request_id = response.headers["x-request-id"]  # 行级：幂等键从响应头取（对账口）
            # 复杂语句（async 推导式+条件）行上：收下全部 "data: " 行——SSE 帧原样可见
            frames = [line async for line in response.aiter_lines() if line.startswith("data: ")]
    _check(frames[-1] == "data: [DONE]", f"完整流该以 [DONE] 收尾，末帧是：{frames[-1]!r}")
    # 复杂语句（推导式+条件）行上：usage 帧=choices 空的载体帧（其余帧 choices 非空）——
    # "usage 出口唯一"（ADR-0008）：透传帧不带 usage 数字，数字只在这一帧
    usage_frames = [
        json.loads(line.removeprefix("data: ")) for line in frames[:-1] if _is_usage_frame(line)
    ]
    _check(len(usage_frames) == 1, f"include_usage 置位该恰 1 帧 usage，实得 {len(usage_frames)}")
    _check(
        _is_usage_frame(frames[-2]), f"usage 帧该紧贴 [DONE] 之前，实得倒数第二帧：{frames[-2]!r}"
    )
    # 复杂语句（推导式）行上：content 帧拼回完整回显——帧是增量、拼起来是那句话
    assembled = "".join(_delta_text(line) for line in frames[:-2])
    _check(assembled == f"fake-reply: {prompt}", f"拼回全文不对：{assembled!r}")
    charges = ledger.charges()
    _check(len(charges) == 1, f"流末回填后用户账该恰 1 笔，实得 {len(charges)} 笔")
    charge = charges[0]
    usage = usage_frames[0]["usage"]
    # 行级：所见即所付——usage 帧数字与账本同一份数字（结算口径，估算或官方）
    _check(usage["total_tokens"] == charge.total_tokens, f"帧账不一致：帧 {usage} vs 账 {charge}")
    # 二次结算（来者不善：携另一组 usage 数字）——幂等 no-op，账本仍恰一笔（账本为准）
    again = ledger.settle(
        request_id,
        "demo-key-c",
        prompt_text=prompt,
        completion_text=f"fake-reply: {prompt}",
        official_usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    _check(len(ledger.charges()) == 1, f"二次结算后该仍恰 1 笔，实得 {len(ledger.charges())} 笔")
    _check(
        again
        == Usage(
            prompt_tokens=charge.prompt_tokens,
            completion_tokens=charge.completion_tokens,
            total_tokens=charge.total_tokens,
        ),
        f"二次结算该返回首次入账的数字（账本为准），实得 {again}",
    )
    print(f"流式请求（include_usage）共 {len(frames)} 帧，content 拼回「{assembled}」")
    print(f"流末两帧：{frames[-2]}  ← usage 帧（恰一帧、紧贴 [DONE] 前）\n          {frames[-1]}")
    print(
        f"流末回填结算：用户账恰 1 笔——prompt={charge.prompt_tokens} "
        f"completion={charge.completion_tokens} total={charge.total_tokens}"
        "（估算口径：fake 官方 usage 为全零占位，对整段文本 tokenize，不按块加总）"
    )
    print("✓ 第三幕通过——二次结算（携来者不善 1/1/2）no-op：账本仍恰 1 笔、数字以首次为准")


def _is_usage_frame(line: str) -> bool:
    """一帧 SSE 是否 usage 载体帧：choices 为空列表且带 usage（OpenAI 兼容形状）。"""
    payload = json.loads(line.removeprefix("data: "))
    return not payload.get("choices") and payload.get("usage") is not None


def _delta_text(line: str) -> str:
    """一帧 SSE 的正文增量（choices[0].delta.content）——content 拼回公式的取材口。"""
    payload = json.loads(line.removeprefix("data: "))
    return payload["choices"][0]["delta"].get("content") or ""


async def act4() -> bool:
    """可选 live 幕（有 key 才碰真网）：估算 vs 官方 usage 的误差百分比。

    返回 True=跑了且过；无 key 返回 False=跳过（三幕结论不受影响）；自检不过抛错
    （与 pytest live 标记同语义：跑了就得过）。口径与 tests/test_dashscope_live.py
    的误差用例同源——同一 prompt、同一条式子：|估算-官方|/官方×100%。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("（无 DASHSCOPE_API_KEY——live 幕自动跳过，三幕的离线结论不受影响）")
        return False
    prompt = "用一句话介绍令牌桶限流"  # 行级：与 live 误差用例同款 prompt——口径同源
    provider = DashScopeProvider(api_key=api_key)  # 行级：不传 transport → 走真网络
    response = await provider.chat(
        ChatRequest(
            model="qwen-turbo",  # 便宜快模型：误差对照只测 tokenizer 口径，不烧贵 token
            messages=[ChatMessage(role="user", content=prompt)],
        )
    )
    official_total = response.usage.total_tokens
    _check(official_total > 0, "真上游 usage.total_tokens 该 > 0——0 说明 usage 没穿过翻译层")
    completion = response.choices[0].message.content
    estimated_total = estimate_tokens(prompt) + estimate_tokens(completion)
    error_pct = abs(estimated_total - official_total) / official_total * 100
    _check(
        error_pct < 100, f"估算 {estimated_total} vs 官方 {official_total}，误差 {error_pct:.1f}%"
    )
    print(f"估算 {estimated_total} vs 官方 {official_total}，误差 {error_pct:.1f}%")
    print("✓ 第四幕（live）通过——「计费误差可量化」的数字出处（进考点卡 06）")
    return True


def _print_footer(live_ran: bool) -> None:
    """收尾横幅：三幕结论 + 面试可念数字；live 幕跑了/跳过单独一行交代，不瞒不漏。"""
    print("\n" + "=" * 60)
    print("三幕全过 ✓  退出码 0。重跑：uv run python -m scripts.demo_w3")
    # 复杂语句（条件表达式）行上：live 幕的两种收场在横幅里都留痕——跳过是常态不算失败
    live_note = "通过，误差数字见上方" if live_ran else "无 DASHSCOPE_API_KEY，自动跳过"
    print(f"第四幕（live）：{live_note}")
    print("面试可念数字（口径与 tests/ 同源）：")
    print("  · 降级不重复扣费：1 逻辑请求含 1 次失败尝试 → 用户账恰 1 笔（式子：N 含 M → 恰 N 笔）")
    print("  · 限流硬保证：并发 8 路同 key，通过恰 3 = 桶容量 C，429 恰 5，上游只见 3 次调用")
    print("  · 计费两段式：流末回填恰 1 笔、二次结算 no-op；[DONE] 前恰 1 帧 usage（所见即所付）")
    print("=" * 60)


async def _amain() -> int:
    """起真 uvicorn，顺序演三幕 + 可选 live 幕；任一幕自检不过 → 退出码 1（全过才是 0）。"""
    server, server_task, port = await _start_server()
    print("=" * 60)
    print("FlowGate W3 可复跑 demo：治理三幕（fake 上游、零外网、真 uvicorn）")
    print(
        f"[准备] uvicorn 监听 127.0.0.1:{port}（OS 分配端口），"
        "三幕上游 = FakeProvider（可控失败形态）"
    )
    print("=" * 60)
    live_ran = False  # 行级：live 幕跑没跑的账——横幅据此交代（跳过是常态，不算失败）
    try:
        for name, run in [("第一幕", act1), ("第二幕", act2), ("第三幕", act3)]:
            print(f"\n── {name} ──────────────────────────────────────")
            try:
                await run(port)
            except Exception as exc:
                # 行级：带异常类型名——TimeoutError 这类空 message 异常只有类型说得清死因
                print(f"\n✗ {name} 自检未通过：{type(exc).__name__}: {exc}")
                return 1
        print("\n── 第四幕（可选 live）───────────────────────────")
        try:
            live_ran = await act4()  # 行级：True=跑了且过、False=无 key 跳过（见 act4 docstring）
        except Exception as exc:
            # 行级：live 幕跑了就得过（同 pytest live 标记语义）——过了才有误差数字可念
            print(f"\n✗ 第四幕（live）自检未通过：{type(exc).__name__}: {exc}")
            return 1
    finally:
        server.should_exit = True
        try:
            # 为什么 wait_for（限用并发原语）：给服务器回收也上闹钟——到点进下面的强拆，
            # 演示脚本绝不挂死在收尾上（沿 demo_w2 纪律）
            await asyncio.wait_for(server_task, timeout=5)
        except TimeoutError:
            server_task.cancel()  # 行级：5s 内退不干净就强拆——演示脚本绝不挂死
    _print_footer(live_ran)
    return 0


def main() -> int:
    """同步入口：asyncio.run 跑三幕 + live 幕，退出码 = 演示自检结果（0=全过/跳过）。"""
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
