"""W2 可复跑 demo：一条命令演完流式三幕（fake 上游、零外网、真 socket）。

对应 .scratch/sse-streaming/issues/05。W2 周交付物 = 测试全绿 + 可演示——本脚本就是
"可演示"那半场，三幕各对应一票机制：
① 逐块 SSE 输出（01 流式闭环，打字机效果可见）；
② 真 socket 中途 abort 后上游被显式取消（02 断连不泄漏，带实测数字）；
③ gap 超时截断、半截回答不带 [DONE]（03 截断语义）。

为什么是 demo 不是测试（spec 分工："测试求确定、demo 求真实"）：第二幕刻意用
真 uvicorn + 真 socket + 真客户端 abort，不用 ASGI 测试桩——代价是墙钟节奏与端口，
这正是确定性测试套件排除的东西。所以三幕的自检直接做在本脚本里（全过退出码 0），
tests/ 仍是零外网、零真实 sleep 的确定性套件；机制的证明在 tests/
（断连/超时用例全在 ASGI 高度，见 tests/test_streaming.py），本脚本只负责
"同一套结论在生产高度也演得出来"。

跑法（一条命令、不用改代码、不用配 key，任何机器可复跑）::

    uv run python -m scripts.demo_w2
"""

import asyncio
import json
import logging
import os
import socket
import sys

import uvicorn

# 行级：装配处（app.main 的 create_provider）在 import 时读环境——先钉死 fake 再 import，
# 保证零外网、零 key 不受宿主环境影响（用户就算配了 DASHSCOPE_API_KEY 也绝不触网）
os.environ["FLOWGATE_PROVIDER"] = "fake"

# 行级：输出编码钉死 UTF-8——评审实测（2026-10-06）GBK 控制台 print ✓/✗ 直接
# UnicodeEncodeError 崩、退出码 1，"任何机器可复跑"不成立；errors="replace" 兜底，
# 演示宁可个别字形缺角，绝不因一个字符中途死掉
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 以下项目内 import 必须垫在环境钉子之后（app.main 的装配处在 import 时读环境）
import app.main as gateway  # noqa: E402
import streaming.sse as streaming_sse  # noqa: E402
from providers.fake import FakeProvider  # noqa: E402

# 行级：uvicorn/ASGI 内部日志全静音——演示输出要干净可念；断连/截断的证据在
# fake 账本与 streaming.sse 截断日志（act3 有专用捕获），不靠 uvicorn 的报错堆栈
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.lifespan"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# 演示句：fake 会把它回显成 "fake-reply: 断连不泄漏" 切 4 块——这句话本身就是本周的故事
DEMO_PROMPT = "断连不泄漏"
# 第二幕并发断连路数：与 tests/test_streaming.py 的并发用例同为 16 路——
# ASGI 高度的测试数字与真 socket 高度的 demo 数字互为印证（口径相同，高度不同）
ABORT_CLIENTS = 16
# 放行节奏：200ms/帧——比这更快打字机就看不清了，再慢又拖演示时长
PACE_SECONDS = 0.2
# 第三幕把 gap 预算从 30s 压到 1s（测试同款手法：patch 常量，不给生产加钩子）——
# 30s 的截断演示要等半分钟，面试现场等不起；尺子变短，截断语义不变
GAP_SECONDS = 1.0
# 单次读 socket 的上限：机制坏了就地红掉，不把演示挂死在台上
READ_TIMEOUT = 5.0


def _check(ok: bool, message: str) -> None:
    """自检断言：不过就带着可念的原因红掉——"退出码 0"是三幕全过的承诺，不是默认值。"""
    if not ok:
        raise RuntimeError(message)


class _WireClient:
    """真 socket 上的极简流式 HTTP 客户端：读响应头、逐帧收 SSE、支持中途硬 abort。

    给初学者的解释（asyncio 流与 TCP 字节流，本脚本首次出现）：open_connection 拿到的
    (reader, writer) 就是一条真 TCP 连接的两端；TCP 只保证字节按序到达、不保证
    "一读一帧"——服务端两次 write 的帧可能挤在同一次 read 里到，所以要自己拿缓冲区
    按 "\\n\\n" 拼帧。abort() 走 transport.abort()：直接掐 socket（RST 级），
    不走"读完剩余数据再挥手"的优雅关闭——第二幕"真客户端 abort"的字面含义就在这。

    为什么不用 httpx：三幕要在字节高度演示（帧何时到、abort 是否真断、半截长什么样），
    httpx 会把分帧与连接管理藏进内部；asyncio + socket 都是 stdlib，零新依赖（依赖红线）。
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """接住一条已连上的真 TCP 连接两端，观测状态从零起步（帧缓冲、响应头待读）。"""
        self._reader = reader
        self._writer = writer
        self._buf = b""  # 行级：跨 read 拼帧的缓冲区——TCP 不按帧分包
        self.status_line = ""
        self.headers = ""
        self._headers_read = False  # 行级：响应头惰性读（见 start 的死锁说明）
        self._t0 = 0.0  # 行级：到达时刻的零点（请求发出时刻），start 里赋值

    async def start(self) -> None:
        """发出流式请求（只发不收）。

        为什么不在这收响应头（首跑死锁教训 2026-10-06）：流式响应的 200 头要等
        首块取到手才发（sse_response 的可报错窗口），首块要等闸门放行——若 start
        在这等头、放行又在 start 返回之后，两边互等直接死锁。收头挪进 next_frame
        惰性做，放行就能先于一切等待发生。
        """
        body = json.dumps(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": DEMO_PROMPT}],
                "stream": True,
            }
        ).encode()
        self._t0 = asyncio.get_running_loop().time()  # 行级：打字机时间戳都相对请求发出时刻
        self._writer.write(
            # 行级：HTTP/1.0 请求——uvicorn 对它回裸 SSE 字节 + Connection: close
            # （探测实验 2026-10-06）：响应体不套 Transfer-Encoding: chunked、以连接关闭
            # 收尾，demo 要看见的是 SSE 帧本身不是 HTTP 分帧；断连语义与 1.1 相同
            b"POST /v1/chat/completions HTTP/1.0\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"\r\n"
            + body
        )
        await self._writer.drain()

    async def _read_headers(self) -> None:
        """读到裸空行为止：响应头收完，剩下的字节全是 SSE 帧（惰性调用，见 start）。"""
        while b"\r\n\r\n" not in self._buf:
            # 为什么 wait_for（限用并发原语，用则配注释）：读 socket 不能无限悬——
            # 到点抛 TimeoutError 红掉自检，演示绝不挂死在台上
            chunk = await asyncio.wait_for(self._reader.read(4096), timeout=READ_TIMEOUT)
            if not chunk:
                raise RuntimeError("响应头没读完连接就关了")
            self._buf += chunk
        head, self._buf = self._buf.split(b"\r\n\r\n", 1)
        self.headers = head.decode("latin-1")
        self.status_line = self.headers.splitlines()[0]
        self._headers_read = True

    async def next_frame(self) -> tuple[float, str] | None:
        """收一帧 SSE → (距请求发出的秒数, 帧文本)；EOF 且无整帧则返回 None。"""
        if not self._headers_read:
            await self._read_headers()  # 行级：首个 next_frame 顺手把响应头收掉（惰性，见 start）
        while b"\n\n" not in self._buf:
            # 为什么 wait_for：同 _read_headers——每次读帧也上闹钟，机制坏了一秒级红掉
            chunk = await asyncio.wait_for(self._reader.read(4096), timeout=READ_TIMEOUT)
            if not chunk:
                # 行级：EOF——正常收尾（[DONE] 后关连接）与截断收尾都在这收口；
                # 缓冲区剩的半帧丢弃（服务端截断发生在帧边界，正常到不了这）
                return None
            self._buf += chunk
        frame, self._buf = self._buf.split(b"\n\n", 1)
        now = asyncio.get_running_loop().time()
        return now - self._t0, frame.decode("utf-8")

    def abort(self) -> None:
        """硬 abort：transport.abort() 直接掐 socket——真客户端中途弃流，不是优雅关闭。"""
        self._writer.transport.abort()

    def close(self) -> None:
        """常规收尾（第一/三幕读完整条流之后用）。"""
        self._writer.close()


class _LogCapture(logging.Handler):
    """把 streaming.sse 的截断日志接到 demo 输出——"原因进日志"（03）在演示里看得见。"""

    def __init__(self) -> None:
        """标准 handler 初始化 + 自备行存：收集的正文逐条进 self.lines（供第三幕打印）。"""
        super().__init__()
        self.lines: list[str] = []  # 行级：收下每条日志的正文（不含堆栈，堆栈太长不适合台上）

    def emit(self, record: logging.LogRecord) -> None:
        """handler 标准入口：记下格式化后的消息正文。"""
        self.lines.append(record.getMessage())


async def _pace(gate: asyncio.Event) -> None:
    """每 PACE_SECONDS 放行一轮的节拍器，被取消即停——第二幕 16 路共用的产块节奏。"""
    while True:
        await asyncio.sleep(PACE_SECONDS)
        gate.set()


async def _open_client(port: int) -> _WireClient:
    """真 socket 连上 demo 服务器、发出流式请求——三幕共用的客户端入口。"""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    client = _WireClient(reader, writer)
    await client.start()
    return client


def _frame_label(frame: str) -> str:
    """一帧 SSE → 打字机行：'data: ' + 增量摘要——逐帧差异落在 delta 上，一眼可见。

    为什么打摘要不打整帧：完整帧是 150 字符的 chat.completion.chunk JSON，7 帧长一个样，
    打字机效果反而看不见；摘要保留 "data: " 前缀与 delta 形状（SSE 形态仍可读），
    完整帧形态在 wire 上原样存在（客户端就是按 "data: {json}\\n\\n" 拼帧的）。
    """
    if frame == "data: [DONE]":
        return "data: [DONE]  ← 完成记号（完整流才有）"
    # 复杂语句（下标链）行上：帧=envelope+choices[0]，三种帧（role/content/finish）各说一句
    choice = json.loads(frame.removeprefix("data: "))["choices"][0]
    delta = choice["delta"]
    if delta.get("role"):
        return f'data: {{"delta":{{"role":"{delta["role"]}","content":""}}}}  ← 首块宣告角色'
    if choice.get("finish_reason"):
        return f'data: {{"finish_reason":"{choice["finish_reason"]}"}}  ← 末块结束信号'
    return f'data: {{"delta":{{"content":"{delta["content"]}"}}}}'


def _assemble(frames: list[tuple[float, str]]) -> str:
    """把收到的 "data: {json}" 帧里的 delta.content 拼回全文。

    复杂语句（单层推导式 + 条件过滤）行上：逐块拼回完整回答，就是"每块都是真增量、
    拼起来等于非流式那句话"的字面证明；[DONE] 帧不是 JSON，过滤掉。
    """
    return "".join(
        (json.loads(t.removeprefix("data: "))["choices"][0]["delta"].get("content") or "")
        for _, t in frames
        if t != "data: [DONE]"
    )


async def act1(port: int) -> None:
    """第一幕：逐块 SSE 输出——帧按放行节奏逐块到达（打字机），完整流以 [DONE] 收尾。"""
    gate = asyncio.Event()  # 行级：闸门从关着起步——产块节奏由本幕握手式放行拿住
    fake = FakeProvider(gate=gate)
    # 行级：装进装配点（同测试的 monkeypatch 手法）——路由、streaming/ 全是生产原样，
    # 要控制的只有 fake 的产块节奏（闸门是 fake 自带的测试设施，不是生产钩子）
    gateway.provider = fake

    client = await _open_client(port)
    frames: list[tuple[float, str]] = []
    while True:
        # 握手节奏：收到一帧才放行下一帧——Event 是闩不是信号量，盲 set 两次只会放行
        # 一块（闩已开再 set 无效），握手让"每帧对应一次放行"变成确定事实
        await asyncio.sleep(PACE_SECONDS)
        gate.set()
        got = await client.next_frame()
        if got is None:
            break
        frames.append(got)
        print(f"[t+{got[0]:.2f}s] {_frame_label(got[1])}")
        if got[1] == "data: [DONE]":
            break
    client.close()

    assembled = _assemble(frames)
    _check(bool(frames), "一帧都没收到（连接上没流出任何 SSE 字节）")
    # 复杂语句（推导式）行上：帧间隔打字机断言——最后一帧时刻减首帧时刻，
    # 一口气全到的话这个跨度约等于 0，"逐块"就只是嘴上说说
    span = frames[-1][0] - frames[0][0]
    _check("200" in client.status_line, f"响应头不是 200：{client.status_line}")
    _check("text/event-stream" in client.headers, "响应不是 text/event-stream")
    _check(assembled == f"fake-reply: {DEMO_PROMPT}", f"拼回全文不对：{assembled!r}")
    _check(frames[-1][1] == "data: [DONE]", "完整流没收尾 [DONE]")
    _check(span >= 0.5, f"帧间隔只有 {span:.2f}s——不是逐块到达")
    _check(fake.stream_endings == ["completed"], f"收尾账本不对：{fake.stream_endings}")
    print(
        f"✓ 第一幕通过：{len(frames)} 帧（6 块 chunk + [DONE]）逐块到达"
        f"（跨度 {span:.1f}s，打字机可见），"
        f"拼回全文「{assembled}」，完整流以 [DONE] 收尾，收尾账本={fake.stream_endings}"
    )


async def _abort_one(port: int) -> int:
    """一路"读两帧就硬 abort"的客户端；返回断前收到的帧数（断言用）。"""
    client = await _open_client(port)
    got = 0
    for _ in range(2):
        if await client.next_frame() is not None:
            got += 1
    client.abort()  # 行级：chunk 全流 6 帧只读 2 帧就掐——断点必在"回答半路"，不是没开始/已说完
    return got


async def _wait_endings(fake: FakeProvider, expected: int) -> None:
    """轮询等 fake 收尾账本记满 expected 条（带上限，不挂死演示）。

    为什么真实轮询：断连的收尾是异步落到服务端的（uvicorn 发现 socket 断了才取消
    请求任务）——demo 求真实，这里没有测试里"app 返回即查账"的确定时刻，
    0.05s 一查的成本就是"求真实"的代价。
    """
    deadline = asyncio.get_running_loop().time() + 5.0
    while len(fake.stream_endings) < expected:
        if asyncio.get_running_loop().time() > deadline:
            return  # 行级：到点不等了——后面的检查会报出缺几条，比挂死强
        await asyncio.sleep(0.05)


async def act2(port: int) -> None:
    """第二幕：16 路真 socket 并发、各读到第 2 帧硬 abort——「断连不泄漏」的实测数字。"""
    gate = asyncio.Event()
    fake = FakeProvider(gate=gate)  # 共享闸：一次放行=每路各进一块（语义见 fake._pass_gate）
    gateway.provider = fake

    # 本幕不能用握手节奏：16 路共用一把闸，谁也说不清"下一帧"是哪路的——
    # 定时节拍器放行，每次 set() 唤醒所有在等的流各进一块，16 路大致齐步走
    # 为什么 create_task（限用并发原语，用则配注释）：节拍器必须与 16 路客户端 IO 并发跑——
    # await 内联就变成"先拍完再连客户端"，顺序反了；后台拍、finally 里 cancel 收
    pacer = asyncio.create_task(_pace(gate))
    try:
        # 复杂语句（gather+推导式）行上：16 路并发真连接真 abort，每路读 socket 有
        # READ_TIMEOUT 兜底——坏一路红一路，不把演示挂死
        counts = await asyncio.gather(*(_abort_one(port) for _ in range(ABORT_CLIENTS)))
    finally:
        pacer.cancel()

    # 行级：断连收尾异步落到服务端，真实轮询等账本记满（口径见 _wait_endings）
    await _wait_endings(fake, ABORT_CLIENTS)

    # 复杂语句（推导式）行上：账本里非 "cancelled" 的条目=收尾记错的泄漏（宁冤勿纵的
    # 记账下这不可能是误报）；总数不足 16=有流根本没记账的泄漏——两条一起看才是 0 泄漏
    uncancelled = [e for e in fake.stream_endings if e != "cancelled"]
    cancelled = fake.stream_endings.count("cancelled")
    _check(set(counts) == {2}, f"有客户端没读满 2 帧就断了：{counts}")
    _check(uncancelled == [], f"有上游没被取消：{fake.stream_endings}")
    _check(cancelled == ABORT_CLIENTS, f"收尾账本不足 {ABORT_CLIENTS} 条：{fake.stream_endings}")
    print(f"{ABORT_CLIENTS} 路并发：各读 2 帧（chunk 全流 6 帧的中途）后 transport.abort() 硬断……")
    print(f"fake 收尾账本（{ABORT_CLIENTS} 路）：cancelled={cancelled}，completed=0")
    print(
        "✓ 第二幕通过——「断连不泄漏」实测数字："
        f"并发 {ABORT_CLIENTS} 路真 socket 中途 abort，未被取消的上游 = 0"
    )
    print(
        "  （口径同 issue 02：账本里非 cancelled 的条目=记错的收尾；"
        "总数不足 N=没记账的泄漏——两条同时成立才算 0 泄漏）"
    )


# 拆不动说明（act3 整块超 40 行，同 providers/fake.py chat_stream / streaming/sse.py
# sse_stream 的先例）：压尺子→握手放行两帧→卡住→验证截断是**同一条**截断故事的
# 走读现场——抽子函数会把"卡住 → gap 耗尽 → 截断"的因果链切成两截，演示讲解反而
# 跳来跳去；打印行就是剧情本身，不为行数挪走。故保持单函数。
async def act3(port: int) -> None:
    """第三幕：首两帧照发、此后上游卡住——gap 超时截断，半截回答不带 [DONE]。"""
    gate = asyncio.Event()
    fake = FakeProvider(gate=gate)
    gateway.provider = fake
    old_gap = streaming_sse.GAP_TIMEOUT_SECONDS
    streaming_sse.GAP_TIMEOUT_SECONDS = GAP_SECONDS  # 行级：压短尺子（同测试手法），语义不变
    log = _LogCapture()
    logging.getLogger("streaming.sse").addHandler(log)
    try:
        client = await _open_client(port)
        frames: list[tuple[float, str]] = []
        # 握手放行恰好两帧（role + 首块内容）：每帧都对应一次放行，卡点位置确定
        for _ in range(2):
            gate.set()
            got = await client.next_frame()
            if got is None:
                break
            frames.append(got)
            print(f"[t+{got[0]:.2f}s] {_frame_label(got[1])}")
        # 第 3 帧起不放行：流停在产块闸上=「上游卡住」，gap 预算（1s）耗尽即截断收尾
        t_stall = asyncio.get_running_loop().time()
        while True:
            got = await client.next_frame()
            if got is None:
                break
            frames.append(got)  # 行级：走到这=卡住后还有帧，与"卡住截断"的剧情矛盾，下方红掉
        elapsed = asyncio.get_running_loop().time() - t_stall
        client.close()
    finally:
        streaming_sse.GAP_TIMEOUT_SECONDS = old_gap  # 行级：尺子用完放回去，不污染后续演示/测试
        logging.getLogger("streaming.sse").removeHandler(log)

    # 复杂语句（推导式）行上：截断语义的三条证据——恰好 2 帧、没有 [DONE]、收尾账本非 completed
    _check(len(frames) == 2, f"卡住后不该再有帧，实收 {len(frames)} 帧")
    _check(all(t != "data: [DONE]" for _, t in frames), "截断流伪带了 [DONE]")
    _check(fake.stream_endings == ["cancelled"], f"收尾账本不对：{fake.stream_endings}")
    _check(any("gap 超时" in line for line in log.lines), f"截断死因没进日志：{log.lines}")
    print(
        f"（第 3 帧起闸门不再放行 = 上游卡住；"
        f"gap 预算压到 {GAP_SECONDS:g}s，约 {elapsed:.1f}s 后截断）"
    )
    print(
        f"收到 {len(frames)}/6 帧 chunk，拼到一半「{_assemble(frames)}」就没了，"
        "**没有 [DONE]** ——半截回答，客户端可辨"
    )
    print(f"截断死因（streaming/sse 日志）：{log.lines[-1]}")
    print(f"fake 收尾账本：{fake.stream_endings}（超时掐流也显式收尾，不悬垂）")
    print("✓ 第三幕通过：对照第一幕——完整流有 [DONE]，截断流没有")


async def _start_server() -> tuple[uvicorn.Server, asyncio.Task, int]:
    """起真 uvicorn（真 socket；端口 0 交给 OS 分配，任何机器不撞端口）。"""
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
    # 为什么 create_task：uvicorn 的 serve() 是长循环，必须当后台任务跑，主线才走得动
    # 三幕；退出时 _amain 里 should_exit + await 回收，跑不干净就 cancel（绝不挂死）
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)  # 行级：serve() 的绑定是异步的，等 socket 真正开听
    return server, task, port


async def _amain() -> int:
    """起真 uvicorn，顺序演三幕；任一幕自检不过 → 退出码 1（全过才是 0）。"""
    server, server_task, port = await _start_server()
    print("=" * 60)
    print("FlowGate W2 可复跑 demo：SSE 流式三幕（fake 上游、零外网、真 socket）")
    print(f"[准备] uvicorn 监听 127.0.0.1:{port}（OS 分配端口），上游 = FakeProvider（带闸门账本）")
    print("=" * 60)
    try:
        for name, run in [("第一幕", act1), ("第二幕", act2), ("第三幕", act3)]:
            print(f"\n── {name} ──────────────────────────────────────")
            try:
                await run(port)
            except Exception as exc:
                # 行级：带异常类型名——TimeoutError 这类空 message 异常只有类型说得清死因
                print(f"\n✗ {name} 自检未通过：{type(exc).__name__}: {exc}")
                return 1
    finally:
        server.should_exit = True
        try:
            # 为什么 wait_for（限用并发原语）：给服务器回收也上闹钟——到点进下面的强拆，
            # 演示脚本绝不挂死在收尾上
            await asyncio.wait_for(server_task, timeout=5)
        except TimeoutError:
            server_task.cancel()  # 行级：5s 内退不干净就强拆——演示脚本绝不挂死
    print("\n" + "=" * 60)
    print("三幕全过 ✓  退出码 0。重跑：uv run python -m scripts.demo_w2")
    print(f"面试可念数字：并发 {ABORT_CLIENTS} 路真 socket 中途断连，未被取消的上游 = 0")
    print("=" * 60)
    return 0


def main() -> int:
    """同步入口：asyncio.run 跑三幕，退出码 = 演示自检结果（0=全过）。"""
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
