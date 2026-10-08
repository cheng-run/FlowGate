"""serve：一条命令把 FlowGate 网关跑起来（w4 issue 07）——DB 分层缺省 + 首启自举 dev key。

为什么 DB 分层缺省（本票核心口径）：同一个 FLOWGATE_BILLING_DB 服务两种场景——测试
要零残留（create_ledger/create_keystore 的 ":memory:" 缺省一字不动，进程内临时库跑完
无痕），生产要落盘（重启后 key 与账还在）。缺省住在**入口**而不是库构造函数：serve
作为生产入口，env 未设时注入 flowgate.db——"生产必须落盘"由入口保证、"测试零残留"
由库缺省保证，各管各。keyctl 的对应口径是**报错**+指向 .env.example（管理面不猜
路径，票 04）——注入缺省是 serve 的专属动作，两个入口刻意不对称。
空串按未设处理：sqlite3.connect("") 会建"进程私有临时库"，key 签进去转眼就消失，
这种静默黑洞比报错危险（keyctl 同款口径）。

为什么自举是安全的：仅**空表首启**一次——库非空永不自举（连撤销过的行都算非空：
表里有行 = 这库有人管，不能悄悄发新凭据）；自举出的 dev key 与手签的 key 同权同形
（name=dev、scope=*），keyctl revoke 可即时作废——它不是后门，是一把写在启动输出里、
可撤销的常规 key。真实部署应 keyctl create 建专属 key 后 revoke dev key。
"""

import argparse
import asyncio
import os
import signal
import socket
import sys

import uvicorn
from fastapi import FastAPI

from keys.store import KeyStore

# 缺省库文件名：相对路径——落在启动时的 cwd（人从仓库根跑 → 仓库根 flowgate.db，
# 已由 .gitignore 的 *.db 兜住；测试子进程 cwd 钉 tmp_path → 零残留）
DEFAULT_DB_PATH = "flowgate.db"
# 自举 dev key 的固定形态（票面钉）：人读标签 dev、scope 通配——一把开箱即用的常规 key
BOOTSTRAP_NAME = "dev"
BOOTSTRAP_SCOPE = "*"
# 缺省端口（票面钉）：8000 好记，可粘贴 curl 不用改
DEFAULT_PORT = 8000


def resolve_db_path() -> str:
    """读 FLOWGATE_BILLING_DB，未设/空串时注入缺省 flowgate.db——分层缺省的入口半边。

    为什么注入要写回 os.environ：装配处（assembly 的 create_ledger/create_keystore）
    在 import 时读 env 建库，本函数必须跑在 import app 之前、且写进 env 它们才看得见；
    显式值（含 ":memory:"）原样尊重、一字不动——缺省只补"没说"，不覆盖"说了"。
    """
    db_path = os.environ.get("FLOWGATE_BILLING_DB", "")
    if db_path:
        return db_path  # 行级：显式设（含 :memory:）尊重原值——测试零残留的通道就靠它
    os.environ["FLOWGATE_BILLING_DB"] = DEFAULT_DB_PATH  # 行级：注入必须在 import app 之前
    return DEFAULT_DB_PATH


def resolve_port(flag: int | None) -> int:
    """端口三档缺省：--port 旗标 > FLOWGATE_PORT env > 8000；0 合法 = OS 分配。

    为什么旗标压过 env：命令行是当场显式意图，env 是环境预设——同 keyctl/装配处
    "配错喊响"纪律，env 非整数响亮报错、不静默回退。
    """
    if flag is not None:
        return flag  # 行级：显式旗标（含 0=OS 分配）最高优先——0 是合法值不是"未给"
    raw = os.environ.get("FLOWGATE_PORT", "")  # 行级：env 档——空串按未设处理（同 DB 口径）
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)  # 行级：env 显式给了就取代缺省——部署口径不用改命令行
    except ValueError as exc:
        # 行级：非整数=配置错，指名 env 变量响亮报错（_build_one"配错喊响"纪律）
        raise RuntimeError(f"FLOWGATE_PORT 配置非法：{raw!r}（应为整数）") from exc


def bootstrap_dev_key(store: KeyStore) -> dict[str, str] | None:
    """key 表空 → 铸一把 dev key 并返回 {key_id, credential}；非空 → None 永不自举。

    为什么判空走 store.list() 而不是 SQL COUNT：自举走 keys 的公开 API（票面红线
    "不直接写 SQL"）——空的定义=表里没有行，list() 含已撤销项，所以"只剩撤销行"
    也算非空（这库有人管过，不能悄悄发新凭据，安全性见模块 docstring）。
    """
    rows = store.list()  # 行级：公开审计面读全量（含已撤销）——一行都没有才是"首启"
    if rows:
        return None  # 行级：库非空永不自举（story 23）——每次启动都发凭据=变相开后门
    return store.create(name=BOOTSTRAP_NAME, scope=BOOTSTRAP_SCOPE)


def _print_boot(created: dict[str, str], port: int) -> None:
    """首启自举的控制台输出：key_id + 凭据串 + 可粘贴 curl + 一句安全边界。

    为什么凭据串在同一次输出里出现两处（凭据行 + curl 的 Bearer）：keyctl 的"明文
    恰一次、用法行占位符"纪律防的是**多次展示**的泄漏面；这里是签发时的同一次输出，
    而票面钉了"可粘贴 curl"——占位符就没法粘贴。展示面仍只有这一次，之后物理上
    取不回（库里只有 hash，见 keys/store.py）。
    """
    credential = created["credential"]
    # 机读行沿 keyctl 输出契约（key_id:/credential: 前缀），smoke 与人读同一份解析
    block = "\n".join(
        [
            f"key_id: {created['key_id']}",
            f"credential: {credential}",
            "已自举 dev key（仅空库首启一次；真实部署请 keyctl create 建专属 key 后 revoke 它）",
            "可粘贴 curl（缺省 fake 上游，零外网）：",
            f"curl http://127.0.0.1:{port}/v1/chat/completions \\",
            f'  -H "Authorization: Bearer {credential}" \\',
            '  -H "Content-Type: application/json" \\',
            """  -d '{"model": "fake-model", "messages": [{"role": "user", "content": "你好"}]}'""",
        ]
    )
    print(block, flush=True)  # 行级：flush——stdout 是管道时（smoke）也要即刻可见


def _load_gateway() -> tuple[FastAPI, KeyStore]:
    """导入 app.main 装出网关——启动自检的集中点（为什么 import 即自检见下）。

    为什么自检=一次 import：装配处（assembly）在 import 时就把上游/账本/key 库装成
    单例，_build_one 的配错（缺 key/base_url、未知上游）当场 RuntimeError——serve 把
    这次 import 摆在起服务之前，配错死在"还没有任何请求可等"的时刻（票面口径）。
    为什么 import 放函数内：必须晚于 resolve_db_path 的 env 注入——放模块顶层会抢在
    注入前跑，DB 分层缺省就失效了（这是顺序敏感点，别"整理"回顶层）。
    """
    from app.main import app, keystore

    return app, keystore


def _bind(port: int) -> tuple[socket.socket, int]:
    """先绑端口再干别的（demo_w2._start_server 先例）：返回 (socket, 实际端口)。

    为什么先绑后传：port=0 时实际端口在 bind 后才知道，可粘贴 curl 与监听行都要它；
    绑定失败（端口占用）发生在铸 key 之前——起不来的服务不该往库里留凭据。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))  # 行级：只听本机——对外暴露是部署层的事，缺省不替人开门
    sock.listen(128)
    return sock, sock.getsockname()[1]


async def _serve(app: FastAPI, sock: socket.socket, port: int) -> None:
    """真 uvicorn 跑在本进程内（demo_w2/w3 的 _start_server 同款形态），跑到信号退出。

    给初学者的解释（asyncio.run + 长循环任务，本文件首次）：uvicorn 的 serve() 是个
    永不返回的 async 函数——create_task 把它当后台任务点火，主线才能在"真的开听了"
    之后打印监听行；Ctrl+C / Ctrl+Break 由 uvicorn 的信号处理器转成 should_exit，
    长循环自己收场，await task 等到那一刻才返回（demo 的收尾纪律：绝不挂死）。
    """
    server = uvicorn.Server(uvicorn.Config(app=app, host="127.0.0.1", port=port))
    task = asyncio.create_task(server.serve(sockets=[sock]))  # 行级：长循环当后台任务点火
    while not server.started:
        await asyncio.sleep(0.01)  # 行级：serve() 的监听是异步完成的，等 socket 真正开听
    print(f"FlowGate listening on http://127.0.0.1:{port}", flush=True)  # 行级：smoke 的就绪信号
    await task  # 行级：跑到 should_exit（信号触发）才返回


def _quiet_sigbreak_rethrow() -> None:
    """Windows 下把 SIGBREAK 的"收场重掷"降级为无事发生——优雅退出=退出码 0 的前提。

    为什么需要：uvicorn 0.54 优雅收场后会还原旧处理器并**重掷**触发的信号（capture_signals
    尾部，让 CLI 像"被信号打死"）——SIGBREAK（Ctrl+Break）的默认动作是杀进程（2026-10-08
    实测退出码 3），"干净退出"就成假红。先把自己的 no-op 装成旧处理器，重掷落在 no-op 上，
    main 正常返回 0。Ctrl+C（SIGINT）不同：重掷走 KeyboardInterrupt，main 已收成 0，不用动。
    """
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, lambda *_: None)  # 行级：no-op 顶替默认"杀进程"动作


def build_parser() -> argparse.ArgumentParser:
    """命令面恰一个旗标 --port（含 0=OS 分配）；其余口径全走 env（.env.example 有入口）。

    为什么不再加 --host/--db：票面只钉了端口旗标；库路径已有 FLOWGATE_BILLING_DB、
    host 缺省 127.0.0.1（见 _bind），多一个旗标多一份口径分叉（Speculative Generality）。
    """
    parser = argparse.ArgumentParser(
        prog="serve", description="一条命令把 FlowGate 网关跑起来（缺省端口 8000）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,  # 行级：None=未给旗标——与 0（OS 分配）严格区分，三档解析在 resolve_port
        help="监听端口（0=OS 分配）；缺省 FLOWGATE_PORT 或 8000",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """一键入口：注入 DB 缺省 → 启动自检 → 绑端口 → 自举打印 → 真 uvicorn 跑到信号退出。

    顺序有讲究（行上逐条标注）：env 注入先于 import（装配处 import 时读 env）、
    绑端口先于铸 key（起不来的服务不往库里留凭据）、铸 key 先于起服务（curl 要端口）。
    """
    args = build_parser().parse_args(argv)
    try:
        port_hint = resolve_port(args.port)  # 行级：--port / FLOWGATE_PORT / 8000 三档定档
        resolve_db_path()  # 行级：注入必须先于 _load_gateway 的 import（装配处读 env 建库）
        app, keystore = _load_gateway()  # 行级：import 即启动自检——配错在这响亮，不等请求
        sock, port = _bind(port_hint)  # 行级：先绑后传——绑定失败就别铸 key（见 _bind docstring）
        created = bootstrap_dev_key(keystore)
        if created is not None:
            _print_boot(created, port)  # 行级：仅首启空库打印——非空库连自举都没有
        _quiet_sigbreak_rethrow()  # 行级：先装 no-op——uvicorn 收场重掷 SIGBREAK 时不背刺退出码
        asyncio.run(_serve(app, sock, port))
        return 0  # 行级：信号触发的优雅退出落这——退出码 0 是"干净退出"的字面兑现
    except RuntimeError as exc:
        # 行级：配置错（_build_one/resolve_port）统一收口——单行响亮报错、退出码 1
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0  # 行级：uvicorn 信号处理器接住 Ctrl+C 之外的打断窗口——CLI 退也要干净


if __name__ == "__main__":
    raise SystemExit(main())
