"""serve 一键启动测试（w4 issue 07）：缝 2 函数直调（自举/分层缺省/端口）+ 缝 4 启动 smoke。

为什么 smoke 走干净解释器子进程：「首启空表自举」与「env 注入先于装配 import」只有
从零起步的解释器才测得到——pytest 进程里 conftest 已 import app.main、单例已按
":memory:" 建好，强行 reload 会毒化全套件共享的 TEST_KEY（本票面约定：conftest
本轮归票 03，不许碰）；与 test_keyctl 同款理由（真入口只有子进程才测得到）。
uvicorn 是真的、跑在 scripts.serve 进程内（demo_w2/w3 的 _start_server 同款形态），
不是 TestClient 桩；cwd 钉 tmp_path 是为了既测"缺省注入 flowgate.db"又零残留。
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from keys.store import KeyStore
from scripts.serve import bootstrap_dev_key, resolve_db_path, resolve_port

# 仓库根：子进程 PYTHONPATH 指到它（cwd 钉 tmp_path 时 `-m scripts.serve` 才找得到包）
REPO_ROOT = Path(__file__).resolve().parents[1]


def _serve_env(tmp_path: Path) -> dict[str, str]:
    """子进程环境底座：UTF-8 钉死、包路径指仓库根、清掉库/端口/上游覆盖（现场确定）。

    为什么 PYTHONPATH=REPO_ROOT：smoke 的 cwd 必须是 tmp_path（测"缺省注入 flowgate.db"
    且零残留），而 `python -m scripts.serve` 靠 cwd/sys.path 找包——两头都要顾。
    为什么清 FLOWGATE_BILLING_DB/FLOWGATE_PORT：宿主机可能设过，现场要从"未设"起步。
    """
    env = dict(os.environ)  # 行级：拷贝一份再改，不污染测试进程自己的环境
    env["PYTHONUTF8"] = "1"  # 行级：子进程标准流钉 UTF-8（test_keyctl 同款口径）
    env["PYTHONPATH"] = str(REPO_ROOT)
    for name in (
        "FLOWGATE_BILLING_DB",
        "FLOWGATE_PORT",
        "FLOWGATE_PROVIDER",
        "FLOWGATE_BUDGET_TOKENS",
    ):
        env.pop(name, None)
    return env


def test_serve_startup_fails_loud_when_upstream_key_missing(tmp_path: Path) -> None:
    """证明：上游配置错在启动时响亮报错（指名缺的 env 变量、退出码 1），不等首个请求。

    怎么证明：干净解释器子进程带 FLOWGATE_PROVIDER=dashscope 但不给 DASHSCOPE_API_KEY
    跑 python -m scripts.serve——断言退出码 1、stderr 点名 DASHSCOPE_API_KEY（_build_one
    纪律）、stdout 没有 listening 行（服务根本没起，"不等首个请求"的字面兑现）。
    """
    env = _serve_env(tmp_path)
    env["FLOWGATE_PROVIDER"] = "dashscope"  # 行级：点名真上游——fake 缺省不会触发配置错
    env.pop("DASHSCOPE_API_KEY", None)  # 行级：现场=缺 key（宿主机可能设过，显式清掉）
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.serve"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",  # 行级：与 PYTHONUTF8 对齐，解码不猜系统码页
        timeout=30,
    )
    assert proc.returncode == 1  # 行级：启动失败=退出码 1（响亮报错的出口形状）
    assert "DASHSCOPE_API_KEY" in proc.stderr  # 行级：报错指名缺的变量（沿 _build_one 纪律）
    assert "listening" not in proc.stdout  # 行级：没监听过——错误发生在首个请求之前


def test_serve_startup_fails_loud_when_upstream_base_url_missing(tmp_path: Path) -> None:
    """证明：缺 base_url 同样启动即响亮报错——票面"缺 key/base_url"的第二种现场（评审补测）。

    怎么证明：kimi 的 base_url 无内置默认（装配处刻意不对称），子进程给 key 不给
    KIMI_BASE_URL 跑 serve，断言退出码 1、stderr 点名 KIMI_BASE_URL。born-green 说明：
    行为由装配处 import 机制先于测试存在，本测试是评审驱动的契约钉，非红绿循环。
    """
    env = _serve_env(tmp_path)
    env["FLOWGATE_PROVIDER"] = "kimi"
    env["KIMI_API_KEY"] = "sk-dummy-for-boot-check"  # 行级：key 有了——红点只剩缺 base_url
    env.pop("KIMI_BASE_URL", None)  # 行级：现场=缺 base_url（宿主机可能设过，显式清掉）
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.serve"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",  # 行级：与 PYTHONUTF8 对齐，解码不猜系统码页
        timeout=30,
    )
    assert proc.returncode == 1  # 行级：启动失败=退出码 1（与缺 key 同一出口形状）
    assert "KIMI_BASE_URL" in proc.stderr  # 行级：报错指名缺的变量（沿 _build_one 纪律）
    assert "listening" not in proc.stdout  # 行级：没监听过——错误发生在首个请求之前


def _start_serve(tmp_path: Path) -> subprocess.Popen[str]:
    """真子进程起 serve（--port 0 = OS 分配）；stdout/stderr 接管给断言，返回在跑的进程。"""
    env = _serve_env(tmp_path)
    env["FLOWGATE_PROVIDER"] = "fake"  # 行级：钉死 fake——smoke 零外网可复跑（测试纪律）
    return subprocess.Popen(
        [sys.executable, "-m", "scripts.serve", "--port", "0"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",  # 行级：与 PYTHONUTF8 对齐，解码不猜系统码页
        # 行级：Windows 下新进程组才收得到 CTRL_BREAK（优雅退出的信号通道）；别处 0=不加旗标
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


def _read_until(proc: subprocess.Popen[str], marker: str, deadline_seconds: float) -> list[str]:
    """等子进程 stdout 出现含 marker 的行，返回已读全部行；到点没有就强拆并红掉。

    为什么读行放 daemon 线程：管道上的 readline 阻塞且无超时——线程一直收到 EOF
    （管道不淤积），主线按截止时间轮询结果，子进程挂死时测试红掉而不是跟着挂死。
    """
    lines: list[str] = []

    def _pump() -> None:
        """收行线程体：把子进程 stdout 逐行攒进 lines，收到 EOF（子进程退出）为止。"""
        for line in proc.stdout:  # 行级：uvicorn 日志也顺带清走——管道永不淤积
            lines.append(line)

    threading.Thread(target=_pump, daemon=True).start()
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:  # 行级：轮询就绪标记——到点强拆红掉，绝不挂死
        if any(marker in line for line in lines):
            return lines
        time.sleep(0.05)
    _force_stop(proc)
    raise AssertionError(
        f"{deadline_seconds}s 内未见就绪标记 {marker!r}\n"
        f"stdout: {''.join(lines)}\nstderr: {proc.stderr.read()}"
    )


def _parse_created(lines: list[str]) -> tuple[str, str]:
    """从启动输出抽 (key_id, credential)——解析形状即 keyctl 同款"两行机读输出"契约。"""
    fields = {}  # 行级：key_id:/credential: 前缀行（与 scripts/serve.py 的打印契约对齐）
    for line in lines:
        if line.startswith("key_id:"):
            fields["key_id"] = line.removeprefix("key_id:").strip()
        elif line.startswith("credential:"):
            fields["credential"] = line.removeprefix("credential:").strip()
    return fields["key_id"], fields["credential"]


def _parse_port(lines: list[str]) -> int:
    """从监听行抽实际端口——--port 0 时 OS 分配的端口只有启动输出说得出。"""
    for line in lines:
        if "FlowGate listening on http://127.0.0.1:" in line:
            return int(line.rsplit(":", 1)[1].strip())  # 行级：端口在最后一个冒号后
    raise AssertionError(f"启动输出里没有监听行：{''.join(lines)}")


def _graceful_stop(proc: subprocess.Popen[str]) -> None:
    """发优雅退出信号（Windows CTRL_BREAK / 别处 SIGINT）并等子进程收场（上限 10s）。"""
    if sys.platform == "win32":
        # 行级：CTRL_BREAK 映射到子进程 SIGBREAK——uvicorn 0.54 的 HANDLED_SIGNALS 已含它
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)  # 行级：超时抛 TimeoutExpired 即红——"干净退出"不许挂死


def _force_stop(proc: subprocess.Popen[str]) -> None:
    """失败收场：强杀子进程——断言红也绝不留孤儿进程（收尾清理纪律）。"""
    proc.kill()
    proc.wait(timeout=5)


def _assert_boot_output(lines: list[str]) -> tuple[str, str]:
    """断言首启打印契约（smoke 清单①）并返回 (key_id, credential)。

    打印契约=key_id 行 + credential 行 + 可粘贴 curl（含真凭据、/v1 路径）——票面
    "打印 key_id + 凭据串 + 可粘贴 curl" 三件缺一即红。
    """
    output = "".join(lines)
    key_id, credential = _parse_created(lines)
    assert key_id.startswith("key_")
    assert credential.startswith("fgk_")
    assert f"Authorization: Bearer {credential}" in output  # 行级：curl 可粘贴=带真凭据
    assert "/v1/chat/completions" in output
    return key_id, credential


def _assert_default_db(tmp_path: Path, key_id: str, credential: str) -> None:
    """断言 DB 缺省生效（smoke 清单②）：flowgate.db 落在 cwd，自举 key 住在里面。"""
    db = tmp_path / "flowgate.db"
    assert db.exists()  # 行级：注入的相对路径落 cwd——缺省注入生效的字面证据
    assert KeyStore(str(db)).verify(credential) == key_id  # 行级：跨进程读同一文件库验 key


def _assert_http(base: str, credential: str) -> None:
    """断言真 HTTP（smoke 清单③）：/health 免认证 200、匿名 401、带自举 key 的请求 200。"""
    assert httpx.get(f"{base}/health", timeout=5).status_code == 200
    payload = {"model": "fake-model", "messages": [{"role": "user", "content": "你好"}]}
    url = f"{base}/v1/chat/completions"
    assert httpx.post(url, json=payload, timeout=10).status_code == 401  # 行级：无 key=拒
    # 行级：带自举凭据敲门——200 证明那把打印出来的 key 真的开通了服务
    ok = httpx.post(
        url, json=payload, headers={"Authorization": f"Bearer {credential}"}, timeout=10
    )
    assert ok.status_code == 200


def test_serve_smoke_first_boot_prints_dev_key_and_serves(tmp_path: Path) -> None:
    """证明：一条命令真跑起来（缝 4 全清单）——首启打印 dev key 与可粘贴 curl、DB 缺省
    落盘生效、/health 200、带自举 key 的请求 200（匿名 401 验 fail-closed）、干净退出。

    怎么证明：干净解释器子进程跑 python -m scripts.serve --port 0（cwd=tmp_path、未设
    FLOWGATE_BILLING_DB）；按①②③三段助手逐项断言（打印/缺省库/HTTP），最后发优雅
    退出信号、等退出码 0（清单④）。拆段说明：三段现场互不复用、拆开各自 ≤40 行。
    """
    proc = _start_serve(tmp_path)
    try:
        lines = _read_until(proc, "FlowGate listening on ", deadline_seconds=30)
        key_id, credential = _assert_boot_output(lines)
        _assert_default_db(tmp_path, key_id, credential)
        _assert_http(f"http://127.0.0.1:{_parse_port(lines)}", credential)
        # 干净退出：优雅信号 → 进程收场且退出码 0
        _graceful_stop(proc)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            _force_stop(proc)  # 行级：断言红掉也不留孤儿——收尾清理纪律


def test_resolve_db_path_injects_default_when_unset(monkeypatch) -> None:
    """证明：FLOWGATE_BILLING_DB 未设 → 返回 flowgate.db 并写回 env（注入语义）。

    怎么证明：先清掉环境变量（宿主机可能设过），调用后断言返回值与 env 里读回来的
    值都是票面钉的缺省字面量——"注入"必须对后续读 env 的装配处可见，只改返回值不算。
    """
    monkeypatch.delenv("FLOWGATE_BILLING_DB", raising=False)
    # 行级：返回值=缺省字面量——独立真源是票面"未设 → flowgate.db"，不是抄实现常量
    assert resolve_db_path() == "flowgate.db"
    # 行级：env 也已写入——装配处（create_ledger/create_keystore）读的是这层
    assert os.environ["FLOWGATE_BILLING_DB"] == "flowgate.db"


def test_resolve_db_path_respects_explicit_memory(monkeypatch) -> None:
    """证明：显式 ":memory:" 原样尊重——分层缺省只补"没说"，不覆盖"说了"（票面）。

    怎么证明：显式设 ":memory:" 后调用，返回值与 env 都必须一字不动；测试零残留的
    通道全靠这条（测试显式传 :memory: 时 serve 不许偷偷换落盘文件）。
    """
    monkeypatch.setenv("FLOWGATE_BILLING_DB", ":memory:")
    assert resolve_db_path() == ":memory:"
    assert os.environ["FLOWGATE_BILLING_DB"] == ":memory:"  # 行级：env 一字未动


def test_resolve_db_path_injects_default_when_empty(monkeypatch) -> None:
    """证明：空串按未设处理 → 仍注入缺省（堵 sqlite3.connect("") 的静默黑洞）。

    怎么证明：显式设空串后调用，断言返回缺省且 env 被改写——.env 里留空值
    （FLOWGATE_BILLING_DB=）不该得到一座"签进去就消失"的进程私有临时库。
    """
    monkeypatch.setenv("FLOWGATE_BILLING_DB", "")
    assert resolve_db_path() == "flowgate.db"
    assert os.environ["FLOWGATE_BILLING_DB"] == "flowgate.db"


def test_bootstrap_dev_key_mints_dev_key_when_store_empty(tmp_path) -> None:
    """证明：空表首启自举一把 dev key（name=dev、scope=*），且它是把立即可用的常规 key。

    怎么证明：全新 KeyStore（tmp 文件库）直调 bootstrap_dev_key（缝 2 函数直调，
    走 keys 公开 API 不碰 SQL）；断言返回 {key_id, credential} 形态、store.verify
    立即认它、list 里恰好一行且标签/scope/状态是票面钉的口径。
    """
    store = KeyStore(str(tmp_path / "keys.db"))  # 行级：全新文件库=空表首启的现场
    created = bootstrap_dev_key(store)
    assert created is not None  # 行级：空表必须自举——返回 None 即票面"首启打印 key"没了来源
    key_id, credential = created["key_id"], created["credential"]
    assert key_id.startswith("key_")  # 行级：公开标识形态（与 store 同款前缀约定）
    assert credential.startswith("fgk_")  # 行级：凭据串形态（store 同款前缀约定）
    # 行级：自举出的 key 立即通过公开校验面——它不是哑数据，是把能用的常规 key
    assert store.verify(credential) == key_id
    # 行级：list 五字段断言标签与 scope（票面 name=dev、scope=*）——推导式行上注释
    assert [(row["name"], row["scope"], row["status"]) for row in store.list()] == [
        ("dev", "*", "active")
    ]


def test_bootstrap_dev_key_skips_when_store_not_empty(tmp_path) -> None:
    """证明：库非空永不自举（story 23）——已有 key 时返回 None，表内一行不多。

    怎么证明：先用公开 create() 铸一把常规 key，再调 bootstrap_dev_key 断言返回
    None，且 list 仍是原来那把——"以后每次启动都悄悄发新凭据"在这个现场必红。
    """
    store = KeyStore(str(tmp_path / "keys.db"))
    existing = store.create(name="ops", scope="*")  # 行级：先造"非空"现场
    assert bootstrap_dev_key(store) is None  # 行级：非空=不自举，返回 None 即"没发新凭据"
    # 行级：仍只有原来那把——键集一字不多（推导式行上注释）
    assert [row["key_id"] for row in store.list()] == [existing["key_id"]]


def test_bootstrap_dev_key_skips_when_only_revoked_rows(tmp_path) -> None:
    """证明："空"指表里没有行——只剩已撤销行也算非空，不借"清一色撤销"悄悄发凭据。

    怎么证明：铸一把再撤销（软删行仍在），调 bootstrap_dev_key 断言返回 None——
    判空若改查"未撤销行数"（WHERE revoked_at IS NULL），这个现场会假绿出新凭据。
    """
    store = KeyStore(str(tmp_path / "keys.db"))
    minted = store.create(name="old", scope="*")
    store.revoke(minted["key_id"])  # 行级：软删后行仍在（审计痕迹）——正是要钉的现场
    assert bootstrap_dev_key(store) is None  # 行级：只剩撤销行 ≠ 首启，不自举
    assert len(store.list()) == 1  # 行级：撤销行还在——现场确实是"有行但全撤销"


def test_resolve_port_defaults_to_8000_when_no_flag_no_env(monkeypatch) -> None:
    """证明：缺省端口 8000（票面：curl 示例好记）——没旗标没 env 时的落点。

    怎么证明：清掉 FLOWGATE_PORT、旗标传 None（argparse 未给 --port 的形态），
    断言回 8000 这个票面字面量。
    """
    monkeypatch.delenv("FLOWGATE_PORT", raising=False)
    assert resolve_port(None) == 8000


def test_resolve_port_env_overrides_default(monkeypatch) -> None:
    """证明：env 覆盖口成立——FLOWGATE_PORT 取代 8000（票面"与 env 覆盖"）。

    怎么证明：设 FLOWGATE_PORT=9000、旗标不给，断言回 9000——只认死 8000 的实现在此必红。
    """
    monkeypatch.setenv("FLOWGATE_PORT", "9000")
    assert resolve_port(None) == 9000


def test_resolve_port_flag_beats_env(monkeypatch) -> None:
    """证明：--port 旗标压过 env——当场显式意图赢过环境预设（优先级口径）。

    怎么证明：env=9000 而旗标=7000，断言回 7000；实现若让 env 先读必红。
    """
    monkeypatch.setenv("FLOWGATE_PORT", "9000")
    assert resolve_port(7000) == 7000


def test_resolve_port_zero_passes_through_for_os_assignment(monkeypatch) -> None:
    """证明：--port 0 原样通过（OS 分配端口）——0 是合法值，不被当"未给"回退 8000。

    怎么证明：旗标 0、env 清空，断言回 0；实现若写 `if flag:`（把 0 当假）必红。
    """
    monkeypatch.delenv("FLOWGATE_PORT", raising=False)
    assert resolve_port(0) == 0


def test_resolve_port_rejects_invalid_env(monkeypatch) -> None:
    """证明：env 非整数响亮报错且指名变量（_build_one"配错喊响"纪律）——不静默回退 8000。

    怎么证明：FLOWGATE_PORT="eighty"，断言 RuntimeError 且消息点名该变量；静默回退的
    实现在此必红（不抛异常直接穿透 pytest.raises）。
    """
    monkeypatch.setenv("FLOWGATE_PORT", "eighty")
    with pytest.raises(RuntimeError, match="FLOWGATE_PORT"):  # 行级：报错须指名 env 变量
        resolve_port(None)
