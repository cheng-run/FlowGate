"""keyctl 子进程 smoke（w4 issue 04）：CLI 管理面的外部行为。

受测面 = CLI 的打印输出与退出码（spec 测试决定：keyctl 是薄壳，库语义已在
tests/test_keys_store.py 钉住，这里只证明"子命令接线 + 跨进程共享同一文件库"）。
为什么全走子进程：票面点名"子进程 smoke"——`python -m scripts.keyctl` 的真入口
（argparse 分派、env 读取、退出码）只有子进程才测得到，直接 import main() 测不到
-m 接线本身。
撤销后的"认证层拒绝"按票面口径降级为 store 层 verify 回 None（02 并行在建，与
HTTP 401 的互证留待合入后集成验收）——子进程写库、测试进程读库，顺带把"跨进程
共享同一文件库"也钉住了。
"""

import os
import subprocess
import sys
from pathlib import Path

from keys.store import KeyStore

# 仓库根：subprocess 以它为 cwd，`-m scripts.keyctl` 才找得到 scripts 包（demo 同款口径）
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(db_path: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    """真子进程跑 keyctl；db_path=None = 模拟"未设 FLOWGATE_BILLING_DB"的宿主机。

    为什么 PYTHONUTF8=1：Windows 管道下子进程默认按系统码页编中文输出，测试断言
    会乱码——钉死 UTF-8 让输出确定（用户终端自用系统编码，不受测试影响）。
    """
    env = dict(os.environ)  # 行级：拷贝一份再改，不污染测试进程自己的环境
    env["PYTHONUTF8"] = "1"  # 行级：子进程标准流钉 UTF-8（为什么见上面 docstring）
    if db_path is None:
        env.pop("FLOWGATE_BILLING_DB", None)  # 行级：显式清掉——宿主机可能设过
    else:
        env["FLOWGATE_BILLING_DB"] = str(db_path)  # 行级：库路径走 env（票面口径）
    # 行级：cwd=仓库根——`-m scripts.*` 靠 cwd 上 sys.path 找到 scripts 包
    return subprocess.run(
        [sys.executable, "-m", "scripts.keyctl", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        encoding="utf-8",  # 行级：与 PYTHONUTF8 对齐，解码不猜系统码页
        timeout=30,
    )


def _parse_created(stdout: str) -> tuple[str, str]:
    """从 create 输出抽 (key_id, credential)——解析形状即"两行机读输出"的契约本身。"""
    fields = {}  # 行级：断言与实现共用的输出契约——key_id 行、credential 行（前缀=字段名）
    for line in stdout.splitlines():
        if line.startswith("key_id:"):
            fields["key_id"] = line.removeprefix("key_id:").strip()
        elif line.startswith("credential:"):
            fields["credential"] = line.removeprefix("credential:").strip()
    return fields["key_id"], fields["credential"]


def _assert_create_prints_once(stdout: str) -> tuple[str, str]:
    """断言 create 输出契约并返回 (key_id, credential)——拆给 smoke，守函数 ≤40 行纪律。"""
    key_id, credential = _parse_created(stdout)
    assert key_id.startswith("key_")  # 行级：公开标识形态（与 store 同前缀约定）
    assert credential.startswith("fgk_")  # 行级：凭据串形态
    # 行级：用法行教"Bearer 怎么带"（票面 create 条目）
    assert any("Bearer" in line for line in stdout.splitlines())
    # 行级：明文在本次输出恰出现一次——story 11"打印一次"的字面钉子（用法行带占位符）
    assert stdout.count(credential) == 1
    return key_id, credential


def test_smoke_create_list_revoke_then_store_rejects(tmp_path: Path) -> None:
    """证明：CLI 全链可用——create 打印一次凭据、list 可见不回显、revoke 后跨进程即拒。

    怎么证明：子进程 create（自定义 name/scope）断言输出契约；list 断言五字段可见、
    凭据串不回显；revoke 后在测试进程直开同一文件 KeyStore.verify 回 None——票面
    "认证层拒绝"的降级口径（401 互证留待 02 合入），也证明 CLI 与服务跨进程同库。
    """
    db = tmp_path / "keys.db"
    created = _run_cli(db, "create", "--name", "demo", "--scope", "model-a,model-b")
    assert created.returncode == 0, created.stderr
    key_id, credential = _assert_create_prints_once(created.stdout)

    listed = _run_cli(db, "list")
    assert listed.returncode == 0, listed.stderr
    # 行级：审计五字段齐（列头=store.list 的字段名）；自定义 name/scope 原样可见
    for field in ("key_id", "name", "scope", "created_at", "status"):
        assert field in listed.stdout
    assert key_id in listed.stdout
    assert "demo" in listed.stdout
    assert "model-a,model-b" in listed.stdout  # 行级：逗号 scope 原样透传（票面"逗号或 *"）
    assert "active" in listed.stdout
    assert credential not in listed.stdout  # 行级：list 绝不回显凭据（story 13）

    revoked = _run_cli(db, "revoke", key_id)
    assert revoked.returncode == 0, revoked.stderr

    # 行级：换进程视角直开同一文件——撤销即时生效 + 软删行保留（跨进程共享的证明）
    store = KeyStore(str(db))
    # 降级口径：store 层"查无"即认证拒绝（与 02 的 HTTP 401 互证合入后集成验收）
    assert store.verify(credential) is None
    rows = store.list()
    assert [row["key_id"] for row in rows] == [key_id]  # 行级：软删不删行（审计痕迹）
    assert rows[0]["status"] == "revoked"

    # 行级：撤销后 list 仍可见（票面"已撤销可见"），且同样不回显凭据
    listed_after = _run_cli(db, "list")
    assert listed_after.returncode == 0
    assert key_id in listed_after.stdout
    assert "revoked" in listed_after.stdout
    assert credential not in listed_after.stdout


def test_revoke_unknown_id_fails_loudly(tmp_path: Path) -> None:
    """证明：对不存在的 key_id 响亮报错（退出码 1 + stderr 点名）——管理员面不防枚举。

    怎么证明：空库上 revoke 一个捏造 id，断言退出码 1、stderr 同时含该 id 与"查无"——
    静默成功是反例（管理员以为撤掉了）；报错不点名 id 也是反例（排错还得翻别的输出）。
    为什么可点名：CLI 是管理员面（票面：不必防枚举），与 verify 统一回"查无"的
    防枚举口径分工不同，因为调用方不同。
    """
    result = _run_cli(tmp_path / "keys.db", "revoke", "key_nope")

    assert result.returncode == 1
    assert "key_nope" in result.stderr  # 行级：报错点名查无的 id（响亮=可定位）
    assert "查无" in result.stderr


def test_missing_db_env_points_to_env_example() -> None:
    """证明：FLOWGATE_BILLING_DB 未设时不猜路径，报错提示指向 .env.example（票面口径）。

    怎么证明：子进程不带 FLOWGATE_BILLING_DB 跑 list，断言退出码 1、stderr 提到
    ".env.example"——不猜是因为猜错会把 key 签进没人读得到的库（sqlite 连空串=私有
    临时库的静默黑洞）；提示指向 .env.example 是票面钉的指引落点。
    """
    result = _run_cli(None, "list")

    assert result.returncode == 1
    assert ".env.example" in result.stderr  # 行级：票面钉的指引落点


def test_revoke_twice_succeeds(tmp_path: Path) -> None:
    """证明：重复撤销回成功、不误报查无——store"行在即 True"幂等契约的 CLI 兑现。

    怎么证明：create 后对同一 key_id 连撤两次，断言两次退出码都是 0——第二次若报
    "查无"，就是把"已撤销过"错当"不存在"（01 契约备忘：revoke 幂等回 True，编排
    已认可，票 04 直接采用）。born-green 契约钉：行为来自 store 层既定契约，
    非本票红绿新增（同 01 评审补测先例）。
    """
    db = tmp_path / "keys.db"
    created = _run_cli(db, "create")
    key_id, _ = _parse_created(created.stdout)

    first = _run_cli(db, "revoke", key_id)
    second = _run_cli(db, "revoke", key_id)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr  # 行级：幂等成功，不当查无报错


def test_create_without_flags_defaults_to_star_scope(tmp_path: Path) -> None:
    """证明：裸 create = name 缺省 default、scope 缺省 *（全模型）——省事缺省口径。

    怎么证明：不带参数 create 后 list 输出含 "default" 与 "*"——scope 缺省通配是
    票面"--scope 逗号或 *"的省事半边（免手敲），name 缺省保证审计行总有可读标签。
    born-green 契约钉：缺省值随 argparse 解析面在首切片即定型。
    """
    db = tmp_path / "keys.db"
    created = _run_cli(db, "create")

    assert created.returncode == 0, created.stderr
    listed = _run_cli(db, "list")
    assert "default" in listed.stdout  # 行级：--name 缺省值
    assert "*" in listed.stdout  # 行级：--scope 缺省=通配（全部 model）
