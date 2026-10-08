"""keyctl：虚拟 key 的 CLI 管理面（w4 issue 04）——create / list / revoke 三子命令。

为什么是 CLI 不是 HTTP 管理端点（spec 决定，ADR 级否决）：管理端点用什么 key 认证是
鸡生蛋问题（得为"能管 key 的 key"再造一层管理面），砍产品面红线也不开工单管理 UI；
管理员在服务器本机跑脚本，安全边界就是 shell 权限。
为什么 argparse 直写、不引 click/typer（依赖红线）：三个子命令的解析量 stdlib 完全够，
引 click 是为十行解析背一整个第三方运行时。
为什么凭据串只打印一次（story 11）：明文此后物理上取不回（库里只有 hash，见
keys/store.py 模块 docstring）——连输出里也只出现一次，用法行用占位符，杜绝"第二次
展示"这个泄漏面。
库路径口径：与 serve/create_ledger 同读 FLOWGATE_BILLING_DB（跨进程共享同一文件库）；
未设时的提示指向 .env.example（票面钉的指引落点）。
"""

import argparse
import os
import sys

from keys.store import KeyStore


def _db_path() -> str:
    """读 FLOWGATE_BILLING_DB——key 库路径与 serve/create_ledger 同一 env 口径。

    未设/空串抛 RuntimeError：管理面不猜路径，提示指向 .env.example（票面口径）。
    空串也算未设：sqlite3.connect("") 会建"进程私有临时库"，key 签进去转眼就消失——
    这种静默黑洞比报错危险得多。
    """
    db_path = os.environ.get("FLOWGATE_BILLING_DB", "")
    if db_path:
        return db_path
    raise RuntimeError(
        "未设置 FLOWGATE_BILLING_DB（key 库的 SQLite 文件路径）。"
        "请先 cp .env.example .env、在其中设置该变量，"
        "再用 uv run --env-file .env python -m scripts.keyctl … 运行"
    )


def _cmd_create(args: argparse.Namespace) -> int:
    """create：签发一把新 key，打印 key_id + 凭据串 + 一行 Bearer 用法——明文仅此一次。

    为什么输出做成"两行机读 + 一行人读"：机读行给脚本/smoke 解析（字段名即前缀），
    用法行人读教一次"Bearer 怎么带"就够——多打一行真串就多一个泄漏面（story 11）。
    """
    store = KeyStore(_db_path())
    # 行级：name/scope 原样透传给 store（store 只存取不解释，解释归 03 授权层）
    created = store.create(name=args.name, scope=args.scope)
    print(f"key_id: {created['key_id']}")
    print(f"credential: {created['credential']}")
    # 行级：用法行带占位符不带真串——明文在输出里只出现一次（story 11 的字面兑现）
    print("用法: 在请求头携带 Authorization: Bearer <凭据串>")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """list：审计视图——key_id/name/scope/created_at/status 五字段，已撤销可见、绝不回显凭据。

    为什么列头直接用 store.list 的字段名：输出契约与 store 返回值一字不差，走读时
    对照 keys/store.py 不用翻译一层；凭据串压根不在返回值里，"绝不回显"是结构保证。
    """
    rows = KeyStore(_db_path()).list()
    if not rows:
        print("（没有 key）")  # 行级：空库也要有话说——静默无输出会被当成命令没跑
        return 0
    _print_table(rows)
    return 0


def _display_width(text: str) -> int:
    """字符串显示宽度：CJK 全角算 2 列——终端对齐靠它，不引 wcwidth（依赖红线）。"""
    # 行级：>0x2E80 覆盖中日韩全角区间（含全角标点）；key_id/时间戳等 ASCII 算 1 列
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度右补空格到 width 列——str.ljust 数字符不数列，中文名会把表挤歪。"""
    return text + " " * (width - _display_width(text))


def _print_table(rows: list[dict[str, str]]) -> None:
    """定宽表打印审计行——列宽按显示宽度取最大，管理面给"人读审计"一个整齐的面。"""
    headers = ("key_id", "name", "scope", "created_at", "status")
    widths = {}  # 行级：每列宽度=表头与全部单元格显示宽度的最大值（含中文名也不歪）
    for header in headers:
        widths[header] = _display_width(header)
        for row in rows:
            widths[header] = max(widths[header], _display_width(row[header]))
    print("  ".join(_pad(header, widths[header]) for header in headers))  # 行级：表头行按列宽补齐
    for row in rows:
        # 行级：数据行同一形状——逐列补到列宽再拼（推导式行上注释，项目注释纪律）
        print("  ".join(_pad(row[header], widths[header]) for header in headers))


def _cmd_revoke(args: argparse.Namespace) -> int:
    """revoke：软删一把 key（置 revoked_at，即时生效）；查无 id 响亮报错（票面）。"""
    store = KeyStore(_db_path())
    if not store.revoke(args.key_id):
        # 行级：store 回 False=查无此行——CLI 是管理员面不必防枚举，直说并给出自查线索
        raise RuntimeError(f"查无 key_id {args.key_id}（keyctl list 可核对公开标识）")
    # 行级：重复撤销 store 仍回 True（幂等，01 契约备忘）——同样走到这行打印成功
    print(f"已撤销 {args.key_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """三个子命令的解析面：create [--name] [--scope] / list / revoke <key_id>（票面恰三）。"""
    parser = argparse.ArgumentParser(
        prog="keyctl", description="虚拟 key 管理面（create/list/revoke）"
    )
    # 行级：required=True——没有子命令就该是用法错误，不做"默认干什么"的猜测
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="签发一把新 key（凭据串只打印这一次）")
    create.add_argument("--name", default="default", help="人读标签（缺省 default）")
    create.add_argument("--scope", default="*", help="模型白名单：逗号分隔或 *（缺省 *）")
    create.set_defaults(handler=_cmd_create)

    list_cmd = sub.add_parser("list", help="列出全部 key（含已撤销；绝不回显凭据串）")
    list_cmd.set_defaults(handler=_cmd_list)

    revoke = sub.add_parser("revoke", help="撤销一把 key（软删，即时生效）")
    revoke.add_argument("key_id", help="公开标识（keyctl list 可查）")
    revoke.set_defaults(handler=_cmd_revoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：argparse 分派到子命令处理器；配置错/查无 key_id → 退出码 1 并 stderr 指名。"""
    args = build_parser().parse_args(argv)
    try:
        # 行级：set_defaults 挂上的 handler——分派不写 if-ladder，加子命令只加一段
        return args.handler(args)
    except RuntimeError as exc:
        # 行级：子命令抛的错统一收口——stderr 打"错误:"前缀、退出码 1（响亮报错的出口）
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
