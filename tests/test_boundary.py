"""边界检查（w4 issue 06）：除装配处与适配器自体外，全仓库无人点名具体适配器。

规则原文（ADR-0001 后果一节与 architecture-notes 的"W4 边界检查"兑现）：
    "除装配处与适配器自体外，全仓库无人点名具体适配器。"

为什么是 pytest 元测试而不是独立脚本：进全绿套件才有守门力（票面 Q7 已裁）——
违规 import 与业务测试同批变红，红了就合不进去；独立脚本没人记得跑等于没有。
为什么用 stdlib ast 而不是 grep：import 有 `import x` / `from x import y` / 别名
多种语法形状，正则误报漏报都难收场；ast.parse 拿到的 Import/ImportFrom 节点是
语法事实，且零新依赖（依赖红线：能 stdlib 就不引库）。

术语（对齐 CONTEXT.md）：具体适配器 = providers/ 包里 fake / dashscope / kimi
三个上游实现与其具体类；providers.base 里的 Provider 协议与 UpstreamError 是**端口**，
不算适配器——核心包 import 端口正是 ports & adapters 想要的依赖方向。
"""

import ast
from pathlib import Path

# 扫描面 = 核心六包（票面 checklist：全量文件、非抽样）——白名单之外的全部代码即它们。
# 等价性注记（走读口径）："扫描面 = 六包"与规则的"全仓库"当下恰好等价——白名单外的
# 代码就是这六包；若将来出现第七个顶层模块，要靠把它加进本元组来维持这份等价
CORE_PACKAGES = ("app", "routing", "streaming", "billing", "ratelimit", "keys")

# 白名单 = 允许点名具体适配器的四个位置（结构性需要，票面 checklist）：
# 根级 assembly = 装配处本尊（ADR-0001 的"唯一点名处"）；providers/ = 适配器自体；
# tests/ = 测试要 isinstance 断言适配器类；scripts/ = demo 要注入 fake 上游
WHITELIST = ("assembly.py", "providers", "tests", "scripts")

# 白名单四类位置各取一个的代表样本（样本测试用）：证明豁免对每类都生效，不止纸面名单
WHITELIST_SAMPLE_PATHS = (
    "assembly.py",
    "providers/fake.py",
    "tests/test_assembly.py",
    "scripts/demo_w2.py",
)

# 具体适配器类名（条款 3 的深度防御清单）：条款 1/2 按模块路径判、新增适配器自动进网；
# 这里枚举只为封"经第三方转手"的走私口——新适配器漏补一行也只漏走私面，正面仍兜住
CONCRETE_ADAPTER_CLASSES = ("FakeProvider", "DashScopeProvider", "KimiProvider")

# 仓库根 = tests/ 的上一级（本文件住 tests/，扫描面相对根目录取）
REPO_ROOT = Path(__file__).resolve().parent.parent


def find_adapter_imports(rel_path: str, source: str) -> list[tuple[int, str]]:
    """检测谓词：返回一个文件里"点名具体适配器"的 import，形如 (行号, import 原文)。

    规则拆成可判定的三条（"点名"的字面含义全覆盖）：
    1. import 的模块路径伸进 providers/ 且越过 base——`providers.fake` 等具体适配器模块；
    2. 从 providers 包直取 base 以外的名字——`from providers import FakeProvider` 等；
    3. 任意模块绑定了具体适配器类名——防经第三方转手的走私（`from x import FakeProvider`）。
    白名单文件（装配处/适配器自体/tests/scripts）直接放行：那四处点名是结构性需要。
    ast 只看 import 语句（票面口径"遍历 import"）；属性链摸到的运行期动态访问不在此列。
    """
    if _is_whitelisted(rel_path):
        return []  # 白名单四处点名是结构性需要，见 WHITELIST 注释
    violations: list[tuple[int, str]] = []
    # 复杂语句（ast.walk 遍历整棵树）行上：嵌套在函数/条件里的 import 同样入网
    for node in ast.walk(ast.parse(source)):
        if _names_concrete_adapter(node):
            # get_source_segment 取 import 原文——失败消息即定位与修复指引
            text = ast.get_source_segment(source, node) or "<import 语句>"
            violations.append((node.lineno, text))
    return sorted(violations)  # 按行号排序：失败消息顺序稳定，对照源码从上往下修


def _is_whitelisted(rel_path: str) -> bool:
    """白名单判定：路径是否落在允许点名适配器的四个位置（Windows 路径归一成 posix）。

    为什么按"路径首段"判：包级豁免（providers/tests/scripts 整包）与根级文件豁免
    （assembly.py）用同一条规则覆盖，不用维护"文件清单"那种会漏新文件的名单。
    """
    posix = rel_path.replace("\\", "/")
    top = posix.split("/")[0]  # 首段 = 包名（providers/tests/scripts）或根级文件名（assembly.py）
    return top in WHITELIST


def _names_concrete_adapter(node: ast.AST) -> bool:
    """单个 import 节点是否点名具体适配器（规则三条的可判定化，见 find_adapter_imports）。

    为什么按节点判而不是在 find_adapter_imports 里摊平：import 有 Import/ImportFrom
    两种节点形状、每种下有多条名字，判定收在一个函数里，规则三条各有自己的落点可读。
    """
    if isinstance(node, ast.Import):
        # 条款 1：import providers.fake / import providers.fake as pf 都算
        return any(_touches_adapter_module(alias.name) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        # 条款 1：from providers.fake import …——模块路径已越过 base
        module = node.module or ""
        if _touches_adapter_module(module):
            return True
        # 条款 2：from providers import 非 base 名字——包直取（类名或子模块名）
        if module == "providers":
            return any(alias.name != "base" for alias in node.names)
        # 条款 3：任意模块绑定具体适配器类名——防经第三方转手的走私
        return any(alias.name in CONCRETE_ADAPTER_CLASSES for alias in node.names)
    return False  # 非 import 节点（import 整模块的类名绑定不成立，条款 3 只管 from-import）


def _touches_adapter_module(module: str) -> bool:
    """模块路径条款：伸进 providers/ 且越过 base = 点名具体适配器模块。

    只放行 providers（包本体）与 providers.base（端口）；providers.base 再往下不存在
    （base 是模块不是包），多点一段都算违规。按模块路径判而不是枚举 fake/dashscope/
    kimi：新增适配器自动进网，不靠人记得来更新名单——漏网就是边界上的口子。
    """
    if module != "providers" and not module.startswith("providers."):
        return False  # 其他模块（billing/routing/…）与适配器无关；providersx 不算 providers
    return module not in ("providers", "providers.base")


def format_violations(violations: list[tuple[str, int, str]]) -> str:
    """失败消息 = 定位即修复指引：文件:行号 指名违规 import，再给规则原文与合法去处。

    为什么消息要写全三要素：红了的人（含未来的自己）只看 pytest 输出就该知道去哪改、
    改成什么样——`from providers.base import …` 是合法替代，不用回查本测试文件。
    """
    # 头行 = 规则原文（票面 checklist：docstring 与失败消息都亮出规则）
    lines = ["边界违规（规则：除装配处与适配器自体外，全仓库无人点名具体适配器）："]
    for rel, lineno, text in violations:
        lines.append(f"  {rel}:{lineno}: {text}")  # 文件:行号 + import 原文 = 定位与指认
    # 修复指引的合法去处由 WHITELIST 拼出：白名单改名时消息跟着走，不给人肉同步留漂移口
    allowed = "、".join(WHITELIST)
    lines.append(
        f"修复指引：核心包只可 `from providers.base import …`（端口/协议）；"
        f"具体适配器只出现在 {allowed}。"
    )
    return "\n".join(lines)


def test_adapter_import_flagged_when_naming_fake_adapter() -> None:
    """证明：核心包里 `from providers.fake import …` 被谓词咬住——检测不是空转。

    怎么证明：喂一段含违规 import 的内联源码（样本即违规现场），断言谓词报出的
    违规恰是那条 import（行号 + 原文）——样本必红，证明扫描真会咬人。
    """
    # 内联样本：一行端口 import（合法）+ 一行点名 fake 适配器（违规）
    source = "from providers.base import Provider\nfrom providers.fake import FakeProvider\n"

    violations = find_adapter_imports("app/main.py", source)

    assert violations == [(2, "from providers.fake import FakeProvider")]


def test_adapter_import_flagged_when_import_providers_kimi() -> None:
    """证明：`import providers.kimi` 整模块导入形状同样被咬——不止 from-import 一种语法。

    怎么证明：样本只含一条整模块 import，断言谓词报出它（行号 1 + 原文）。
    """
    source = "import providers.kimi\n"

    violations = find_adapter_imports("routing/chain.py", source)

    assert violations == [(1, "import providers.kimi")]


def test_adapter_import_flagged_when_taking_name_from_providers_package() -> None:
    """证明：从 providers 包直取具体名字（类或子模块）也算点名——封住绕开子模块路径的口子。

    怎么证明：样本 `from providers import DashScopeProvider`（模块路径只是 providers
    包本体，走不了模块路径条款），断言谓词仍报出违规——直取条款生效。
    """
    source = "from providers import DashScopeProvider\n"

    violations = find_adapter_imports("billing/settlement.py", source)

    assert violations == [(1, "from providers import DashScopeProvider")]


def test_adapter_class_flagged_when_smuggled_via_other_module() -> None:
    """证明：类名条款咬人——经第三方模块转手的具体适配器类名同样是"点名"。

    怎么证明：样本 `from routing.helpers import KimiProvider`（模块路径与 providers
    无关，条款 1/2 都不触发），断言谓词仍报违规——封住"改个 import 路径绕开扫描"的口子。
    """
    source = "from routing.helpers import KimiProvider\n"

    violations = find_adapter_imports("app/main.py", source)

    assert violations == [(1, "from routing.helpers import KimiProvider")]


def test_port_import_allowed_when_core_only_names_providers_base() -> None:
    """证明：负对照——谓词不是见 providers 就咬的空转器（防止"永远红"冒充"会咬人"）。

    怎么证明：样本只 import providers.base 的端口名（Provider/UpstreamError，核心包
    现实中的合法用法），断言谓词返回空列表——与上面四条必红样本合成完整真值表。
    """
    # 端口 import + 普通跨包 import（billing 的类不是适配器）——都该放行
    source = (
        "from providers.base import Provider, UpstreamError\n"
        "from billing.ledger import BillingLedger\n"
    )

    assert find_adapter_imports("app/main.py", source) == []


def test_adapters_allowed_when_file_in_whitelist() -> None:
    """证明：白名单豁免真实生效——装配处/适配器自体/tests/scripts 点名不算违规。

    怎么证明：把与必红样本同款的违规 import 喂给四个白名单位置，断言谓词全部放行——
    豁免不是文档摆设；tests/ 要 isinstance、scripts/ 要注入 fake，都是结构性需要。
    """
    violating = "from providers.fake import FakeProvider\n"

    # 复杂语句（推导式）行上：四个白名单位置逐个过一遍违规样本，必须全放行
    results = [find_adapter_imports(path, violating) for path in WHITELIST_SAMPLE_PATHS]

    assert results == [[], [], [], []]  # 装配处/providers/tests/scripts 各一份空违规


def test_violation_message_names_file_and_line_when_flagged() -> None:
    """证明：失败消息 = 定位即修复指引（票面 checklist）——文件:行号、import 原文、规则去处齐全。

    怎么证明：喂一条 (路径, 行号, import 原文) 违规给格式化器，断言消息里三要素都在，
    且带规则原文与合法去处——修的人拿到消息不用回查本文件。
    """
    message = format_violations([("app/main.py", 2, "from providers.fake import FakeProvider")])

    assert "app/main.py:2" in message  # 文件 + 行号 = 定位
    assert "from providers.fake import FakeProvider" in message  # import 原文 = 指认
    assert "除装配处与适配器自体外" in message  # 规则原文 = 解释为什么红


def test_adapter_imports_absent_when_scanning_core_packages() -> None:
    """证明：核心六包全量文件零点名具体适配器——边界物理成立，这条红了就是有人越界。

    怎么证明：六包逐包 rglob 全量 .py（非抽样）过检测谓词，汇总违规断言为空；扫描面
    缺包同样失败（防覆盖面静默缩水）。非空时失败消息按 文件:行号 指名 import 原文。
    为什么进全绿套件而不是独立脚本：与业务测试同批跑才有守门力（票面 Q7 已裁）。
    """
    violations: list[tuple[str, int, str]] = []
    for pkg in CORE_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        # 行级：六包一个都不能少——目录消失会让边界静默漏检，宁可红着喊出来
        assert pkg_dir.is_dir(), f"扫描面缺包：{pkg}（边界检查的覆盖面不许静默缩水）"
        for path in sorted(pkg_dir.rglob("*.py")):  # 全量文件，非抽样（票面 checklist）
            rel = path.relative_to(REPO_ROOT).as_posix()  # posix 形态：消息与白名单统一口径
            # 显式 utf-8：源码是 utf-8，Windows 缺省 GBK 会把中文注释读坏
            source = path.read_text(encoding="utf-8")
            for lineno, text in find_adapter_imports(rel, source):
                violations.append((rel, lineno, text))
    assert not violations, format_violations(violations)
