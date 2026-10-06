# FlowGate

## Agent skills

### Issue tracker

Issues 以 markdown 文件存于本仓库 `.scratch/<feature>/`（本地，无远程 tracker）。See `docs/agents/issue-tracker.md`.

### Triage labels

默认五个标准标签（needs-triage / needs-info / ready-for-agent / ready-for-human / wontfix）。See `docs/agents/triage-labels.md`.

### Domain docs

Single-context：根目录 `CONTEXT.md` + `docs/adr/`。See `docs/agents/domain.md`.

## 项目规范

> 目标：**简历项目，通过面试是第一目标**；代码必须让用户本人看得懂。周计划与面试防御见 `docs/plan.md`。以下规范覆盖全部代码（含测试）与提交。

### 注释（两层）

- **行级**（行上为主）：写"这行在干什么、为什么这么写"，允许偏 what——看得懂优先。这是对全局"注释只写为什么"规则的**项目级例外**，全局规则不动。
- **块级/函数级**（docstring + 段注释）：写**为什么**——trade-off、反例、关联考点（如"选令牌桶不选漏桶，见 docs/adr/0002-…"）。
- 密度：函数/类 100% 有中文 docstring；逻辑块 100% 有意图注释；复杂语句（嵌套条件、async、推导式、正则、切片）行上必须有注释；自明行（import、常量、简单 return/赋值）可省，拿不准就写。
- 质量线 = **走读**：用户任何一行看不懂即注释缺陷，补到能讲为止。
- 注释与 docstring 用简体中文；标识符用英文。

### 简单（代码不炫技）

- 直白优先：单层推导式、短 lambda、普通函数与类。躲开 metaclass、运算符重载、`eval/exec`、装饰器工厂；装饰器、`contextlib`、`gather` 之外的并发原语属限用（用则配解释注释）；3.14 新语法不为用而用。
- `async/await` 与 async 生成器是考点必用：每个 async 模式**首次出现**配一段"给初学者的解释"。
- 函数 ≤40 行、嵌套 ≤3 层，超限先拆；拆不动则注释写明原因。

### 依赖

- 基础设施可用：FastAPI、Pydantic、uvicorn、httpx、pytest、PyYAML。
- **核心机制一律自建**：限流、计费、路由 fallback、流式管理、key 管理——不引现成实现（含 slowapi/limits/litellm/tenacity/各家官方 SDK），上游用 httpx 裸调 OpenAI 兼容 API。
- 新增任何依赖，在 commit 或注释里写一句"为什么不用 stdlib/自建"。

### 测试

- 测试同享注释规范（断言等自明行可省行级注释）；命名 `test_<行为>_<条件>`（英文）。
- 每个测试函数中文 docstring：证明什么、怎么证明的。
- 可复跑、不依赖外网；真实上游测试显式标记、可跳过。fake 上游同标准注释。

### 开发方法

**红-绿-重构（TDD）默认**：先测试后实现。例外只有探索性 spike，合入主线前测试必须补齐全绿。

### 机械兜底

- 每次改完过 ruff（format + check，配置在 pyproject.toml）；注释密度不进 lint，靠走读验收。
- 边界检查（`app/` 不 import 具体适配器）W4 收尾做。

### 验收（每模块三件套）

1. **走读**：用户复述模块干什么；
2. **自测**：2~3 题，答不出的地方 = 代码/注释不合格；
3. **考点卡**：一句话故事 + trade-off + "怎么测的"数字 → `docs/interview-cards/NN-模块名.md`。

### ADR

每个已定设计决策一份：`docs/adr/NNNN-标题.md`（背景/决定/备选/取舍/后果，一两页内，中文），随做随写。ADR 就是面试故事提纲。

### Git 与 README

- commit message 中文祈使句，首行 ≤50 字，正文写为什么；一个 commit 一个可讲的点；不 squash，完整历史即证据。
- `docs/` 整目录**不入库**（.gitignore 已排除，2026-10-06 用户定）：计划/自测/考点卡/ADR 等教练材料私有，不进 git 历史——将来公开仓库永远见不到它们。仓库公开面 = 代码 + 测试 + README；AGENTS.md 里指向 docs/ 的材料仅本地有效。
- README 中文为主、术语保留英文。敏感信息（key、真实中转地址）永不入库。
