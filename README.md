# FlowGate

**教学级 Python LLM 流式网关**——从零自建、每个机制都带可复跑的实测数。
一条命令启动，一个进程 + 一个 SQLite 文件；curl 或任何 OpenAI SDK 调
`POST /v1/chat/completions` 即得 SSE 流式回答。背后一整套可走读的治理机制：
虚拟 key 认证、令牌桶限流、顺序 fallback（降级不重复扣费）、usage 流末落账。

## 快速开始（fake 上游，零 key、零外网）

```bash
uv sync                      # 装依赖（需要 Python 3.14 与 uv）
cp .env.example .env         # 配置样例落一份——快速开始用缺省值就够，接真实上游时再改它
uv run python -m scripts.serve
```

首次启动（空库）会自举一把 dev key 并打印可粘贴 curl——凭据串只打印这一次，
库里只存 sha256 hash，此后物理上取不回。输出形如（key_id 与凭据串每次启动都不同）：

```
key_id: key_xxxxxxxxxxxxxxxx
credential: fgk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
已自举 dev key（仅空库首启一次；真实部署请 keyctl create 建专属 key 后 revoke 它）
可粘贴 curl（缺省 fake 上游，零外网）：
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer fgk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-model", "messages": [{"role": "user", "content": "你好"}]}'
FlowGate listening on http://127.0.0.1:8000
```

把 curl 照贴（Bearer 后换成启动输出里那一行 credential）→ fake 上游的回显：

```json
{"id":"fake-xxxxxxxx","object":"chat.completion","model":"fake-model","choices":[{"index":0,"message":{"role":"assistant","content":"fake-reply: 你好"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":5,"total_tokens":7}}
```

**dev key 安全边界**：仅本地开发用；真实部署请 `keyctl create` 建专属 key 并 `revoke` dev key
（`uv run --env-file .env python -m scripts.keyctl …`，与网关共享同一个库文件）。

> Windows 终端小提示（2026-10-08 实测）：上面的 curl 块是 bash 续行写法。macOS/Linux 照贴；
> PowerShell 先拼成一行再贴（行内中文没问题）；Git Bash 照贴时行内中文会被参数编码搅坏
> （服务端回 400 body 解析错误），把 content 换成 ASCII（如 `"hello"`）即可。

## 九模块导览

| 目录 | 干什么 | 面试考点 |
|---|---|---|
| `app/` | FastAPI 路由 + Pydantic 契约：认证/限流门卫、错误翻译（401/403/422/429/502/504） | 接口契约设计 |
| `providers/` | 上游适配：fake（离线回显）/ DashScope / Kimi，OpenAI 兼容 REST 裸调（httpx） | 抽象 seam 放哪 |
| `streaming/` | SSE 流式透传、客户端断连清理、gap 超时截断 | 背压、async 生成器、CancelledError |
| `routing/` | 重试 + 顺序 fallback 链 + request_id 幂等 | 超时预算分配、不重复扣费 |
| `ratelimit/` | 令牌桶 per key_id（容量 + 速率可配） | 令牌桶 vs 漏桶 |
| `billing/` | 预算前置检查、流末回填结算、SQLite 账本（用户账 + 损耗账） | 流式 token 计数口径 |
| `keys/` | 虚拟 key 签发/校验/撤销（只存 hash）+ scope 模型白名单 | scope 与撤销语义 |
| `tests/` | pytest + fake 上游 + 并发/断连/计费确定性测试 | 怎么证明它对 |
| `scripts/` | `serve` 一键启动、`keyctl` 管理面、`demo_w2/w3` 可复跑演示 | 分层缺省与启动自检 |

根目录 `assembly.py` 是**装配处**——全仓唯一点名具体上游适配器类的地方（composition root），
由 `tests/test_boundary.py` 的 AST 边界检查钉死依赖方向。配置走 `.env`（12-factor：
代码只读环境变量，不引 dotenv）。

## 与 LiteLLM / otari 的区别

| | FlowGate | LiteLLM | otari |
|---|---|---|---|
| 定位 | 教学级网关核心（核心 2~3 千行级，不含 tests） | 产品级 LLM 网关 | 产品级 LLM 网关（Mozilla AI） |
| 形态 | 一个进程 + 一个 SQLite 文件 | 服务集群 + 外部存储 | 产品 + 全套管理面 |
| 上游 | 3 个自建适配（fake / DashScope / Kimi） | 40+ 上游 | 多上游 |
| 管理面 | 无 dashboard；keyctl CLI 三命令 | dashboard、多租户、key 管理 | 全套管理面 |
| 限流/计费/keys | 三个机制全部自建，接口小、可逐行走读 | 产品内建 | 产品内建 |
| 证据 | 三个工程点都带实测数 + 可复跑 demo | — | — |
| 代码来源 | 从零自写（不 fork、不抄实现） | 开源 | 开源 |

分野：LiteLLM / otari 面向生产部署，FlowGate 面向教学与走读——规模刻意小、机制
全部自建、关键结论都可复跑。架构形状参考 otari（resolve 返回有序 attempts 列表、
Protocol seam、端口命名纪律），实现全部自写。

## 三个工程点 + W2/W3 实测数

1. **断连不泄漏**（`streaming/`）：客户端中途断开，上游流被显式取消（CancelledError
   沿 async 生成器传播），不留悬垂连接。
   实测：**16 路**真 socket 并发各读 2 帧后硬 abort → 上游收尾账 cancelled=**16**、
   completed=**0**，未被取消的上游 = **0**。口径：账里非 cancelled 条目 = 记错的收尾，
   总数不足 16 = 没记账的泄漏——两条同时成立才算 0 泄漏。
   复跑：`uv run python -m scripts.demo_w2`（第二幕）；同口径确定性测试
   `uv run pytest tests/test_streaming.py`。
2. **降级不重复扣费**（`routing/` + `billing/`）：一个逻辑请求 = 一笔用户账（request_id
   幂等键 + DB 唯一约束去重）；失败尝试只进损耗账，不向调用方收费。
   实测：**N=3** 发逻辑请求、共 **M=6** 次失败尝试 → 用户账恰 **3** 笔、损耗账恰 **6** 条。
   口径：三棒链两棒恒失败，每发恰 2 次失败尝试。
   复跑：`uv run pytest tests/test_billing.py`（N=3/M=6 用例）；demo 小口径 N=1、M=1：
   `uv run python -m scripts.demo_w3`（第一幕）。
3. **计费误差可量化**（`billing/`）：usage 流末回填结算——官方 usage 优先，缺失才对
   整段文本估算（不按块加总）。实测（真网 live 幕，qwen-turbo，2026-10-07）：
   估算 74 vs 官方 61 → 误差 **21.3%**（式子 |估算−官方|/官方×100%，数值随模型输出浮动）。
   复跑：`uv run --env-file .env python -m scripts.demo_w3`（第四幕 live，需
   DASHSCOPE_API_KEY）。

另一组 W3 治理硬数字（`ratelimit/`）：令牌桶 per key_id——**8 路**同 key 并发 → 200 恰 **3**、
429 恰 **5**，上游只见 **3** 次调用（被拒的 5 发不碰上游）。复跑：
`uv run python -m scripts.demo_w3`（第二幕）/ `uv run pytest tests/test_ratelimit.py`。

## 测试与 demo

```bash
uv run pytest                        # 全量：离线、确定性（2026-10-08 快照 201 passed）
uv run python -m scripts.demo_w2     # W2 流式三幕：逐块输出 / 16 路断连 / 截断语义
uv run python -m scripts.demo_w3     # W3 治理三幕：降级不重复扣费 / 限流 8 路恰 3 / 流末回填
```

真网 smoke 用例带 `live` 标记、缺上游 key 自动跳过，全量跑永不触网；显式跑：
`uv run --env-file .env pytest -m live`。

## 配置（.env）

所有配置走环境变量（12-factor），样例与注释见 `.env.example`。要点：

- `FLOWGATE_PROVIDER`：`fake`（缺省，离线回显）| `dashscope` | `kimi`；逗号表 = 顺序
  fallback 链（如 `dashscope,kimi`）。
- `FLOWGATE_BILLING_DB`：key 库 + 账本的 SQLite 文件路径。serve 未设时缺省
  `flowgate.db`（生产入口给落盘缺省）；keyctl 未设则报错并指向 `.env.example`
  （管理面不猜路径）。
- `FLOWGATE_PORT`：端口三档——`--port` 旗标 > `FLOWGATE_PORT` > 缺省 8000；
  `--port 0` = OS 分配。
- 上游凭据（`DASHSCOPE_API_KEY` / `KIMI_API_KEY` / `KIMI_BASE_URL` 等）只活在 `.env`，
  永不进 git。

## 接真实上游

1. `cp .env.example .env`（已做过可跳过），在 `.env` 里填上游凭据：
   - **DashScope**：百炼控制台获取 `DASHSCOPE_API_KEY`（形如 `sk-xxxx`）；前缀地址
     有内置缺省，走中转才覆盖 `DASHSCOPE_BASE_URL`。
   - **Kimi**：`KIMI_API_KEY` + `KIMI_BASE_URL`（中转地址，两者都必填；地址只放 `.env`）。
2. 设 `FLOWGATE_PROVIDER=dashscope` 或 `kimi`（或 `dashscope,kimi` 顺序 fallback）。
3. `uv run --env-file .env python -m scripts.serve`——启动自检会在起服务前响亮报错
   （缺 key/缺 base_url 都当场指名变量），不等第一个请求才发现。

## 已知边界

- **单进程单 worker**：令牌桶与预算都在进程内，跨进程/多 worker 共享限流不在 v1。
- **撤销即时生效、痕迹保留**：撤销是软删（审计可见、与从未存在同一条 401）；限流桶
  与账本历史行不清理。
- **Out of scope（刻意砍产品面）**：dashboard、workspace/members、MCP、代码执行、
  guardrails——v1 要深度不要广度。
- **上游凭据只转发不治理**：DashScope/Kimi 的 API key 由部署方放进 `.env` 自己管，
  网关只持有并转发、不治理其生命周期。
