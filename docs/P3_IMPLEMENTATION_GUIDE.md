# P3 实施指南：本地作品集 Demo（合成 fixture · fake-only · 回环 HTTP）

本文档描述 P3 阶段**实际实现**的本地演示能力：一个可以用浏览器打开并操作的
作品集 Demo，默认使用手写合成 fixture 与本地 fake callable，**不调用真实
LLM、不联网、不读取 API key、不下载数据、不访问 Dataset 真实资产**。

诚实声明（与界面常驻标注一致）：

- Demo 中所有评论、用户、商品、日期与统计均为**手写合成示例**，不是真实用户
  或真实消费记录；
- 四种模式的对比数字是 **fake 运行的计数**，不构成准确率、RMSE 或"记忆提升率"
  结论；真实 LLM 实测尚未运行；
- P2 评测模块仍在独立开发中：Demo 不导入 P2 接口，UI 中 P2 面板明确标注
  「评测集成待验收 / 真实 LLM 实测未运行」；
- 安全实现是**演示级的边界**（回环绑定、Host/Origin 校验、CSRF token、
  请求体上限、限流与并发上限），不宣称生产级完整安全；CSRF token 是浏览器
  同源初始化机制，不用于对抗本机其他进程；
- 本文档不声称独立 review 已通过。

## 1. 学习顺序与模块位置

```text
P0（已验收）amazon_p0.py 等：严格 prepare / provider / 基线闸门
P1（已验收）reliable_agent.py + temporal_context.py：
    有界证据工具、受限渲染、严格校验、四状态、预算账本、memory 闸门
P3（本阶段）：
    demo_fixtures.py   手写合成 fixture + 固定场景目录 + 模式合同（fake-only）
    demo_app.py        回环 HTTP 服务（stdlib）+ 执行投影 + CSV/报告数据 + CLI --check
    web_demo/          中文静态界面（index.html / app.js / styles.css）
    tests/test_demo_app.py  回环 HTTP 与场景合同测试（无外网）
P2（并行开发中）：p2_evaluation/… 由独立会话负责；P3 不导入、不依赖。
```

P3 只使用已验收 P1 的**稳定默认接口**：`run_reliable_suite`、`BudgetLedger`、
`ReliableMemoryStore`、`RenderConfig`、`BudgetSpec`，以及
`temporal_context.TEMPORAL_ABLATION_CONTRACTS` 的四模式合同。P3 不修改
P0/P1 模块、不修改 P2 任何文件。

## 2. 运行方式

```powershell
# 启动本地 Demo（默认 127.0.0.1:8290；仅回环可访问）
python demo_app.py

# 指定端口 / IPv6 回环
python demo_app.py --host 127.0.0.1 --port 8311
python demo_app.py --host ::1 --port 8311

# 离线自检：场景合同（9 场景 x 4 模式）+ 回环 HTTP 自检，跑完即退出
python demo_app.py --check
```

浏览器打开 `http://127.0.0.1:8290/` 后：左侧选择场景与模式（默认"对比全部
四模式"），点击"运行合成演示"。页面**不会**发起任何外部请求（无 CDN、无字体、
无图片外链）。

测试（独立 basetemp，避免与并行的 P2 工作互相干扰）：

```powershell
python -m pytest tests/test_demo_app.py tests/test_demo_app_frontend.py -q `
  --basetemp=C:\Users\Lenovo\AppData\Local\Temp\opencode\p3-revised-coder-20261010
```

`tests/test_demo_app_frontend.py` 使用 Node 内存 DOM stub 回归 JS 行为与移动端 CSS
契约（node 缺失时自动跳过）；**真实浏览器渲染（含 375x812、post-run、详情
Combined、预算子页）由 main 复验**。

## 3. 架构与数据流

```text
浏览器（web_demo）
  GET  /api/meta      -> 固定场景/模式枚举、CSRF token、安全说明（同源初始化）
  GET  /             -> index.html（CSP: default-src 'none'; script-src 'self' …）
  POST /api/run       -> {"scenario": 枚举, "mode": 枚举或 "all_modes"}
        │  Origin（若有）必须回环同源；X-Demo-CSRF 必须匹配 server-issued token
        ▼
demo_app.execute_demo(scenario, mode)
  per request：新建临时目录 -> demo_fixtures.build_demo_provider
      手写合成 tasks/groundtruth/reviews -> build_temporal_manifest
      -> TemporalContextProvider 全量校验（与 P0/P1 同一套闸门）
  per mode：新建 ScriptedDemoLLM（确定性 fake）+ BudgetLedger + ReliableMemoryStore
      run_reliable_suite(contexts, include_trusted_stats=合同值, memory_store=合同值)
        - 无生成记忆模式：memory_store=None（召回全 0）
        - 有生成记忆模式：仅最终验证成功写一次；严格更早事件时间才召回
  投影（safe projection）：
      - 从"faked callable 实际收到的 prompt"中提取真实渲染的证据行
        （证据面板显示的就是模型真正看到的内容，非事后重建）
      - 引用注册表：result.evidence_refs + P1 diagnostics 的 cited_evidence
      - 记忆漏斗：candidates → time_eligible → selected → prompt_included → write
      - 每任务用量差值 task_usage_delta（相邻 run 累计快照的差值；首任务以 0 为
        基线；非负且总和等于 run total），budget_after_task 明确标注为"截至本任务
        累计"；prompt 字节/输出 token 是"预算占用/预留"，不是实际计费
      - 四模式对比：同一 context_scope_sha256，仅两个开关不同
  _assert_no_groundtruth_marker：响应 JSON 出现 GROUNDTRUTH 标记即失败关闭
```

### 3.1 固定场景（9 个）

| scenario_id | 预期合同 | 关键点 |
| --- | --- | --- |
| `cross_date_success` | 成功 x3 | 跨日期召回；同日只成为候选 |
| `same_day_isolation` | 成功 x2 | 同 UTC 日条目 eligible=0 |
| `user_isolation` | 成功 x2 | 另一用户候选恒为 0 |
| `repair_once` | 成功（repaired） | 恰好 1 次 repair，共 2 次尝试 |
| `transport_failed` | 失败 | `llm_transport_error`，不写记忆 |
| `budget_stop` | 降级 + 拒绝 | 共享预算只允许 1 次调用；第 2 任务调用前拒绝 |
| `insufficient_evidence` | 拒绝 | 0 次 LLM 调用 |
| `render_gate_refusal` | 拒绝 | 渲染预算 1 字节，无行可引用 |
| `injection_filtered` | 成功 | 注入行被过滤并计数，prompt 不含注入文本 |

非成功场景在 UI 中作为**预期合同**展示（明确标注），永远不计为 success。

### 3.2 四模式合同（直接来自 temporal_context）

| 模式 | generated memory | trusted stats |
| --- | --- | --- |
| No_Memory | 关 | 关 |
| Generated_Review_Memory | 开 | 关 |
| Trusted_History_Stats | 关 | 开 |
| Combined | 开 | 开 |

四种模式使用**完全相同**的 provider 上下文（真实历史/参考逐字节一致），只切换
两个开关；`context_scope_sha256` 在四模式间一致，UI 显式展示。

## 4. API 合同

- `GET /api/meta`：返回场景/模式枚举（固定）、`all_modes_id`（默认选中项，固定为
  `"all_modes"`）、`csrf_token`、安全说明、P2 占位状态、限额（含
  `max_connections`）。响应无 CORS 头。元数据不完整时前端禁用运行并给出友好
  失败文案（不会"假就绪"）。
- `POST /api/run`：body 恰为 `{"scenario": <枚举>, "mode": <枚举|"all_modes">}`。
  不接受任意 user/item/as_of/path/model/URL/代码/upload/secret；未知字段、
  缺失字段、错误类型、非枚举值全部 400 且不反射原始输入。
  成功响应为脱敏 JSON：`simulated=true`、`synthetic_session_id`（临时合成）、
  每模式 `run_hash`（records sha256）、逐任务实际/预期合同、`task_usage_delta`
  （每任务差值）与 `budget_after_task`（截至本任务累计）、memory 漏斗、
  渲染证据与引用注册表；不含 GT、prompt 原文、真实数据或路径。
- 静态：仅 `/`、`/app.js`、`/styles.css` 三个精确 allowlist 路由，无目录遍历；
  其他路径 404，方法不允许 405。

## 5. 安全边界（演示级，诚实说明）

| 边界 | 实现 |
| --- | --- |
| 绑定 | 仅 `127.0.0.1` / `::1`；其他 host 直接拒绝；端口 0-65535 |
| Host/Origin | Host 与（存在的）Origin 必须是回环名且端口一致；否则 403 |
| CSRF | 每个 server 实例随机 token；POST 必须携带 `X-Demo-CSRF`；无 Origin 的 CLI 请求也必须显式携带 token |
| 请求体 | Content-Length 必须为单个整数字符串；≤16 KiB；拒绝 `Transfer-Encoding`；Content-Type 必须 `application/json` |
| JSON | 单对象、拒绝重复键/NaN/Infinity、深度 ≤8、未知字段拒绝 |
| 响应 | 安全类别码；无 traceback/路径/prompt/密钥/原始输入反射；`_assert_no_groundtruth_marker` 二次防泄漏 |
| 并发/限流 | **连接级信号量先于线程创建**（默认 16，硬上限 32；超限安全 503 或关闭，绝不无限建线程）+ 解析后 inflight 信号量（默认 4）+ 滑动窗口限流（默认 60/60s，硬上限 600/3600s）；handler socket 超时 15s |
| 错误日志 | `handle_error` 覆写为固定安全类别（`connection_error category=handler_failure`），不输出 traceback/路径/请求内容 |
| 前端 | 所有动态 DOM 用 `textContent` 创建；无 raw HTML 注入；CSP 固定；无外部资源；tab 键盘（Arrow/Home/End）+ 焦点保留 + roving tabindex；≤375px 真收缩（flex-wrap/min-width:0），不用 overflow 遮罩 |

**非目标**：这不是生产服务，也不是对抗本机其他进程的沙箱。token 只防止
跨站页面借浏览器发起写请求；本机进程可以直接读取 token。

## 6. 合成 vs 真实（必须保持的口径）

- fixture：`demo_fixtures.py` 手写 14 行合成评论 + 6 个任务 + 6 条 GT 标记行，
  由现有 manifest/provider 全量核验；GT 文本带 `SYNTHETIC_GROUNDTRUTH` 标记，
  响应与 UI 永远不会出现；
- 模型：注入式确定性 fake callable；无真实网络客户端、无 API key 入口、
  无 ENV 读取；
- 统计面板：准确率固定显示「未实测」；记忆数字标注「召回次数（不是提升率）」；
- P2 面板：固定文案「评测集成待验收 / 真实 LLM 实测未运行」，不预写 P2 已完成，
  不塞任何 fake RMSE。

## 7. CI 与本地验证

`.github/workflows/ci.yml` 新增两步（全部离线、无 API key、无 Simulator 框架）：

```yaml
- name: P3 demo contract + loopback self check (offline)
  run: python demo_app.py --check
- name: P1 synthetic smoke (offline fake callables)
  run: python p1_smoke.py synthetic --output-dir "$RUNNER_TEMP/p1-smoke-out"
```

说明：CI 需要在 GitHub 上实际跑一次才能最终确认（本阶段不 push；本地已用等价
命令验证）。`--check` 不启动长期 server，退出码 0/2。

本地验证命令（均在 Windows 上实测通过）：

```powershell
python demo_app.py --check
python -m pytest tests/test_demo_app.py tests/test_demo_app_frontend.py -q --basetemp=<独立目录>
python -m pytest tests/test_p1_workflow.py tests/test_reliable_agent.py -q
node --check web_demo/app.js
```

`tests/test_demo_app_frontend.py` 在 Node 缺失时跳过；真实浏览器（375x812、
post-run、详情 Combined、预算子页）由 main 复验。

## 8. P2 完成后的薄 UI 集成建议（不在本阶段实施）

P2 评测模块经 main 独立验收后，可以在**不破坏现有边界**的前提下接入：

1. 新增只读端点（如 `GET /api/p2/status`），仅返回验收后的聚合状态与版本；
   仍走同一 Host/Origin/CSRF/限流中间件；
2. 若要触发评测运行，使用独立的固定枚举 POST（不要把 P2 自由参数透传到 P3
   API）；运行入口必须仍是离线可替换的注入式 callable；
3. UI 单独分区展示 P2 结果，标明「已验收 / 真实运行范围 / 未覆盖范围」，
   未验收前继续显示占位文案；
4. 报告拼接：P3 的脱敏报告保持独立，P2 指标以独立段落附上并带验收标注，
   避免把评测数字与合成演示数字混在一张表里。

## 9. 限制

- 场景与 fixture 数量有限，刻意保持小规模、可读、可核验；
- fake 运行只证明"合同与闸门形状"，不证明任何模型质量；
- HTTP 安全为演示边界，未做 TLS、认证、配额持久化等生产特性；
- 本地 Windows 端口探测使用 `SO_EXCLUSIVEADDRUSE`；跨平台语义已在代码中
  按平台可用性处理，但 CI 上仍需实际运行确认；
- 移动端与交互行为的自动化覆盖是 Node DOM-stub + CSS 契约；真实浏览器渲染
  （375px、post-run、Combined、预算子页）由 main 复验，本文档不声称浏览器
  验收已完成。

## 10. 独立 review 返修记录（2026-10-10）

- meta 增加 `all_modes_id`；前端一致排序与默认选中，元数据不完整时禁用运行；
- 详情模式切换同时重渲染 stats（run 级、不随任务切换），Node 测试区分
  No_Memory（未启用）与 Combined（候选 2 / 进入 prompt 2 / 写入 3）；
- 新增 `task_usage_delta` 每任务差值（总和==run total），预算文案改为
  "占用/预留"，禁用记忆显示"未启用"；
- 报告迁至 gitignored `results/P3_CODER_REPORT_20261010.md`；
- 连接级信号量先于线程创建，`handle_error` 安全日志，限流/连接配置硬上限；
- 375px 移动端真收缩布局；tab 焦点保留 + Arrow/Home/End + roving tabindex；
- 本地 favicon 消除 404；新增 Node DOM-stub 与 CSS 契约测试。
