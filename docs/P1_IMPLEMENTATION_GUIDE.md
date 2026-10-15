# P1 实施指南：可靠单 Agent 受控工具工作流

本文档描述 P1 阶段**实际实现并通过离线测试**的能力，以及它与 P0 的衔接方式。
P1 只覆盖「单 Agent + 受控工具 + 结构化预测 + 有限修复 + 明确状态 + 受控写入」，
不包含 P2 付费模型消融、P3 UI、不接入真实在线模型、不下载数据、不改变既有
P0 产物。所有示例命令只使用注入的 fake callable 与合成/本地已核验数据。

诚实声明（与代码中的 `HONESTY_NOTES` 一致）：

- 这是**工具增强的单一 Agent 工作流**，不是自主 multi-agent 系统；
- 离线合成测试与 fake callable **不测真实 LLM 准确率**，不产出准确率或
  Memory 增益结论；
- provenance 校验证明的是「引用范围与渲染一致」，**不等于语义/事实正确**；
- 注入过滤是有限启发式，不是全面安全保证；
- `TaskEvidenceTool` 是**执行器内部的受界对象，不是外部权限沙箱**：它没有查询
  参数面、payload 有硬上限、读取前必须预留预算；但进程内恶意调用者仍可自行
  构造另一个工具对象，本模块不承诺防住这种情况；
- generated memory 只作表达风格参考，**绝不是可信评分统计或 groundtruth**；
- `degraded` 统计回退永远不会被当作 LLM 成功。

## 1. P0 → P1 总体数据流

```text
P0 download（唯一联网操作，显式调用）
  -> raw 压缩源（gitignored Dataset/ 下）
P0 prepare
  -> normalized_reviews.jsonl   全量规范化来源（本地、ignored）
  -> runtime_reviews.jsonl      冻结历史前缀 + 选中目标源行
  -> tasks/ groundtruth/        身份 vs 被扣留标签（本地、ignored）
  -> manifest.json              严格时间清单（schema 2）
  -> p0_policy.json             适配器/协议/先验元数据
P0 run（离线）
  -> 工件尺寸/行数闸门（解析前）-> 策略参数校验 -> 选择重建
  -> eligible 前缀先验重算校验 -> TemporalContextProvider 严格重建
  -> 四个基线预测（预测函数不接触 groundtruth）-> 打分 -> 脱敏报告

P1（reliable_agent.py，离线）
  -> load_verified_prepared_dataset：
       逐步复用 P0 的全部闸门（artifact_gate → policy → selection →
       prior → provider → split 计数），不直接 json.load 绕过
  -> run_reliable_suite（按 provider execution_order 串行）：
       begin_run：拒绝 run_id 不匹配/已使用/非全新的 memory store
       每任务：
         validate_memory_selection：当前用户 + 严格更早事件时间 + 可信
           origin + run 绑定；任何 forged/foreign recall 在渲染前失败
         BudgetLedger.reserve_tool_calls(4)（原子预留，先于任何读取）
         TaskEvidenceTool 四个有界读取（history/refs/stats/prior）
           -> ToolPayload：条数/文本/总字节上限、full digest + 字符范围
           -> EvidenceRegistry：task-scoped evidence_id、注入/空行过滤
         render_task_prompt（固定指令 + JSON 转义证据块 + 字节/条数/文本上限）
           -> RenderedPrompt：实际渲染集合；0 行可引用 -> refused，不发 LLM
         BudgetLedger.reserve_llm_attempt（检查 prompt 字节与 per-call
           output token 上限；额度不足/deadline 已过不发请求）
         draft → 严格校验 → 质量问题检查 →（可选 1 次 repair）→ 四状态结果
         success 时且仅当 success 时：ReliableMemoryStore.remember_success 写一次
       end_run
  -> summarize_reliable_run：run_id 哈希、固定 model_source 枚举、
     仅安全数字/枚举/sha 的聚合

P1 CLI（p1_smoke.py，离线）
  synthetic 模式：自建合成 provider + 脚本化 fake，覆盖跨日期/同日/不同用户/
                   成功/失败/repair/预算停止/注入过滤/证据不足/渲染闸门
  prepared 模式：读取本地已验收 P0 prepared 目录（必须经过 load_verified_prepared_dataset），
                   用确定性 fake 运行，输出 shape/gate 核验报告
```

同一份 P0 产物既不会被 P1 重写，也不会因 P1 改变字节：P1 只读取、只做与
`amazon_p0.run_baselines` 相同的校验后使用 provider 上下文。

## 2. 模块与主要职责

| 模块 | 职责 |
| --- | --- |
| `reliable_agent.py` | P1 核心：有界证据工具、受限渲染、严格校验、四状态、预算账本、memory 闸门与召回校验、run 编排、P0 prepared 校验加载 |
| `p1_smoke.py` | 离线 smoke CLI（`synthetic` / `prepared`），fake-only，失败非零 |
| `amazon_p0.py` | P0 适配器/协议/基线 runner（**未改动**；P1 复用其内部校验函数） |
| `rating_baselines.py` | 标签安全的四个确定性评分基线（P1 degraded 回退复用） |
| `temporal_context.py` | 严格 provider；P1 复用 `TemporalContextProvider`，并修复了 ablation 的 generated memory 二次写入闸门 |
| `local_memory.py` | 旧实验内 memory（未改动；P1 使用独立的 `ReliableMemoryStore`） |
| `improved_agent_with_quality.py` | 旧课程入口；P1 阶段修复了 JSON 类型强转、截断、日志隐私、reflection transport 与 memory 污染漏洞，返回形状保持兼容 |
| `comprehensive_evaluation.py` | 旧 Simulator runner；`DeepSeekLLM` 增加 typed transport failure，不做大面积重写 |

### 2.1 P1 核心类/函数调用链

```text
run_reliable_suite(contexts, llm=..., budget=..., global_prior=..., memory_store=...)
  ├─ model_source 枚举校验
  ├─ memory_store.begin_run(run_id)        # 不匹配/复用/非全新 -> 拒绝
  └─ 每个 context（execution_order 升序）：
       memory_store.recall_before(user, target_timestamp, limit)
       run_reliable_task(context, llm, budget, global_prior, memory_recall, expected_run_id)
         ├─ BudgetLedger.reserve_tool_calls(4)      # 原子预留，先于任何读取
         ├─ TaskEvidenceTool（有界读取 x4）          # 无查询参数面
         ├─ build_evidence_registry(payloads)        # candidate→tool→registered
         ├─ render_task_prompt(...)                  # 校验 recall + 渲染
         │    └─ 0 个可引用行 -> refused（不调用 LLM）
         ├─ BudgetLedger.reserve_llm_attempt(draft)  # 调用前 reserve
         ├─ _invoke_llm(...)                         # 异常→llm_transport_error
         ├─ validate_model_output(...)               # 字节/重复/深度/类型/引用
         ├─ generated_quality_issues(...)            # 注入/重合
         ├─（失败时）1 次 repair：reserve → call → 再校验
         └─ ReliableResult(status/reason/stars/review/evidence_refs/
                           simulated/prediction_origin/diagnostics/usage)
       success → memory_store.remember_success(...)  # 恰好一次
  └─ summarize_reliable_run(...)                     # 脱敏聚合
```

## 3. 有界证据工具与证据注册表

### 3.1 诚实的权限边界

`TaskEvidenceTool` 的公开 getter **没有任何调用方可传参数**（没有
`user_id` / `item_id` / `as_of` / `path` / `task_index` 查询面），并且它由
执行器为**唯一一个** provider 任务构造。这是「结构上没有查询面」的约束，
不是 token/沙箱安全边界：代码中已删除 token 授权与相关强断言，文档也不再
声称「伪造 token 访问不可能」。进程内调用者可以自行构造别的工具对象；本
模块实际保证的是：没有查询参数、payload 有硬上限、读取前必须预留预算。

### 3.2 ToolPayload：真实条数、真实字节、真实截断

`TaskEvidenceTool` 常量上限：

```text
MAX_TOOL_ROWS_PER_KIND          = 64        # 每类最多返回行
MAX_TOOL_TEXT_CHARS             = 4000      # 单条文本在工具边界截断
MAX_TOOL_PAYLOAD_BYTES_PER_KIND = 256 KiB   # 每类 payload 总字节上限
```

- history 采用**最新优先**抽样（event_time、source_row_index 降序），refs
  保持 provider 顺序；超出上限的行计入 `skipped_by_row_cap` 或
  `skipped_by_payload_bytes`，绝不先复制全部上下文再截断；
- `ToolPayloadLimits` 只允许**更小**：三个字段分别不得超过
  `MAX_TOOL_ROWS_PER_KIND` / `MAX_TOOL_TEXT_CHARS` /
  `MAX_TOOL_PAYLOAD_BYTES_PER_KIND`，超限直接
  `tool_limits_hard_limit_exceeded`；
- 文本截断时**保留全文 digest**（`_temporal_text_sha256`）、原始字符数
  （`_temporal_text_chars`）与截断标记（`_temporal_text_truncated`），
  registry 的 `EvidenceItem` 同时记录 `text_chars_total`/`text_truncated`；
- `ToolPayload` 暴露 `source_rows` / `returned_rows` / `skipped_*` /
  `text_truncated_rows` / `payload_bytes`，registry 的漏斗统计直接来自这些
  **真实动作数字**，不是「非空校验数」之类的代理值。

### 3.3 漏斗式诊断：候选 → 工具返回 → 注册 → 渲染 → 引用

`diagnostics["evidence"]` 与 `diagnostics["rendered"]` 分别记录：

- `candidate_rows`：provider 可见行（工具看到的源行数）；
- `tool_returned_rows`：工具实际返回行；
- `registered_rows`：过滤空文本/注入启发式命中后注册的行；
- `filtered_empty_rows` / `filtered_injection_rows` / `skipped_by_tool_cap_rows`
  / `tool_text_truncated_rows` / `tool_payload_bytes`；
- `history_rendered` / `reference_rendered` / `history_truncated` /
  `reference_truncated`（**只统计实际渲染的行**）/ `*_skipped_by_budget`；
- `cited_rows` 与 `cited_evidence`（每条引用的 id/kind/source_row_index/
  event_time/text_sha256/text_chars_total/text_truncated/stars）。

引用校验**只对照实际渲染集合** `RenderedPrompt.rendered_ids`：被预算截断或
未渲染的行、别的任务的 id、同一文本另一来源的 id 都不可能通过。代码不做
任何「全文 substring 反推来源」。

### 3.4 evidence_id 与结构化渲染

`evidence_id = "ev_" + sha256("p1-evidence\0" + task_scope + kind + source_row_index + text_digest)[:16]`，
`task_scope = sha256(context_sha256 + manifest_context_sha256)`。同一行在不同
任务中 id 不同；同一文本不同来源 id 不同；同一任务重建稳定。

prompt 由固定可信指令 + 证据区 + generated memory 区 + 统计区 + 先验区 +
输出契约组成。证据/记忆行是**逐行 JSON**（引号与换行转义），untrusted 文本
无法破坏结构。渲染预算：总 prompt 字节上限、每区条数上限、单条文本字符上限、
**每区字节上限**（`max_history_chars` / `max_reference_chars` /
`max_memory_chars` 字段名沿用 chars，实际按 UTF-8 编码字节数比较）；超出的行
不渲染并计数。**注册行存在但实际渲染为 0 行时直接
`refused`（`no_rendered_evidence`），不调用 LLM、不写记忆。** 固定文本本身
超预算时在调用前直接 `refused`（`prompt_budget_exceeded`）。已知注入模式在
注册阶段命中即过滤，并明确「有限启发式，不是安全保证」。

## 4. 四状态决策树

| 状态 | reason（安全类别） | 触发条件（固定策略） |
| --- | --- | --- |
| `success` | `validated_model_output` / `validated_model_output_repaired` | 严格 JSON 契约 + 整数 1..5 评分 + 非空未超长文本 + 质量检查通过 + 引用全部命中实际渲染集合 |
| `degraded` | `statistical_baseline_fallback` | 草稿与（允许的）repair 都未通过校验/质量检查，但上下文有可见证据：用**实际渲染行**的 stars 计算 `rating_baselines` 固定回退，`review=None`，`prediction_origin=statistical_baseline`，引用贡献行 |
| `refused` | `insufficient_evidence` / `no_rendered_evidence` / `budget_exhausted` / `deadline_exceeded` / `prompt_budget_exceeded` / `output_token_limit_exceeded` | 证据为空、注册行无法渲染、或在任何调用之前预算/deadline/per-call 上限不允许；这是「拒绝执行」，不是错误 |
| `failed` | `llm_transport_error` / `invalid_llm_return` / `internal_error` | 注入 callable 抛异常、返回非字符串/空串，或内部意外错误 |

关键点：

- 布尔、NaN、Infinity、超大整数、范围外、非整数评分、重复 JSON 字段、
  超深 JSON、超字节输出、超量引用、空/超长文本、错字段/多字段、非 JSON、
  散文包 JSON、无效/未渲染/跨任务引用，**都不会被 snap/clamp 成 success**；
  最多一次 repair，绝不无限循环；
- `diagnostics["llm"]` 区分 `repair_requested`（校验失败后想修）与
  `repair_attempted`（reserve 成功、真正发出）以及 `repair_reserve_failure`；
- `refused` / `failed` 的 `stars` 与 `review` 均为 `None`；`degraded` 的评分
  出处是**明确标注的统计基线**，其参数固定为 `rating_baselines` 默认值
  （`lambda_user=5`、`lambda_item=5`、`weight_user=0.5`）；传入非默认
  `BaselineParams` 在调用前直接拒绝（`unsupported_baseline_params`），
  `diagnostics.baseline.params` 报告实际数值，不会把自定义参数伪报成默认；
- `simulated` 恒为 `True`，`prediction_origin ∈ {llm, statistical_baseline, none}`。

## 5. 预算与限额（调用前 reserve）

`BudgetSpec` 固定上限：每任务/整 run 的最大 LLM 次数（默认每任务 2：draft 1 +
repair ≤ 1）、工具调用次数、单次/整 run prompt 字节、单次/整 run 输出 token、
可选 `deadline_seconds`。所有字段有**显式硬上限**（`BUDGET_HARD_LIMITS`：
每任务 LLM ≤ 2、per-call prompt ≤ 64 KiB、per-call output ≤ 4096 token、
deadline ≤ 24h 等），`RenderConfig` 同样有 `RENDER_CONFIG_HARD_LIMITS`
（review ≤ 8000 字符、输出字节 ≤ 64 KiB、引用 ≤ 16 等），不存在无限可调预算。

`BudgetLedger` 语义：

- **调用前 reserve 并检查**：额度不足、deadline 已过、per-call prompt 字节
  超限、**per-call output token 超限**都直接 `refused`，不发请求；
- **工具读取先预留**：`reserve_tool_calls(task, 4)` 原子检查并一次性计入；
  失败时**一个 getter 都不会执行**，不读取、不复制任何 payload；
- **失败也计 attempt**：reserve 先自增计数，异常只记录 failure，不退还次数；
  prompt 字节与请求的 `max_tokens` 在 reserve 时按最坏情况计费；
- 整 run 计数器在一个共享账本中，**不会每任务重置**；repair 只在剩余额度
  允许时发送，否则走明确标注的统计回退；
- deadline 只在**调用前/调用之间**检查，不做线程强制取消（阻塞中的外部
  调用无法被本仓库取消）；调用 adapter 负责显式 timeout 契约；
- usage 分离：`prompt_tokens_estimate = ceil(utf8_bytes / 2)` 是**启发式估计**
  （字段与文档都明确说明不是计费上界、未含 chat-message framing）；
  `prompt_bytes_charged` 是硬字节上限；`output_token_budget_charged` 是请求
  上限；`actual_token_usage` 在 callable 未提供时为 `"unknown"`；
  `estimated_cost_usd` 恒为 `"unknown"`，不伪造美元花费。

## 6. 双层 Memory 写入闸门与召回校验

### 6.1 旧入口（legacy）闸门

`ImprovedSimulationAgent.workflow()` 保持课程兼容返回形状
`{"stars": float, "review": str}`，但新增 `quality_status`：

- JSON 星级**先做原始类型检查再交给 Pydantic**：`true` / `"4"` / `NaN` /
  `Infinity` 等非法 JSON 星级不会被强转成 success；review 文本尽量保留，
  评分回退为 3.0 并标记 `invalid_json_stars` → `degraded`；
- `success` 仅当结构化/标签解析完整、无注入、无重合告警、无评分 snap、
  无超长截断；
- `degraded` 覆盖：`legacy_parse_fallback`、`legacy_invalid_json_stars`、
  `legacy_stars_snapped`、`legacy_review_truncated`、
  `generated_injection_pattern`、`reference_overlap`；
- `failed`：异常路径，诊断只含安全类别与 `failure_stage`
  （`draft`/`reflection`），无异常文本/路径/traceback；日志只写固定类别，
  Pydantic 校验错误与解析异常消息**不再被记录**（可能含私有 marker）。

只有 `quality_status == "success"` 才写 `LocalMemoryStore`。
`ReasoningWithQualityAwareness` 在 draft/reflection 调用外层记录
`last_transport_failure` 并**原样抛出**，绝不吞掉异常返回错误文本；workflow
异常分支据此给出 `llm_transport_error` 诊断。`temporal_context.
run_temporal_ablation` 的第二次 generated memory 写入要求
`quality_status == "success"` **且**全部安全 flags 显式为 False
（fallback/snap/invalid/truncated/injection/leakage）、`review_origin ==
"model_output"`、review 必须是字符串、评分为整数网格 1..5；缺键即 fail-closed。
通用 API 错误文本因此不会传到下一个任务。

### 6.2 P1 入口闸门

`ReliableMemoryStore`：

- 新 run 新建 store；`begin_run(run_id)` 拒绝 run_id 不匹配、已 begin/end、
  或已有条目/写入尝试的**非全新 store**；run 结束后 `end_run` 标记；复用
  store 跨 run/config 会被直接拒绝，而不是仅靠文档约定；
- `remember_success` 只接受最终 `success` + `prediction_origin == llm` +
  非空 review + 合法有限评分 + `simulated is True`；同一 (task, execution)
  重复写入直接 `duplicate_write` 拒绝（repair 成功也只写一次）；
- 条目记录 origin、task_index、execution_order、event_time、provenance；
  每用户条数上限有硬上限 `MAX_MEMORY_ENTRIES_PER_USER_HARD = 64`（可更小）；
- `recall_before(user, as_of)` 按 **UTC 日历日**判定（P0 冻结日协议）：只有
  entry 的 UTC 日期**严格早于** target 的 UTC 日期才可召回；同一 UTC 日内
  即使有时刻先后也不建立顺序，带时区偏移的字符串先换算 UTC 再取日期；
  最新优先；
- **召回对象不被信任**：`render_task_prompt` / `run_reliable_task` 在渲染前
  调用 `validate_memory_selection`，逐条校验当前用户、UTC 日严格更早、
  可信 origin、run 绑定（提供 expected_run_id 时）、非空文本、合法评分、
  序列不重复；任何 forged/foreign recall 在产生 prompt 之前抛错（fail
  closed），文本不可能泄漏给别的用户/任务；
- 候选/时间合格/选择/实际 prompt 包含/写入门槛计数分别进入 run 报告；
- generated memory 文本与星级只用于表达一致性渲染，绝不进入统计回退
  的评分计算（`degraded` 只用实际渲染的真实证据行）。

## 7. 既有路径修复（P1 附带的安全边界）

- `improved_agent_with_quality.workflow`：完整安全诊断模板、
  `quality_status=failed`、`memory_stored=False`、无敏感文本日志；JSON 类型
  强转、超长截断、reflection transport 均按 6.1 处理。
- `temporal_context.run_temporal_ablation`：generated memory 写入加显式成功
  与安全 flags 闸门，记录 `generated_memory_write`。
- `comprehensive_evaluation.DeepSeekLLM`：`LLMTransportError`（仅安全 code），
  网络/解析/空 completion 都是 typed failure；不再返回 `"（API 错误）"` 假
  completion，也不打印异常详情；旧 runner 结构不变。
- 兼容边界：新 `reliable_agent` 不消费 legacy 输出；legacy 入口的新
  `quality_status` 只用于自身诊断与 temporal runner 闸门，不改变旧返回契约；
  真实 Simulator 路径保持 fail-closed。

## 8. 离线核验与 CLI

```powershell
# 7 个场景的合成契约 smoke（fake-only；失败非零；expected failure 只算 contract_passed）
.\.venv\Scripts\python.exe p1_smoke.py synthetic --output-dir <新目录>

# 本地已验收 P0 prepared 目录 + 确定性 fake（复用 P0 全部闸门，不联网）
.\.venv\Scripts\python.exe p1_smoke.py prepared `
  --prepared-dir Dataset\amazon_musical_instruments_p0\prepared_v2 `
  --output-dir <新目录> --max-tasks 12
```

语义与隐私：

- 输出目录必须新建，已存在即失败并拒绝覆盖；入口/启动异常会清理新建目录，
  而**契约检查失败时保留报告用于诊断**（两者语义不同，报告中 `gate_semantics`
  字段有说明）；
- synthetic 报告区分 `contract_passed_scenarios`（含预期失败场景）与
  `prediction_successes_total`（真实成功数），不把预期失败的通过当预测成功；
- prepared 是**非预期失败**的闸门：`checks.no_unexpected_failed_tasks` 要求
  `failed == 0` 并纳入 `contract_passed`，任何 transport/internal failed 都会让
  prepared CLI 非零退出；合法的 `refused`（证据不足/预算）与 `degraded` 不
  判失败，但会在 `prediction_coverage` 中单列计数与 `success_fraction`，
  **绝不把 refused/degraded 计入成功、绝不宣称 100% 成功**；该闸门只证明
  shape/gates，不证明准确率（`accuracy_claim: none`）；
- prepared 的引用核验按**每条 record 自己的 prompt**（由
  `diagnostics.rendered.prompt_sha256` 定位）检查引用 id 是否出现在该任务
  实际渲染的 prompt 中，不用全 run prompt union，避免另一任务或 repair
  输出串台；
- prepared 报告只含安全数字/常量/枚举/sha：状态计数、预算快照、memory 计数
  （含哈希后的 run id）、provider 校验链、目标文本 prompt 泄漏计数（仅计数）；
- `run_id` 在聚合报告中只出现 `run_id_sha256`；`model_source` 只接受固定枚举；
- `--save-records` 若保存模拟文本/明细，只允许写入 gitignored 的
  `results/` 或 `Dataset/`，否则非零退出；
- 调用 `p1_smoke.py prepared` 前必须存在真实 prepared 目录；测试中使用合成
  P0 fixture，不依赖本地真实数据。

## 9. 测试地图与主要反例

新增测试文件：

- `tests/p1_fixtures.py`：共享合成 provider 与脚本化 fake。
- `tests/test_reliable_agent.py`（62 个）：工具无查询参数面且不是沙箱、
  ToolPayload 条数/文本/字节上限与真实统计（含硬上限参数化测试）、full digest
  + 范围、registry 漏斗与 payload 一致、evidence_id 跨任务隔离与同文本区分、
  注入/空行过滤、渲染字节/条数/截断（只计实际渲染行）与未渲染引用拒绝、
  固定文本超预算拒绝、0 渲染行 refused 且不调用、严格校验参数化反例（布尔/
  NaN/Infinity/字符串/半星/越界/空/超长/缺字段/多字段/坏引用/散文包 JSON/
  尾随内容）、输出字节闸门、重复 JSON 字段、超深 JSON、超大整数、引用数量
  上限、预算/Render 硬上限、成功引用 provenance、伪造引用 repair 后 degraded、
  传输异常脱敏、repair_requested/attempted/reserve 失败区分、repair 成功仅写
  一次、无证据不调用、工具预算**先预留后读取**（spy 证明拒绝时零读取）、
  deadline 同样先检查、run 预算不重置且失败计 attempt、per-call output token
  上限、degraded 只用渲染证据且忽略 memory 星级（并校验报告的实际参数数值）、
  非默认 baseline 参数拒绝、memory 各写入闸门拒绝矩阵、UTC 日粒度召回（同日
  带小时拒绝/相邻 UTC 天合格/时区跨日转换/非 UTC 目标偏移）、用户隔离、
  每用户条数硬上限、foreign/同日/未来/origin/run 召回在渲染前拒绝、
  suite 拒绝 run_id 不匹配/复用/非全新 store、聚合报告 run id 哈希与枚举校验、
  可复现、四状态常量。
- `tests/test_p1_workflow.py`（27 个）：legacy 异常安全诊断与不写记忆、
  fallback/注入/重合不写记忆、snap 标记 degraded、bool/string/NaN 星级
  degraded 且不写、校验错误与异常日志不含私有 marker、超长截断 degraded
  不写、reflection transport failure 显式传播、legacy 成功仅写一次且返回
  契约不变、temporal runner 显式成功 + 安全 flags 闸门（通用错误文本、
  非字符串 review、缺 flags 均 fail-closed）、P0 prepared → 共享闸门 → P1
  fake 贯通、artifact gate 先于解析、prepared 缺工件/篡改先验拒绝、
  label perturbation 不进入 prompt、跨任务/同文本另一来源引用拒绝、
  CLI synthetic/prepared 通过与非零失败（含 unexpected transport 全失败
  prepared CLI 非零且报告保留）、prepared 每条 record 对照自身 prompt 的
  引用核验（手造串台反例证明 run-union 会漏检）、合法 refused 单列
  coverage 不误报 100%、CLI 记录目录闸门、prompt 不含目标文本、
  DeepSeekLLM 空 completion typed failure、gitignore 保持。

阅读顺序建议：先读 `docs/P0_IMPLEMENTATION_GUIDE.md`（数据与协议），再读
`reliable_agent.py` 模块 docstring → `BudgetSpec/BudgetLedger` →
`TaskEvidenceTool`/`ToolPayload`/`build_evidence_registry` →
`render_task_prompt`/`validate_memory_selection` → `validate_model_output` →
`ReliableMemoryStore` → `run_reliable_task` → `run_reliable_suite` →
`load_verified_prepared_dataset`，最后对照 `tests/test_reliable_agent.py`
的任一反例验证语义。

## 10. 尚未验证 / 明确边界

- 真实 LLM 的行为、准确率、成本、时延均未验证；本阶段只有 fake callable；
- 任何真实模型调用与原始评论发第三方都需要另行预算、许可与批准；
- 在线 provider 的 usage/计费字段未接入；`actual_token_usage` 在 fake 下为
  `unknown`；`prompt_tokens_estimate` 只是启发式估计，真实计费 unknown；
- deadline 不强制取消阻塞中的外部调用；只有调用前/调用间检查；
- 注入过滤与重合阈值是启发式，不是语义安全证明；工具对象不是进程内沙箱；
- P0 真实数据的 `unixReviewTime` 事件时间语义仍依赖调用方断言；
- P1 不改变 Simulator 的 fail-closed（`--temporal-manifest` 依旧拒绝真实路径）。
