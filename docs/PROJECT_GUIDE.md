# 项目构造指南

本文件面向"全面理解这个仓库"：结构、数据流、模块职责、产物格式、运行方式、
测试策略、已修 bug、诚实边界和扩展点。代码与注释为英文，说明为中文。

## 1. 项目是什么

- CS245 Agent Society Track A 的 LLM 用户模拟 Agent。
- 输入：用户历史评论、商家信息、目标商家的参考评论。
- 输出契约：`{"stars": 1.0–5.0 的整数星, "review": "评论文本"}`。
- 运行在 AgentSociety 模拟框架（`websocietysimulator==1.0.0a30`）之上：
  框架负责任务加载、交互工具（`get_user` / `get_item` / `get_reviews`）与
  模拟调度；本仓库只包含个人 Agent、评测、校准与测试代码。

## 2. 仓库目录

```text
llm-user-simulation-agent/
├─ improved_agent_with_quality.py   # Agent 主实现：校验、画像、质量、推理、解析、基线
├─ local_memory.py                   # 单次实验内、按 user_id 隔离的本地 Memory
├─ comprehensive_evaluation.py      # 实验运行器 + 指标 + 配对 bootstrap + CLI
├─ score_calibration.py             # 评分校准（bias / linear）
├─ inspect_agent_output.py          # 单任务人工观察工具（未 CLI 化）
├─ final_project.ipynb              # 课程实验 Notebook（历史执行记录，勿改）
├─ ablation_results_*_clean.json    # Yelp / Amazon / Goodreads 历史结果快照
├─ tests/                           # 130 个离线测试（不装框架、不联网）
│  ├─ fakes.py                      # FakeLLM / FakeInteractionTool / make_review
│  ├─ test_agent_workflow_offline.py# workflow 端到端
│  ├─ test_agent_offline.py         # 画像、质量阈值、参考选择、解析与归一
│  ├─ test_structured_output.py     # JSON 契约、Pydantic、反思策略
│  ├─ test_agent_components.py      # 混合检索、注入/泄漏、确定性 baseline
│  ├─ test_score_calibration.py     # 校准与验证集切分
│  ├─ test_statistics.py            # 逐任务记录、paired bootstrap
│  ├─ test_run_artifacts.py         # run 目录、metadata、逐任务持久化
│  ├─ test_evaluation_config.py     # CLI 默认值、实验列表、dry-run
│  ├─ test_llm_client.py            # seed/JSON payload、usage 统计
│  └─ test_synthetic_memory.py      # 重复用户 Memory 消融
├─ docs/
│  ├─ PROJECT_SCOPE.md              # 归属边界与扩展路线图
│  ├─ DEVLOG.md                     # 逐步开发记录（学习/讲解材料）
│  ├─ PROJECT_GUIDE.md              # 本文件
│  ├─ SYNTHETIC_MEMORY_ABLATION.md  # 重复用户 Memory 验证记录
│  └─ INTERVIEW_GUIDE.md            # 面试问答材料
├─ requirements.txt                 # 运行时依赖（框架、numpy、requests、pydantic）
├─ requirements-dev.txt             # 离线测试所需（不含框架）
├─ pytest.ini                       # testpaths / pythonpath / basetemp 在项目内
├─ ruff.toml                        # lint 配置（排除 .venv、.pytest_tmp、notebook）
├─ .github/workflows/ci.yml         # CI：ruff + pytest + dry-run smoke
└─ .env.example / README.md / LICENSE / THIRD_PARTY_NOTICES.md
```

## 3. 单任务数据流（Agent）

```text
Simulator 注入 task={user_id,item_id} 与 interaction_tool
        │
        ▼
EnhancedPlanning（固定 3 步计划）
        │  步骤匹配：user → get_user；review → get_reviews(item)；business/item → get_item
        ▼
include_context == True ?
        │是                                        │否（No_Context baseline）
        ▼                                          ▼
get_reviews(user) → UserProfileAnalyzer      build_minimal_prompt
extract_query_terms(business_info)
get_relevant_reviews（混合排序）
looks_like_prompt_injection 过滤参考评论
ReviewQualityAnalyzer（useful/funny/cool 示例）
MemoryDILU（可选）
build_prompt
        │
        ▼
ReasoningWithQualityAwareness
  ① draft（默认 temperature 0.4，要求 JSON）
  ② draft_needs_reflection(draft)?
       ok → 直接返回初稿；否则 reflection（默认 temperature 0.2）
        │
        ▼
parse_review_result_with_status
  JSON → AgentOutput(Pydantic) → 失败则标签格式 → 再失败默认值（记录 used_fallback）
        │
        ▼
后处理：512 截断 → 生成结果注入检查 → 参考评论 n-gram 泄漏检查
        │
        ▼
{"stars": float, "review": str} + last_diagnostics
```

## 4. 模块详解

### 4.1 improved_agent_with_quality.py

**数值与解析基础**

| 名称 | 作用 |
| --- | --- |
| `VALID_STAR_VALUES` | `(1.0, 2.0, 3.0, 4.0, 5.0)` |
| `normalize_stars(x)` | half-up 归一到 1–5 星（`floor(x+0.5)` + clamp），改动时 warning |
| `STARS_PATTERN` / `REVIEW_PATTERN` | 定位 `stars:/星级:` 与 `review:/评论:`，容忍 Markdown 加粗 |
| `AgentOutput` | Pydantic 输出契约：`stars` 自动归一、`review` 非空 |
| `extract_json_payload` | 从纯 JSON、```json 代码块、散文嵌入中提取第一个 JSON 对象 |

**检索（T6）**

- `extract_query_terms(business_info)`：从 name/categories/city/state 提取长度 ≥2 的词项。
- `score_reference_review(review, query_terms)`：
  `0.5*参与度 + 0.3*词项重合 + 0.2*长度`；参与度 = `log1p(useful+funny+cool)` 饱和到 50。
- `ImprovedSimulationAgent.get_relevant_reviews(reviews, top_k, query_terms)`：
  长评论优先（≥3 条时只用长评论），按分数降序；同分保持原顺序（稳定排序）。

**安全（T10）**

- `looks_like_prompt_injection(text)`：中英文指令注入短语检测。
- `reference_overlap_ratio(review, refs, ngram=8)` / `check_output_leakage`：
  生成评论与参考评论的字符 n-gram 重合率，≥0.5 告警。

**推理（T9）**

- `draft_needs_reflection(draft)`：初稿为空 / 无 stars / 非整星 / 空评论 / 短于 40 字时返回
  `(True, reason)`，否则 `(False, "ok")`。
- `ReasoningWithQualityAwareness(reflection_decider=...)`：
  draft 一次；decider 说 ok 就跳过第二次调用，`stats` 记录
  `draft_calls / reflection_calls / reflection_skipped`。

**Agent**

- `ImprovedSimulationAgent(llm, enable_reflection, use_memory, max_reference_reviews, include_context, memory_store, memory_limit)`。
- `workflow()`：见第 3 节；结束后写入 `last_diagnostics`
  （`used_parse_fallback`、`skipped_injection_reviews`、`injection_warning`、
  `leakage_warning`、`reflection_stats`）。
- `memory_store` 只保存本次运行生成的评论，按 `user_id` 隔离；`ExperimentRunner`
  为每个启用 memory 的实验配置创建新的 store，不写磁盘也不保存 groundtruth。
- `parse_review_result_with_status(result)`：返回 `(stars, review, used_fallback)`。
- `parse_review_result(result)`：兼容旧调用，返回 `(stars, review)`。

**基线（T12）**

- `DeterministicBaseline.predict(reviews_user)`：历史平均星，归一；空历史返回 3.0。
- `DeterministicSimulationAgent`：框架兼容壳，`workflow()` 直接返回基线结果。

### 4.2 comprehensive_evaluation.py

**LLM 客户端**

- `DeepSeekLLM(api_key, base_url, chat_model, embedding_model, seed)`：
  - `_build_payload()`：seed 非空时写入 API payload。
  - `usage_stats()`：`calls / errors / latency_seconds`（T13）。
- `DeepSeekEmbeddingModel`：历史遗留，当前无调用点。

**指标与统计（T11）**

- `extract_prediction(output)`：兼容 `{"output": {...}}` 与平铺 `{"stars": ...}`。
- `build_per_task_records(outputs, groundtruths)`：逐任务 `predicted/actual/error/squared_error`。
- `calculate_additional_metrics()`：exact、±0.5、±1.0、均值/方差、Pearson。
- `paired_bootstrap_test(records_a, records_b, metric="error", n_boot, seed)`：
  配对差值 `mean(a)-mean(b)`、95% 百分位 CI、双侧 p 值
  （`p = min(1, 2*min(P(boot>=0), P(boot<=0)))`）。

**实验运行（T4/T12）**

- `ExperimentConfig(name, ..., include_context, agent_kind)`。
- `ExperimentRunner(data_dir, task_set, api_key, num_tasks, max_workers, output_dir,
  chat_model, base_url, seed)`：
  - 每次运行创建 `run_<timestamp>` 目录；
  - 每个 memory-enabled 配置创建一个新的 `LocalMemoryStore`，并共享给该配置的所有任务 Agent；
  - `per_task_records` 内存中保存逐任务记录；
  - `_extract_groundtruths` 兼容三种框架属性名；
  - `_save_run_metadata` / `_save_per_task_records` / `_save_intermediate_results`；
  - `compare(name_a, name_b)` 调用 paired bootstrap。
- `ResultsAnalyzer`：Markdown 对比表、消融分析、最佳配置。

**CLI（T3/T12）**

- `default_experiments()`：`Deterministic`、`Baseline`、`No_Context`、`Full`、
  `No_Reflection`、`No_Memory`、`Fewer_References`。
- `build_parser()`：`--data-dir/--task-set/--num-tasks/--max-workers/--seed/
  --api-key/--chat-model/--base-url/--output-dir/--experiment/--dry-run`，以及
  temperature、JSON mode 和 held-out calibration 参数。
- `main(argv=None)`：返回退出码；dry-run 不导入框架、不调 API、不建目录。

### 4.3 score_calibration.py（T8）

- `ScoreCalibrator(method="bias"|"linear")`：`fit` / `calibrate` / `calibrate_many` /
  `to_dict` / `from_dict`；输出自动归一 + clamp。
- `split_pairs(predictions, actuals, validation_ratio, seed)`：确定性切分。
- `rmse(predictions, actuals)`：便于比较校准前后。

### 4.4 tests/

- `fakes.py` 提供 `FakeLLM`（队列响应、记录调用）与 `FakeInteractionTool`。
- 全部测试满足两个约束：不 import `websocietysimulator`（缺失时走 stub）、不联网。

## 5. 产物格式

一次真实运行后：

```text
results/run_YYYYmmdd_HHMMSS/
├─ metadata.json              # run_id、数据集、任务数、workers、模型、seed、实验列表
├─ per_task_<Config>.json     # [{index, predicted, actual, error, squared_error}, ...]
├─ results_<Config>.json      # 单实验指标 + config + llm_usage
├─ experiment_report.md       # 对比表 / 消融 / 最佳配置
├─ comparisons.json           # Full vs Deterministic / Baseline 的 bootstrap 结果
└─ all_results_<timestamp>.json
```

`comparisons.json` 条目字段：`metric, n_tasks, mean_difference, ci95_low,
ci95_high, p_value, n_boot`。

## 6. 运行手册

```bash
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 离线验证
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe comprehensive_evaluation.py --dry-run

# 真实实验（需要 API key 与数据集资产）
$env:DEEPSEEK_API_KEY = "..."
.\.venv\Scripts\python.exe comprehensive_evaluation.py --task-set yelp `
    --num-tasks 100 --max-workers 5 --seed 42 --experiment Full Baseline
```

校准模块的离线用法：

```python
from score_calibration import ScoreCalibrator, split_pairs, rmse

train, validation = split_pairs(predictions, actuals, validation_ratio=0.5, seed=1)
calibrator = ScoreCalibrator(method="bias").fit(*train)
raw = rmse(*validation)
calibrated = rmse(calibrator.calibrate_many(validation[0]), validation[1])
print(raw, calibrated)
```

## 7. 测试策略

- 红-绿流程：先写回归测试证明 bug，再修代码（`docs/DEVLOG.md` 有完整记录）。
- 离线原则：LLM 与交互工具都用 fake；`--dry-run` 走 subprocess 做无网络 smoke。
- `pytest.ini` 把 basetemp 指到项目内 `.pytest_tmp/`，避免写系统临时目录。
- CI 在每次 push 跑 lint + 130 个测试 + dry-run。

## 8. 已修复的真实 Bug

| Bug | 影响 | 修复 |
| --- | --- | --- |
| 多行 review 只保留首行 | 生成文本被截断、评测文本质量失真 | 正则定位标签后取全部剩余文本 |
| `round()` 银行家舍入 | `4.5→4.0`、`2.5→2.0` 静默改变预测 | half-up + 越界 clamp + warning |
| 空 review 通过 | 空字符串进入结果 | `if not review_text` 回退默认文本 |
| 计划第 3 步被误判为 business | 重复 `get_item`、评论抓取落空 | 分支顺序 user → review → business/item |
| `calculate_statistical_significance` 只是相对差异 | 无法支撑显著性结论 | 替换为逐任务 paired bootstrap |
| 裸 `except`、无占位符 f-string、未用导入 | lint/维护性问题 | ruff 全量清理 |

## 9. 诚实边界

- calibration 与 bootstrap 只有离线/合成数据验证，**没有**在真实任务资产上验证过。
- 参考评论检索是"参与度 + 词项重合 + 长度"的透明混合排序，**不是**语义 embedding RAG。
- 历史 Yelp/Amazon/Goodreads 数字来自旧代码路径，不能声称由当前清洗后的代码复现。
- 逐任务延迟拿不到（框架 API 无此信息）；只有实验总时长和 LLM 调用累计延迟。
- 没有数据资产与新的 API key 时，只能验证离线链路，不能产生新的实验数字。

## 10. 扩展点

- **新数据集**：`--task-set` + 按 README 的 Data Layout 放置 tasks/groundtruth。
- **新指标**：在 `calculate_additional_metrics` 增加聚合，或在 `paired_bootstrap_test`
  上用 `squared_error` 做 RMSE 级比较。
- **语义检索**：为 `get_relevant_reviews` 增加可选 `embedding_model` 参数，
  在 `score_reference_review` 之外叠加余弦相似度；当前刻意未接入。
- **新 Agent 变体**：继承 `ImprovedSimulationAgent`，或给 `ExperimentConfig.agent_kind`
  增加分支；`DeterministicSimulationAgent` 就是示例。
