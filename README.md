# LLM User Simulation Agent

## 上游框架与致谢

本项目直接依赖 [tsinghua-fib-lab/AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge) 发布的 PyPI 包 `websocietysimulator==1.0.0a30`（WWW'25 AgentSociety Challenge，MIT License，Copyright (c) 2024 FIB LAB, Tsinghua University）。上游框架提供模拟环境、Agent 基类、任务交互工具与框架评测接口。本项目通过 `requirements.txt` 以 pip 依赖方式使用该框架，**未复制（vendored）上游源码**。

本项目作者在上游之外的个人实现增量主要包括：用户画像与参考评论筛选、条件反思/修复、运行内 Memory，以及后续的本地离线数据协议、结果校验、预算与失败处理等可靠性工程。上游框架、示例、文档及评测接口的成果归其原作者所有，不应被表述为本项目的个人原创；本项目也不暗示获得上游作者或机构的认可或背书。上游框架采用 MIT License，这不改变 Amazon、Yelp、Goodreads 等数据集各自的许可与使用条件；本仓库不分发这些数据。

## 项目定位与当前状态（2026-10-14）

**项目 A：根据安全可见的用户历史，模拟个性化评分与评论。**

独立的[评测与诊断项目 B](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)负责评测这些预测，不再生成另一套评论。两项目通过版本化结果契约协作；契约入口正在开发，尚未独立验收。

```text
真实历史 → 时间安全上下文 → 画像/参考筛选 → 模拟评分与评论
                                       → 结果与运行诊断 → 项目 B
隔离的目标标签 ──────────────────────────────────────→ 末端评分器
```

**发布说明：本地升级代码、离线验收记录与 GitHub 已发布版本不是同一状态。** 未提交模块不代表远端 clone 后可运行；README 更新不等于代码发布。下方原课程与历史实现说明保留，不合并不同实验的成绩。

| 能力 | 已实现／已验证 | 尚未证明 |
| --- | --- | --- |
| AgentSociety 课程生成链路 | 保留画像、参考筛选、条件反思/修复和课程记录 | 当前真实 Simulator 的时序及返回顺序契约未核验 |
| Amazon 小数据闭环 | 本地核验乐器 10,261 条评论、400 个目标及四种统计基线 | 基线成绩不是 LLM 成绩 |
| 可靠生成路径 | 四态结果、实际渲染引用、有限修复、预算与成功写入有离线测试 | 引用正确不等于事实正确；真实模型行为待测 |
| Memory | 运行内、按用户隔离；可靠路径按 UTC 日召回更早生成结果 | 不跨进程持久化，不是用户真实反馈，不证明有收益 |
| Demo | 合成数据/fake-only；浏览器交互与移动端有独立验收记录 | 无真实 LLM 准确率、成本或延迟结论 |

当前定位是**有可靠性工程的用户模拟原型**，不是已经证明个性化效果的生产系统，也不是自主多 Agent。

### 怎么衡量，而不是凭一条示例判断

- **评分**：RMSE、MAE、±1 星准确率、预测减实际的平均有符号偏差；对比固定全局/用户/物品/组合基线。
- **评论**：相对安全历史的长度偏差、参考字符 n-gram 重合率；另做盲评的用户风格匹配、评分—评论一致性。
- **可靠性**：首次格式通过率、修复请求/实际调用/成功数、最终成功/降级/拒绝/失败比例、调用与耗时。
- **Memory**：同任务、同顺序、同设置比较有/无记忆；记录真正入 prompt 的条目及用户覆盖，按用户做配对 bootstrap。
- 分开报告成功预测、带统计回退的系统结果和全请求覆盖率；不能删掉失败任务后宣称整体改善。

完整实测汇总仍待补齐。旧课程汇总不能当新版成绩，fake 格式通过不能当真实准确率。

### 数据集、幻觉与记忆的学习入口

- [数据集怎么选、哪些尚未适配、许可边界](docs/DATASETS.md)
- [当前能力、指标定义、幻觉识别与记忆实现](docs/AGENT_CAPABILITIES.md)
- [逐步开发计划与验收标准](docs/DEVELOPMENT_PLAN.md)
- [P0 数据协议](docs/P0_IMPLEMENTATION_GUIDE.md) / [P1 可靠性机制](docs/P1_IMPLEMENTATION_GUIDE.md) / [合成 Demo](docs/P3_IMPLEMENTATION_GUIDE.md)

真实评论、标签、prompt、逐任务结果和实验报告不公开上传。真实模型调用、第三方数据传输、人工评审或付费服务需先确认数据使用范围与预算；模拟内容不能冒充真实用户发布。

---

## 原课程与历史实现说明（保留）

An LLM-based user simulation agent developed for the CS245 Agent Society
course project, Track A.

This repository is a clean, independent extraction of the personal agent and
evaluation work from the original course workspace. The upstream
`websocietysimulator` framework is used as a dependency and is not vendored
here.

## What It Does

The agent simulates a Yelp-style user who must produce a rating and review for
an item. It combines:

1. Historical user-review statistics for personalization.
2. Review-quality signals such as `useful`, `funny`, and `cool`.
3. Reference reviews for the target item.
4. Optional cross-task memory.
5. A two-stage LLM process: draft generation followed by reflection and
   revision.
6. Structured parsing and validation of the final rating and review.

The required output is:

```json
{
  "stars": 4.0,
  "review": "..."
}
```

## Architecture

```text
Task
  -> EnhancedPlanning
  -> user/item/review retrieval through the simulator tool
  -> hybrid reference-review ranking (engagement + lexical overlap + length)
  -> prompt-injection filtering
  -> UserProfileAnalyzer
  -> ReviewQualityAnalyzer
  -> prompt construction
  -> draft generation
  -> conditional reflection (only when the draft fails checks)
  -> structured JSON parsing with label-format fallback and validation
  -> leakage check against reference reviews
  -> structured output
```

### Personal Implementation

- `improved_agent_with_quality.py`
  - `EnhancedPlanning`
  - `UserProfileAnalyzer`
  - `ReviewQualityAnalyzer`
  - `ReasoningWithQualityAwareness` (conditional reflection)
  - `ImprovedSimulationAgent` (JSON output contract, hybrid retrieval,
    injection/leakage checks, no-context mode, optional local memory)
  - `AgentOutput` (Pydantic validation), `normalize_stars`
  - `DeterministicBaseline` / `DeterministicSimulationAgent`
- `local_memory.py`
  - bounded, in-process, user-scoped memory for within-run ablations
- `comprehensive_evaluation.py`
  - experiment configuration and CLI (`--dry-run`, `--seed`, `--experiment`,
    temperature, JSON mode and calibration options)
  - simulator execution and LLM usage tracking
  - fresh local memory store per memory-enabled experiment configuration
  - RMSE, MAE, accuracy, distribution and correlation metrics
  - per-task records and diagnostics, legacy random-split calibration (not temporal),
    paired bootstrap tests
  - comparison and ablation report generation
- `memory_audit.py`
  - offline repeated-user and review-leakage audit with redacted output
  - time-filtered split manifest (event-time semantics require owner confirmation)
    and optional prompt-trace inspection
  - per-task Memory candidate/recall/prompt-inclusion analysis
- `temporal_context.py`
  - manifest revalidation against current local task/review sources
  - fail-closed context provider and framework-free, serial ablation prototype
  - `No_Memory`, `Generated_Review_Memory`, `Trusted_History_Stats`, `Combined`
- `amazon_p0.py`
  - offline Amazon 2014 5-core adapter with an explicit, pinned, size-capped
    download command and strict bounded parsing (shared 64 MiB / 100k-row
    gates, no `eval`)
  - field mapping with locally stable pseudonyms, dropped non-verifiable
    metadata, rejection/duplicate audits and desensitized source policy
  - frozen-history day-cut task construction, repeated-user multi-task
    selection, paired task/groundtruth files and a canonical strict manifest
    consumed by the existing `TemporalContextProvider`
  - no-LLM runner producing aggregate redacted `report.json` / `report.md`
    plus local per-task prediction files in new, refuse-to-overwrite dirs
- `rating_baselines.py`
  - four deterministic label-safe baselines (global, shrunk user, shrunk
    item, fixed combined) that only accept provider-visible contexts and a
    verified global prior; prediction functions never receive groundtruth
    ratings, and actual stars are read only by the scorer after predictions
  - deterministic user-cluster bootstrap with shared resamples per baseline
    and paired differences against the global mean
- `score_calibration.py`
  - bias and linear score calibrators; explicit temporal validation/test helper
- `reliable_agent.py`
  - P1 reliable single-agent tool workflow over strict provider contexts
    (honestly a tool-augmented workflow, not an autonomous multi-agent
    system): executor-internal evidence tool with no query surface and
    bounded row/text/byte payloads, reserve-before-read tool budgets,
    task-scoped `evidence_id` registry, bounded structured rendering with a
    no-rendered-evidence refusal, strict output validation (byte/duplicate/
    depth/type/reference gates) with at most one repair, explicit
    `success`/`degraded`/`refused`/`failed` statuses, pre-call budget
    reservations with hard ceilings, sanitized typed failures, and a
    success-only generated-memory gate with recall validation and one-shot
    run binding
  - `load_verified_prepared_dataset` reuses every P0 gate (artifact size/row
    gates before parsing, policy validation, selection reconstruction,
    eligible-prefix prior re-verification, strict provider revalidation)
  - offline synthetic tests and injected fake callables only; no real-LLM
    accuracy is measured or claimed
- `p1_smoke.py`
  - offline P1 smoke CLI: `synthetic` runs the required contract scenarios
    (cross-date same-user memory, same-day isolation, user isolation,
    success/failure/repair/budget stop, injection filtering, insufficient
    evidence, render-gate refusal); `prepared` revalidates a local P0
    prepared directory and runs a deterministic fake callable through the
    shared gates
  - prepared is a non-expected-failure gate: any unexpected transport/
    internal `failed` task makes the contract fail and the CLI exit nonzero
    (the diagnostic report is kept), while legitimate `refused`/`degraded`
    outcomes are reported separately in `prediction_coverage` and are never
    counted as successes; citations are checked per record against that
    task's own rendered prompt, not a run-wide union
  - expected-failure scenarios count as `contract_passed`, never as
    prediction successes; failures exit nonzero; reports contain only safe
    numbers/enums/digests
- `inspect_agent_output.py`
  - interactive single-task inspection
  - predicted-vs-ground-truth comparison
  - generated review inspection
- `tests/`
  - 344 offline tests covering the agent, evaluation, calibration, temporal
    context prototype, the offline P0 Amazon rating baselines and the P1
    reliable tool workflow (with fakes only)
- `final_project.ipynb`
  - course experiment notebook and execution record
- `docs/SYNTHETIC_MEMORY_ABLATION.md`
  - deterministic and live repeated-user memory validation
- `ablation_results_*_clean.json`
  - archived Yelp, Amazon and Goodreads experiment snapshots

## Framework Boundary

The following framework is external to this repository:

- PyPI package: `websocietysimulator==1.0.0a30` (pinned in `requirements.txt`)
- Upstream repository:
  [tsinghua-fib-lab/AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)
- Original project: WWW'25 AgentSociety Challenge
- Upstream scope: simulation environment, agent base classes, task interaction
  tools and framework evaluation interfaces
- License: MIT for the framework code; attribution and copyright details are
  recorded in `THIRD_PARTY_NOTICES.md`

The framework is installed from PyPI as a dependency and is not vendored. The
original framework source, examples, tutorials, static assets, generated data
and course task files are intentionally not copied into this repository.
The framework's MIT license does not set the license terms for Amazon, Yelp or
Goodreads data, which are not distributed here.

## Archived Results

The original project README reported the following headline results on 300
tasks per dataset:

| Dataset | RMSE | MAE | Pearson | Overall quality |
| --- | ---: | ---: | ---: | ---: |
| Yelp | 0.968 | 0.745 | 0.691 | 82.8% |
| Amazon | 1.024 | 0.798 | 0.658 | 80.3% |
| Goodreads | 0.892 | 0.681 | 0.723 | 85.0% |

The JSON files in this repository are archived snapshots from a related
evaluation run and should be treated as reproducibility evidence, not as a
claim that every configuration can be reproduced without the original task
assets.

The historical run reported approximately 162 seconds for 300 tasks. The
current copied evaluation harness defaults to 10 tasks and 5 workers for a
small local smoke run.

## Setup

Python 3.10 or newer is recommended.

```bash
py -m venv .venv
\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Set the LLM key in the shell before running an experiment:

```bash
$env:DEEPSEEK_API_KEY = "your-key"
```

Do not commit `.env` files, API keys, datasets or generated results.

## Running Tests

The 344 offline tests cover the agent workflow (with fakes), profile
statistics, quality thresholds, hybrid reference ranking, injection and
leakage checks, conditional reflection, structured output, calibration,
per-task records, paired bootstrap tests, strict temporal contexts, the
offline P0 Amazon rating baselines and the P1 reliable tool workflow
(executor-fixed evidence scope, rendered-citation validation, strict result
statuses, budget reservations, memory write gates and the offline smoke
CLI). They require neither the simulator framework nor an API key:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

CI runs the same lint, tests and a `--dry-run` smoke check on every push
(`.github/workflows/ci.yml`).

## Data Layout

The task data is not included in this repository. The simulator harness
expects the original course/competition assets in a layout similar to:

```text
Dataset/
  item.json
  review.json
  user.json
example/
  track1/
    yelp/
      tasks/
      groundtruth/
    amazon/
      tasks/
      groundtruth/
    goodreads/
      tasks/
      groundtruth/
```

After installing the dependency and restoring authorized task assets:

```bash
# Validate the configuration without API calls or the simulator framework
py comprehensive_evaluation.py --dry-run

# Run the ablation suite (defaults to the yelp task set, 10 tasks, 5 workers)
py comprehensive_evaluation.py --task-set yelp --num-tasks 10 --max-workers 5
py comprehensive_evaluation.py --experiment Baseline Full --output-dir results

py inspect_agent_output.py
```

Every path, task count, worker count and model setting is now a CLI argument;
see `py comprehensive_evaluation.py --help`. `--dry-run` neither imports the
simulator framework nor calls the LLM, so it doubles as a no-network smoke
check. `inspect_agent_output.py` is preserved from the course experiment and
still assumes the original simulator task layout and an API key.

## Colab Quickstart

The course datasets are large, so Google Colab with the data mounted from
Google Drive is the recommended way to run full experiments.

```python
# 1. Mount the Drive folder that holds the course assets
from google.colab import drive
drive.mount("/content/drive")

# 2. Inspect the Drive layout and make the clone the working directory
COLAB_ROOT = "/content/drive/MyDrive/society agent"  # adjust to your folder
!ls "$COLAB_ROOT"
!find "$COLAB_ROOT" -maxdepth 3 -type d -name tasks
%cd $COLAB_ROOT/llm-user-simulation-agent

# 3. Install dependencies (the simulator framework is the heavy part)
!python -m pip install -q -r requirements.txt
!python -m pip install -q -r requirements-dev.txt

# 3b. Colab 2026 ships transformers 5.x / openai 2.x / langchain 1.x, which
#     break `import websocietysimulator`. Only these packages are needed by
#     the framework's import chain; chroma-based memory is optional and
#     would otherwise force numpy<2 on Colab:
!python -m pip install -q "lmdb<2.0,>=1.6.2" "langchain-openai<0.3,>=0.2.14" "langchain-core<0.4" "openai<2.0,>=1.58.1" "transformers>=4.47,<5"
!python -c "import websocietysimulator; print('framework OK')"

# 4. Validate offline before spending any API budget
!python -m pytest -q
!python comprehensive_evaluation.py --dry-run

# 5. Load the API key from Colab Secrets, never paste it into the notebook
import os
from google.colab import userdata
os.environ["DEEPSEEK_API_KEY"] = userdata.get("DEEPSEEK_API_KEY")
```

If the assets are not inside the cloned folder, pass absolute Drive paths
instead of relying on the defaults:

```bash
python comprehensive_evaluation.py \
  --data-dir "/content/drive/MyDrive/society agent/Dataset" \
  --task-dir "/content/drive/MyDrive/society agent/example/track1/yelp/tasks" \
  --groundtruth-dir "/content/drive/MyDrive/society agent/example/track1/yelp/groundtruth" \
  --task-set yelp --num-tasks 30 --max-workers 1 --seed 42 \
  --experiment Baseline No_Context Full No_Memory \
  --draft-temperature 0.4 --reflection-temperature 0.2 --json-mode \
  --calibration-method bias \
  --output-dir "/content/drive/MyDrive/society agent/results"
```

Notes:

- Prefer a GPU or high-RAM runtime: importing the framework loads TensorFlow,
  PyTorch and transformers, and the default CPU runtime can disconnect under
  that load. The environment setup itself is verified on the default runtime
  (344 offline tests plus `--dry-run` pass).
- Without `langchain`/`langchain-chroma` (step 3b), the optional framework
  `MemoryDILU` enhancement is skipped; this does **not** disable memory. The
  evaluation runner's `use_memory` configurations always use the dependency-free
  `LocalMemoryStore`: it recalls the same user's newest generated reviews before
  generation, writes the new review afterward, is bounded and user-scoped, and is
  discarded after each experiment.
- `--task-set` accepts `yelp`, `amazon` and `goodreads`; each needs its own
  `tasks/` and `groundtruth/` folders, and a processed data directory with
  `item.json`, `review.json` and `user.json` for that dataset.
- Run 30-50 paired tasks for each of `yelp`, `amazon` and `goodreads` before
  spending the budget on a full run.
- In the legacy Simulator path, `--max-workers 1` alone does not prove source-
  order submission/return or temporal safety. The new temporal prototype controls
  order itself and only exposes generated reviews from strictly earlier target
  timestamps; same-time tasks do not exchange generated memory.
- Keep `--max-workers` modest on Colab for general runs to avoid rate limits and
  memory pressure.
- Write `--output-dir` to Drive so results survive runtime resets. The runner
  saves metadata, per-experiment metrics and per-task JSON after every
  experiment, so an interrupted session keeps the results already produced;
  re-running starts a new run (there is no resume flag yet).

## Current Scope and Honest Limitations

Implemented and unit-tested offline:

- user profiling, review quality analysis, within-run local memory, conditional
  reflection and structured output parsing with label-format fallback;
- hybrid reference-review ranking (engagement + lexical overlap + length),
  prompt-injection filtering and generated-review leakage checks;
- a deterministic baseline and a no-context LLM baseline in the ablation suite;
- `score_calibration.py` with bias and linear calibrators;
- per-task error records and a paired bootstrap test replacing the old
  relative-difference helper;
- `memory_audit.py` for redacted repetition/leakage checks, optional exact
  prompt-trace inspection, time-filtered manifests, rating-bias intervals and
  per-task candidate/recall/prompt-inclusion diagnostics;
- per-run artifacts (`metadata.json`, per-task JSON, comparisons) and LLM
  usage counters;
- an offline P0 Amazon 2014 Musical Instruments pipeline (strict adapter,
  frozen-history tasks, strict provider consumption, four label-safe
  baselines, user-cluster bootstrap and redacted reports) with dedicated
  synthetic tests; a small real-file run was executed locally with only
  aggregate outputs, kept in gitignored directories;
- a P1 reliable single-agent tool workflow (`reliable_agent.py`) over the
  strict P0/temporal provider contexts: executor-internal evidence tool with
  no query surface and bounded payloads (every fixed read reserved before it
  runs), task-scoped evidence registry and rendered-citation validation,
  bounded structured rendering, strict validation with at most one repair,
  explicit `success`/`degraded`/`refused`/`failed` results, pre-call budget
  reservations with failed attempts counted and hard ceilings, sanitized
  typed LLM failures, and a success-only generated-memory gate with
  strictly-earlier event-time recall, per-entry recall validation and
  one-shot run binding; plus an offline `p1_smoke.py` CLI
  (`synthetic` / `prepared`, fake callables only, failures nonzero, redacted
  reports).

Honest limitations that remain:

- P1 is a tool-augmented single-agent workflow, not an autonomous multi-agent
  system; it has been exercised only with injected fake callables and
  synthetic/offline fixtures. No real-LLM behavioral, accuracy, latency or
  cost result exists, and any paid-model run or sending raw reviews to a
  third party needs separate budget and permission;
- provenance validation proves citation scope against the actually rendered
  registry, not semantic or factual correctness; injection filtering and the
  leakage threshold are bounded heuristics, not security guarantees;
- generated memory is style reference only: it is never used as a rating
  statistic, and a `degraded` statistical fallback is computed only from
  actually rendered real evidence rows and is never presented as model
  success;
- deadline handling checks before requests and between requests only; an
  in-flight blocking call is not force-cancelled. Online usage reporting is
  not wired, so `actual_token_usage` is `unknown` for fake callables and no
  dollar cost is ever estimated;
- the P0 Amazon dataset has no explicitly verified license
  (`license_status = not_explicitly_verified`, research use pending
  confirmation); its source file, tasks, labels and per-task outputs are not
  tracked or distributed in this repository - they remain local and
  gitignored and must not be redistributed;
- P0 event-time semantics rest on a caller assertion for the documented
  `unixReviewTime` field at UTC-day granularity; the target cohort is a
  conditional repeated-user sample after the frozen cut day, not a
  representative sample of the raw file, and its test split is not the raw
  file's last 20%; P0 baseline numbers are static-policy measurements, not
  Memory or LLM effects;
- the temporal context provider, four ablation contracts and time-filtered
  manifest have synthetic/offline tests only. The real Simulator path is not
  time-safe and is fail-closed for `--temporal-manifest` / `--temporal-ablation-mode`;
- the temporal prototype is deliberately fixture-sized: it rejects review
  sources above 64 MiB on disk or after decompression, or 100,000 parsed rows.
  A known 4.22 GB review file is outside its supported input range; no large-data
  usability is claimed. It streams records, keeps bounded task-relevant metadata, uses
  per-user/item indexes rather than rescanning the full corpus per task, and
  retains raw text only for selected context rows. Source bytes are parsed in a
  constant number of passes, not once per task;
- temporal manifest split boundaries keep identical target timestamps together;
  multi-record task/groundtruth files require unique matching task indexes,
  and the manifest includes a canonical parsed-record source fingerprint plus
  context-row event-time/text-digest/rating fingerprints without embedding raw
  reviews in the manifest;
- `Trusted_History_Stats` is an offline prototype and requires data-owner
  confirmation that the selected timestamps are event times; it uses only
  provider-filtered source reviews, never current or other groundtruth labels;
- the legacy `calculate_calibrated_metrics` uses a random task split and emits
  `time_safe: false`; temporal calibration is a separate explicit-split helper
  and is not connected to the Simulator runner;
- user-cluster bootstrap and time-filtered manifest construction have
  synthetic/offline tests but still need validation on authorized real task
  assets; the focused live synthetic run validates memory wiring, not model
  quality;
- the course task/groundtruth/processed-review data and the 2026 per-task run
  artifacts are not tracked or distributed in this repository; source
  leakage and target-time semantics
  therefore remain unverified here;
- session-provided 2026 paired aggregates showed actual Memory recall in 0/400
  Yelp, 0/400 Amazon and 2/400 Goodreads tasks; those aggregate differences are
  not evidence that Memory improved accuracy. See the audit report;
- reference selection is a transparent lexical hybrid, not embedding-based
  semantic retrieval, and no embedding endpoint is wired in;
- the archived Yelp/Amazon/Goodreads numbers come from the earlier course run
  and were not produced by this cleaned-up code path;
- per-task latency cannot be measured through the simulator API; only
  per-experiment wall time and aggregate LLM call latency are saved;
- local memory is intentionally in-process only: it does not persist across
  experiments or Colab runtime restarts and does not store groundtruth.

See `docs/PROJECT_SCOPE.md` for the ownership inventory, `docs/PROJECT_GUIDE.md`
for the full architecture, `docs/INTERVIEW_GUIDE.md` for talking points, and
`docs/MEMORY_EVALUATION_AUDIT_2026-09.md` for the no-API memory audit, data
limitations and next-step experiment protocol.

## Offline Memory and Leakage Audit

The audit CLI does not import the simulator, call an LLM, or print user IDs,
item IDs, comments, prompts or input paths. The temporal manifest path streams
JSON arrays/JSONL and retains only bounded, task-relevant metadata; before
parsing it rejects source files above 64 MiB (including decompressed `.gz`
content), and it stops at 100,000 records.
These are safety gates for a small offline prototype, not capacity estimates.
The separate `dataset` audit keeps its read-only aggregate-audit semantics.
`--prompt-file` accepts local prompt traces for exact-text checks; only counts
are emitted. Prompt traces and datasets remain sensitive local data and must
not be committed.

```powershell
# Repeated-user opportunity and source-corpus leakage evidence
py memory_audit.py dataset `
  --task-dir example/track1/goodreads/tasks `
  --groundtruth-dir example/track1/goodreads/groundtruth `
  --review-data Dataset/review.json `
  --output results/goodreads_memory_audit.json

# Time-filtered manifest: only strictly earlier, non-target real-review rows.
# Add the assertion flag only after a data owner confirms `date` is event time.
py memory_audit.py manifest `
  --task-dir example/track1/goodreads/tasks `
  --groundtruth-dir example/track1/goodreads/groundtruth `
  --review-data Dataset/review.json `
  --train-ratio 0.6 --validation-ratio 0.2 `
  --confirm-target-timestamp-semantics `
  --output results/temporal_manifest_goodreads.json
```

The assertion is not inferred from field names: `--confirm-target-timestamp-semantics`
records only that the caller claims a data-owner review. Missing/ambiguous target
rows, or target source rows without matching event times, are ineligible. A
well-paired zero-eligible manifest is a valid *audit-only* output that exposes
rejection reasons (`runtime_eligible: false`); missing sources, conflicting
task/groundtruth identity, or unreliable alignment make the manifest CLI exit
nonzero. `TemporalContextProvider` rejects zero tasks, empty review sources and
zero eligible contexts. It revalidates a canonical fingerprint of every parsed
review record and streams the source again while retaining only the selected
raw context rows. Per-task visibility uses per-user/item indexes, not repeated
full-corpus scans. Prompt provenance is recorded while the actual history and
reference sections are rendered; a row with no uniquely tracked source index is
reported as unknown rather than inferred from a text match elsewhere in the
prompt. This remains limited to 64 MiB / 100,000-row synthetic or small offline
fixtures and does not support the known 4.22 GB review asset. The external
Simulator adapter remains unverified; requesting temporal mode from
`comprehensive_evaluation.py` exits before creating results or importing the
framework. See the audit report for strict prerequisites and remaining gates.

## Offline P0 Amazon Rating Baselines (No LLM)

`amazon_p0.py` builds a field-traceable, time-safe, label-isolated offline
task set from the public Amazon 2014 Musical Instruments 5-core file, and
`rating_baselines.py` scores four deterministic rating baselines against it.
The pipeline is fully offline apart from the explicitly invoked `download`
subcommand; it makes no LLM or Simulator calls and keeps every real review,
task, label and per-task result inside gitignored `Dataset/` directories.

Design essentials (details in `docs/P0_IMPLEMENTATION_GUIDE.md`):

- strict bounded parsing of `.jsonl/.json/.gz` with the shared 64 MiB
  (including decompressed) / 100,000-row gates, no `eval`, per-record caps,
  explicit rejection/duplicate audits and locally pseudonymized identifiers;
- a frozen observed-history prefix cut at a day boundary, and target
  selection from repeated users with at least two distinct UTC dates each
  (never one task per user; budgets that cannot satisfy the contract fail
  closed), with a runtime review source of frozen history plus selected
  target rows only;
- the global prior is computed from the eligible frozen prefix only: selected
  target ids and matchable target-text copies are masked out, prefix/excluded/
  eligible counts are recorded, and the runner recomputes and verifies the
  prior and the recorded selection from the validated artifacts;
- the existing strict `TemporalContextProvider` withholds every selected
  target (all splits) and duplicate/matching target texts from every context;
  the canonical manifest partitions the selected target cohort only, not the
  raw file;
- global/user/item/fixed-combined baselines with documented shrinkage
  (`lambda = 5`, `weight_user = 0.5`), explicit cold-start fallback counts and
  continuous 1..5 predictions; prediction functions never receive groundtruth
  ratings; groundtruth is parsed for verification/exclusion, while the scorer
  uses target ratings only after predictions are complete;
- deterministic user-cluster bootstrap with one shared resample per
  iteration for every baseline, paired differences against the global mean,
  and no fabricated intervals for empty or single-cluster tests;
- run-time artifact size/row gates before any parsing, bounded readers,
  policy parameter/protocol validation with report fields rebuilt from fixed
  safe constants, and aggregate `report.json` / `report.md` with no
  identifiers, review text, unvalidated free text or paths, written to new
  refuse-to-overwrite directories;
- the `download` subcommand accepts only the exact pinned official HTTPS
  archive, bounds `--max-bytes`, fully validates the gzip stream (CRC,
  truncation, decompression cap) before an atomic publish, and never
  overwrites existing files.

The dataset page was not found to state an explicit data license, so
`license_status` is recorded as `not_explicitly_verified` (research use
pending confirmation), and local data is not redistributed. Reported
baseline numbers are what they are - static offline rating policies - and
are not Memory or LLM results.

```powershell
# Explicit download of only the pinned official archive (gitignored path);
# every later command is offline.
py amazon_p0.py download `
  --url https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz `
  --output Dataset\amazon_musical_instruments_p0\raw\reviews_Musical_Instruments_5.json.gz

py amazon_p0.py prepare `
  --source Dataset\amazon_musical_instruments_p0\raw\reviews_Musical_Instruments_5.json.gz `
  --work-dir Dataset\amazon_musical_instruments_p0\prepared_v2 `
  --history-ratio 0.6 --max-tasks 400 --seed 42 `
  --confirm-target-timestamp-semantics

py amazon_p0.py run `
  --prepared-dir Dataset\amazon_musical_instruments_p0\prepared_v2 `
  --output-dir Dataset\amazon_musical_instruments_p0\baseline_v2 `
  --n-boot 2000 --seed 42
```

## Offline P1 Reliable Tool Workflow (Fake LLM Only)

`reliable_agent.py` turns the strict P0/temporal provider contexts into a
single-agent, tool-augmented workflow with an explicit contract: verified
evidence -> structured prediction -> strict validation and at most one repair -> explicit status -> one memory write only on validated success. It reuses the
existing P0 gates through `load_verified_prepared_dataset` and never modifies
the P0 artifacts.

Design essentials (details in `docs/P1_IMPLEMENTATION_GUIDE.md`):

- the executor-internal `TaskEvidenceTool` has no caller-settable
  `user`/`item`/`as_of`/`path` queries and is **not** an external permission
  sandbox: it is an in-process object for one fixed task whose getters return
  bounded `ToolPayload`s (row/text/byte caps, newest-first history sampling,
  full-source digest and character range preserved on truncated text). Every
  fixed tool read is reserved atomically before any getter runs, so an
  unaffordable or expired budget reads nothing. Evidence rows get task-scoped
  `evidence_id`s tied to kind, source row and text digest, so cross-task refs
  and same-text other-source refs cannot pass;
- prompts are rendered in bounded structured form (fixed trusted
  instructions, JSON-escaped untrusted evidence lines, per-section entry/text
  and total byte caps); citations are validated against the **actually
  rendered** registry, never inferred by substring search; a task whose
  registered rows cannot be rendered at all is refused
  (`no_rendered_evidence`) before any LLM call;
- strict validation byte-gates the raw completion before parsing and rejects
  booleans, NaN/Infinity, huge-integer overflow, off-grid/out-of-range stars,
  duplicate JSON keys, deeply nested JSON, too many or unknown refs, empty or
  overlong reviews, wrong/extra fields and non-JSON output; at most one
  repair is sent (`repair_requested`/`repair_attempted`/reserve failure are
  reported separately), and an unrepairable draft with visible evidence
  degrades to a labeled statistical baseline computed only from rendered real
  evidence rows;
- budgets are reserved per call against per-task and whole-run caps with
  explicit hard ceilings (failed attempts count, run caps never reset per
  task, a requested per-call output token cap is checked before any attempt,
  repairs only with remaining budget), and typed `LLMTransportError` failures
  never leak exception text, paths or keys;
- the generated-memory store writes only final validated model successes
  once, records origin/order/event-time/provenance, validates every recalled
  entry (current user, strictly earlier event time, trusted origin, run
  binding) before rendering, and refuses a store that is not fresh or not
  bound to the exact run id; aggregate reports hash the caller run label and
  accept only a fixed `model_source` enum. Generated text/stars are style
  reference only, never rating statistics.

```powershell
# Synthetic contract scenarios with scripted fakes (7 scenarios; expected
# failures count as contract_passed, never as prediction successes).
.\.venv\Scripts\python.exe p1_smoke.py synthetic --output-dir results\p1_smoke_run

# Local P0 prepared directory through every shared gate + deterministic fake
# (offline; no API key; report has safe aggregates only, accuracy_claim=none).
.\.venv\Scripts\python.exe p1_smoke.py prepared `
  --prepared-dir Dataset\amazon_musical_instruments_p0\prepared_v2 `
  --output-dir results\p1_prepared_smoke --max-tasks 12
```

Expected-failure scenarios and contract checks keep the smoke honest: the
synthetic expected-failure scenarios count as contract passes but never as
prediction successes; the prepared mode treats any unexpected
transport/internal `failed` task as a contract failure and exits nonzero
(keeping the report for diagnosis), while legitimate `refused`/`degraded`
outcomes are counted separately in `prediction_coverage` and are never
reported as successes. No accuracy, Memory-gain or real-LLM claim is
produced.

## Attribution

This project was developed for the CS245 Agent Society course project, Track
A. It builds on the MIT-licensed AgentSociety simulation framework. See
`THIRD_PARTY_NOTICES.md` for upstream attribution.

## License

The original work in this repository is released under the MIT License. See
`LICENSE`.
