# Development Log

本文件记录每一步开发的「目标 → 概念 → 代码 → 验收」，用于复习、讲解和面试准备。
代码与注释保持英文，说明使用中文。

## Step 2 — 修复 `parse_review_result`（多行丢行 / 静默舍入）

### 1. 目标

| 问题 | 修复前现象 | 修复后行为 |
| --- | --- | --- |
| 多行 review 丢行 | `stars: 4.0\nreview: A\nB\nC` 只得到 `A` | 得到 `A\nB\nC` |
| `review:` 后正文换行 | 返回空字符串 | 返回下一行开始的完整正文 |
| 非整星静默舍入 | `4.5 → 4.0`、`2.5 → 2.0`，无日志 | half-up：`4.5 → 5.0`、`2.5 → 3.0`，并记录 warning |
| 越界评分 | `9 → 5.0` 但无日志 | `9 → 5.0` 且日志记录原始值 |
| 空 review | 返回 `""` | 回退到默认文本并告警 |

### 2. 概念

#### 2.1 旧解析为什么丢行

旧实现逐行扫描，每行独立判断。遇到 `review:` 时只取**本行**冒号后的内容：

```python
elif 'review:' in line_lower or '评论:' in line_lower:
    parts = line_stripped.split(':', 1)
    if len(parts) >= 2:
        review_text = parts[1].strip()
```

下一行的正文不会再被写入 `review_text`，所以多行评论只剩首行。

#### 2.2 正则版解析思路

不再“按行取值”，而是“按标签位置取值”：

- `STARS_PATTERN.search(result)`：在全文定位第一个 `stars:` / `星级:` 标签，抓取标签后的数字。
- `REVIEW_PATTERN.search(result)`：只负责定位 `review:` / `评论:` 标签的位置。
- 正文 = `result[review_match.end():].strip()`，即**标签之后的全部文本**，换行因此天然保留。

同时用 `\**` 容忍模型常见的 Markdown 加粗写法（如 `stars: **4.5**`）。

#### 2.3 银行家舍入 vs half-up

Python 内置 `round()` 使用 banker's rounding（四舍六入五成双）：

```python
round(4.5)  # 4  ← 期望 5
round(2.5)  # 2  ← 期望 3
```

对评分任务来说，这会静默把接近正确的预测系统性拉低。改用 half-up 再 clamp 到 `[1, 5]`：

```python
snapped = float(min(5, max(1, math.floor(stars + 0.5))))
```

覆盖关系：`4.5 → 5.0`、`2.5 → 3.0`、`3.7 → 4.0`、`0.4 → 1.0`、`9.0 → 5.0`。

#### 2.4 附带修复

- `if not review_text` 同时覆盖 `None` 和 `""`，空评论不再静默通过。
- 无占位符的 f-string 日志改为 %-style（原代码 `f"无法解析stars..."` 没有插值，属于误用）。
- 越界/非整星值记录原始值：`stars 4.50 is not on the 1-5 grid, snapped to 5.0`。

### 3. 代码切片

新增模块级常量与纯函数（`improved_agent_with_quality.py`）：

```python
VALID_STAR_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)

STARS_PATTERN = re.compile(
    r'(?:stars?|星级)\s*:\s*\**\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
REVIEW_PATTERN = re.compile(r'(?:review|评论)\s*\**\s*:', re.IGNORECASE)


def normalize_stars(stars: float) -> float:
    if stars in VALID_STAR_VALUES:
        return stars

    snapped = float(min(5, max(1, math.floor(stars + 0.5))))
    logging.warning(
        "stars %.2f is not on the 1-5 grid, snapped to %.1f", stars, snapped
    )
    return snapped
```

解析主体（Before → After 核心差异）：

```python
# Before: 逐行 split，只取本行冒号后的内容
for line in lines:
    if 'review:' in line_lower:
        review_text = line_stripped.split(':', 1)[1].strip()

# After: 定位标签位置，标签之后全部内容都是正文
review_match = REVIEW_PATTERN.search(result)
review_text = result[review_match.end():].strip() if review_match else None
```

### 4. 验收

```text
Red   （修改前新增回归测试）: 6 failed, 22 passed
Green （修改后全量测试）    : 28 passed in 0.06s
```

实测原问题用例：

```text
input : "stars: 4.5\nreview: Great food.\nThe service was slow.\nWould return."
before: (4.0, 'Great food.')
after : (5.0, 'Great food.\nThe service was slow.\nWould return.')
log   : stars 4.50 is not on the 1-5 grid, snapped to 5.0
```

### 5. 文件清单

| 文件 | 改动 |
| --- | --- |
| `improved_agent_with_quality.py` | 新增 `math` 导入、`VALID_STAR_VALUES`、两个正则和 `normalize_stars`；重写 `parse_review_result` |
| `tests/test_agent_offline.py` | 新增 6 个回归用例（多行、正文换行、冒号保留、空正文、half-up、告警） |
| `docs/DEVLOG.md` | 本文件 |

### 6. 面试可讲的一句话

> 输出解析按标签定位而不是逐行截取，所以多行评论不会丢；评分归一采用确定性
> half-up 并记录告警，避免用 Python 默认的银行家舍入静默改变预测值。

## Step 3 — 评测 CLI 化与离线 dry-run

### 1. 目标

| 问题 | 修复前 | 修复后 |
| --- | --- | --- |
| 路径/任务数/workers 硬编码在 `main()` | 改参数必须动代码 | CLI 参数 |
| 没有 dry-run | 只能真跑（需 API key + 框架 + 数据） | `--dry-run` 零依赖校验 |
| 顶层 `import matplotlib/seaborn` 未使用 | 拖慢导入、增加依赖 | 已删除 |
| 顶层 `from websocietysimulator import Simulator` | 没装框架时 `--help` 都跑不了 | 懒加载，仅执行实验时导入 |
| 结果散落在仓库根目录 | `results_*.json` 与报告混在根目录 | 统一写入 `--output-dir` |

### 2. 概念

**2.1 懒加载（lazy import）**

把 `from websocietysimulator import Simulator` 从模块顶层移进 `run_experiment()`。
Python 在函数被调用时才执行该 import，所以只做参数校验的 `--dry-run` 不需要安装框架。

```python
def run_experiment(self, config: ExperimentConfig) -> Dict:
    # Imported lazily so --dry-run works without the simulator framework.
    from websocietysimulator import Simulator
    from improved_agent_with_quality import ImprovedSimulationAgent
```

**2.2 CLI 单一入口**

`build_parser()` 返回 `argparse.ArgumentParser`；`main(argv=None)` 接收参数列表。
这样测试可以直接 `main(["--dry-run", ...])`，不必启动子进程。

**2.3 实验选择与校验**

`--experiment Baseline No_Memory` 先与 `default_experiments()` 的名字求差集，
未知名字通过 `parser.error()` 以退出码 2 结束，避免拼错名字后静默跑错配置。

```python
known = {config.name for config in experiments}
unknown = [name for name in args.experiment if name not in known]
if unknown:
    parser.error(...)
```

**2.4 退出码约定**

| 退出码 | 含义 |
| --- | --- |
| 0 | 成功（含 dry-run） |
| 2 | 参数错误或缺 API key |

### 3. 验收

```text
$ python comprehensive_evaluation.py --dry-run --task-set amazon --num-tasks 3 --experiment Baseline No_Memory
DRY RUN - no LLM calls will be made
  data_dir    : Dataset
  task_set    : amazon
  num_tasks   : 3
  ...
  api_key     : missing
  experiments : 2
    - Baseline: Baseline: reflection=False, memory=False, refs=3
    - No_Memory: No_Memory: reflection=True, memory=False, refs=5
```

- 无框架、无 API key、无数据集即可执行
- 不创建 `results/` 目录（dry-run 无副作用）
- `pytest -q`: 34 passed（新增 6 个 CLI 测试，其中 `test_cli_dry_run_runs_as_script`
  通过 subprocess 端到端验证脚本可作为无网络 smoke 使用）

### 4. 文件清单

| 文件 | 改动 |
| --- | --- |
| `comprehensive_evaluation.py` | 删除未用导入；`ExperimentRunner` 接收 `max_workers/output_dir/chat_model/base_url`；懒加载框架；`default_experiments()`、`build_parser()`、`print_dry_run()`、`main(argv)` |
| `tests/test_evaluation_config.py` | 新增 6 个 CLI 离线测试 |
| `requirements-dev.txt` | 增加 `numpy`、`requests`（仅为离线导入评测模块） |
| `pytest.ini` | `--basetemp=.pytest_tmp`，测试临时文件留在项目内（E 盘） |
| `README.md` / `docs/PROJECT_SCOPE.md` | 同步 CLI 用法与进度 |

### 5. 面试可讲的一句话

> 评测入口是 CLI 化的：路径、任务数、worker 和模型都可配置，`--dry-run`
> 不导入模拟器框架也不调用 API，因此可以在 CI 或无网络环境做冒烟校验；
> 框架依赖被延迟到真正执行实验时才加载。

## Step 4 — Agent 加固与评测科学性（T4–T14）

一次迭代完成 11 个任务，全部有离线测试（`pytest -q`：107 passed）。

| 任务 | 改动 | 关键文件 |
| --- | --- | --- |
| T5 | FakeLLM/FakeInteractionTool 端到端 workflow 测试；修复计划第 3 步被误判为 business 导致重复 `get_item` 的 bug | `tests/fakes.py`、`tests/test_agent_workflow_offline.py` |
| T7 | JSON 输出契约 + Pydantic 校验（`AgentOutput`），JSON 解析失败回退标签格式 | `improved_agent_with_quality.py` |
| T6 | 参考评论混合排序：`0.5*参与度 + 0.3*词项重合 + 0.2*长度`，同分保持稳定顺序 | 同上 |
| T10 | 注入短语过滤（跳过可疑参考评论）+ 生成结果的 n-gram 泄漏检查 | 同上 |
| T9 | 条件反思：初稿缺分/空评论/过短/非整星才触发第二次调用，统计跳过次数 | 同上 |
| T12 | `Deterministic` / `No_Context` 两种 baseline；`--seed` 贯通 API payload 与运行元数据 | 同上 + 评测 |
| T8 | `ScoreCalibrator`（bias/linear）+ 验证集切分 + RMSE 对比 | `score_calibration.py` |
| T11 | `build_per_task_records` + `paired_bootstrap_test` 替换相对差异，输出 95% CI 与 p 值 | `comprehensive_evaluation.py` |
| T4 | run 目录（`results/run_<id>/`）：metadata、逐任务 JSON、report、comparisons | 同上 |
| T13 | LLM 调用数/错误数/累计延迟统计，写入实验元数据 | 同上 |
| T14 | ruff lint + GitHub Actions CI（lint + pytest + dry-run smoke） | `ruff.toml`、`.github/workflows/ci.yml` |

新增测试文件：`test_structured_output.py`、`test_agent_components.py`、
`test_score_calibration.py`、`test_statistics.py`、`test_run_artifacts.py`、
`test_llm_client.py`、`test_agent_workflow_offline.py`。

面试可讲的一句话：

> 我把"能跑"的课程代码补成了"可验证"的工程：输出有 JSON schema、检索有透明
> 排序、反思按需触发、评测有逐任务误差和配对 bootstrap，而且这些路径全部能在
> 无框架、无 API key 的离线测试里验证。
