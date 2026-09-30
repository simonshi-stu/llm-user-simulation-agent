# 用户模拟 Agent：Memory 评测审计与下一阶段结果

日期：2026-09-29

## 摘要

本阶段交付的是可离线复核的审计工具、严格时间切分清单、运行时记忆诊断、评分偏差统计和合成测试；**没有运行 LLM/API 评测，也没有用合成数据代替真实规模或真实效果**。

当前能支持的核心结论是：2026-09-26 的配对评测中，Full 每任务都写入了生成评论，但 Yelp/Amazon 没有任务真正召回，Goodreads 仅有 2/400 个任务召回。因而该轮 Full 与 No_Memory 的整体差异不能被解释为“Memory 提升准确率”。更早的本地 300 任务汇总是另一批历史结果，不能与本轮 400 任务做改动归因。

## 数据与证据盘点

### 当前本地可见内容

- 当前 checkout 不含课程 `tasks/`、`groundtruth/`、处理后的 `user/item/review` 数据，也没有 2026-09-26 的逐任务结果、诊断或 prompt trace。
- 仓库只找到三份已跟踪的旧汇总 JSON：Yelp、Amazon、Goodreads 各 300 条，payload 时间戳为 `20251127_072955`。它们不含逐任务索引、Memory 召回记录或配对误差；用途仅是历史快照。
- 2026-09-26 Colab/Drive 逐任务产物在此 Windows 环境不可访问。以下该轮数字来自本次任务提供的观测摘要，没有在本地逐例复算。
- 指定的桌面故事线 Word 文档在给定位置不可访问（文件不存在）；虽已确认本机安装 `python-docx`，但没有读取或改写该文档。
- 本环境没有安装 `websocietysimulator`，所以不能在这里检查上游真实任务加载/调度实现。

**因此，真实数据规模之外的重复率、两个 Goodreads 召回任务的索引/上下文/预测误差、时间字段可信度、目标/未来评论是否实际进入 Simulator 返回值，均仍待核查。**没有据此作数据泄漏已发生或已排除的断言。

### 代码已确认与尚待数据/框架确认

代码可确认的事实：`ImprovedSimulationAgent` 生成前按 user ID 从本地 store 召回，生成后写入 Agent 自己生成的评论和星级；store 最多保留 8 条、按用户隔离。同一用户之后的任务才有机会读取。Memory 提示文字标明“仅作风格参考”，这不是反馈标签。普通用户历史与画像照常出现在 prompt 中；MemoryDILU 是可选增强，不是 LocalMemoryStore 的开关。

Agent 会把 `get_reviews(user_id)` 的返回内容用于画像并选取用户历史例子，也会从 `get_reviews(item_id)` 选目标对象参考评论。仓库没有 Simulator 的查询实现，因此只有在确认 adapter 返回了目标/未来行后，才能把“源数据中有该行”提升为“该行实际进入 prompt”。源码路径说明了**条件性泄漏面**，不单独证明本轮实际泄漏。

## 2026-09-26 已报告观测（非本地重算）

设置：Full 与 No_Memory 对同索引有效任务配对，每个数据集 400 个；seed=42、max-workers=1、JSON mode；六个数据集×配置运行单元成功、API errors=0。Full 每集写入 400 次。下表的指标按任务提供的观测摘要记录：

| 数据集 | Memory 实际召回 | Full / No_Memory RMSE | Full / No_Memory MAE | Full / No_Memory ±0.5 | Full / No_Memory Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yelp | 0 / 400 | 0.977241 / 0.978519 | 0.695 / 0.6925 | 0.4325 / 0.4375 | 0.7104 / 0.7086 |
| Amazon | 0 / 400 | 0.869267 / 0.873570 | 0.58625 / 0.59375 | 0.4925 / 0.4850 | 0.56794 / 0.56115 |
| Goodreads | 2 / 400 | 1.007472 / 1.017349 | 0.705 / 0.710 | 0.4425 / 0.4425 | 0.43244 / 0.42157 |

当轮 Full 的预测均值/真值均值差分别是 Yelp `+0.455`、Amazon `+0.42125`、Goodreads `+0.58`，提示整体有正向评分偏差。没有逐任务数据时，不能给出按真实星级分组的偏差、区间或重复用户区间。

Yelp/Amazon 的 0 召回意味着这两组整体指标差异不是“跨任务生成评论记忆被读取”的证据。Goodreads 只有 2 个任务实际召回，也不足以把 400 任务总体 RMSE 差归因为记忆。模型非确定性、提示差异、调度/任务对齐等均需逐任务配对核对。更早的 300 条归档数值不得与本轮合并或用于证明这次工程改动有效。

## 本阶段工程交付

### 1. `memory_audit.py`：无 LLM、隐私安全审计

- `dataset`：自然排序配对 tasks/groundtruth，统计全部可用任务中的用户首次/复现率、串行顺序下的潜在召回机会；可扫描 JSON 数组、JSONL 和常见压缩文件。处理 review 数据时逐行解析，并只在内存保留与任务用户/目标物品有关的字段、时间戳及文本摘要，不输出原文、ID 或输入路径。
- 源数据泄漏证据分开报告：目标评论文本是否出现在同用户/目标物品的处理后评论源行、同用户/目标物品是否存在目标时点或未来的行。源行命中不等同于实际 Simulator 返回或 prompt 纳入。
- 可用 `--prompt-file` 检查本地 prompt trace：优先用任务文件中唯一的 source task index 关联；若 trace 的 `task_index` 实际是文件顺序而不是 source index，必须确认后加 `--prompt-index-matches-task-order`。无索引但条数吻合的 prompt 列表会明确标为顺序假设。工具核对目标评论、未来用户评论、未来目标物品评论的精确文本，只输出计数；不能识别改写、截断或未提供的数据。
- `run`：汇总每任务 task index/实际执行顺序、候选数、实际召回数、写入前已存在的 memory sequence、真正渲染到 prompt 的 sequence、Full/No_Memory 数值预测与误差。默认不把 source task index、执行序和 per-task 记录位置互相推定；只有在核验 source-index/record-index 映射或 Simulator 输出顺序后才用显式 CLI 确认开关关联。Full/No_Memory 配对 bootstrap 另需确认两份 per-task index 是同一批目标任务。对老诊断缺字段时输出 `unknown`，不按列表位置偷偷配对。

CLI 不保存任何私有输入；`--output` 只写脱敏汇总。输入 trace 本身仍可能含用户数据，必须留在本地/Drive，不应提交或分享。

### 2. 串行运行诊断

新的逐任务诊断记录 0-based `execution_order`（workflow 开始次序），以及 Simulator 若提供的 `task_index`；用本次进程内 HMAC 伪名代替原始 user/item ID；另记 task fingerprint、首现/复现分层、memory 候选数、召回数和 sequence、被召回 memory 的来源执行序、实际 prompt 纳入数/sequence、prompt SHA-256。**不保存原始评论或 prompt。**HMAC key 不写入 run metadata，哈希只用于同一 runner 进程内配对，不保证跨运行可关联。来源执行序还能标记并行执行下“较晚启动任务先完成、被较早任务读到”的未来记忆竞态。

这解决了“enabled/write 不等于读取有效”的诊断歧义。max-workers=1 的 FakeSimulator 测试证明本仓库 runner 在一个明确串行 fake 调度中按调用顺序写入并读取；这**没有证明**未安装的真实 Simulator 在 max-workers=1 时必然按源任务顺序提交/返回。未来真实运行必须检查 `execution_order`、`memory_sequence` 和 source `task_index`；若缺少 task index，审计器不会把诊断顺序当成 groundtruth 顺序。

### 3. 严格时间清单与校准 helper

`build_temporal_manifest()` 为每个目标任务要求用户、物品、可解析目标时点及**唯一匹配的目标源评论行**；用户历史与目标物品参考只能使用严格早于目标时点的行。无法唯一匹配目标源行的任务不进入严格清单。所有能从 groundtruth/任务匹配到的目标交互都会从**所有任务**上下文剔除，避免较早测试目标被误用作后续真实反馈；同时间戳也不可见。目标评论文本摘要还做第二层排除检查。每任务清单包含 train/validation/test 时间序、首现/复现层、上下文源行序号/数量和哈希，不包含真值星级、原始 ID 或评论。

限制：通用别名（`date`、`timestamp` 等）只是候选字段，无法从字段名证明它代表真实评论发生时间。manifest 明示 `target_timestamp_semantics_verified=false`，需数据拥有者确认语义；遇到缺失用户/物品/时间或匹配时间冲突会把任务排除并统计原因。对无可信时间的旧数据，严格方案是禁用用户评论历史、由全量评论派生的画像统计和目标物品评论，只保留确定在预测前可见的静态输入；仅“留一条目标评论”不够排除未来交互。

`score_calibration.evaluate_temporal_calibration()` 只用 manifest 的 validation 行拟合预先选定的校准器，train 不用于拟合，test 只做一次末端评估；测试证明更改测试标签不影响拟合参数。现有综合 runner 的旧校准开关仍使用随机 train/validation 切分，尚未接入时间 manifest，也没有本地真实集成验证。

### 4. 分数诊断和配对不确定性

- 新增 `rating_diagnostics()`：真值星级分组 signed bias/MAE/RMSE、预测 1–5 星计数、均值与 bootstrap 95% 区间；first-seen/repeat 汇总仅使用与评分行显式对齐的诊断标签，不从评分记录索引推断出现顺序。runner 写入整体及真值分组统计，但暂不把诊断匿名用户组映射到评分行；只有 `memory_audit run` 在显式核验 index 映射后才计算用户分层，避免假设 Simulator 的返回顺序。
- `paired_bootstrap_test()` 现在检查已存在的 task index/fingerprint，且支持按 user cluster 重采样。默认仍是 task-level bootstrap；runner 在没有可靠逐任务匿名用户映射时不会声称 cluster CI。
- 需要严格评测时，两条件使用完全相同的 task manifest、顺序、seed、模型设置和截至时点上下文；对 Full vs No_Memory 的差值做配对分析，重复用户则以用户为 bootstrap cluster。零实际召回的样本须单独标注，不能以开关名称解释因果。

评分准确性与评论风格/质量是不同终点：分数使用 bias、MAE/RMSE、分布校准和置信区间；风格/质量应另行盲评语言、长度、语气一致性、具体性、无依据事实和多样性。不能用更接近星级的 RMSE 代替评论质量。

## 离线复现命令

在已恢复且获准使用的数据目录下运行；示例不会调用 API：

```powershell
# 源数据/复现率/泄漏候选审计。输出只有计数和证据状态。
& .\.venv\Scripts\python.exe memory_audit.py dataset `
  --task-dir example/track1/goodreads/tasks `
  --groundtruth-dir example/track1/goodreads/groundtruth `
  --review-data Dataset/review.json `
  --output results/goodreads_memory_audit.json

# 若有本地 prompt trace（task_index + prompt），也检查真实 prompt 纳入。
# task_index 若对应任务文件中的 source index，无需额外开关；若它只是
# natural file order，只有核实后一致才加 --prompt-index-matches-task-order。
& .\.venv\Scripts\python.exe memory_audit.py dataset `
  --task-dir example/track1/goodreads/tasks `
  --groundtruth-dir example/track1/goodreads/groundtruth `
  --review-data Dataset/review.json `
  --prompt-file $env:LOCAL_PROMPT_TRACE `
  --output results/goodreads_prompt_audit.json

# 生成可用于所有消融共享的严格时间任务/上下文清单；不启动模型。
& .\.venv\Scripts\python.exe memory_audit.py manifest `
  --task-dir example/track1/goodreads/tasks `
  --groundtruth-dir example/track1/goodreads/groundtruth `
  --review-data Dataset/review.json `
  --train-ratio 0.6 --validation-ratio 0.2 `
  --output results/temporal_manifest_goodreads.json

# 本地逐任务 run 产物的隐私安全汇总。替换成实际 run 目录中的文件。
& .\.venv\Scripts\python.exe memory_audit.py run `
  --full-diagnostics results/run_x/per_task_diagnostics_Full.json `
  --full-records results/run_x/per_task_Full.json `
  --no-memory-diagnostics results/run_x/per_task_diagnostics_No_Memory.json `
  --no-memory-records results/run_x/per_task_No_Memory.json `
  --output results/goodreads_memory_cases.json

# 只有在核验同一 index 对应相同目标任务后，才加 --paired-task-indexes-confirmed。
# 只有在核验 source task index == per-task record index 后，才加
# --source-task-index-matches-record-index；或者独立确认输出顺序后用
# --serial-output-order-confirmed。否则保持默认 unknown / not_computed。
```

`results/` 与额外的本地审计/逐任务文件模式已加入 `.gitignore`。不要把数据集、prompt trace、逐任务误差或生成结果复制到 tracked 文件。

## 尚未完成与执行顺序

1. **阻塞：恢复获准的本地/Drive 资产。**需要 tasks、groundtruth、处理后 review 数据（含 user/item/review ID、stars、可信事件时间）、本轮 Full/No_Memory 逐任务结果与诊断；逐例 Goodreads 还需可证明实际 prompt 内容的本地 trace，或用新诊断重跑小样本。当前 Drive 产物不可访问。
2. 运行 `dataset` 审计，先审核 task/groundtruth 文件自然排序是否与实际加载顺序一致，核验 timestamp 字段含义、任务索引、数据切分和提示词工具视图；检查 target/future 精确匹配及 prompt trace。
3. 安装项目指定 Simulator 后，以确定性 fake 和一个明确小样本核对源 task index、workflow `execution_order`、返回输出顺序、groundtruth 序和 memory sequence；不得仅因 `max_workers=1` 推定这些序相同。
4. 如果时间字段可信，人工审核 temporal manifest 的 eligible/blocked 数量和上下文行，再把同一 manifest 接到每个实验配置。若无可信时间，使用不含评论历史/目标评论参考的严格替代上下文，避免声称时间安全的个性化效果。
5. 先运行 Goodreads 实际召回案例的离线逐任务对照，再讨论任何在线试验；本会话没有启动新的 API 实验。任何付费/在线扩大样本实验需另行确认预算、模型与 API 权限。
6. 注册四种预先声明的消融：无跨任务记忆；仅先前生成评论（无真值反馈）；仅截至时点可信真实历史统计；真实历史统计 + 生成评论。所有处理共用相同任务、顺序、seed、任务可见上下文及模型设置。验证集拟合/选定校准后，仅在保留测试集报告一次；同时给整体与 first-seen/repeat 分层指标、user-cluster paired bootstrap。

## 面试可讲的真实故事

> 我先把跨任务 Memory 接进了 Agent，并用写入次数和小型重复用户 fake 确认它能工作。全面配对结果出来后，我没有把微小 RMSE 差异包装成效果：逐任务诊断显示 Yelp/Amazon 实际召回为零，Goodreads 也只有 2/400。于是我把问题从“Memory 是否提升评分”重新拆成“候选→实际读取→prompt 纳入→同任务误差”，补上无原文的执行序号/匿名分组诊断、泄漏审计和时间安全 manifest，并单独检查了评分正偏差。结论是机制 wiring 已有合成测试，真实收益与 Goodreads 两例归因仍待本地逐任务产物和可信时间数据验证。

不要说“Memory 已提升准确率”或“已证明无泄漏”。现在可验证的交付是评测可证伪性与复现基础，而不是性能胜利。

## 离线验证记录

本阶段交付验收（含三项回归测试）：`pytest -q` **150 passed**；`ruff check .` 通过；`git diff --check` 通过；`comprehensive_evaluation.py --dry-run --task-set goodreads --num-tasks 2 --max-workers 1 --experiment Full No_Memory` 通过。上述检查不需要 Simulator、网络、API key 或真实数据。Luna 阶段还对修改的 Python 模块执行过 `compileall`。
