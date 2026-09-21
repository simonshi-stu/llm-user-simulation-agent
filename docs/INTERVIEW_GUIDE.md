# 面试指南（CS245 Agent Society · Track A）

本文件用于面试准备：项目介绍、贡献边界、高频追问与诚实回答、数字口径。
原则：**只讲代码里真实存在的东西；未验证的一律标明"已实现但未验证"。**

## 1. 一分钟项目介绍

> 这是一个基于 AgentSociety 模拟框架的 LLM 用户模拟 Agent：给定用户的历史评论、
> 商家信息和该商家的参考评论，让 LLM 模拟这位用户生成星级和评论文本。我实现了
> 用户画像、评论质量分析、条件反思、结构化输出和参考评论混合检索，并搭建了评测
> 框架：逐任务误差、配对 bootstrap 显著性检验、确定性 baseline 和 no-context
> baseline。所有核心路径都有离线测试，不装框架、不联网也能验证。

## 2. 贡献边界（主动说清楚）

| 属于我的 | 不属于我的 |
| --- | --- |
| `improved_agent_with_quality.py` 的四个模块与输出链路 | `websocietysimulator` 上游框架 |
| `comprehensive_evaluation.py` 实验与统计 | 课程数据管线 `data_process.py` |
| `score_calibration.py` 校准模块 | 原始数据集、任务文件、groundtruth |
| `tests/`、CLI、dry-run、CI 配置 | Notebook 中调用的框架教程代码 |
| 三个数据集的历史结果快照 | 框架自带的 planning/reasoning 基类 |

一句话：**在框架之上扩展 Agent，而不是重新造模拟器。**

## 3. 高频追问与回答

### Q1 这跟"直接让大模型写一条评论"有什么区别？
五个差异：①结构化用户画像（评分分布、长度、useful/funny/cool 倾向）；
②参考评论的质量与相关性筛选；③两阶段生成（初稿 + 条件反思）；④Pydantic
输出契约与兜底解析；⑤可重复的评测闭环（baseline、逐任务误差、bootstrap）。

### Q2 用户画像怎么构建？
统计历史评分均值/评论长度/星级分布，阈值判定评分倾向（乐观/中立/挑剔）、
评论风格（详细/简洁）、engagement_style（informative/entertaining/insightful/
straightforward），渲染成结构化 prompt。冷启动（无历史）返回"新用户"默认画像。

### Q3 Reflection 为什么后来又做成条件化？
初稿常有缺评分、空评论、过短、非整星的问题，二次调用能修复格式与质量。
条件化只在初稿不合格时触发，避免每次都付双倍 token。离线测试验证了
"好初稿只调用 1 次、坏初稿调用 2 次"；真实节省比例尚未在真实数据上测量。

### Q4 输出解析为什么 JSON + Pydantic，还保留标签兜底？
JSON 契约 + Pydantic 校验从根上消灭"自由文本解析"的脆弱性；但模型偶尔不守
格式，因此保留 `stars:/review:` 标签兜底和默认值，并用 `used_parse_fallback`
记录回退率，避免评测中断又保持可观测。

### Q5 参考评论是 RAG 吗？
不是。当前是透明混合排序：`0.5*参与度(log1p(useful+funny+cool)) +
0.3*与商家信息的词项重合 + 0.2*长度适配`，同分保持稳定顺序。语义 embedding
检索是明确的扩展点，但因为仓库里没有可用的 embedding 端点，我没有把未验证的
功能说成已有。

### Q6 校准模块具体做什么？有效果吗？
`ScoreCalibrator` 支持两种映射：bias（加平均偏差）和 linear（最小二乘
`intercept + slope*raw`），先用 `split_pairs` 切验证集，再用 RMSE 比较校准前后。
**诚实口径**：模块与离线测试已完成；因为缺少真实模型输出，尚未在真实数据上
验证提升幅度。

### Q7 统计检验怎么做？为什么不用 t 检验？
保存每个任务的绝对误差，`paired_bootstrap_test` 对配对差值做 10000 次重采样，
输出均值差、95% 百分位 CI 和双侧 p 值（`p = min(1, 2*min(P(boot≥0), P(boot≤0)))`）。
任务样本非正态、方差未知，配对 bootstrap 不依赖正态假设，而且与逐任务记录
天然兼容。原来的"相对差异"函数已删除。

### Q8 数据泄漏和 prompt 注入怎么防？
参考评论是其他用户对同一商家的评论，模型可能照抄：用字符 8-gram 重合率检查，
超过 0.5 告警。参考评论与生成结果都会做中英文指令注入短语检测，可疑参考评论
直接跳过并在 `last_diagnostics.skipped_injection_reviews` 计数。

### Q9 成本与延迟？
每个任务 1 次 LLM 调用（初稿通过检查）或 2 次（触发反思）。`DeepSeekLLM`
统计调用数、错误数、累计延迟，写入实验结果。历史记录 300 任务约 162 秒是旧
代码路径的数据；当前仓库没有重新测量。

### Q10 300 任务的结果现在能复现吗？
不能直接复现：课程任务资产与有效 API key 都不在仓库里。当前能验证的是离线
链路：107 个测试、dry-run、无 API 的 workflow 端到端。要复现需要恢复数据资产
并配置新的 key，然后按 README 的命令跑。

### Q11 如果继续做，下一步是什么？
①恢复数据 + key，跑通 300 任务并记录逐任务结果；②在真实输出上验证校准与
bootstrap；③为检索加 embedding 相似度做 A/B；④把条件反思的触发策略与
成本-质量帕累托曲线做出来；⑤用成对任务集重跑全部消融。

### Q12 这个项目最大的工程教训？
不要相信"能跑过"的自由文本解析。多行丢失和银行家舍入两个 bug 都会静默改变
预测值，直接污染评测指标；测试先行的价值就在这里。

## 4. 数字口径（别说错）

| 说法 | 正确表述 |
| --- | --- |
| 提交数 | 不要提"183 次提交"，那是上游历史；个人贡献是 Agent、评测、Notebook、结果 |
| 历史结果 | Yelp/Amazon/Goodreads 各 300 任务（RMSE 0.968/1.024/0.892），来自旧课程代码，本仓库未复现 |
| 耗时 | 历史约 162 秒/300 任务，旧环境记录 |
| 当前验证 | 107 个离线测试通过、ruff 无告警、CI 已配置（未观察远端运行） |
| 消融名称 | 当前代码是 `Deterministic / Baseline / No_Context / Full / No_Reflection / No_Memory / Fewer_References`；历史 JSON 的 calibration 字段与当前代码不符，不要引用 |

## 5. 可现场演示

```bash
.\.venv\Scripts\ruff.exe check .          # 秒级
.\.venv\Scripts\python.exe -m pytest -q   # ~1.5s，107 passed
.\.venv\Scripts\python.exe comprehensive_evaluation.py --dry-run
```

离线 workflow 演示：`tests/test_agent_workflow_offline.py` 用 FakeLLM 与
FakeInteractionTool 完整跑通 `workflow()`，并断言 `last_diagnostics`。

## 6. 三个可讲的故事（STAR）

1. **解析器 bug**：发现多行 review 丢行与银行家舍入 → 先写 6 个失败测试
   （红）→ 正则定位 + half-up 归一 → 28 个测试全绿。
2. **评测科学性**：原 `calculate_statistical_significance` 只是相对差异 →
   改为逐任务记录 + 配对 bootstrap，并保留 CI 与 p 值。
3. **可复现性**：没有框架/数据/key 时无法验证 → CLI 化、懒加载、fakes、
   dry-run、CI，让核心路径在没有外部依赖时也能验证。
