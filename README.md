# LLM 用户模拟 Agent

[简体中文](README.md) | [English](README.en.md)

这是一个研究原型：根据某个用户过去的行为，模拟他面对一件新商品时可能给出的星级和评论。它只读取该用户更早的历史，生成结果后交给独立的评测项目检查，而不是让生成代码自己给自己打分。

它不是用来冒充真人发评论的机器人，也不声称知道用户真实的想法。

## 0. 引用与致谢

- **[AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety)**（清华大学 FIB Lab）。本项目明确致谢并沿用其 LLM 社会模拟研究语境；本仓库没有复制或内嵌 AgentSociety 本身。
- **[AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)**（清华大学 FIB Lab，MIT License）。早期 Agent A 课程链路调用的直接框架：通过 PyPI 安装 `websocietysimulator==1.0.0a30`，使用其模拟环境、Agent 基类、任务交互工具和框架评测接口。
- **Amazon Product Data（2014）Musical Instruments 5-core 评论**，由加州大学圣地亚哥分校 Julian McAuley 团队维护，数据入口在[这里](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。本仓库不分发原始评论、目标标签或私有运行文件。“网上可以下载”不等于可以自由再分发或商用。
- **[项目 B：Amazon 评论评测与诊断](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)**。同一项工作的独立评测侧。

以上个人和机构提供了框架代码、数据或托管能力，但没有参与本项目的设计，也不为项目结论背书。

在上游框架之外，本项目增加的内容包括用户画像、参考评论筛选、条件反思／修复、运行内 Memory，以及后续的时间安全、证据检查、预算和失败处理。上游代码和示例仍归原作者所有。

## 1. 这个项目有什么用？为什么需要它？

一段模拟评论可以读起来很自然，但星级完全错了；也可能大段复制参考评论、编造购买经历，或者跑到一半失败。只看一条示例，或者只看一个准确率数字，这些区别都会被盖住。

这个项目想把用户模拟变成可以检查的东西：

- **个性化预测。** 给定用户的安全历史和目标商品，预测他可能给出的评分与评论。
- **冷启动和历史长度研究。** 比较历史很长、很短、甚至没有可用历史的用户，而不是假设所有人都一样。
- **离线评测。** 为实验生成模拟反馈，不必每次找真人重新写评论。
- **把结果干净地交给评测端。** 导出版本化结果，让项目 B 在不读取生成代码的情况下检查和评分。

所有输出都会标记为模拟内容。它不是真实顾客的意见，不能当作真实评论发布。

## 2. 怎么使用？整个流程是什么样？

一次典型运行是这样的：

```text
安全的用户历史 + 目标商品
        ↓
画像统计与参考评论筛选
        ↓
只包含实际渲染证据、长度受限的 Prompt
        ↓
LLM 初稿 -> 校验 -> 必要时最多修复一次
        ↓
成功 / 降级 / 拒绝 / 失败
        ↓
评分、评论与运行诊断
        ↓
项目 B：独立评分与诊断
```

关键在于模型**拿不到**什么：未来评论、目标评分、其他用户的历史。启用 Memory 时，也只召回同一用户更早生成的记录，并且只作为表达风格参考，不能当作可信评分标签。

这套设置可以回答这些问题：

- 面对一件新商品，这个用户会比平时打得更高还是更低？
- 换一个 Prompt 或检索方式，在同一批任务上有没有变好？
- Memory 真的进入 Prompt 后有没有帮助，还是只增加了成本？
- Agent 能不能泛化到没见过的用户？

## 3. 项目目前做到哪一步？

**本次公开快照已包含**

- 保留 AgentSocietyChallenge 链路：用户画像、参考评论排序、条件反思／修复，以及课程评测入口。
- 发布 Agent A 的可靠生成侧代码：`improved_agent_with_quality.py`、`temporal_context.py`、`comprehensive_evaluation.py`、`local_memory.py` 和隐私审计工具。
- 发布 Amazon 乐器源适配与时间安全 runner：`amazon_p0.py`、`amazon_full_evaluation.py`、`rating_baselines.py`。支持旧/新字段和秒、毫秒、微秒、纳秒时间戳，并把失败记录归为明确状态。
- 在 Drive/Colab 原始源上完成 source-only dry-run：解析 `3,017,439` 行，有效 `3,017,381` 行，`58` 行因 `invalid_timestamp` 明确拒绝；有效日期为 `2001-09-17` 至 `2023-09-12`。原始数据没有进入仓库。
- 离线验证覆盖传输失败隔离、解析状态、时间安全、Memory 闸门、评分基线和隐私边界；这些验证不等于真实模型质量。
- 2026-10-28 在公开仓库快照上运行离线测试：`186 passed`；没有调用真实模型。

**还没有完成**

- 没有完成当前 GPT-5.6 Luna 的真实全量 LLM 评测，所以目前没有新的准确率、风格、成本或延迟成绩。
- 之前的 DeepSeek 部分运行已停止，不能当作完整评测结果；后续全量评测不使用 DeepSeek。
- 没有做真人风格评审，也没有评“评分和评论是否一致”。
- Memory 只在单次运行内有效，重启就没了，也没有测出收益。
- 当前公开仓库发布的是生成侧；独立评测侧仍在 [项目 B](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)。

一句话：**源解析和离线工程检查已有可复核结果；当前模型的全量质量仍没有证据。**

## 4. 当前开发路径与 A/B 合并范围

开发 workspace 将以下内容放在同一套验收上下文中：

- **A：生成侧**：用户画像、参考评论、条件反思／修复、运行内 Memory、时间安全上下文、Amazon 源适配和四种确定性评分基线。
- **B：评测侧**：独立结果契约、评分/覆盖/失败诊断、证据约束和导出验收；B 的代码和 Git 历史仍保留在独立仓库。
- **公共边界**：只交换版本化预测结果和诊断，不交换目标标签、Prompt、原始评论或逐记录私有输出。

这次发布是**经过审查的源码快照整合，不是把两个仓库的 Git 历史合并**。本仓库发布 A 侧代码和公开说明；项目 B 不会因为开发 workspace 的组合而变成这个仓库的一部分。

下一步按以下顺序进行：

1. 已完成：源字段/时间归一化、source-only 全量解析验证、离线安全测试和公开文档更新。
2. 明天：先取得私有的当前 GPT-5.6 Luna 批量调用路径，固定 max/Absolute 配置、预算、数据外发许可和停止条件；未取得前不调用真实模型。
3. 批量评测：仅在 Drive/Colab 保存 raw、标签、Prompt、模型输出和逐记录结果；只向公开仓库发布去标识的汇总状态。
4. 独立交给项目 B 评分和诊断，再做有/无 Memory 的同任务配对；没有分母和覆盖率时不报告收益。

## 5. 怎么使用？

这里**没有插件，也没有在线后端**。它是一个本地运行的 Python 项目，也可以在 notebook／Colab 里跑。真实生成需要接 LLM；测试和契约路径使用 fake，不需要 API key。

### 安装和测试

需要 Python 3.10 或更新版本。

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

上游框架从 PyPI 安装，没有复制进本仓库。API key 放在本地 `.env` 或环境变量里，不要提交到仓库。

### 运行已发布的 source-only 预览

原始源必须位于本地受控的 Drive/Colab staging 目录；下面的命令不调用 LLM，也不把源文件复制到仓库：

```powershell
python amazon_full_evaluation.py --raw-root "<Drive/Colab raw-root>" --dry-run
```

完整模型运行暂不提供公开命令，等待当前模型的私有批量接口和批准配置。

### 运行本地的结果契约预览

下面的命令属于更宽的本地开发 workspace，不属于当前公开 `main`：

```powershell
python -m user_simulator_project --dry-run
python -m user_simulator_project --synthetic --max-tasks 3 --output-dir results/stage1_demo
```

输出目录必须是全新的，并且位于仓库里被忽略的 `results/` 或 `Dataset/` 目录下。导出器会先写临时文件、校验，再以不覆盖的方式发布；已有文件不会被替换。本地目录可用时，在项目 B 里验证：

```powershell
python -m amazon_eval_agent validate --input "<绝对路径>/simulation_run.v1.json" --json
```

退出码 `0` 只表示结果文件符合契约，不代表预测准确。

### 运行 Demo

公开仓库没有在线用户模拟 Demo。开发 workspace 做过本地合成 Demo，但它不是在线服务，也不是当前公开部署。

如果本地要起 Demo 服务，只绑定 `127.0.0.1`，不要暴露到公网。Demo 只用合成内容和 fake 模型输出，不能接真实评论或目标标签。

### 不该做什么

- 不要用这个 Agent 冒充真人发布评论。
- 未经许可、没有明确预算，不要把原始评论、目标标签、Prompt 或运行结果发给第三方模型。
- 不要把 fake 跑通或契约测试通过，当成真实模型效果。

## 6. 继续阅读

- [数据集选择与许可说明](docs/DATASETS.md)
- [能力、指标、幻觉检查与 Memory](docs/AGENT_CAPABILITIES.md)
- [分步开发计划与验收标准](docs/DEVELOPMENT_PLAN.md)
- [2026-10-28 公开状态报告](docs/STATUS_2026-10-28.md)
- [P0 数据协议](docs/P0_IMPLEMENTATION_GUIDE.md) / [P1 可靠工作流](docs/P1_IMPLEMENTATION_GUIDE.md) / [合成 Demo](docs/P3_IMPLEMENTATION_GUIDE.md)
- [项目 B：评测与诊断](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)

如果只记住一句话：**这是一个受控的用户模拟原型，不是已经证明有效的个性化推荐系统。**
