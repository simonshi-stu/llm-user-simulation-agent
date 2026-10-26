# LLM 用户模拟 Agent

[简体中文](README.md) | [English](README.en.md)

这是一个研究原型：根据某个用户过去的行为，模拟他面对一件新商品时可能给出的星级和评论。它只读取该用户更早的历史，生成结果后交给独立的评测项目检查，而不是让生成代码自己给自己打分。

它不是用来冒充真人发评论的机器人，也不声称知道用户真实的想法。

## 0. 引用与致谢

- **[AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety)**（清华大学 FIB Lab）。更广泛的 LLM 社会模拟研究平台；本仓库没有复制或内嵌 AgentSociety 本身。
- **[AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)**（清华大学 FIB Lab，MIT License）。项目直接依赖的模拟框架：通过 PyPI 安装 `websocietysimulator==1.0.0a30`，使用其模拟环境、Agent 基类、任务交互工具和框架评测接口。
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

关键在边界：用户个人历史只取当前用户更早的记录；目标评分、未来评论和已知目标文本副本不会进入上下文。同商品参考评论可以来自其他用户的安全历史，但不包含目标行、未来行或目标文本副本。启用 Memory 时，只召回同一用户更早生成的记录，并只作为表达风格参考，不能当作可信评分标签。

这套设置可以回答这些问题：

- 面对一件新商品，这个用户会比平时打得更高还是更低？
- 换一个 Prompt 或检索方式，在同一批任务上有没有变好？
- Memory 真的进入 Prompt 后有没有帮助，还是只增加了成本？
- Agent 能不能泛化到没见过的用户？

## 3. 项目目前做到哪一步？

**已经能用**

- 保留原 AgentSociety 课程链路：用户画像、参考评论排序、条件反思／修复，以及课程评测入口。
- 本地核验过 Amazon 乐器数据：10,261 条评论、400 个目标任务和四种固定统计基线。数据只留在本地，不再分发；这是 2014 年固定研究数据协议，不是“2026 最新数据”。
- 新增的可靠工作流有离线测试覆盖：受限证据、四种结果状态、最多一次修复、调用预算，以及只在成功时写 Memory。
- 版本化的 `simulation_run.v1` 结果契约、本地 synthetic/prepared fake 导出入口和独立校验入口已整理进当前发布候选；正式发布前不称已发布。
- 本地合成浏览器 Demo 的代码已包含在仓库中，可用 `python demo_app.py --check` 自检。它不是在线服务，也不接真实模型。

**还没有完成**

- 没有跑新版真实 LLM 评测，所以目前没有新的准确率、风格、成本或延迟成绩。
- 没有做真人风格评审，也没有评“评分和评论是否一致”。
- Memory 只在单次运行内有效，重启就没了，也没有测出收益。
- 当前是开源代码候选，不是在线服务；没有真实用户泛化或生产可用性结论。

一句话：**工程骨架和离线检查是真的；但“模拟得准不准、够不够个性化”还没有证据。**

## 4. 怎么使用？

这里**没有插件，也没有在线后端**。它是一个本地运行的 Python 项目，也可以在 notebook／Colab 里跑。真实生成需要接 LLM；测试和契约路径使用 fake，不需要 API key。

### 安装和测试

需要 Python 3.10 或更新版本。

```powershell
# 运行测试、lint 和离线 smoke：安装开发依赖（含 pytest/ruff）
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

`requirements.txt` 是运行真实课程链路时使用的依赖（包含 `websocietysimulator`）；只跑离线测试和 fake/契约路径时用 `requirements-dev.txt` 就够了。上游框架从 PyPI 安装，没有复制进本仓库。API key 放在本地 `.env` 或环境变量里，不要提交到仓库。

公开测试只使用 synthetic fixture，不读取真实评论、目标标签或 per-task 结果，也不访问模型 API。

### 运行本地的结果契约预览

以下命令在 fresh clone 的仓库根目录可运行：

```powershell
python -m user_simulator_project --dry-run
python -m user_simulator_project --synthetic --max-tasks 3 --output-dir results/stage1_demo
```

输出目录必须是全新的，并且位于仓库里被忽略的 `results/` 或 `Dataset/` 目录下。导出器会先写临时文件、校验，再以不覆盖的方式发布；已有文件不会被替换。

若本机同时有项目 B，验证命令要在 **B 的 checkout** 里运行。A 的 fresh clone 根目录没有 `amazon_eval_agent` 包，直接 `python -m amazon_eval_agent` 不能 import 兄弟仓库：

```powershell
# 先切到 B 的仓库，不要留在 A 根目录
$BRepo = "C:\path\to\amazon-instruments-evaluation-agent"
$Artifact = "C:\path\to\A\results\stage1_demo\simulation_run.v1.json"
Set-Location $BRepo
python -m amazon_eval_agent validate --input $Artifact --json
```

如果 B 使用独立 venv，也可以显式调用 B 的解释器：

```powershell
& "$BRepo\.venv\Scripts\python.exe" -m amazon_eval_agent validate --input $Artifact --json
```

退出码 `0` 只表示结果文件符合契约，不代表预测准确。没有 B sibling 时，跨仓契约测试会显式 skip。

### 运行 Demo

```powershell
# 离线自检：场景合同 + 回环 HTTP，自检后退出
python demo_app.py --check

# 本地合成 Demo，仅绑定回环地址
python demo_app.py --host 127.0.0.1 --port 8290
```

浏览器打开 `http://127.0.0.1:8290/`。Demo 只用合成内容和 fake 模型输出，不接真实评论或目标标签，也不会发起外部请求。它只绑定 `127.0.0.1` / `::1`，不要暴露到公网。

### 不该做什么

- 不要用这个 Agent 冒充真人发布评论。
- 未经许可、没有明确预算，不要把原始评论、目标标签、Prompt 或运行结果发给第三方模型。
- 不要把 fake 跑通或契约测试通过，当成真实模型效果。

## 5. 继续阅读

- [数据集选择与许可说明](docs/DATASETS.md)
- [架构、模块职责与标签边界](docs/ARCHITECTURE.md)
- [能力、指标、幻觉检查与 Memory](docs/AGENT_CAPABILITIES.md)
- [分步开发计划与验收标准](docs/DEVELOPMENT_PLAN.md)
- [P0 数据协议](docs/P0_IMPLEMENTATION_GUIDE.md) / [P1 可靠工作流](docs/P1_IMPLEMENTATION_GUIDE.md) / [合成 Demo](docs/P3_IMPLEMENTATION_GUIDE.md)
- [项目 B：评测与诊断](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)

如果只记住一句话：**这是一个受控的用户模拟原型，不是已经证明有效的个性化推荐系统。**
