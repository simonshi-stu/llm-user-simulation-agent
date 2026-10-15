# user_simulator_project（阶段 1 独立导出入口）

本包是用户模拟项目（A）的阶段 1 独立入口：**离线**跑通已验收的 P1 可靠工作流，
用确定性 fake callable 生成结果，并按共享契约 `SimulationRunV1` 导出一份版本化
artifact，供独立的评测诊断项目（B）校验。不调用真实模型、不读 API key、不联网。

## 命令（A 仓库根目录，PowerShell）

```powershell
# 只打印安全计划：不调用任何 LLM、不创建文件
.\.venv\Scripts\python.exe -m user_simulator_project --dry-run

# 合成 fixture + fake callable -> fake_on_synthetic
.\.venv\Scripts\python.exe -m user_simulator_project --synthetic --output-dir results\stage1_synthetic

# 本地已验收 prepared 目录 + fake callable -> fake_on_real_context（默认前 3 个任务）
.\.venv\Scripts\python.exe -m user_simulator_project --prepared-dir <prepared 目录> --output-dir results\stage1_prepared --max-tasks 3
```

## 规则与边界

- 输出目录必须**新建**，且位于**本仓库根目录**（模块锚定，不依赖当前工作目录）
  下被忽略的 `results/` 或 `Dataset/` 中；其他位置的 `results/` 一律拒绝。已存在
  即失败，失败只清理本次新建的目录。stdout 只打印数字、固定枚举与 sha：不打印
  评论、标签、路径或私有错误文本；
- artifact 以**本次专属**的 `.part` 文件写出（独占创建；已存在的 part/final 一律
  拒绝且绝不删除），对这份精确字节完整复检后再以 **hard link 原子发布**为最终文件
  （不覆盖；崩溃不会在最终文件名下留下半成品）；不支持 hard link 时 **fail-closed**
  `atomic_publish_unavailable`，绝不降级为非原子写。失败清理只删除本次运行的已知
  产物与空目录，未知/晚建文件保留；
- `--prepared-dir` 必须通过 `reliable_agent.load_verified_prepared_dataset` 的
  全部门禁（工件尺寸/行数闸门先于解析、policy、选择重建、先验复核、严格 provider、
  split 计数、目标副本排除），不绕过标签排除、UTC 日规则或 Memory 预算；
- 固定 `--seed 42`；`--max-tasks` 默认 3（上限 12）；串行执行（max_workers=1）；
- 产物只含共享契约字段：契约没有 groundtruth / 目标文本 / 原始 prompt / 密钥 /
  任意路径字段；`user_group` 是确定性**伪名**标签（保留"同一模拟用户跨任务同
  标签"的配对关系），不声称不可逆匿名；`review` 是生成自由文本、可能含任意词，
  不做内容级脱敏声明；
- **诚实声明**：P1 预测函数不接收 groundtruth；groundtruth 行仅由共享 P0/P1
  机制用于身份/时间/文本核验与目标排除，绝不喂给预测、统计先验或任何统计；
  **本阶段不做评分**（未来评分属于评测项目 scorer）。两种模式都只用 fake
  callable，产物只能证明形状与门禁通过，**不能测真实 LLM 收益、准确率或
  Memory 效应**。

## 来源与版本

- 契约镜像：`user_simulator_project/simulation_run_v1.py`；canonical 位于
  `amazon-instruments-evaluation-agent/amazon_eval_agent/contracts/simulation_run_v1.py`。
  镜像 `schema_hash()` 必须与 canonical 一致（跨仓库测试钉住）；
- 复用的既有能力：`p1_smoke.build_synthetic_provider` / `ScriptedFakeLLM`、
  `reliable_agent` 的 suite/budget/memory store 与 prepared 校验加载器；
  未改动任何既有核心模块与历史结果。
