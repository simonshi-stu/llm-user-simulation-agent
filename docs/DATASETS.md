# 数据集选择：先匹配任务，再核验许可

核对日期：2026-10-14。本文件是数据选择指南，不是法律意见；没有执行新数据下载或接受协议。

## 现在可以用到哪里

| 数据 | 对用户模拟 A 的用途 | 对评测诊断 B 的用途 | 当前接入状态与许可 |
| --- | --- | --- | --- |
| 手写合成 fixture | 校验四态、修复、证据、Memory 与预算 | 工具合同、手算指标和异常路径 | 已用于离线测试；不是真实模型成绩 |
| Amazon 2014 Musical Instruments 5-core | 用户评分和评论风格历史 | 时间审计、四基线、后续模拟预测评测 | 本地已核验 10,261 行；400 目标；许可未明确核实，数据不再分发 |
| Amazon 2018 Software 小子集 | 第二领域复现候选 | 跨领域协议与指标复现 | 官方表列 12,805 行；独立 adapter 尚未接入；许可待确认 |
| MovieLens 100K | 用户偏好、评分与时间序列记忆 | 评分基线与冷启动诊断 | 100,000 评分/943 用户/1,682 电影；没有评论正文；研究用途，引用要求，商业与再分发需额外许可 |
| Goodreads Book Graph | 阅读偏好、评分、长文本风格 | 评论风格及跨领域评分候选 | 有公开书评；官方明确 academic use only，不得再分发或商业使用；未接入新版 |
| Yelp Open Dataset | 本地商家评论、画像与风格候选 | 与旧课程任务领域对照 | 当前官方入口返回 403，未核验当前数据协议；不可按镜像/旧课程许可推定已获授权 |
| Amazon Reviews 2023 乐器分类 | 较新领域扩展、评论和元数据 | 大规模扩展与事实证据候选 | 官方列约 300 万乐器评论；不能直接塞入 64 MiB/10 万行管线；许可及 adapter 待核验 |
| UCI Online Retail | 真实购买/取消事件的行为记忆候选 | 事件序列与记忆检索诊断 | 541,909 交易记录，CC BY 4.0；无星级与评论，不能直接作为评分/风格评测标签 |

**技术上适合、许可允许、当前代码支持，是三个不同条件。**当前通用闭环只核验了 Amazon 2014 乐器，不宣称其他数据可直接传入。

## 推荐顺序

1. 合成 fixture 验证协议与安全性。
2. 用现有 Amazon 2014 乐器完成 A→B 互通和评分/评论评测。
3. 许可确认后，用 2018 Software 或 MovieLens 小数据做一次领域复现。
4. 需要真实行为事件时再适配 UCI Online Retail；需要长文本风格时再考虑 Goodreads。
5. 暂不下载 Amazon 2023 全量或 GB 级历史文件；扩容需要另外设计索引与资源边界。

## 新 adapter 的最小验收

- 核对用户/物品/评分/正文/事件时间字段，明确评分尺度、缺失值和时间单位。
- 使用字段白名单；姓名、累计 helpful/vote、抓取时的平均评分不能直接作为历史特征。
- 先冻结历史与任务清单，保持标签单独存放；目标、同日/未来行及可匹配副本不能进上下文。
- 2023 的 timestamp 示例可能为毫秒，新旧 asin/parent_asin 语义也不同，不能只改字段名。
- 商品元数据采集时点未证明早于目标时，不能当时序安全产品事实；当前页面平均评分是潜在未来信息。
- 检查实际用户重复机会，不从“5-core”名称推定样本满足所有协议。
- 文件、解压、行数与单记录资源门槛保留；不使用发布页旧示例中的 `eval`，不默认执行 `trust_remote_code=True`。
- 保存本地来源/版本/哈希/许可状态；公开仓库只提供说明及合成样例。

## 已核对的官方来源

- Amazon 2014：[UCSD 发布页](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。本次未找到明确开放许可。
- Amazon 2018：[UCSD 发布页](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)。Software 小文件的行数来自官方表，不是新版适配结果。
- Amazon 2023：[发布页](https://amazon-reviews-2023.github.io/) / [官方数据卡](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)。没有据镜像或工具代码许可证推定评论版权开放。
- MovieLens：[100K 发布页](https://grouplens.org/datasets/movielens/100k/) / [使用许可与字段](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)。默认提供的随机划分不能直接当时间安全划分。
- Goodreads：[UCSD Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)。官方写明学术用途、禁止再分发和商业使用。
- Yelp：[官方入口](https://www.yelp.com/dataset)。本次 403；通用服务条款不能代替数据集协议。
- UCI：[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)。页面明确 CC BY 4.0，引用 DOI：10.24432/C5BW33。

所有原始数据和本地实验继续放在 E 盘。公开可见也可能包含个人信息；真实模型外发、人评展示或公开 Demo 必须另行确认用途和数据范围。
