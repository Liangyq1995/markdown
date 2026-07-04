---
name: ml-systems-report
description: >-
  将 Machine Learning Systems (Vol.1/Vol.2) PDF 按章节生成中文 HTML 阅读报告。
  使用 Catppuccin 深色主题、MathJax、Mermaid、ECharts。
  当用户要求阅读 mlsysbook、ML Systems 教材、Machine-Learning-Systems PDF 并生成章节报告时使用。
---

# ML Systems 章节报告生成

## 输出规范

| 项 | 值 |
|---|---|
| 输出目录 | `reports/ml-systems/` |
| 命名 | `第{N}章_{中文标题}.html` |
| 语言 | 简体中文 |
| 参考模板 | `reports/ml-systems/第1章_引言.html` |
| 样式规范 | `reports/CLAUDE.md` |

## 内容来源（优先级）

1. **在线版**：`https://mlsysbook.ai/vol1/contents/vol1/{slug}/{slug}.html`
2. **本地 PDF**：仓库根目录 `Machine-Learning-Systems-Vol1.pdf`（Read 工具可读）
3. 用户已确认的第 1 章报告作为结构与深度基准

## 技术细节 enrich（网搜补充）

当原书/PDF **只点到技术名、缺少实现细节** 时，用 WebSearch 补充后再写入报告。

### enrich 合格标准（用户明确要求）

每条 enrich **至少满足以下 2 项**，否则不写：

| 必须有 | 示例 |
|--------|------|
| **具体数值** | Jetson Nano 4 GB 统一内存、INT8 模型 3.5 MB、p99 &lt; 200 ms |
| **公式 + 代入算例** | PSI 四箱手算 → 0.044；FedAvg 3 客户端各 1000 样本 |
| **可执行配置/命令** | Istio `weight: 95/5`；`mlflow.log_param("data_hash", "a3f2…")` |
| **明确阈值与触发动作** | PSI ≥ 0.25 → 48h 内重训；错误率 &gt; 基线 0.3% 持续 5 min → 回滚 |
| **与本章案例的直接挂钩** | 所有 enrich 须回答「DR 512MB 场景下怎么用」 |

### 不合格 enrich（禁止）

- ❌ 「使用 K8s + Istio 做流量切分」——无权重、无回滚条件
- ❌ 「MLflow 记录 params/metrics/artifacts」——无字段名、无绑定方式
- ❌ 「PSI 监控数据漂移」——无公式、无算例、无阈值动作
- ❌ 百科式「X 是一种…用于…」且与章节决策无关

### 写法

- 融入 `tech-card` 或正文，**不单独贴「延伸」标签凑字数**
- 优先官方文档、论文、成熟工程实践
- 原书已充分展开的不重复网搜

**典型需 enrich 的技术**（按章）：

| 章 | 可 enrich 主题 | 合格 enrich 应含 |
|----|----------------|------------------|
| 3 Workflow | PSI/KS、Canary、MLflow | 手算 PSI、Istio weight 95/5、run tag 链 |
| 4 Data Eng | Feature Store、Delta Lake | 具体 schema、partition、time travel 命令 |
| 11 HW | Roofline | 算例：峰值 FLOPS、内存带宽、ridge point |
| 13 Serving | PagedAttention | block size、KV cache 字节数公式 |

## Vol.1 章节映射

| 章 | 英文 | 中文标题 | URL slug |
|----|------|----------|----------|
| 1 | Introduction | 引言 | introduction/introduction |
| 2 | ML Systems | ML系统 | ml_systems/ml_systems |
| 3 | ML Workflow | ML工作流 | ml_workflow/ml_workflow |
| 4 | Data Engineering | 数据工程 | data_engineering/data_engineering |
| 5 | Neural Computation | 神经计算 | dl_primer/dl_primer |
| 6 | Network Architectures | 网络架构 | dl_primer/dl_primer |
| 7 | ML Frameworks | ML框架 | ml_frameworks/ml_frameworks |
| 8 | Model Training | 模型训练 | model_training/model_training |
| 9 | Data Selection | 数据选择 | data_selection/data_selection |
| 10 | Model Compression | 模型压缩 | model_compression/model_compression |
| 11 | Hardware Acceleration | 硬件加速 | hw_acceleration/hw_acceleration |
| 12 | Benchmarking | 基准测试 | benchmarking/benchmarking |
| 13 | Model Serving | 模型服务 | model_serving/model_serving |
| 14 | ML Operations | ML运维 | ml_ops/ml_ops |
| 15 | Responsible Engineering | 负责任工程 | responsible_ai/responsible_ai |
| 16 | Conclusion | 总结 | conclusion/conclusion |

## HTML 模板要点

复制 `第1章_引言.html` 的 `<head>` CSS 与 `<script>` 块，替换 `<main>` 内容。

### 必须包含

- 固定左侧 TOC（240px）+ scroll-spy（见第 1 章末尾 script）
- 页眉：`h1` + `.meta`（书名、Vol、生成日期）
- 开篇摘要表：核心主线 + 学习目标
- 中文节标题：`h2` 一/二/三…，`h3` x.1/x.2…
- 至少 1 个 Mermaid 图 + 1 个表格 + 关键公式（MathJax）
- `tech-card` 高亮核心定义/案例
- 章节总结 + 下一章预告
- 页脚来源链接 `mlsysbook.ai`

### 公式规则

- 块级：`<div class="math-block">$$...$$</div>`
- **仅**在 `$...$` / `$$...$$` 内：`<` 写 `\lt`，`>` 可写 `\gt`（防 HTML 截断）
- **普通正文**（`<p>`、`<td>`、`<li>` 等，非数学模式）：用 HTML 实体 `&lt;`、`&gt;`，**禁止** `\lt`/`\gt`（会原样显示）

### 代码块规则

- 必须用 `<pre><code>...</code></pre>`，**不是图片**
- CSS 须含 `pre` / `pre code` 样式（见第 1 章模板：背景 `#313244`、等宽字体、明确 `color`）
- 行内注释（`# ...`）放 `<pre>` 外另起 `<p>` 说明，避免 MathJax/渲染干扰
- MathJax 配置 `skipHtmlTags` 含 `pre`、`code`

### Mermaid 限制

- 使用 `graph TD/LR`、flowchart
- **禁止** timeline、mindmap、radar、多系列 bar

### ECharts（可选）

- 容器需 `style="height:Xpx"`
- `backgroundColor:'#24243e'`

## 工作流程

```
1. 确认章节号与是否已有报告
2. 获取章节内容（WebFetch 或 Read PDF），提取全部 ## 节
3. 识别「仅点名、缺细节」的技术点 → WebSearch enrich（见上表）
4. 用中文重写：概念 + 定量/公式 + 案例 + enrich 技术细节 + 工程启示
5. 写入 reports/ml-systems/第{N}章_{标题}.html
6. 人工确认后再写下一章；全部完成后更新 index.html
```

## 质量红线（用户明确要求）

- **宁可慢，不可水**：未精读源章节前禁止写 HTML
- **无用报告 = 浪费时间**：仅列标题/bullet、无推导无案例 = 不合格，须重写
- **验收标准**：读者不读原书也能掌握该章核心论点、定量关系与工程决策方法
- **禁止**：Python 批量脚本、子任务并行凑数、未读 PDF 就动笔

**必须**：
1. 先完整读取源章节（WebFetch 或 Read PDF），提取全部 `##` 节
2. 以 `第1章_引言.html` 为质量基准（~450 行、有深度叙述）
3. 每节含：概念解释 + 定量数据/公式 + 工程启示（非 bullet 堆砌）
4. 保留书中关键案例（Napkin Math、Definition、War Story、Fallacy）
5. 术语首次出现保留英文原名
6. 不复制 Self-Check 原题，可提炼 1–2 条 insight
7. 单章 HTML **≥ 400 行**（短章如 Conclusion 可 ≥ 250 行）；**禁止用泛泛 enrich 凑行数**
8. enrich 须过「合格标准」自检（见上表），不合格则删或不写
9. **逐章生成，人工确认后再写下一章**（除非用户明确要求批量/不再人工检查）

## 索引页

全部章节完成后创建 `reports/ml-systems/index.html`：列出 16 章链接，按 Part I–IV 分组。
