# markdown

个人技术笔记与 **HTML 阅读报告** 仓库：机器学习笔记、推荐 / 强化学习资料，以及集中维护的综述精读页面与周报合集。

---

## 目录速览

| 路径 | 说明 |
|------|------|
| **`reports/reports/`** | 主阅读区：LLM / Agent / RL / 模型技术解析 / 博客摘要 / 教材章节等（深色主题自包含 HTML）。索引见 [`reports/reports/README.md`](reports/reports/README.md)。 |
| **`reports/weekly-papers/`** | HuggingFace 每周论文中文阅读报告（按周归档）。见 [`reports/weekly-papers/README.md`](reports/weekly-papers/README.md)。 |
| **`reports/`**（顶层） | 少量独立 HTML（如 RL 工作流、训练与 TensorBoard、HTTP Manager 等）。 |
| **`机器学习算法/`** | 笔记与专题：推荐系统、强化学习等；插图多引用仓库根下的 `picture/`、`pictures/`。 |
| **`picture/`**、**`pictures/`** | 笔记与报告共用的本地配图目录。 |
| **`Apache-Spark技术报告.html`** | Spark 技术报告（单页 HTML）。 |

> 历史上曾使用 **`技术报告合集/`** 存放与报告库重复的 HTML，现已删除副本，**以 `reports/reports/` 为准**。

---

## 如何阅读

1. **克隆仓库**（需 Git）。
2. 用浏览器直接打开：
   - 专题报告：`reports/reports/` 下任意 `.html`（可从该目录 [`README.md`](reports/reports/README.md) 索引进入）
   - 周报：`reports/weekly-papers/<周次>/` 下报告，或从 [`weekly-papers/README.md`](reports/weekly-papers/README.md) 进入
   - 根目录 `Apache-Spark技术报告.html`、以及 `机器学习算法/` 下的学习报告
3. 若页面引用 **`../../picture/`**、**`../../pictures/`** 等相对路径，请保持本地目录结构与仓库一致，避免移动单文件导致裂图。

部分报告中的外链插图（如 `raw.githubusercontent.com`）不依赖本仓库图片，仅需联网加载。

---

## 技术说明（报告页）

- 主题：深色阅读样式（Catppuccin Mocha 系）。
- 公式：MathJax 3（`$...$` / `$$...$$`）。
- 图表：常见页面含 Mermaid、ECharts（以各文件实际引用为准）。

---

## 许可证与声明

内容仅供个人学习整理；转载、引用请遵守原作者与数据源许可。仓库内报告多为读书笔记，**不构成**任何产品或投资建议。
