# markdown

个人技术笔记与 **HTML 阅读报告** 仓库：机器学习笔记、推荐/强化学习资料、Spark 技术报告，以及集中维护的 `reports/` 综述与精读页面。

---

## 目录速览

| 路径 | 说明 |
|------|------|
| **`reports/`** | 主阅读区：LLM 综述、论文精读、Agent、RL、模型技术解析、博客摘要、教材章节报告等（深色主题自包含 HTML）。详见 [`reports/README.md`](reports/README.md)。 |
| **`机器学习算法/`** | 笔记与专题：推荐系统（Markdown）、强化学习（`.md` + 学习报告 HTML）等；文中插图多引用仓库根下的 `picture/`、`pictures/`。 |
| **`picture/`**、`**pictures/**` | 笔记与报告共用的本地配图目录。 |
| **`Apache-Spark技术报告.html`** | Spark 技术报告（单页 HTML）。 |
| **`_prune_unused_images.py`** | 可选维护脚本：根据全库 `.md`/`.html` 及本仓库 `origin` 的 GitHub raw 链接判断引用后，清理 `picture/`、`pictures/` 中未被引用的图片（默认干跑，加 `--apply` 才删除）。 |

> 历史上曾使用 **`技术报告合集/`** 存放与 `reports/` 重复的 HTML，现已删除副本，**以 `reports/` 为准**。

---

## 如何阅读

1. **克隆仓库**（需 Git）。
2. 用浏览器 **直接打开** `reports/` 下任意 `.html`，或打开根目录的 `Apache-Spark技术报告.html`、各专题学习报告。
3. 若页面引用 **`../../picture/`** 等相对路径，请保持本地目录结构与仓库一致，避免移动单文件导致裂图。

部分报告中的外链插图（如 `raw.githubusercontent.com` 上其他开源项目）不依赖本仓库图片，仅需联网加载。

---

## 技术说明（与 `reports/` 一致）

- 主题：深色阅读样式（Catppuccin Mocha 系）。
- 公式：MathJax 3（`$...$` / `$$...$$`）。
- 图表：常见页面含 Mermaid、ECharts 等（以各文件实际引用为准）。

---

## 许可证与声明

内容仅供个人学习整理；转载、引用请遵守原作者与数据源许可。仓库内报告多为读书笔记，**不构成**任何产品或投资建议。
