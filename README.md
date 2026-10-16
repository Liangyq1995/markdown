# markdown

个人技术笔记与 **HTML 阅读报告** 仓库：机器学习笔记、推荐 / 强化学习资料，以及集中维护的综述精读页面与周报合集。

| 内容 | 位置 | 规模 |
|------|------|------|
| HTML 阅读报告 | `reports/` | 180 篇 |
| Markdown 笔记 | `机器学习算法/` | 15 篇 |
| 配图 | `pictures/` | 178 张（`000` – `177`） |

所有报告都是**自包含单页 HTML**，用浏览器直接打开即可，不需要本地服务器。

---

## 目录结构

```
md/
├── reports/                      # 阅读报告（HTML）
│   ├── reports/                  #   主阅读区：按主题分类 + 顶层综合报告
│   │   ├── rl-training/          #     RL 训练与 LLM 后训练（25）
│   │   ├── paper-reading/        #     论文精读（18）
│   │   ├── model-reports/        #     模型技术解析（17）
│   │   ├── llmbook/              #     LLM 教材章节（14）
│   │   ├── blog-digests/         #     博客摘要（8）
│   │   ├── agent-systems/        #     Agent 系统（5）
│   │   ├── ml-foundations/       #     ML 基础（4）
│   │   ├── *.html                #     6 篇跨类别综合报告
│   │   ├── README.md             #     ← 报告总索引
│   │   └── CLAUDE.md             #     ← 报告模板与排版规则
│   ├── weekly-papers/            #   HuggingFace 每周论文报告，按周归档
│   └── *.html                    #   3 篇顶层独立报告
├── 机器学习算法/                  # Markdown 笔记
│   ├── 强化学习/                  #   4 篇
│   └── 推荐系统/                  #   11 篇
├── pictures/                     # 配图目录，统一编号
├── Apache-Spark技术报告.html      # 单页技术报告
└── .gitignore
```

---

## 目录速览

| 路径 | 说明 | 索引 |
|------|------|------|
| **`reports/reports/`** | 主阅读区：LLM / Agent / RL / 模型技术解析 / 博客摘要 / 教材章节，深色主题自包含 HTML | [README](reports/reports/README.md) |
| **`reports/weekly-papers/`** | HuggingFace 每周论文中文阅读报告，按周归档（当前 `2026-W21` – `2026-W29`） | [README](reports/weekly-papers/README.md) |
| **`reports/`**（顶层） | 3 篇独立 HTML：RL 工作流、训练与 TensorBoard、HTTP Manager | — |
| **`机器学习算法/`** | 笔记与专题：推荐系统、强化学习；插图统一引用仓库根下的 `pictures/` | — |
| **`pictures/`** | 笔记与报告共用的本地配图目录 | — |
| **`Apache-Spark技术报告.html`** | Spark 技术报告（单页 HTML） | — |

> 历史上曾使用 **`技术报告合集/`** 存放与报告库重复的 HTML，现已删除副本，**以 `reports/reports/` 为准**。

---

## 如何阅读

1. **克隆仓库**（需 Git）。
2. 用浏览器直接打开：
   - 专题报告：从 [`reports/reports/README.md`](reports/reports/README.md) 索引进入，或直接打开 `reports/reports/` 下任意 `.html`
   - 周报：从 [`reports/weekly-papers/README.md`](reports/weekly-papers/README.md) 进入，或打开对应周次目录
   - 根目录 `Apache-Spark技术报告.html`
3. 报告中的插图有两种来源：
   - **本仓库配图** —— 统一引用根目录 `pictures/`（例如 `../../pictures/000.png`）。请保持目录结构不变，移动单个文件会导致裂图。
   - **外链插图** —— 形如 `raw.githubusercontent.com`、`*.github.io`，不依赖本仓库，仅需联网加载。
4. 部分报告依赖 CDN 加载 Mermaid / ECharts / MathJax，**首次打开需要联网**。

> 根目录的教材 PDF（`Machine-Learning-Systems-Vol1/2.pdf`）体积较大，**未纳入版本库**（见 `.gitignore`）。

---

## 配图约定

`pictures/` 是笔记与报告共用的配图目录，图片已统一重编号：

- **命名**：`000.png` … `177.png`，三位补零、连续无缺号；扩展名保持原样（其中 `073` 为 `.jpg`）
- **引用**：一律使用**相对路径**，不使用绝对路径
- **新增**：接着当前最大编号往后排（下一张为 `178.png`），避免与已有编号冲突

> 此前 `picture/`（含 `1/`、`2/` 子目录）与 `pictures/` 两个目录并存，且存在 **26 组同名但内容不同**的图片。现已合并为单一目录并重编号，全部引用同步改写为相对路径。

---

## 写作与生成约定

- **报告命名**：`{论文名}-阅读报告.html`（论文精读）或 `{主题}-技术解析.html`（技术解析）
- **报告存放**：按主题放入 `reports/reports/` 下对应子目录；跨类别综合报告放在该目录顶层
- **报告样式**：Catppuccin Mocha 深色主题 + MathJax 公式 + Mermaid 图 + ECharts 可视化 + 左侧固定目录
- **模板与排版规则**：公式写法、Mermaid 陷阱、ECharts 要求、布局约定等详见 [`reports/reports/CLAUDE.md`](reports/reports/CLAUDE.md)
- **周报索引**：`reports/weekly-papers/README.md` 由脚本自动维护，**请勿手动编辑**分隔线以下部分

---

## 技术说明（报告页）

- 主题：深色阅读样式（Catppuccin Mocha 系）
- 公式：MathJax 3（`$...$` / `$$...$$`）
- 图表：常见页面含 Mermaid、ECharts（以各文件实际引用为准）

---

## 许可证与声明

内容仅供个人学习整理；转载、引用请遵守原作者与数据源许可。仓库内报告多为读书笔记，**不构成**任何产品或投资建议。
