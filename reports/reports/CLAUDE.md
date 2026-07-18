# Reports Directory — Claude Code Context

## Directory Purpose
This directory contains HTML reading reports and technical analyses for LLM/AI research papers, organized by topic category. All reports use a dark-theme (Catppuccin Mocha) layout with MathJax formulas, Mermaid diagrams, ECharts visualizations, and a fixed left-side TOC.

## Structure
```
reports/
├── paper-reading/        # Deep-dive reading reports: individual papers + inference/system topics
├── agent-systems/        # LLM-based agent architectures, MCP, multi-agent (5 reports)
├── rl-training/          # RL algorithms + LLM post-training
│   ├── easy-rl-reports/  # easy-rl textbook chapter reports (13 chapters)
│   └── reinforce/        # Standalone RL algorithm deep-dives (10 reports)
├── model-reports/        # Specific model families (DeepSeek, Kimi, LLaMA, Qwen, Claude, etc.)
├── ml-foundations/       # Math, statistics, deep learning fundamentals
├── blog-digests/         # Blog series summaries
│   └── lilianweng-reports/   # 8 thematic volumes
└── llmbook/              # LLM textbook chapter reports (14 chapters)
```

Top-level HTML files (cross-category or multi-topic):
- `Agent-RL训练深度报告.html`
- `Gateway-in-AgentRL-深度报告.html`
- `LLM-Gateway-深度技术报告.html`
- `LLM-MTP训练技术报告.html`
- `llm-tool-server-evaluation.html`   # LLM 工具服务器评估指南（使用方视角）
- `MoE-LLM-RL训练报告.html`

## Report Generation Workflow

### Step 1 — Research (when needed)
Use **Tavily MCP tools** (not WebSearch) for web research:
```
mcp__tavily__tavily_search   → broad topic scan, 6–8 results
mcp__tavily__tavily_extract  → fetch full content from key URLs
mcp__tavily__tavily_research → deep research (rate-limited: 20 req/min, may hit quota)
```
For PDF papers: extract with `pdfminer.six`, read with `Read` tool.

### Step 2 — Skill
Always invoke `llm-paper-reading` skill before generating a report.

### Step 3 — Generate & Save
- **Save location**: appropriate subdirectory based on content type (see Structure above)
- **Naming**: `{PaperName}-阅读报告.html` for reading reports, `{Topic}-技术解析.html` for tech analyses

---

## HTML Template

Full skeleton to use every time:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>报告标题</title>
  <style>
    :root {
      --bg: #1e1e2e; --bg-code: #313244; --bg-nav: #181825;
      --text: #cdd6f4; --heading: #89b4fa; --link: #89dceb;
      --border: #45475a; --quote-bar: #fab387; --strong: #a6e3a1;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.8; }
    #toc { position: fixed; top: 0; left: 0; width: 220px; height: 100vh; background: var(--bg-nav); overflow-y: auto; padding: 20px 12px; font-size: 13px; border-right: 1px solid var(--border); }
    #toc h2 { color: var(--heading); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
    #toc a { display: block; color: var(--text); text-decoration: none; padding: 3px 8px; border-radius: 4px; margin-bottom: 2px; transition: background 0.15s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #toc a:hover { background: var(--border); }
    #toc a.active { background: var(--heading); color: #1e1e2e; font-weight: 600; }
    #toc a.toc-h3 { padding-left: 20px; font-size: 12px; opacity: 0.8; }
    main { margin-left: 240px; max-width: none; padding: 40px 64px; }
    h1 { color: var(--heading); font-size: 1.9em; margin-bottom: 8px; line-height: 1.3; }
    h2 { color: var(--heading); font-size: 1.4em; margin: 2em 0 0.6em; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
    h3 { color: var(--heading); font-size: 1.1em; margin: 1.4em 0 0.4em; }
    p { margin: 0.6em 0; }
    a { color: var(--link); }
    strong { color: var(--strong); }
    blockquote { border-left: 3px solid var(--quote-bar); padding: 8px 16px; margin: 1em 0; background: #2a2a3e; border-radius: 0 4px 4px 0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.93em; }
    th { background: #2a2a3e; color: var(--heading); padding: 8px 12px; text-align: left; border: 1px solid var(--border); }
    td { padding: 7px 12px; border: 1px solid var(--border); }
    tr:hover td { background: #2a2a3e; }
    pre { background: var(--bg-code); border-radius: 6px; padding: 16px; overflow-x: auto; margin: 1em 0; font-size: 0.88em; }
    code { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
    p code, li code { background: var(--bg-code); padding: 2px 5px; border-radius: 3px; font-size: 0.88em; }
    .mermaid { background: #2a2a3e; border-radius: 6px; padding: 16px; margin: 1em 0; text-align: center; }
    .echarts-container { margin: 1em 0; border-radius: 6px; overflow: hidden; }
    hr { border: none; border-top: 1px solid var(--border); margin: 2em 0; }
    ul, ol { padding-left: 1.5em; margin: 0.6em 0; }
    li { margin: 0.3em 0; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <script>window.MathJax = { tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] } };</script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  <nav id="toc"><h2>目录</h2></nav>
  <main>
    <!-- 报告正文 -->
  </main>
  <script>
    mermaid.initialize({ startOnLoad: true, theme: 'dark' });
    const toc = document.getElementById('toc');
    document.querySelectorAll('main h2, main h3').forEach((h, i) => {
      if (!h.id) h.id = 'section-' + i;
      const a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      if (h.tagName === 'H3') a.classList.add('toc-h3');
      toc.appendChild(a);
    });
    const observer = new IntersectionObserver(entries => {
      entries.forEach(e => {
        const link = toc.querySelector(`a[href="#${e.target.id}"]`);
        if (link) link.classList.toggle('active', e.isIntersecting);
      });
    }, { rootMargin: '-10% 0px -80% 0px' });
    document.querySelectorAll('main h2, main h3').forEach(h => observer.observe(h));
  </script>
</body>
</html>
```

---

## Component Templates

### Math Block
```css
.math-block { background: #2a2a3e; border-radius: 6px; padding: 14px 20px; margin: 1em 0; overflow-x: auto; }
.math-block mjx-container[display="true"] { display:block!important; text-align:left!important; margin:0.4em 0!important; line-height:1.6!important; }
.math-block mjx-container[display="true"] mjx-mtr { line-height:1!important; }
```
Usage: `<div class="math-block">$$...$$</div>` for block math, `$...$` for inline.

### Algo-box (pseudocode)
```css
.algo-box { background:#1a1a2e; border:1px solid var(--border); border-radius:6px; padding:18px 22px; margin:1em 0; font-size:.9em; line-height:1.9; }
.algo-box .algo-title { color:var(--heading); font-weight:700; font-size:1em; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid var(--border); }
.algo-box .kw  { color:#cba6f7; font-weight:700; }   /* keywords: for/if/return */
.algo-box .cm  { color:#6c7086; font-style:italic; } /* comments */
.algo-box .var { color:#94e2d5; }                    /* variables/tensors */
.algo-box .fn  { color:#a6e3a1; }                    /* function calls */
.algo-box .num { color:#f9e2af; }                    /* numbers/hyperparams */
```

### HTML Card Timeline (replaces Mermaid `timeline`)
```html
<style>
  .timeline-wrap { overflow-x: auto; margin: 1em 0; }
  .timeline { display: flex; gap: 0; min-width: 900px; }
  .tl-col { flex: 1; display: flex; flex-direction: column; }
  .tl-period { background:#2a2a3e; border:1px solid var(--border); text-align:center; font-weight:700; color:var(--heading); padding:10px 6px; font-size:0.95em; border-radius:6px 6px 0 0; }
  .tl-body { flex:1; padding:10px 8px; border-left:1px solid var(--border); border-right:1px solid var(--border); border-bottom:1px solid var(--border); border-radius:0 0 6px 6px; }
  .tl-item { background:#313244; border-radius:5px; padding:7px 10px; margin-bottom:8px; font-size:0.88em; line-height:1.5; }
  .tl-tag { display:inline-block; font-size:0.75em; font-weight:700; padding:1px 6px; border-radius:10px; margin-bottom:4px; background:var(--heading); color:#1e1e2e; }
  .tl-tag.orange { background:#fab387; }
  .tl-tag.green  { background:#a6e3a1; }
  .tl-tag.purple { background:#cba6f7; }
</style>
```

---

## Rules & Gotchas

### Math Formula Rules (critical)
| Wrong | Correct |
|-------|---------|
| `$o_{<t}$` | `$o_{\lt t}$` — `<` in any formula must be `\lt` |
| `$$o_{<t}$$` | `$$o_{\lt t}$$` — same for block math |
| `y^*` | `y^{*}` — bare `*` needs `{}` |
| `\(...\)` | `$...$` |
| `\min\!\left(` | `\min\left(` — remove `\!` |
| `_{\mathrm{old}}` | `_\mathrm{old}` |

Block math placement: `<div class="math-block">$$...$$</div>` or `<p>$$...$$</p>`.

### Mermaid Rules
| Use case | Correct syntax | Avoid |
|----------|---------------|-------|
| Timeline | HTML card template above | `timeline` (broken multi-event) |
| Flow / pipeline | `flowchart TD` or `LR` | — |
| Architecture | `graph TD` | — |
| Sequence | `sequenceDiagram` | participant names with `（）` |
| Multi-model comparison | ECharts | `radar`, multi-series `bar` |
| Tech roadmap | `graph TD` or HTML table | `mindmap` (unstable CJK) |

Mermaid blocks: always `<pre class="mermaid">...</pre>`, never code fences.

Node labels containing `（）`, `/`, `→`, `<br>` must be quoted: `["label"]`.

Subgraph titles: plain single-line text only.

### ECharts Rules
- Container: `<div id="chart-X" class="echarts-container" style="height:Xpx"></div>` — **height is mandatory**
- Background: `backgroundColor: '#2a2a3e'` (not `#24243e`, not `#1a1a2e`)
- Use ECharts when data series ≥ 2 AND data points ≥ 3; otherwise use HTML table

### Layout Rules
- `main`: `margin-left: 240px; max-width: none; padding: 40px 64px;` — **never set max-width** (causes blank areas on wide screens)
- TOC sidebar: `width: 220px` fixed left

### code blocks in `<pre><code>`
- `<` inside code blocks → `&lt;` (e.g., `y_{&lt;t}` in comments), otherwise browser truncates rest of page

---

## PDF Source Files
Original PDFs are in: `~/Downloads/pdf_files/`

```bash
# Extract text
python3 -c "
from pdfminer.high_level import extract_text
text = extract_text('path.pdf')
print(text[:12000])
"

# Save full text for large PDFs
python3 -c "
from pdfminer.high_level import extract_text
text = extract_text('path.pdf')
with open('/tmp/paper_text.txt','w') as f: f.write(text)
print('saved, len=', len(text))
"
# Install: pip3 install pdfminer.six -q
```
