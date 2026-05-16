# Reports Directory — Claude Code Context

## Directory Purpose
This directory contains HTML reading reports and technical analyses for LLM/AI research papers, organized by topic category. All reports use a dark-theme (Catppuccin Mocha) layout with MathJax formulas, Mermaid diagrams, ECharts visualizations, and a fixed left-side TOC.

## Structure
```
reports/
├── llm-surveys/          # Survey papers on LLMs (broad coverage, multi-topic)
├── paper-reading/        # Deep-dive reading reports for individual papers
├── agent-systems/        # LLM-based agent architectures and memory systems
├── rl-training/          # Reinforcement learning: algorithms, LLM post-training
│   ├── easy-rl-reports/  # easy-rl textbook chapter reports
│   └── reinforce/        # Standalone RL algorithm deep-dives
├── model-reports/        # Specific model families (DeepSeek, Kimi, LLaMA, etc.)
├── ml-foundations/       # Math, statistics, deep learning fundamentals
├── blog-digests/         # Blog series summaries
│   ├── sebastianraschka-reports/
│   └── lilianweng-reports/
└── llmbook/              # LLM textbook chapter reports (14 chapters)
```

## Report Generation Rules
When generating new HTML reports for this directory:

1. **Format**: Use the standard dark-theme template (see existing files for reference).
2. **Save location**: Place in the appropriate subdirectory based on content type.
3. **Naming**: `{PaperName}-阅读报告.html` for reading reports, `{Topic}-技术解析.html` for tech analyses.

### HTML Template Key Points
- CSS vars: `--bg:#1e1e2e`, `--heading:#89b4fa`, `--strong:#a6e3a1`, etc.
- Fixed TOC: 240px left sidebar with scroll-spy
- Scripts: mermaid@11, echarts@5, mathjax@3 (all from CDN)
- Mermaid: use `<pre class="mermaid">`, NOT code fences
- ECharts: container must have explicit `style="height:Xpx"`, `backgroundColor:'#24243e'`

### Math Formula Rules (critical)
- Inline: `$...$`, Block: wrap in `<div class="math-block">$$...$$</div>`
- **`<` in formulas → must write `\lt`** (prevents HTML parser truncation)
- `.math-block` CSS must include MathJax override rules:
  ```css
  .math-block mjx-container[display="true"] { display:block!important; text-align:left!important; margin:0.4em 0!important; line-height:1.6!important; }
  .math-block mjx-container[display="true"] mjx-mtr { line-height:1!important; }
  ```
- `background:#2a2a3e` (not `#1a1a2e`) for math blocks — matches reference style

### Algo-box (pseudocode)
```css
.algo-box { background:#1a1a2e; border:1px solid var(--border); border-radius:6px; padding:18px 22px; }
.algo-box .kw { color:#cba6f7; font-weight:700; }   /* keywords */
.algo-box .cm { color:#6c7086; font-style:italic; } /* comments */
.algo-box .var { color:#94e2d5; }                   /* variables */
.algo-box .fn { color:#a6e3a1; }                    /* functions */
```

## PDF Source Files
Original PDFs are in: `~/Downloads/pdf_files/`
Extract with: `python3 -c "from pdfminer.high_level import extract_text; print(extract_text('path.pdf')[:12000])"`

## What NOT to do
- Do not use `timeline`, `mindmap`, `radar` in Mermaid (unstable rendering)
- Do not use multi-series `bar` in Mermaid (use ECharts instead)
- Do not put `<` directly in `$...$` or `$$...$$` — use `\lt`
- Do not set `max-width` on `main` (causes large blank areas on wide screens)
