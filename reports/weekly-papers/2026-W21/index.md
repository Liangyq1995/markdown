# HuggingFace 论文周报 2026-W21

> 抓取日期：2026-05-29  来源：https://huggingface.co/papers/week/2026-W21

| 排名 | 标题 | arXiv | Upvotes | 报告 |
|------|------|-------|---------|------|
| 1 | Code as Agent Harness | [2605.18747](https://arxiv.org/abs/2605.18747) | 285 | [阅读报告](code-agent-harness-阅读报告.html) |
| 2 | CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence | [2605.12882](https://arxiv.org/abs/2605.12882) | 269 | [阅读报告](citevqa-document-attribution-阅读报告.html) |
| 3 | DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards | [2605.21467](https://arxiv.org/abs/2605.21467) | 204 | [阅读报告](delta-token-credit-rlvr-阅读报告.html) |
| 4 | AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration | [2605.20025](https://arxiv.org/abs/2605.20025) | 185 | [阅读报告](auto-research-claw-multiagent-阅读报告.html) |
| 5 | TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation | [2605.22355](https://arxiv.org/abs/2605.22355) | 175 | [阅读报告](transitlm-mapfree-routing-阅读报告.html) |
| 6 | Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality? | [2605.22109](https://arxiv.org/abs/2605.22109) | 169 | [阅读报告](personality-mllm-prejudice-阅读报告.html) |
| 7 | When Vision Speaks for Sound | [2605.16403](https://arxiv.org/abs/2605.16403) | 149 | [阅读报告](vision-sound-audiovisual-阅读报告.html) |
| 8 | Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining | [2605.14747](https://arxiv.org/abs/2605.14747) | 144 | [阅读报告](video2gui-interaction-pretraining-阅读报告.html) |
| 9 | PhysBrain 1.0 Technical Report | [2605.15298](https://arxiv.org/abs/2605.15298) | 143 | [阅读报告](physbrain-embodied-vlm-阅读报告.html) |
| 10 | Mega-ASR: Towards In-the-wild² Speech Recognition via Scaling up Real-world Acoustic Simulation | [2605.19833](https://arxiv.org/abs/2605.19833) | 131 | [阅读报告](mega-asr-robust-recognition-阅读报告.html) |

---

## 本周亮点

本周论文以 **Agent系统**、**推理强化学习**、**多模态理解** 和 **具身智能** 为主要议题：

- **🏆 最高关注**：*Code as Agent Harness*（285 upvotes）——系统综述代码作为智能体载体的统一框架，三层结构涵盖接口、机制、多智能体协作
- **📊 基准贡献**：*CiteVQA*（269 upvotes）——揭示文档问答中"答对引错"的系统性幻觉，SAA 指标首次联合评估答案与引用证据
- **🔬 训练优化**：*DelTA*（204 upvotes）——从判别器视角重新解读 RLVR，通过 Token 系数重加权在7个数学基准上提升 2-3 分
- **🤖 自主科研**：*AutoResearchClaw*（185 upvotes）——多智能体辩论 + 自愈执行器，在 ARC-Bench 上超越 AI Scientist v2 达 54.7%
- **🗺️ 城市导航**：*TransitLM*（175 upvotes）——用 1300 万条路线记录训练无地图公交规划大模型
- **👁️ 人格感知**：*MM-OCEAN*（169 upvotes）——51% 正确评分无法溯源行为证据，揭示 MLLM 人格理解"偏见鸿沟"
- **🔊 音视频对齐**：*When Vision Speaks for Sound*（149 upvotes）——Thud 框架检测并修复 Clever Hans 效应，提升 28pp
- **🖥️ GUI Agent**：*Video2GUI*（144 upvotes）——从 5 亿视频元数据自动提取 1200 万条 GUI 交互轨迹（ICML 2026）
- **🦾 具身智能**：*PhysBrain*（143 upvotes）——人类第一视角视频注入物理常识，SimplerEnv 域外泛化 SOTA
- **🎙️ 语音识别**：*Mega-ASR*（131 upvotes）——渐进式声学-语义训练，复合场景 WER 相对降低 30%+
