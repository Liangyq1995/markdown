# HuggingFace 论文周报 2026-W27

> 抓取日期：2026-07-03  来源：https://huggingface.co/papers/week/2026-W27

| 排名 | 标题 | arXiv | Upvotes | 机构 | 报告 |
|------|------|-------|---------|------|------|
| 1 | Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent | [2606.30616](https://arxiv.org/abs/2606.30616) | 195 | InternScience | [阅读报告](agents-a1-horizon-scaling-阅读报告.html) |
| 2 | Orca: The World is in Your Mind | [2606.30534](https://arxiv.org/abs/2606.30534) | 194 | 多机构（57位作者） | [阅读报告](orca-world-foundation-model-阅读报告.html) |
| 3 | Agentic Abstention: Do Agents Know When to Stop Instead of Act? | [2606.28733](https://arxiv.org/abs/2606.28733) | 137 | University of Washington | [阅读报告](agentic-abstention-agent-stop-阅读报告.html) |
| 4 | Dockerless: Environment-Free Program Verifier for Coding Agents | [2606.28436](https://arxiv.org/abs/2606.28436) | 98 | ByteDance | [阅读报告](dockerless-env-free-verifier-阅读报告.html) |
| 5 | DOPD: Dual On-policy Distillation | [2606.30626](https://arxiv.org/abs/2606.30626) | 89 | 多机构（16位作者） | [阅读报告](dopd-dual-onpolicy-distillation-阅读报告.html) |
| 6 | LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing | [2606.26740](https://arxiv.org/abs/2606.26740) | 81 | 清华大学 | [阅读报告](liveedit-streaming-video-edit-阅读报告.html) |
| 7 | BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding | [2606.31315](https://arxiv.org/abs/2606.31315) | 78 | — | [阅读报告](blockpilot-speculative-decoding-阅读报告.html) |
| 8 | PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation | [2606.28128](https://arxiv.org/abs/2606.28128) | 72 | DAGroup-PKU / NVIDIA | [阅读报告](physisforcing-robot-world-model-阅读报告.html) |
| 9 | Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs | [2606.27378](https://arxiv.org/abs/2606.27378) | 68 | UBC Okanagan | [阅读报告](latent-thoughts-axioms-llm-阅读报告.html) |
| 10 | GEAR: Guided End-to-End AutoRegression for Image Synthesis | [2606.32039](https://arxiv.org/abs/2606.32039) | 57 | Tencent Hunyuan | [阅读报告](gear-autoregression-image-synthesis-阅读报告.html) |

## 本周主题速览

本周（2026-W27）论文涵盖以下核心方向：

### 🤖 Agent 能力与安全
- **Agents-A1**：用35B MoE + 长轨迹扩展超越万亿参数模型性能
- **Agentic Abstention**：Agent的"及时停止"能力评测与 CONVOLVE 方法
- **Dockerless**：无 Docker 环境的代码补丁验证，支持完整的无环境后训练

### 🌍 世界模型
- **Orca**：通过无意识+有意识双模态学习构建统一世界基础模型
- **PhysisForcing**：物理约束注入扩散视频生成，提升机器人操控世界模拟器的物理合理性

### 🧠 训练方法
- **DOPD**：双重在策略蒸馏，破解特权幻觉问题
- **GEAR**：VQ分词器与AR生成器的端到端联合训练，收敛速度提升10×

### ⚡ 推理加速与视频生成
- **BlockPilot**：实例自适应推测解码，4.20×推理加速
- **LiveEdit**：三阶段蒸馏实现12.66 FPS实时流式视频编辑（ECCV 2026）

### 🔬 基础研究
- **Latent Thoughts Axioms**：LLM潜在思维的公理化评测框架，揭示结构性表征缺陷
