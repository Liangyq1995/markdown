# LLM/AI 技术报告库

> 基于 Catppuccin Mocha 深色主题的 HTML 阅读报告集合，覆盖 LLM 后训练、推理系统、Agent、RL 算法、模型架构等核心领域。
> 所有报告含 MathJax 公式、Mermaid 流程图、ECharts 数据可视化、左侧固定目录导航，可直接用浏览器打开。

---

## 目录

- [顶层综合报告](#顶层综合报告)
- [RL 训练与 LLM 后训练](#rl-训练与-llm-后训练)
- [论文精读](#论文精读)
- [Agent 系统](#agent-系统)
- [模型报告](#模型报告)
- [ML 基础](#ml-基础)
- [博客摘要](#博客摘要)
- [LLM 教材（14章）](#llm-教材14章)

---

## 顶层综合报告

> 根目录下的跨类别综合报告

| 报告 | 内容 |
|------|------|
| [Agent RL 训练深度报告](Agent-RL训练深度报告.html) | Agent 与 RL 训练结合的综合解析 |
| [Gateway in AgentRL 深度报告](Gateway-in-AgentRL-深度报告.html) | Gateway 在 AgentRL 中的角色 |
| [LLM Gateway 深度技术报告](LLM-Gateway-深度技术报告.html) | LLM Gateway 系统设计 |
| [LLM MTP 训练技术报告](LLM-MTP训练技术报告.html) | Multi-Token Prediction 训练技术 |
| [LLM 工具服务器评估指南](llm-tool-server-evaluation.html) | 使用方视角的 LLM 工具服务器评估 |
| [MoE LLM RL 训练报告](MoE-LLM-RL训练报告.html) | MoE 模型的 RL 训练 |

---

## RL 训练与 LLM 后训练

> 路径：`rl-training/`

### 综合技术解析

| 报告 | 核心内容 |
|------|----------|
| [LLM 后训练技术全景](rl-training/LLM后训练技术全景-技术解析.html) | SFT/RLHF/DPO/GRPO/主流模型实践（网络调研版）|
| [LLM RL 训练技术深度解析](rl-training/LLM-RL训练技术深度解析.html) | PPO、GRPO、DeepSeek-R1、Qwen、LLaMA 后训练 |
| [强化学习算法演进报告](rl-training/强化学习算法演进报告.html) | RL 核心算法历史演进 |
| [强化学习与 LLM 技术报告](rl-training/强化学习与LLM技术报告.html) | RL 在 LLM 中的应用综述 |
| [奖励模型演进报告](rl-training/奖励模型演进报告.html) | Reward Model 设计与演进 |
| [verl RL 算法全景分析](rl-training/verl-rl-algorithms-report.html) | verl 框架 RL 算法全景解析 |

### Easy-RL 教材系列（13章）

> 路径：`rl-training/easy-rl-reports/`

| 报告 | 章节 |
|------|------|
| [第1章 强化学习基础](rl-training/easy-rl-reports/第1章-强化学习基础.html) | MDP、价值函数、策略 |
| [第2-3章 MDP与表格型方法](rl-training/easy-rl-reports/第2-3章-MDP与表格型方法.html) | Q-Learning、SARSA |
| [第4-5章 策略梯度与PPO](rl-training/easy-rl-reports/第4-5章-策略梯度与PPO.html) | REINFORCE、PPO 详解 |
| [第6-7-8章 DQN系列](rl-training/easy-rl-reports/第6-7-8章-DQN系列.html) | DQN、Double DQN、Dueling DQN |
| [第9章 演员评论员算法](rl-training/easy-rl-reports/第9章-演员评论员算法.html) | A2C、A3C、SAC |
| [第10-11章 稀疏奖励与模仿学习](rl-training/easy-rl-reports/第10-11章-稀疏奖励与模仿学习.html) | HER、GAIL |
| [第12章 DDPG与TD3](rl-training/easy-rl-reports/第12章-DDPG与TD3.html) | 连续动作空间 RL |
| [第13章 AlphaStar解读](rl-training/easy-rl-reports/第13章-AlphaStar解读.html) | 星际争霸 AI 技术 |

### Reinforce 独立算法系列

> 路径：`rl-training/reinforce/`

| 报告 | 内容 |
|------|------|
| [初探强化学习](rl-training/reinforce/初探强化学习-技术解析.html) | RL 入门概念 |
| [多臂老虎机](rl-training/reinforce/多臂老虎机-技术解析.html) | Bandit 问题 |
| [马尔可夫决策过程](rl-training/reinforce/马尔可夫决策过程-技术解析.html) | MDP 数学基础 |
| [动态规划与时序差分](rl-training/reinforce/动态规划与时序差分-技术解析.html) | DP、TD、MC |
| [策略梯度与AC](rl-training/reinforce/策略梯度与AC-技术解析.html) | REINFORCE、Actor-Critic |
| [TRPO与PPO](rl-training/reinforce/TRPO与PPO-技术解析.html) | 信任域策略优化 |
| [DQN算法](rl-training/reinforce/DQN算法-技术解析.html) | Deep Q-Network |
| [DDPG与SAC](rl-training/reinforce/DDPG与SAC-技术解析.html) | 连续控制算法 |
| [模仿与基于模型的RL](rl-training/reinforce/模仿与基于模型的RL-技术解析.html) | IL、Model-Based RL |
| [离线目标导向多智能体RL](rl-training/reinforce/离线目标导向多智能体RL-技术解析.html) | Offline MARL |

---

## 论文精读

> 路径：`paper-reading/`

### 推理系统

| 报告 | 核心内容 |
|------|----------|
| [LLM 推理 PD 分离](paper-reading/LLM推理PD分离-技术解析.html) | Prefill-Decode 解耦：DistServe/Splitwise/Mooncake/Dynamo |

### Agent 与规划

| 报告 | 论文 |
|------|------|
| [Tree of Thoughts](paper-reading/Tree-of-Thoughts-阅读报告.html) | ToT 推理框架 |
| [Agent World](paper-reading/Agent-World-阅读报告.html) | Agent 与世界模型 |
| [ATLAS](paper-reading/ATLAS-阅读报告.html) | ATLAS 论文精读 |
| [AutoSearch](paper-reading/AutoSearch-阅读报告.html) | 自动搜索策略 |
| [SKILL0](paper-reading/SKILL0-阅读报告.html) | Zero-shot Skill 学习 |

### 记忆与上下文

| 报告 | 论文 |
|------|------|
| [Rethinking Memory Mechanisms](paper-reading/Rethinking-Memory-Mechanisms-阅读报告.html) | LLM 记忆机制再思考 |
| [The Latent Space](paper-reading/The-Latent-Space-阅读报告.html) | 潜在空间理论 |
| [LORE](paper-reading/LORE-阅读报告.html) | 长期记忆表示 |

### RL 与偏好优化

| 报告 | 论文 |
|------|------|
| [FIPO](paper-reading/FIPO-阅读报告.html) | Fine-grained IPO |
| [TRPO 精读](paper-reading/TRPO-阅读报告.html) | Trust Region Policy Optimization |
| [Reward Models in DRL](paper-reading/Reward-Models-in-DRL-阅读报告.html) | 深度 RL 中的奖励模型 |
| [Weak-Driven Learning](paper-reading/Weak-Driven-Learning-阅读报告.html) | 弱监督学习 |
| [R3](paper-reading/R3-阅读报告.html) | R3 论文精读 |

### 架构与效率

| 报告 | 论文 |
|------|------|
| [MoBA](paper-reading/MoBA-阅读报告.html) | Mixture of Block Attention |
| [On-Policy Distillation 综合](paper-reading/On-Policy-Distillation-综合阅读报告.html) | 在线策略蒸馏综合解读 |
| [Meta Harness](paper-reading/Meta-Harness-阅读报告.html) | Meta 训练框架 |

---

## Agent 系统

> 路径：`agent-systems/`

| 报告 | 核心内容 |
|------|----------|
| [LLM Agent RL 深度解析](agent-systems/LLM-Agent-RL深度解析.html) | Agent + RL 结合 |
| [LLM Agent 进展报告](agent-systems/llm-agent-进展报告.html) | Agent 领域最新进展 |
| [LLM Powered Autonomous Agents 技术解析](agent-systems/LLM-Powered-Autonomous-Agents-技术解析.html) | Lilian Weng 博文技术解析 |
| [MCP Atlas 阅读报告](agent-systems/MCP-Atlas-阅读报告.html) | Model Context Protocol |
| [Perplexity Agent Skills](agent-systems/Perplexity-Agent-Skills报告.html) | Perplexity 的 Agent 技能体系 |

---

## 模型报告

> 路径：`model-reports/`

### 开源基础模型

| 报告 | 模型 |
|------|------|
| [DeepSeek 技术详解](model-reports/DeepSeek-技术详解报告.html) | DeepSeek 系列全解析 |
| [Kimi 技术详解](model-reports/Kimi-技术详解报告.html) | Kimi/Moonshot AI |
| [LLaMA 技术详解](model-reports/LLaMA-技术详解报告.html) | LLaMA 1/2/3 系列 |
| [Qwen 技术详解](model-reports/Qwen-技术详解报告.html) | Qwen 系列 |
| [GLM 系列技术详解](model-reports/GLM-系列技术详解报告.html) | GLM/ChatGLM 系列 |
| [Transformer 架构演进](model-reports/Transformer-架构演进报告.html) | Transformer 发展历程 |
| [多模态模型综合研究](model-reports/多模态模型-综合研究报告.html) | 视觉语言模型 |

### 训练与推理工程

| 报告 | 内容 |
|------|------|
| [Ultrascale Playbook](model-reports/ultrascale-playbook-技术解析.html) | 超大规模训练工程 |
| [Smol Training Playbook](model-reports/smol-training-playbook-技术解析.html) | 小模型高效训练 |
| [HF Context Course](model-reports/HF-Context-Course-技术解析.html) | HuggingFace 上下文课程 |
| [Diffusion Models](model-reports/diffusion-models-技术解析.html) | 扩散模型技术解析 |

### 语音与专项

| 报告 | 内容 |
|------|------|
| [CosyVoice 系列](model-reports/CosyVoice-系列-阅读报告.html) | 语音合成模型 |
| [Qwen3 ASR/TTS](model-reports/Qwen3-ASR-TTS-阅读报告.html) | 语音理解与合成 |

### Claude / Claude Code

| 报告 | 内容 |
|------|------|
| [Claude Code Harness](model-reports/Claude-Code-Harness-技术解析.html) | Claude Code 技术架构 |
| [Claude Skills Deep Dive](model-reports/claude-skills-deep-dive-技术解析.html) | Claude 技能系统深度解析 |

---

## ML 基础

> 路径：`ml-foundations/`

| 报告 | 内容 |
|------|------|
| [数学基础学习报告](ml-foundations/数学基础-学习报告.html) | 线代、概率、优化数学基础 |
| [统计学习算法综合报告](ml-foundations/统计学习算法-综合报告.html) | SVM、决策树、集成学习等 |
| [深度学习综合报告](ml-foundations/深度学习-综合报告.html) | 神经网络、CNN、RNN 等 |
| [图神经网络综合报告](ml-foundations/图神经网络-综合报告.html) | GNN、GCN、GAT 等 |

---

## 博客摘要

> 路径：`blog-digests/`

### Lilian Weng 博客系列（8卷）

> 路径：`blog-digests/lilianweng-reports/`

| 报告 | 主题 |
|------|------|
| [01 强化学习基础](blog-digests/lilianweng-reports/01-强化学习基础.html) | RL 基础、策略梯度、Actor-Critic |
| [02 Meta 学习与迁移](blog-digests/lilianweng-reports/02-Meta学习与迁移.html) | MAML、元学习、迁移学习 |
| [03 Transformer 与 LLM 架构](blog-digests/lilianweng-reports/03-Transformer与LLM架构.html) | Attention、位置编码、LLM 架构 |
| [04 LLM 应用与对齐](blog-digests/lilianweng-reports/04-LLM应用与对齐.html) | RLHF、对齐、Agent、Prompt |
| [05 生成模型](blog-digests/lilianweng-reports/05-生成模型.html) | VAE、GAN、Diffusion、Flow |
| [06 自监督学习与数据](blog-digests/lilianweng-reports/06-自监督学习与数据.html) | Contrastive Learning、数据增强 |
| [07 目标检测与视觉](blog-digests/lilianweng-reports/07-目标检测与视觉.html) | YOLO、ViT、视觉架构 |
| [08 ML 基础理论](blog-digests/lilianweng-reports/08-ML基础理论.html) | 优化理论、泛化界、PAC 学习 |

---

## LLM 教材（14章）

> 路径：`llmbook/`，基于《大语言模型》教材系统梳理

| 报告 | 章节内容 |
|------|----------|
| [第1章 概述](llmbook/第1章_概述.html) | LLM 发展背景与全景 |
| [第2章 大语言模型背景](llmbook/第2章_大语言模型背景.html) | NLP 基础、预训练范式 |
| [第3章 大语言模型系统](llmbook/第3章_大语言模型系统.html) | 系统架构与工程 |
| [第4章 数据工程](llmbook/第4章_数据工程.html) | 预训练数据处理 |
| [第5章 模型架构](llmbook/第5章_模型架构.html) | Transformer 变体与创新 |
| [第6章 模型预训练](llmbook/第6章_模型预训练.html) | 预训练算法与工程 |
| [第7章 指令微调](llmbook/第7章_指令微调.html) | SFT、数据构造 |
| [第8章 人类反馈强化学习](llmbook/第8章_人类反馈强化学习.html) | RLHF、PPO、DPO |
| [第9章 应用配置与推理](llmbook/第9章_应用配置与推理.html) | 推理优化、量化、部署 |
| [第10章 提示学习](llmbook/第10章_提示学习.html) | Prompt Engineering、CoT |
| [第11章 规划与 Agent](llmbook/第11章_规划与Agent.html) | ReAct、工具调用、规划 |
| [第12章 评测](llmbook/第12章_评测.html) | 评估基准与方法 |
| [第13章 应用](llmbook/第13章_应用.html) | 垂直场景应用 |
| [第14章 总结与展望](llmbook/第14章_总结与展望.html) | 挑战、趋势与未来 |

---

## 生成说明

所有报告由 **Claude Code + `llm-paper-reading` skill** 生成，采用：
- 🌙 **主题**：Catppuccin Mocha 深色
- 📐 **公式**：MathJax 3（含 `\lt` 转义规则）
- 🔀 **流程图**：Mermaid 11
- 📊 **图表**：ECharts 5
- 📚 **来源**：原始 PDF（pdfminer.six）+ Tavily 网络调研

更多生成规则参见 [CLAUDE.md](CLAUDE.md)。
