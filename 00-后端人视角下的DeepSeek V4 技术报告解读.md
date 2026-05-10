## 🧭 阅读说明
DeepSeek V4 技术报告总共分为五章：
1. Introduction，全文总览
1. Archietecture，介绍了从V3时代继承过来的算法设计，和V4引入的新算法。数学含量很高较为抽象，但数学概念理解本身门槛不高。<text underline="true">***阅读门槛：高***</text>
1. General Infrastrures，介绍从第2章新提出的算法方案出发，落地到真实的GPU上的训练中的工程方案，如无相关知识基础理解门槛极高。<text underline="true">***阅读门槛：高***</text>
1. Pre-Training，介绍预训练中的工程方案。<text underline="true">***阅读门槛：适中***</text>
1. Post-Trainig，介绍后训练中的工程方案，如果有Agent开发基础，理解会容易不少。<text underline="true">***阅读门槛：适中***</text>
具体重点如下：
```markdown
第2章：Architecture
  - CSA / HCA （长上下文KV cache压缩机制）
  - mHC （残差流与层间连接稳定性优化）
  - Muon (大规模训练优化器）

第3章：General Infrastructures
  - kernel （GPU算子与训练吞吐优化）
  - FP4 QAT （面向低精度训练/推理的量化感知训练）
  - 训练框架 （分布式训练、并行策略与调度优化）
  - KV cache 管理框架（长上下文推理的显存与吞吐优化）

第4章：Pre-Training
  - 数据 （预训练数据组织与长上下文数据构造）
  - 训练 schedule （预训练阶段的训练节奏与阶段安排）
  - 稳定性处理 （训练不稳定与 loss spike 处理）
  - base model 评测

第5章：Post-Training
  - GRM (unverified task 的奖励建模/评价优化）
  - OPD（多专家策略到统一模型的在线策略蒸馏）
  - reasoning / agent / coding 对齐 （面向Agent推理、工具使用和代码任务的行为对齐）
```


#### 阅读指引
- 对算法本身感兴趣的，从 **第2章/Architecture **开始读
- 对数学/算法没兴趣，但对GPU基建有兴趣的，从 **第3章/General Infrastructure **开始读
- 对数学/算法/GPU基建工程没兴趣，但对模型预训练感兴趣的，从 **第4章/Pre-Training** 开始读
- 对模型后训练感兴趣的，从 **第5章/Post-Training** 开始读
- 对模型的所有算法/GPU基建/模型训练均无兴趣，只想看看在Agent中怎么用的，挑着看 **第5章/Post-Training** 即可

**本文关心：**
1. DeepSeek V4的训练全过程中的算法和工程要点
1. DeepSeek V4的训练全过程中算法和工程优化对Agent开发的影响和意义
**本文不关心：**
1. 具体的benchmark跑分/具体coding或agent case的表现
1. 官方/非官方公布的模型能力水平
1. DeepSeek V4如何接入现有coding agent（cc, codex等）
**本文可能有以下帮助：**
1. 如何更好在agent中使用DeepSeek V4
1. 解释一些DeepSeek V4在agent中使用的issue

<text underline="true">***阅读 本文全文 的必须前置知识（有这些知识仅能保证阅读能继续，不代表可以顺畅理解）***</text>：
1. Dense模型 / MoE模型 的特征和优劣
1. LLM的基础概念：Transformer / Prefill / KV Cache 等
1. 后训练基础概念： SFT / RL 等
1. 分布式训练的基础概念
1. agent / react loop 的基础概念

本文后续以尽可能通俗+提纲挈领的方式，说明DeepSeek V4在算法和工程上具体做了什么，意义是什么

## 🌍 核心问题
<text underline="true">***DeepSeek V4报告中，全程围绕这个核心问题展开：超长上下文、长程推理和可承受成本三件事从算法+工程上系统性的给出可行方案***</text>

与 DeepSeek V3 时代相比，V4 面临的问题发生了变化：
1. **长上下文从附加能力变成基础设施。**
V3 发布时最一开始只支持32k上下文，当时对长上下文的需求还不显著，deepseek提供一个廉价的模型服务已经是大功一件；V4 时代，1M 上下文已经成为一线大模型很普遍的支持情形，deepseek必须面对超长上下文下的推理成本、KV Cache 成本和服务成本。
2. **Agent 从边缘场景变成核心检验场景。**
模型不再只是回答单轮问题，而是要在 ReAct loop、tool call、多轮状态和长任务轨迹中持续工作。因此，模型能否在长历史中低成本提取关键信息，成为影响 agent 体验的核心因素。
3. **推理预算变得更重要，但也更昂贵。**
给模型更多上下文、更多中间步骤和更多推理时间，通常能提升复杂任务效果；但在真实服务中，长 prefill、长 decode、KV Cache 驻留和多轮工具调用都会迅速放大成本。V4 的关键目标之一，就是一方面能让模型能支持长推理，同时控制成本

对问题进行拆分：
1. 模型需要带着很长的历史上下文继续推理，但 KV Cache、FLOPs、显存和服务成本不能随上下文长度失控增长。国内显卡资源紧张，且DeepSeek V4在预期之内的要大规模私有部署甚至用华为硬件，必须考虑硬件成本
2. 在 agent、coding、reasoning 这类超长轨迹任务中，模型需要低成本地找到整个 session 中对当前 run 最有用的信息，而不是把所有历史 token 都以同等成本重新读取。
3. 这些设计不能只停留在算法层面，还必须能在真实 GPU 训练、推理和服务系统中稳定运行。
