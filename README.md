# DeepSeek V4 技术报告导读

这是一个面向软件工程师和 Agent 开发者的 DeepSeek V4 技术报告中文导读项目。

原始技术报告来自 DeepSeek AI 在 Hugging Face 发布的 DeepSeek V4 系列模型仓库。本仓库保留了报告 PDF，并围绕报告中的架构、训练基础设施、预训练、后训练和 Agent 对齐部分做了分章节解读。

> 本项目不是报告全文翻译，也不是 benchmark 排名分析，而是尝试从工程视角理解 DeepSeek V4 为什么要这样设计，以及这些设计如何影响长上下文、推理成本和 Agent 使用体验。

## 适合谁阅读

- 有后端工程或系统工程经验；
- 理解 LLM / Transformer / KV Cache / Agent / ReAct loop 等基础概念；
- 想理解 DeepSeek V4 技术报告中的核心工程问题；
- 对算法 / LLM训练 / Agent 工程有兴趣

如果你已经熟悉大模型训练、分布式训练和推理框架，本项目可以作为快速导览，2分钟扫完即可；

如果你主要是业务开发或 Agent 应用开发者，本项目会尽量把算法和训练工程术语翻译成更直观的工程问题，可以精读。

## 核心问题

DeepSeek V4 报告可以围绕一个主线来理解：

> 如何同时支持超长上下文、长程推理和可承受成本？

围绕这个问题，报告中的很多设计会自然连在一起：

- 1M context 不只是“上下文窗口变长”，还会带来 prefill 成本、KV Cache 显存、服务吞吐和私有化部署压力；
- CSA / HCA 等注意力压缩机制，是为了让长上下文在推理阶段仍然可承受；
- mHC、Muon、SwiGLU clamp、loss spike 处理等设计，指向大规模训练稳定性；
- FP4 QAT、TileLang、deterministic kernels、训练框架和推理框架，指向低精度与高吞吐的系统落地；
- GRM、OPD、reasoning effort、tool call schema、interleaved thinking，指向后训练阶段对 Agent、代码和工具使用能力的统一吸收。

## 阅读入口

建议先读总览，再按兴趣进入对应章节。

| 文件 | 内容 |
|---|---|
| [00-后端人视角下的DeepSeek V4 技术报告解读.md](./00-%E5%90%8E%E7%AB%AF%E4%BA%BA%E8%A7%86%E8%A7%92%E4%B8%8B%E7%9A%84DeepSeek%20V4%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A%E8%A7%A3%E8%AF%BB.md) | 全文总览、阅读路径、核心问题和前置知识 |
| [01-第二章 Architecture.md](./01-%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Architecture.md) | MoE、mHC、CSA / HCA、Muon 等架构设计 |
| [02-第三章 General Infrastructure.md](./02-%E7%AC%AC%E4%B8%89%E7%AB%A0%20General%20Infrastructure.md) | Expert Parallelism、TileLang、deterministic kernels、FP4 QAT、训练和推理基础设施 |
| [03-第四章 Pre-Training.md](./03-%E7%AC%AC%E5%9B%9B%E7%AB%A0%20Pre-Training.md) | 数据构造、Flash / Pro 规模差异、训练节奏、训练稳定性和 base model 评测 |
| [04-第五章 Post-Training.md](./04-%E7%AC%AC%E4%BA%94%E7%AB%A0%20Post-Training.md) | 后训练 pipeline、GRM、OPD、RL 基础设施、Agent / coding / tool use 评测 |
| [模型参数配置表与导读的对照关系.md](./%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE%E8%A1%A8%E4%B8%8E%E5%AF%BC%E8%AF%BB%E7%9A%84%E5%AF%B9%E7%85%A7%E5%85%B3%E7%B3%BB.md) | 将 `config.json` 中的关键字段映射到导读中的概念 |

也可以按兴趣选择入口：

| 关注点 | 建议入口 |
|---|---|
| 算法架构 | 第 2 章 Architecture |
| GPU / 训练基础设施 | 第 3 章 General Infrastructure |
| 预训练组织方式 | 第 4 章 Pre-Training |
| 后训练、Agent、工具调用 | 第 5 章 Post-Training |
| 只想理解 DeepSeek V4 对 Agent 的影响 | 先读总览，再重点读第 5.1 节 |

## 仓库结构

```text
.
├── DeepSeek_V4.pdf
├── 00-后端人视角下的DeepSeek V4 技术报告解读.md
├── 01-第二章 Architecture.md
├── 02-第三章 General Infrastructure.md
├── 03-第四章 Pre-Training.md
├── 04-第五章 Post-Training.md
├── 模型参数配置表与导读的对照关系.md
├── assets/
│   └── image/
├── config/
│   ├── deepseek-v4-flash-config.json
│   └── deepseek-v4-pro-config.json
├── scripts/
│   ├── affinity_scores_sample.py
│   ├── mHC_sample.py
│   └── newton_schulz_sample.py
└── requirements.txt
```

## 配置文件和脚本

`config/` 中放置了 DeepSeek V4 Flash 和 Pro 的模型配置，用于对照理解模型结构参数。建议在读完架构章节后，再回到配置表看字段含义。

`scripts/` 中是几个帮助理解数学过程的 toy examples：

- `affinity_scores_sample.py`：用于观察 affinity score 相关函数形态；
- `mHC_sample.py`：用于辅助理解 mHC 中的连接和约束直觉；
- `newton_schulz_sample.py`：用于演示 Newton-Schulz 迭代的基本过程。

这些脚本只用于帮助理解，不代表 DeepSeek V4 的真实训练或推理实现。

运行脚本前可以先安装依赖：

```bash
pip install -r requirements.txt
```

## 原始资料

- Hugging Face 模型仓库：[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- Hugging Face 模型仓库：[deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- Hugging Face 技术报告 PDF：[DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)

## 免责声明

- 本项目包含大量个人主观理解和工程化解释，不能替代英文技术报告原文。
- 为了可读性，部分数学过程、训练细节和系统实现做了简化。
- 如果导读内容与原文存在歧义，应以 DeepSeek AI 发布的英文技术报告为准。
- 本项目不讨论模型能力排名，也不提供 DeepSeek V4 接入具体 coding agent 的教程。
