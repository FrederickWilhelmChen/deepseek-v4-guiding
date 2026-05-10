# 🌱 第 4 章 Pre-Training：预训练工程

> 本章回答的是：在算法结构和 GPU 基础设施准备好之后，DeepSeek V4 如何组织真正的预训练过程。

---

## 🧭 本章速览

这一章主要讨论三件事：

| 模块 | 核心问题 | 读者应关注什么 |
|---|---|---|
| Data Construction | 用什么数据训练 | 数据清洗、长文档、FIM、sample-level attention mask |
| Model / Training Setups | 怎么安排训练节奏 | Flash / Pro 差异、序列长度递进、CSA 引入时机 |
| Training Instability | 训练不稳定怎么办 | Loss spike、Anticipatory Routing、SwiGLU Clamping |
| Evaluation | 基模效果怎么看 | 不盯分数，重点看不同 benchmark 反映的能力差异 |

```mermaid
flowchart TD
    A[Pre-Training] --> B[Data Construction]
    A --> C[Model Setups]
    A --> D[Training Setups]
    A --> E[Stability]
    A --> F[Evaluation]

    B --> B1[清洗自动生成 / 模板化网页]
    B --> B2[保留数学 / 代码 / agentic / long text]
    B --> B3[Tokenizer / FIM / Packing]
    B --> B4[Sample-level Attention Mask]

    C --> C1[Flash]
    C --> C2[Pro]

    D --> D1[AdamW + Muon]
    D --> D2[4K -> 16K -> 64K -> 1M]
    D --> D3[CSA warmup]

    E --> E1[Anticipatory Routing]
    E --> E2[SwiGLU Clamping]
```

---

# 1️⃣ Data Construction：预训练数据选择和组织

DeepSeek 在 V3 预训练数据基础上，构造了一个更多样、更高质量、有效上下文更长的数据集。

最终：

- V4 Flash 数据集规模超过 **32T token**。
- V4 Pro 数据集规模超过 **33T token**。

---

## 🧹 1.1 过滤批量自动生成和模板化网页内容

报告原文提到，过滤这两类网页内容主要是为了防止 **model collapse**。

这里可以通俗理解为：

> 如果训练数据里充满 LLM 自动生成的模板化内容，模型会把这些垃圾样本当成真实世界的常见分布，最后学到失真的语言和知识结构。

这和工程里常说的 **Garbage in, Garbage out** 是一个道理。

当前模型厂商已经普遍意识到：模型生成的大量模板化内容会污染后续训练，必须做专项清洗。

---

## 🧮 1.2 保留数学、编程、Agentic、长文档语料

DeepSeek 更重视这些语料：

- 数学语料；
- coding 语料；
- agentic data；
- long text，例如论文、技术报告等。

这里特别值得注意的是 **长文档数据**。

对长上下文训练而言，语料组织大体有两种方式：

| 方式 | 特点 | 问题 |
|---|---|---|
| 多段短文本拼接 | 可以快速构造长序列，提高 token 利用率 | 不同段落语义未必连续 |
| 原生长文本 | 逻辑自然连续，更适合训练长程依赖 | 高质量长文档语料稀缺 |

所以，从训练质量看，原生长文本当然更好；但从数据规模看，拼接 packing 仍然是广泛采用的工程手段。

---

## 🧩 1.3 Tokenizer、Token-splitting 和 FIM

DeepSeek 沿用了 V3 tokenizer，仍保持 **128K 词表规模**。

同时加入了一些 special token，并继承了：

- token-splitting；
- FIM；
- 文档 packing 策略。

### Token-splitting：提高 tokenizer 鲁棒性

Token-splitting 可以理解为一种 tokenizer 鲁棒性训练。

DeepSeek tokenizer 中存在一些“标点 + 换行”一类的组合 token，用来提高压缩效率。但这种组合 token 也可能让模型对特定 token 边界产生偏置。

因此训练时随机拆分一部分组合 token，让模型适应多种等价切分方式，减少换行、标点、代码格式造成的边界敏感性。

### FIM：Fill in Middle

FIM 非常匹配 coding 场景。

因为在真实编码中，很多任务不是在文件尾部续写，而是在已有代码的两行之间插入一段新代码。

也就是说，训练阶段需要让模型学会：

```plaintext
给定上文 + 给定下文 -> 生成中间内容
```

这比单纯“从左到右续写”更符合代码编辑任务。

---

## 🎭 1.4 Sample-level Attention Mask

训练过程中，为了提高效率，也为了组装长上下文，常常会把多段不相关文本 packing 到同一个 sequence 里训练。

但对模型来说，它不知道这些文本本来毫无关系。如果不加限制，模型可能从这些无关文本之间学出奇怪关联。

DeepSeek 的方案是：

> **一个 batch sequence 里可以 pack 多个样本，但 attention 上限制不同 sample 之间互相看不到。**

```mermaid
flowchart LR
    subgraph Packed Sequence
        A1[Sample A token 1]
        A2[Sample A token 2]
        B1[Sample B token 1]
        B2[Sample B token 2]
        C1[Sample C token 1]
    end

    A1 --> A2
    B1 --> B2
    A2 -.不能看.-> B1
    B2 -.不能看.-> C1
```

传统 causal mask 解决的是：预测当前位置时不能看未来 token。

Sample-level attention mask 进一步解决的是：

> **不同样本虽然被工程上 pack 到一起，但语义上仍然互相隔离。**

---

# 2️⃣ Model Setups：Flash 和 Pro 的规模差异

DeepSeek V4 发布了 Flash 和 Pro 两个版本。

## ⚡ Flash

```plaintext
43 层 Transformer
总参数 284B，每 token 激活 13B
CSA top-k = 512
HCA 压缩率 = 128
MoE：1 个 shared expert + 256 个 routed experts
每 token 激活 6 个 routed experts
前两层用 dense 的 pure sliding window attention
后续层交替使用 CSA 和 HCA
```

## 🧠 Pro

```plaintext
61 层 Transformer
总参数 1.6T，每 token 激活 49B
CSA top-k = 1024
HCA 压缩率 = 128
MoE：1 个 shared expert + 384 个 routed experts
每 token 激活 6 个 routed experts
前两层直接用 HCA
后续层交替使用 CSA 和 HCA
```

## 🔍 Flash / Pro 前两层 attention 差异

两个模型在前两层注意力构造上有一个值得注意的区别：

- Flash 前两层使用 dense pure sliding window attention。
- Pro 前两层直接使用 HCA。

我对可能原因的推测如下，报告中没有直接说明：

1. Flash 更强调低成本，滑动窗口不需要压缩机制，也不需要配套工程处理。
2. HCA 更像粗粒度全局背景，DeepSeek 可能认为 Flash 低层没有必要太早建立这套机制。
3. Sliding window attention 是稠密且相对简单的，对 Flash 的训练稳定性更友好。

---

# 3️⃣ Training Setups：训练节奏

报告原文把这部分放在 4.2 Model Setups 下面，主要讲三件事。

## 🧰 3.1 优化器组合：AdamW + Muon

Flash 和 Pro 采用相同策略：

- 大多数参数使用 **Muon**。
- embedding、prediction head、所有 RMSNorm 权重使用 **AdamW**。

这体现了一个工程取舍：

> 主要矩阵参数使用 Muon 来提升收敛和稳定性，但敏感参数仍然保留传统 AdamW。

## 📏 3.2 递进的序列长度

Flash 和 Pro 都支持 1M context，但训练不是一开始就直接上 1M。

而是递进式扩展：

```plaintext
4K -> 16K -> 64K -> 1M
```

这很好理解：从零开始训练的模型不可能一开始就读 1M context，就像不可能直接让婴儿读四大名著。

## 🧭 3.3 Sparse Attention 的引入时机

这部分可以理解为：DeepSeek 选择在训练什么阶段真正引入 CSA。

### Flash 的策略

- 前 1T tokens 使用 dense attention warmup。
- 序列长度扩大至 64K 时引入 CSA。
- 引入 CSA 时，先短暂 warmup CSA indexer。
- 然后进入大部分 sparse attention 训练。

### Pro 的策略

Pro 与 Flash 类似，但 dense attention 阶段更长。

## ✅ 训练节奏的意义

这套安排说明两件事：

1. 稠密注意力的基座打底过程必不可少，先在较短文本上学稳定 token 表示和分布，再引入压缩与筛选。
2. CSA indexer 不是一开始就会选择，它需要先从稳定 attention 分布里学习怎么筛选。过早引入 CSA，可能让模型训练早期就走错路。

---

# 4️⃣ Mitigating Training Instability：训练稳定性问题

DeepSeek 在报告中提到，训练过程中遇到了 **loss spike** 问题。

回滚可以暂时恢复，但不能阻止 spike 再次发生。

## 📈 什么是 loss spike

训练的本质，是不断降低模型预测结果与标签之间的差异，这个差异通常用 loss 衡量。

正常训练中，loss 应该总体下降。

而 **loss spike** 是指：

> 本来逐步下降的 loss 中，突然出现显著上扬的尖峰。

这个尖峰可能后续自动消失，也可能直接把训练带崩。

---

## 🔁 4.1 Anticipatory Routing

DeepSeek 从经验角度发现：spike 和 MoE 层里的异常值绑定，而 expert routing 机制似乎会加剧异常值出现。

异常反馈环路可以简化为：

```mermaid
flowchart TD
    A[某个 Expert 出现短暂异常] --> B[中间层输出异常值]
    B --> C[影响下一轮模型参数]
    C --> D[Router 基于受影响参数计算路由]
    D --> E[更多 token 被送到异常 Expert]
    E --> F[异常被自强化]
    F --> G[Loss Spike]
```

DeepSeek 的解决方案是打破这个环路：

```mermaid
flowchart TD
    A[Expert 可能出现短暂异常] --> B[当前轮不直接用当前参数决定路由]
    B --> C[使用若干轮之前的模型参数计算路由]
    C --> D[Router 不被当前异常立即污染]
    D --> E[异常反馈环路被减速或打断]
    E --> F[训练重新稳定]
```

需要强调的是：

> DeepSeek 不是在所有训练中都使用 Anticipatory Routing，而是在 loss spike 发生后，先回滚，再开启 Anticipatory Routing，维持一段时间后，再回到正常训练行为。

---

## ✂️ 4.2 SwiGLU Clamping

如果把 Anticipatory Routing 类比成后端系统里的熔断、回滚、降级、恢复，那么 SwiGLU Clamping 更像是数值稳定性的提前限流。

它的思路不是等异常反馈环路形成后再处理，而是在 FFN 的关键中间变量上直接限幅，防止异常激活值继续放大和扩散。

### SwiGLU 是什么

可以简化理解为：

- SwiGLU 是 Transformer 中常见的门控激活结构。
- 它用来决定哪些信息应该被放大、哪些信息应该被抑制。
- SwiGLU 里存在一个乘法门控结构。
- 某个分支如果偶发出现极端大值，和另一个分支相乘后可能进一步放大，形成异常值。

### Clamping 的作用

既然这些异常值可能污染后续计算，那直接限幅是一个非常工程化的处理方式。

报告原文中设置阈值为 **10**：超过阈值的值直接截断。

这样即使中间变量偶发异常，也会被截断在可控范围内，降低异常值扩散并诱发 loss spike 的概率。

## ⚠️ 对这两个方案的读法

DeepSeek 报告中也提到，这两个方案不是严谨的数学优化，而是经验化工程方案。

也就是说：

- 原因归纳是经验式的；
- 不是完整数学推导；
- 但工程上被证明有效。

---

# 5️⃣ Evaluations：Base Model Benchmark

本节主要是对预训练基模的评测结果。

如导读最开始所说，benchmark 本身不是本文关心的重点，但报告中有一些现象值得拿出来说。

## 🧪 Coding 评测中的有趣现象

对于预训练基模，在 coding 任务上，V4 两个版本和 V3.2 相比并不是一边倒。

报告中提到：

- BigCodeBench 上，V3.2 Base 比 V4 Flash 和 V4 Pro 都更强。
- HumanEval 上，V4 两个版本相较 V3.2 都有明显优势。

这说明不同 coding benchmark 测的不是同一种能力。

## 🧩 HumanEval vs BigCodeBench

| Benchmark | 更像什么 | 主要考察 |
|---|---|---|
| HumanEval | 短函数级代码生成 / 补全 | 基础算法、局部代码生成、语法正确性、简单逻辑 |
| BigCodeBench | 更接近现实开发中的小任务 | 复杂指令理解、Python 库组合、diverse function calls |

HumanEval 更像传统代码生成能力测试：给一个函数签名或说明，让模型补一个较短、相对自包含的 Python 函数。

BigCodeBench 更接近现实开发中的小任务：要求模型理解复杂指令，并组合使用大量 Python 库和函数调用。

## ✅ 小结

这里的分数只是基模评分。

真正开放出来的模型能力，还需要经过大量后训练。

因此这一节更适合拿来观察：

> **不同 benchmark 到底在测什么能力，而不是简单得出“谁更强”的结论。**
