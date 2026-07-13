# GLM-5 SAO：单 Rollout 异步强化学习，以及它与 DeepSeek V4 后训练路线的差异

> 本文是 [《第 5 章：Post-Training 后训练，Agent 行为对齐和 RL 范式转变》](./04-%E7%AC%AC%E4%BA%94%E7%AB%A0%20Post-Training.md) 的补充阅读。
>
> 文中将尽量区分论文事实与个人判断。SAO 与 CompactionRL 的公开材料目前均为 2026 年 7 月发布的 arXiv v1，后续实现和结论仍可能调整。

---

## 🧭 先给结论：SAO 和 OPD 不是一组直接竞争的方案

看到 GLM-5.2 的 SAO 后，最容易产生的一种理解是：

```text
DeepSeek V4：GRPO / OPD
GLM-5.2：放弃 GRPO，改回 PPO / SAO
```

这个概括只说对了一半。

SAO 解决的问题是：

> 在长程 Agent 任务中，rollout 长短差异巨大，环境交互昂贵，并且 rollout 与训练并行推进时，如何用单条 trajectory 稳定地做异步强化学习。

DeepSeek V4 的 OPD 解决的问题则是：

> 数学、代码、Agent、通用指令等不同领域的奖励目标互相冲突时，如何先分别训练专家，再把多个专家的能力合并进一个统一模型。

因此，两者所处层次不同：

| 方法 | 主要解决的问题 | 所处阶段 |
|---|---|---|
| GRPO / SAO | 一个策略模型如何根据环境奖励做 RL 更新 | 专家能力训练或单一领域策略优化 |
| OPD | 多个已经训练好的专家如何合并到统一模型 | 多专家能力整合 |
| CompactionRL | 超长 Agent 轨迹经过上下文压缩后如何继续做 RL | 长程轨迹组织与 credit assignment |

更准确的比较关系是：

1. **SAO 可以和 DeepSeek V4 专家训练阶段使用的 GRPO 比较；**
2. **SAO 与 OPD 大体正交，甚至可以组合使用；**
3. **CompactionRL 与 DeepSeek V4 的百万 token 长上下文路线，则代表两种不同的长程 Agent 上下文策略。**

---

# 一、SAO 为什么要放弃 group-wise optimization

## 1.1 GRPO 中的 group 与 batch 不是一回事

传统 GRPO 通常会针对同一个 prompt 生成多条 rollout，再利用组内奖励计算相对 advantage。

例如：

```text
16 个 prompt
每个 prompt 生成 8 条 rollout
总 batch size = 16 × 8 = 128
```

同一个 prompt 的 8 条 rollout 构成一个 group。只有当整个 group 都完成后，才能计算组内均值和标准差：

\[
A_i = \frac{R_i - \operatorname{mean}(R_1,\ldots,R_8)}
{\operatorname{std}(R_1,\ldots,R_8)}
\]

随后，128 条样本仍然作为一个训练 batch 进入优化。

SAO 并不是取消 batch，而是将 group size 降为 1：

```text
128 个 prompt
每个 prompt 生成 1 条 rollout
总 batch size = 128
```

所以 SAO 的准确描述不是“一条 rollout 到了就立即做一次 batch size=1 的参数更新”，而是：

> 一条 rollout 完成后，无需等待同 prompt 的其他 rollout，可以立即进入 learner 的训练队列；learner 仍然会将不同 prompt 的 trajectory 组成 batch 更新参数。

SAO 论文的公开实验中，SAO 使用 `batch size = 128, group size = 1`；GRPO 对照组使用 16 个 prompt、每个 prompt 8 条 rollout，同样组成 128 条样本。

---

## 1.2 长程 Agent 任务中的 group barrier

在数学题等短任务中，同一 prompt 生成 8 条回答的长度通常不会相差几个数量级。

但 coding agent 或工具型 Agent 的轨迹可能是：

```text
rollout 1：20 个交互回合，10 分钟结束
rollout 2：80 个交互回合，30 分钟结束
rollout 3：300 个交互回合，持续运行测试和修改代码
...
rollout 8：因为环境故障或复杂任务，数小时后才结束
```

GRPO 必须等待 rollout 8，才能计算整个 group 的相对 advantage。已经完成的 rollout 只能滞留在队列中。

这会产生三个问题：

1. **straggler 问题：**整个 group 被最慢的一条轨迹拖住；
2. **资源利用率问题：**rollout engine 和 learner 难以持续流水；
3. **policy lag 问题：**短轨迹完成后等待太久，真正训练时当前 policy 已经发生变化。

对于真实在线反馈，问题更明显。一个用户任务通常只会真实执行一次，并不会为了凑满一个 GRPO group，把同一任务在生产环境里并行运行八遍。

因此，SAO 的核心变化不是单纯“PPO 比 GRPO 更好”，而是：

> **放弃必须等待同题多条 rollout 的数据组织方式，用 critic 为每条独立 trajectory 提供 baseline，从而适配异步 Agent RL。**

---

# 二、SAO 的完整训练流程

可以把 SAO 的训练管线简化为：

```text
多个 rollout worker 持续与环境交互
        │
        ├─ trajectory A 完成 ─┐
        ├─ trajectory B 完成 ─┤
        ├─ trajectory C 完成 ─┤ → 异步训练队列
        └─ trajectory D 仍在运行
                              │
                              ▼
                    learner 组成训练 batch
                              │
              ┌───────────────┴──────────────┐
              ▼                              ▼
        当前 actor 前向                 critic 前向
      计算 current logprob              估计 state value
              │                              │
              └───────────────┬──────────────┘
                              ▼
                GAE + importance ratio + DIS
                              │
                              ▼
                    更新 actor 与 critic
```

rollout 阶段需要保存的核心信息包括：

- 实际生成的 token id；
- 每个生成 token 在 rollout policy 下的 log probability；
- 环境 observation；
- 环境或 verifier 给出的 reward；
- 哪些 token 是模型 action、哪些是环境 observation；
- trajectory / segment 的边界与版本信息。

这里不需要保存每个 token 对完整词表的 logits，只需要保存实际采样 token 的 logprob。

---

# 三、critic 到底做什么

## 3.1 reward 不是 critic 给出的

假设 coding agent 完成了一个任务：

```text
查看代码
→ 修改文件
→ 运行测试
→ 修复失败
→ 所有测试通过
```

最终 reward 可能来自：

- 单元测试是否通过；
- SWE-Bench patch 是否正确；
- 数学答案是否正确；
- 环境反馈；
- reward model 或 LLM judge。

critic 并不负责决定最终任务是否成功，而是估计：

> 在当前状态下，按照现有策略继续执行，预计最终能够获得多少回报？

记作：

\[
V_\phi(s_t)
\]

假设某状态下 critic 预测最终回报为 0.4，而实际最终回报是 1：

\[
A_t \approx 1 - 0.4 = 0.6
\]

这表示该状态之后采取的行为比原本预期更好，应当提高相关 action 的概率。

如果 critic 预测 0.8，但最终失败得到 0：

\[
A_t \approx 0 - 0.8 = -0.8
\]

则相关 action 比预期更差。

实际 SAO 使用 GAE，而不是简单的 `最终 reward - value`，但 critic 的基本职责没有变化：

> **critic 提供状态相关的 baseline，降低单条 rollout policy gradient 的方差。**

GRPO 不需要 critic，是因为它用同 prompt 的其他 rollout 构造 baseline。SAO 放弃 group 后，需要重新引入 value model。

---

# 四、importance ratio 为什么存在

## 4.1 异步训练必然产生 policy lag

假设一条 rollout 开始生成时，rollout worker 使用模型版本 `v100`。

在它进行长时间环境交互期间，learner 已经根据其他 trajectory 更新到了 `v108`。

于是训练时出现：

```text
数据由 rollout policy v100 生成
当前要优化的是 policy v108
```

这条数据不是从当前 policy 直接采样的，属于 off-policy 数据。

对每个实际生成 token，SAO 计算：

\[
r_t =
\frac{\pi_{\text{current}}(a_t\mid s_t)}
{\pi_{\text{rollout}}(a_t\mid s_t)}
\]

工程实现通常写成：

\[
r_t = \exp\left(
\log \pi_{\text{current}}(a_t\mid s_t)
-
\log \pi_{\text{rollout}}(a_t\mid s_t)
\right)
\]

rollout 阶段保存旧 logprob；训练阶段则把已有 trajectory 用 teacher forcing 输入当前 actor，再计算 current logprob。

这不是重新执行工具、测试或沙箱，只是对已知 token 序列做一次当前模型前向。

---

## 4.2 ratio 的数学意义是换分布

假设数据由旧策略 \(\mu\) 采样，但我们希望估计当前策略 \(\pi\) 下的期望：

\[
\mathbb{E}_{a\sim\pi}[f(a)]
\]

因为手里的样本来自 \(\mu\)，可以改写为：

\[
\mathbb{E}_{a\sim\pi}[f(a)]
=
\mathbb{E}_{a\sim\mu}
\left[
\frac{\pi(a)}{\mu(a)}f(a)
\right]
\]

因此，probability ratio 不是一个经验性的打分项，而是 importance sampling 中将旧分布样本重新加权到新分布时自然出现的密度比。

简化后的 actor 更新信号可以理解为：

\[
\text{update signal}
=
\underbrace{A_t}_{\text{行为比预期好或差多少}}
\times
\underbrace{r_t}_{\text{旧样本在当前策略下应占多大权重}}
\]

其中：

- advantage 决定奖励或惩罚的方向；
- importance ratio 修正样本由旧 policy 生成带来的分布差异。

---

## 4.3 SAO 的 DIS：超出可信区间直接丢弃 token

普通 PPO 使用 clipped surrogate objective，限制新 policy 相对 old policy 的变化幅度。

SAO 面对的异步 policy lag 更严重，因此采用 Direct Double-Sided Importance Sampling（DIS）：

```text
ratio 位于可信区间
→ token 参与梯度计算

ratio 超出可信区间
→ token 被 mask，不参与梯度计算
```

它不是判断 token 内容“好不好”，而是在判断：

> 这条由旧 policy 产生的经验，与当前 policy 是否已经偏离到不值得继续学习。

这种做法牺牲一部分严格无偏性，换取异步训练中的稳定性。

---

# 五、Skip-Observation GAE：跳过的是 RL 时间步，不是工具结果上下文

Agent trajectory 通常是：

```text
Action 1：调用测试工具
Observation：返回 10,000 token 测试日志
Action 2：根据错误修改代码
```

需要区分三个层面：

1. **工具结果仍然保存在 trajectory 中；**
2. **工具结果是后续 action 的输入上下文；**
3. **工具结果不是模型生成的 action，不应进入 actor policy loss。**

如果朴素地对完整 transcript 做 token-level GAE，10,000 个 observation token 会被视为 10,000 个 RL 时间步，最终 reward 传播回 Action 1 时可能发生严重衰减。

这会产生一个不合理的偏差：

```text
工具只返回 10 个 token
→ 前一个 action 获得较强 credit

工具返回 10,000 个 token
→ 前一个 action 获得的 credit 大幅减弱
```

但工具输出长度不等于 Agent 做出了更多决策。

Skip-Observation GAE 将 observation 看作一次环境转换：

```text
Action 1 末尾
──────────────→ Action 2 开头
  observation 不作为独立的 GAE 时间步
```

需要注意，Action 2 的 value 仍然基于已经包含完整工具结果的上下文计算。因此它不是忽略 observation，而是：

> **让 observation 影响下一状态的价值判断，但不让外部文本长度扭曲 reward 的传播距离。**

为什么早期 RLHF 不普遍讨论这个问题？因为传统单轮 RLHF 往往只有：

```text
prompt → assistant response → reward
```

没有多轮 action / observation 交错结构。GRPO 又经常将 sequence-level advantage 广播到整段回答，并不显式使用 token-level critic 和 GAE。

只有当训练同时具备以下条件时，这个问题才变得突出：

- 多轮 Agent 环境；
- token-level value；
- critic-based PPO；
- 大量非模型生成的 observation token。

---

# 六、SAO 如何让 critic 跟得上 actor

single-rollout 的最大代价是方差更大。如果 critic 不准确，advantage 会成为高噪声信号，直接破坏 actor。

SAO 使用了几项很工程化的设计：

## 6.1 critic 更新频率高于 actor

公开实验中，每个 batch 执行：

```text
critic 更新 2 次
actor 更新 1 次
```

目的是让 value model 更快跟上不断变化的 policy distribution。

## 6.2 冻结 critic attention

论文观察到 critic 的梯度不稳定主要来自 full-attention 层，而 MoE projection 相对稳定。因此训练 critic 时冻结 attention，只更新 MoE projection。

这不是一个普适理论结论，更像是针对其 MoE 模型结构得到的工程正则化方案。

## 6.3 扩大 value pretraining

critic 如果冷启动过差，训练初期就会给出错误 advantage。SAO 因此强调扩大 value model 预训练数据，为 single-rollout RL 提供更好的初始 baseline。

这也揭示了 SAO 的真实成本：

> GRPO 省掉了 critic，但需要同 prompt 多次 rollout；SAO 省掉了 group sampling 和等待屏障，却重新承担了 value model 的显存、计算、预训练和稳定性成本。

---

# 七、CompactionRL：为什么完整 rollout 会变成多个 segment

SAO 论文与 CompactionRL 论文是两项不同但配套的工作，二者都被用于 GLM-5.2 的 Agent RL 管线。

CompactionRL 处理的问题是：长程 Agent 在任务尚未完成时，context 已经接近上限。

此时模型生成 summary，并用 summary 与最近历史重建上下文：

```text
Segment 1：
原始任务 + 大量 action / observation
→ 生成 Summary 1

Segment 2：
原始任务 + Summary 1 + 最近历史
→ 继续执行
→ 生成 Summary 2

Segment 3：
原始任务 + Summary 2 + 最近历史
→ 完成任务
```

从任务角度看，这仍然是一条 rollout；从模型实际条件分布看，它已经是多个不同上下文下生成的训练 segment。

因此，segment 切分的首要目标不是加快训练，而是：

1. 在固定峰值 context budget 下继续长程任务；
2. 忠实复现 rollout 时实际看到的压缩上下文；
3. 将 summary generation 本身也作为可训练策略；
4. 让有效任务 horizon 超过单窗口长度。

---

## 7.1 为什么 compaction 进一步削弱了 GRPO 的适配性

同一个 prompt 的 8 条 rollout 可能得到不同数量的 segment：

```text
rollout 1：1 个 segment
rollout 2：3 个 segment
rollout 3：2 个 segment
rollout 4：4 个 segment
...
```

如果 segment 独立进入训练：

- segment 多的 rollout 会在 group 统计中重复出现，权重过高；
- 只在完整 rollout 层计算 group advantage，又无法为独立训练的 summary / execution segment 给出细粒度 advantage。

因此 CompactionRL 采用 PPO critic，并使用：

- **token-level loss normalization：**避免 segment 数量和长度造成样本权重偏差；
- **cross-trajectory GAE：**在 segment 边界之间传播最终 reward，并根据后续真实 trainable token 距离折扣早期 segment。

如果把最终 reward 简单放到每个 segment 末尾，早期 segment 会误以为最终成功距离自己很近，从而被过度奖励。cross-trajectory GAE 正是在修正这个距离错觉。

---

# 八、SAO 的实验结果与边界

SAO 公开可控实验主要基于 Qwen3-30B-A3B，而不是直接公开 GLM-5.2 750B-A40B 的完整消融。

论文报告：

| 方法 | AIME2025 | BeyondAIME | HMMT | IMOAnswerBench | SWE-Bench Verified |
|---|---:|---:|---:|---:|---:|
| GRPO + DIS | 93.5 | 70.8 | 84.0 | 70.0 | 27.0 |
| SAO | **97.3** | **74.8** | **88.3** | **74.0** | **29.8** |

这些结果说明 SAO 在论文设定下优于经过 DIS 改进的 GRPO，但仍要注意：

1. 论文主要验证 coding agent 与工具增强数学推理；
2. single-rollout 的效果强依赖 critic 质量；
3. value model 大体增加了一套模型级别的训练开销；
4. token-level importance ratio 无法完整修正整条 trajectory 的状态分布变化；
5. DIS 通过 mask 极端 token 换取稳定性，也会丢弃训练数据；
6. GLM-5.2 规模上的完整训练曲线、预算与消融没有完全公开；
7. 当前论文仍是 arXiv v1，独立复现证据有限。

因此，合理结论是：

> SAO 证明了 critic-based single-rollout PPO 在长程、异步、工具化 Agent RL 中具有明显吸引力；它并没有证明 GRPO 在短程 RLVR、低成本多采样或同步训练中已经过时。

---

# 九、DeepSeek V4 的后训练路线

DeepSeek V4 的后训练分成两个主要阶段：

```text
第一阶段：分别训练领域专家
数学专家 / 代码专家 / Agent 专家 / 指令专家
    SFT + 领域 reward + GRPO

第二阶段：统一模型能力合并
学生模型生成自己的 rollout
→ 多个专家在学生 trajectory 上给出完整词表分布
→ 使用 reverse KL 做 On-Policy Distillation
```

这里有一句非常容易误读的原文：

> DeepSeek V4 将 mixed RL stage entirely replaced by OPD。

它不是说 DeepSeek V4 完全不再使用 GRPO。报告明确说明，各领域 specialist 仍然通过 SFT 和 GRPO 训练。

被 OPD 替代的是：

> 把数学、代码、Agent、指令跟随等异质目标直接混在一个最终 mixed RL 阶段里共同优化。

---

## 9.1 为什么 DeepSeek 需要 OPD

不同领域的奖励目标可能直接冲突：

| 领域 | 可能偏好的行为 |
|---|---|
| 数学推理 | 长推理、严谨展开、唯一正确答案 |
| coding agent | 工具调用、测试闭环、环境修复 |
| 普通对话 | 简洁、低延迟、不过度推理 |
| 写作任务 | 风格、审美、个体偏好 |

如果将所有 reward 混在一个 RL 阶段，可能产生：

- 不同能力互相污染；
- reward 权重难以平衡；
- 强领域能力被平均化；
- 一个领域的优化损害另一个领域。

DeepSeek 的路线是先把冲突隔离：

```text
每个领域独立训练到足够强
→ 再通过教师分布将能力压回统一学生模型
```

---

## 9.2 OPD 为什么叫 On-Policy

学生模型先按照自己的当前 policy 生成 rollout：

\[
y \sim \pi_\theta(\cdot\mid x)
\]

然后专家模型在学生实际到达的状态上给出 next-token distribution。

学生优化加权 reverse KL：

\[
\mathcal{L}_{OPD}(\theta)
=
\sum_i w_i
D_{KL}\left(
\pi_\theta \| \pi_{E_i}
\right)
\]

它与传统 offline distillation 的关键区别是：

- offline distillation 通常在固定 teacher 数据或 teacher trajectory 上学习；
- OPD 在学生自己当前会到达的状态上询问 teacher。

因此，OPD 主要解决 student distribution drift：当学生走到自己特有的中间状态时，教师仍然能对该状态给出完整分布监督。

但 OPD 的 on-policy 与 SAO 的在线环境反馈不是同一件事：

- OPD 的 on-policy 指训练状态由当前 student policy 产生；
- SAO 的 single-rollout online learning 强调每次真实环境只返回一条 trajectory 和 reward，policy 根据持续到达的反馈更新。

---

## 9.3 DeepSeek V4 的 GRM

对于数学和单元测试这类任务，可以构造明确 reward。

写作、设计、复杂指令跟随等 hard-to-verify 任务则缺乏唯一正确答案。DeepSeek V4 引入 Generative Reward Model（GRM），让模型根据 rubric 生成评价、批评和判断，而不是只输出一个黑盒标量。

因此，在 DeepSeek 路线中：

- GRPO / RL 负责训练领域专家；
- GRM 负责提供难验证任务的评价信号；
- OPD 负责将多个专家整合回统一模型。

这是一条能力分解与再合并的后训练路线。

---

# 十、DeepSeek V4 的 rollout 基础设施

DeepSeek V4 没有像 SAO 论文一样公开宣称完全转向 single-rollout asynchronous PPO，但它同样针对长轨迹做了大量工程优化。

## 10.1 可抢占与容错 rollout service

DeepSeek 为每个 generation request 建立 token-granular Write-Ahead Log：

```text
每生成一个 token
→ 立即写入 WAL

任务被抢占
→ 保存 KV cache 并暂停

恢复任务
→ 使用 WAL 与 KV cache 继续生成
```

如果硬件彻底失败，也可以用已持久化 token 重新 prefill，恢复 KV cache。

报告特别强调，不能简单地丢弃未完成请求并从头重新采样，因为这会产生 length bias：长回答更容易在运行中被中断并丢弃，数据集最终会偏向短回答。

## 10.2 百万 token RL 数据组织

DeepSeek 将 rollout 数据拆成：

- 轻量 metadata；
- 重型 per-token fields。

metadata 可以整体加载，用于全局 shuffle 和 packing layout；per-token 字段通过 shared-memory loader 按 mini-batch 加载并立即释放，从而降低 CPU / GPU 内存压力。

这说明 DeepSeek 与 GLM 都意识到：

> 长程 Agent RL 的问题已经不仅是算法 objective，而是 rollout、存储、调度、容错、数据加载和训练流水线的共同设计。

---

# 十一、SAO 与 DeepSeek V4 专家 GRPO 的直接比较

| 维度 | GLM SAO | DeepSeek V4 专家训练中的 GRPO |
|---|---|---|
| 主要任务 | 长程 Agent、工具推理、异步环境 | 各领域 specialist 的独立能力强化 |
| 每个 prompt rollout 数 | 1 | 多条组成 group |
| advantage baseline | critic / value model / GAE | 同 prompt group reward 均值与标准差 |
| 是否需要 critic | 需要 | 通常不需要 |
| 是否等待同组最慢样本 | 不需要 | 需要 |
| 异步 policy lag | 明确使用 rollout logprob + DIS 控制 | 报告未公开同等级别的 SAO 式设计 |
| online 单次反馈适配 | 自然 | 不自然，需要同 prompt 多次采样 |
| 主要成本 | value model、critic 预训练与更新 | 多 rollout 生成成本与 group 等待 |
| 主要风险 | critic 不准导致高方差和错误更新 | group 同质化、等待屏障、长尾 rollout |
| 适合场景 | rollout 昂贵、轨迹长短差异大、反馈一次性 | 可验证 reward、同题可低成本生成多解 |

这才是 SAO 最直接的比较对象。

---

# 十二、SAO 与 OPD 的区别

| 维度 | SAO | DeepSeek V4 OPD |
|---|---|---|
| 方法类型 | 强化学习优化方法 | 多教师 on-policy 蒸馏方法 |
| 优化信号 | 环境 reward + critic advantage | 专家完整词表 distribution |
| 训练目标 | 提高 policy 的预期环境回报 | 让统一 student 在自身状态上接近领域专家 |
| 是否依赖环境 verifier | 通常依赖 | 不一定，主要依赖 teacher |
| 是否需要多个领域专家 | 不需要 | 需要 |
| 是否解决多目标能力冲突 | 不是主要目标 | 核心目标 |
| 是否解决异步 rollout 等待 | 是 | 不是主要目标 |
| 是否需要 critic | 需要 | 不需要 PPO critic，但需要 teacher inference |
| 核心昂贵资源 | value model 与异步 RL 系统 | 多个超大 teacher 的 full-vocabulary logits |

因此，不能直接得出“GLM 用 SAO、DeepSeek 用 OPD，所以两家公司选择了相反 RL 算法”的结论。

一个完全可能的组合是：

```text
数学专家：使用适合数学任务的 GRPO / SAO 训练
代码专家：使用 SAO + CompactionRL 训练长程 coding agent
Agent 专家：使用 SAO 处理真实环境单 rollout 反馈
写作专家：使用 GRM / judge 提供 reward

最后：
所有专家通过 OPD 合并进统一模型
```

这是一种基于两份报告做出的组合推演，并不是 GLM 或 DeepSeek 已公开宣布的实际方案。

---

# 十三、GLM 与 DeepSeek 在长上下文上的路线差异

GLM-5.2 的 CompactionRL 与 DeepSeek V4 的百万 token 架构形成了一个很有意思的对照。

## GLM：压缩运行状态

```text
固定峰值 context budget
→ 上下文接近上限时总结
→ 用 summary + 最近历史继续执行
→ 训练 summary policy 与跨 segment credit assignment
```

重点是：

> 不提高单窗口峰值，也可以延长有效任务 horizon。

## DeepSeek：扩大并降低长上下文成本

```text
原生 1M context
→ CSA / HCA 压缩注意力和 KV cache
→ on-disk KV cache
→ 百万 token RL 数据加载与 rollout 容错
```

重点是：

> 让模型尽可能在一个超长上下文中保留完整历史，同时把 FLOPs、KV cache 和数据加载成本压下来。

两者也不是完全排斥：

- 原生 1M context 仍可能被更长任务耗尽；
- compaction 仍会丢失细节，长窗口可以降低压缩频率；
- 最现实的长程 Agent 很可能同时需要长上下文、结构化外部状态和阶段性 compaction。

但两家公司公开材料中的优先级明显不同：

- GLM 更关注在固定上下文预算下学习压缩和续跑；
- DeepSeek 更关注把超长上下文本身做成可承受的基础设施能力。

---

# 十四、从两条路线看到的后训练范式变化

## 14.1 GLM：围绕真实环境数据形态重构 RL

SAO 的出发点不是重新证明 PPO 理论优于 GRPO，而是承认真实 Agent 数据具有以下形态：

- 一次任务只产生一条真实执行轨迹；
- 轨迹长度高度不均衡；
- rollout 和 learner 必须并行；
- 环境 observation 很长且不可控；
- context 可能在执行中被压缩重建；
- policy lag 不可避免。

因此它愿意重新支付 critic 成本，换取 single-rollout 和异步流水。

## 14.2 DeepSeek：围绕能力冲突重构后训练 pipeline

DeepSeek V4 的核心判断是：

- 通用模型的领域目标过于异质；
- 最终 mixed RL 很难同时调和所有 reward；
- 不如先训练多个领域专家，再让统一学生在自己的状态分布上向专家学习。

因此它愿意支付多 teacher 推理和 full-vocabulary distillation 的成本，换取能力隔离与合并。

## 14.3 两者共同说明了什么

两条路线共同指向一个趋势：

> 后训练已经不再是选择 PPO、GRPO 或某个 loss function 的单点问题，而是根据 Agent runtime 的真实数据形态，联合设计 rollout、上下文、reward、critic、teacher、存储、调度和容错系统。

---

# 十五、如何判断自己的任务更适合哪种方法

| 任务条件 | 更自然的方向 |
|---|---|
| 数学题、短代码题，同题可低成本采样多次 | GRPO / group-based RL |
| 长程 coding agent，轨迹长短差异巨大 | SAO / critic-based asynchronous PPO |
| 真实用户任务，一次只有一条环境反馈 | single-rollout RL |
| 上下文会在任务中被压缩或重建 | CompactionRL 类 segment-aware PPO |
| 已经拥有多个能力很强的领域专家 | OPD / multi-teacher distillation |
| 多领域 reward 直接混训会互相污染 | specialist training + OPD |
| 没有资源维护 critic | GRPO 或其他 critic-free baseline 更现实 |
| 没有资源长期运行多个超大 teacher | OPD 的工程成本可能不可接受 |

SAO 并没有让 GRPO 失去价值，OPD 也没有取代所有 RL。真正决定方法的，不是算法名称，而是：

1. 一个 prompt 能否重复执行很多次；
2. rollout 是否昂贵；
3. 轨迹是否长且高度不均衡；
4. reward 是否可验证；
5. 是否存在多个相互冲突的领域目标；
6. 是否有能力训练 critic 或维护多个 teacher。

---

# 十六、个人判断

## 1. SAO 最值得关注的不是“PPO 回归”

它真正有价值的部分是：

- 将 group size 与 batch size 解耦；
- rollout 完成后立即进入数据流；
- 用 rollout logprob 直接处理异步 policy lag；
- 对过时 token 做双侧 mask；
- 为 Agent observation 重新定义 GAE 时间轴；
- 为 critic 设计专门的训练节奏和正则化。

所以 SAO 更像一套适合长程 Agent 的 actor-learner 训练协议，而不只是一个新的 loss 名称。

## 2. SAO 的真正瓶颈可能是 critic，而不是 actor objective

single-rollout 没有组内平均来降方差，critic 一旦不准，policy update 很容易被错误 advantage 带偏。

论文中 critic 更新两次、冻结 attention、扩大 value pretraining 都不是附属优化，而是 SAO 能否成立的核心条件。

因此，社区复现时最需要观察的不是只抄 DIS 公式，而是：

- value model 如何初始化；
- critic 数据如何构造；
- explained variance 是否稳定；
- critic 是否真正跟上 policy shift。

## 3. DeepSeek V4 与 GLM-5.2 选择的是不同的“昂贵部分”

- GLM 省掉同 prompt 多 rollout 和 group 等待，但增加 critic；
- DeepSeek 隔离 mixed RL 冲突，但增加多个 teacher 与 full-vocabulary OPD；
- GLM 用 compaction 延长 horizon，但需要训练 summary 和跨 segment credit；
- DeepSeek 原生支持 1M context，但需要复杂注意力、KV cache 和数据基础设施。

这不是谁找到了免费午餐，而是两家公司将成本放在了不同位置。

## 4. 对 Agent 工程最重要的启发

训练方法必须尽量匹配最终 runtime：

- 线上一次只运行一条任务，就不要默认训练数据必须同题生成八次；
- 线上会发生工具返回，就要正确建模 observation；
- 线上会裁剪或压缩上下文，就应在训练中暴露这种状态变化；
- 线上模型运行在低精度环境，就应考虑 rollout 与部署精度一致；
- 线上任务目标高度异质，就要考虑专家隔离与能力合并，而不是只调 reward 权重。

换句话说：

> **后训练应该学习真实 Agent runtime 中会发生的事情，而不是只在一个方便计算的离线数据结构中优化漂亮的 loss。**

---

# 📚 原始资料

1. GLM / Z.AI，**Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning**  
   https://arxiv.org/abs/2607.07508

2. GLM / Z.AI，**CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents**  
   https://arxiv.org/abs/2607.05378

3. Z.AI，**GLM-5.2: Built for Long-Horizon Tasks**  
   https://z.ai/blog/glm-5.2

4. DeepSeek AI，**DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**  
   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

5. Schulman et al.，**Proximal Policy Optimization Algorithms**  
   https://arxiv.org/abs/1707.06347

6. Schulman et al.，**High-Dimensional Continuous Control Using Generalized Advantage Estimation**  
   https://arxiv.org/abs/1506.02438
