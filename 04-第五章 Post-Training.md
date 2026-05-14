## 第 5 章：Post-Training 强化学习策略改良

### 章节导读

本章前两节聚焦于 V4 的后训练工程。5.1 和 5.2 的逻辑关系类似于第二章和第三章之间的关系，都是先讲技术路径，再讲工程实操。后两节 5.3 和 5.4 的内容主要在做 benchmark 评测，本身不是我们关注的重点。

5.1.1 这部分不仅在训练工程上做了很多阐述，还提到了很多和 agent 使用 / agent 开发有关的内容，可以说是所有 Agent 开发者都值得一读的部分，因此在这一节导读尽可能忠于报告原文，不做概念简化。

- 5.1 Post-Training Pipeline 后训练总管线
  - 5.1.1 Specialist Training 专家模型训练，这部分很多处理与 Agent 工程强相关
  - 5.1.2 On-Policy Distillation 基模与专家的对齐和跟随

- 5.2 RL and OPD Infrastructure RL 和 OPD 的训练框架，和第三章的结构思路类似
  - 5.2.1 FP4 Quantization Integration 第三章提及的 FP4 量化集成进训练框架
  - 5.2.2 Efficient Teacher Scheduling for Full-Vocabulary OPD 训练的工程处理
  - 5.2.3 Fault-Tolerant Rollout Service 资源冲突 / 故障时的处理策略
  - 5.2.4 Million-Token RL Framework 长上下文下的 RL 训练框架
  - 5.2.5 Sandbox Infrastructure 沙箱执行环境

- 5.3 Evaluation 后训练完毕后最终的模型评测

- 5.4 Real-world Evaluation 真实世界任务的模型表现评估

### 5.1 Post-Training Pipeline 后训练范式的变化

报告这一节中的逻辑组织有些分散，可以大致划分为两个部分：

#### 面向 Agent 推理、工具使用和代码任务的行为对齐

这一部分相对庞杂，报告中提到了多个点，而这些均可以为 Agent 工程提供启发。

- Reasoning Effort 推理强度

![Reasoning Effort 推理强度](./assets/image/dsv4-reasoning.PNG)

DS 把 flash 和 pro 两个版本均设置了三档推理强度：Non-think / Think High / Think max。在后训练过程中，DS 就对这三种不同推理强度，给予了不同的推理上下文预算，但都使用 `<think></think>` 来包裹推理文本。

特别地，对于 Think max 模式，**DS 在开头注入了一段特殊的系统提示词**，这段系统提示词的报告原文如下：

```plaintext
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked

推理努力：绝对达到最大值，不允许任何捷径。
你必须非常彻底地思考，全面分解问题以解决根本原因，严格地对你的逻辑进行压力测试，考察所有潜在路径、边界情况和对抗性场景。
明确写出你整个审议过程，记录每一个中间步骤、考虑过的替代方案和被否定的假设，以确保没有任何假设被放过
```

这一点其实已经和 Claude / GPT 系列模型相仿，提供不同档位的推理强度，用推理预算来换取思维强度并提高任务实现能力。有的通过控制 reasoning 的 token 预算来控制，有的则是直接提供不同的推理档位。

GPT 系列的模型，推理内容全部是内化的，也就是说 Chain of Thought 本身不会暴露给外部作为 message 吐出模型。Claude 系列模型虽然会展示 thinking 的过程，但也不是把原始的 CoT 暴露出来，而是通过受控的 extended thinking / thinking blocks 机制呈现或管理。

相反地，国内模型如 kimi / qwen / deepseek 均是默认会用不同的格式，显式地吐出 reasoning 过程，这会带来后续 agent 交互的问题，我们后续马上就会提及。

- Tool Call Schema and Special Token 特殊的 tool call 格式

V4 中，DS 引入了一种新的工具调用模式，该模式使用特殊的 `|DSML|` 标记，并采用基于 XML 的格式进行工具调用。报告中提到，经过他们的实验，发现 XML 格式能够有效减轻转义失败问题，并降低工具调用错误，为模型与工具的交互提供了更稳健的接口。

![Tool Call Schema and Special Token](./assets/image/dsv4-toolcal-schema.PNG)

这一点其实与 Claude Code 官方推荐的 prompt 撰写方式不谋而合，Anthropic 的官方文档中推荐使用 XML 格式的提示词组装，以此获得更好的指令跟随能力。OpenAI 在历史模型上也有过类似的表述，但在最新的 GPT-5.4 和 GPT-5.5 中已经大大淡化了这部分的描述。

如果我们注意看非常著名也非常好用的 superpowers 的提示词撰写，里面也使用了 `<hard-gate></hard-gate>` 这样的 XML 模式组装，大大提高了 agent 使用 skill，并依据 skill 做事的指令跟随能力。

同时 DS 用 `|DSML|` 这种特殊的标记来强化 tool call schema 的稳定性，可能隐含的是：我们使用 V4 模型在**某些使用非官方 tools 参数**的场景下，**使用 `|DSML|` 这个标记可能对 tool call 的稳定性和指令跟随有正向作用，因为模型和 tokenizer 在训练时就天然地更“认识”这样一个标记，并且做过大规模的指令对齐。**

这部分内容以个人推测为主，大家可以自行测试。

- Interleaved Thinking 交错推理模式

这一部分在我看来是最值得 agent 开发工程师看的一小节，这部分我直接引用翻译后的报告原文：

```plaintext
DeepSeek-V3.2 引入了一种上下文管理策略，该策略在工具结果回合之间保留推理痕迹，但在新的用户消息到来时会丢弃它们。虽然有效，但在复杂的智能代理工作流中，这仍然会导致不必要的token浪费——每一个新的用户回合都会清空所有累积的推理内容，迫使模型从头重建其问题解决状态。利用DeepSeek-V4系列扩展的 1M token上下文窗口中，我们进一步完善了这一机制，以最大化代理环境中交错思维的有效性：

• 工具调用场景。整个对话过程中所有推理内容都完全保留。不同于DeepSeek-V3.2在每个新用户轮次时会丢弃思维痕迹，DeepSeek-V4系列在所有轮次中保留完整的推理历史，包括跨用户消息边界的内容。这使得模型能够在长期代理任务中保持连贯、累积的思维链。

• 一般对话场景。原策略被保留：当新用户消息到来时，会丢弃前一轮的推理内容，以在持续推理痕迹提供有限收益的情况下保持上下文简洁。

与 DeepSeek-V3.2 一样，通过user message（这里推测可能也包括了所谓的tool message）来模拟工具交互的Agent框架（例如 Terminus）可能不会触发工具调用的上下文路径，因此可能无法从增强的推理持久性中受益。我们仍然建议针对这种架构使用非思考模型。
```

![Interleaved Thinking](./assets/image/dsv4-interleaved-thinking.png)

应当说，这一部分最有价值的点在于：reasoning content 已经不再仅仅是模型自己的思考，而是已经被整合进了 tool call 这个行为本身之中。DeepSeek V4 API 文档中明确提到：如果 thinking mode 下发生了 tool call，那么该轮 assistant 产生的 `reasoning_content` 必须完整参与后续上下文拼接，并在之后所有用户交互轮次里回传给 API；否则会 400 报错。

事实上这也早已不是 DeepSeek 的专利，从 kimi 和 qwen 均有类似的要求，而现在普遍使用的诸如 dify / LangChain / DeepAgents / CrewAI 这类开箱即用的 Agent 框架，往往容易出现强依赖 `reasoning_content` 回传的操作，必须通过 provider / proxy 层面的特殊处理才能不报错。

然而，即使通过 proxy、LiteLLM provider 或 CustomLLM 可以做到在请求层补齐 `reasoning_content` 回传的补丁，也并不等价于 Agent 框架原生支持 thinking model，现有 Agent 框架的 ReAct loop 并没有把 `reasoning_content` / `thinking_blocks` 当作一等状态建模。于是 proxy 只能在框架外维护旁路；一旦发生历史裁剪、流式中断、subagent 派发、会话恢复或工具调用结构变化，仍然可能丢失 reasoning state，或者只能通过丢弃旧上下文来恢复，这本身其实是对旧 ReAct loop 范式越来越大的挑战。

- Quick Instruction 快速指令

这部分同样直接给出翻译后的报告原文：

```plaintext
在聊天机器人场景中，许多辅助任务（例如，确定是否触发网页搜索、意图识别等）必须在生成响应之前执行。传统上，这些任务由一个独立的小模型处理，需要进行冗余的预填充，因为它无法重用现有的 KV 缓存。为了解决这个限制，我们引入了快速指令。我们将一组专用的特殊标记直接附加到输入序列中，每个标记对应一个特定的辅助任务。通过直接重用已经计算的 KV cache，这种机制完全避免了冗余的预填充，并允许某些任务（如生成搜索查询以及确定权限和领域）并行执行。因此，这种方法显著减少了用户感知的首次生成 token 时间（TTFT），并消除了维护和迭代额外小模型的工程负担。
```

DS 支持的快速指令标记如图所示：

![Quick Instruction](./assets/image/dsv4-quick-instruction.png)

这部分内容对 Agent 开发最大的启发是：使用单独的子 Agent 来做这些高频 / 短输出 / 强依赖上下文的前置工作，诸如意图识别 / query rewrite / 信安判断，做成一个独立的子 Agent 可能不是一个好的实践。相反，它们应该变成主 Agent 共享上下文下的快速判定过程，这样更省 token，而不是一般意义上认为派发子 agent 更省 token。

还需要注意的是，这些 Special Token 并不一定意味着在 prompt 中加上这些标记就能获得相应的正向效果，因为这部分只能保证模型的确见过这类 token，在硬件推理时并不能保证能获得原文提到的预填充效果。可能需要更进一步的实验才能有定论。

#### 后训练过程中 RL 范式转变

**前置知识：RL 基础概念，PPO / GRPO 基本思想**

传统后训练中使用 RL 时需要设定一个奖励目标，基模围绕这个奖励目标迭代训练来获得奖励，得到更高的分数。

问题在于，对于一个通用大模型而言，不同领域任务的目标是完全不一样的：

- 数学任务希望长推理、严谨、可验证
- 代码任务希望通过测试、能执行、能修复
- agent 任务希望多轮规划、工具调用和环境反馈闭环
- 普通指令跟随和对话又往往要求简洁、稳定、低延迟

更进一步地，并不是所有任务都有一个非常明确的正确答案。数学计算有对错，代码任务可以通过测试用例验证，但对于写作、页面设计这类任务来说，没有一个固定的答案，也很难设计一个通用的奖励目标。总结下来就是：

1. 通用大模型面临着极大规模不同目标之间的调和问题，调和得不好非常容易出现冲突和能力互相污染。
2. 通用大模型面临的任务中很多没有标准答案。

对于第一个问题，DS 的方案是 On-Policy Distillation（OPD）；对于第二个问题，DS 的方案是 Generative Reward Model（GRM）。

##### OPD

相比把多个领域、多种奖励信号混在同一个 RL 阶段里联合优化，DS 采用 OPD，核心思路是：不同领域的专家模型先各自在自己的领域上训练到极强，然后使用模型蒸馏（也就是 OPD 的 D——Distillation）的方式，让专家模型作为老师，基模则作为学生模型。

对于一个特定的输入，基模自己先生成 rollout，然后拿自己的 rollout 作为上下文状态，再在这些上下文状态上读取专家模型的 next-token distribution / logits distribution。专家不需要生成完整 rollout，也可以对学生 rollout 的每一步给出概率分布标签。在 OPD 中，专家自己不一定要生成一个完整的 rollout 让基模学习，而是只负责在基模的 rollout 中给出监督信号。

![OPD vs Mixed RL](./assets/image/opd-vs-mixed-rl.png)

##### GRM

GRM 策略是专门应用于 hard-to-verify 任务的。原先这些难以直接给出评价的任务，使用的是 RLHF（基于人类反馈的强化学习）的策略，依赖于大量有人工标注的数据。DS 在后训练阶段摒弃了这一做法，设计了基于评分标准的 RL 数据，并使用 GRM 来评估策略轨迹。报告中提到，DS 发现这样做只需要少量人类标注的数据就可以获得相当好的性能。

关键是，我们将强化学习优化直接应用于 GRM 本身。在这种范式下，执行网络原生地充当 GRM，使模型的评估（判断）能力能够与其标准生成能力共同优化。

![GRM vs RLHF](./assets/image/grm-vs-rlhf.png)

用稍微具体一点的例子来说明 GRM 和 RLHF 之间的区别：

- 传统 RLHF + reward model 更像一个打分器：

```plaintext
LLM 生成多个回答
->
人类对回答做偏好标注、排序，或在某些设置中给出质量判断
->
这些标注用于训练一个独立的 scalar reward model
->
后续 RL 阶段中，reward model 给 policy trajectory 输出一个分数奖励
->
主模型再通过 RL 学习如何提高这个奖励。
```

- GRM 更像一个会自己给自己写评语的评审：

```plaintext
输入：
  prompt + answer + rubric

输出：
  evaluation / critique / judgment
  可能再转成 reward signal

可能生成：
  该回答正确识别了用户的核心诉求；
  但没有区分事实与推断；
  对工具调用失败场景覆盖不足；
  整体可执行性较强，给 0.76。
```

需要说明的是，GRM 的思路天生带有自增强的效果。如果在初期的自评审过程没有做好，可能带来的不是进化而是自坍缩。因此 DS 报告也仅提到可以减少人工标注工作量，而不能替代这一过程。

### 5.2 后训练框架

如前文所述，从 5.1 到 5.2 的逻辑顺序和从第二章到第三章一样，从谈方法论到谈工程落地。

#### FP4 量化应用于后训练

第三章中已经详细提及了 FP4 量化在整个框架中的作用，以及多种类型的精度在不同层之间的流转。在后训练这里，报告原文明确说在后训练阶段所有 rollout 轨迹的生成，也就是仅推理阶段，全部采用 FP4 量化。这里的 rollout 不仅包括专家教师模型，也包括基模，这说明 DS 在后训练阶段开始，进一步强化了 FP4 低精度的使用，这本身也非常合理，因为后训练阶段本身就是让基模更加适应真正的现实环境和现实任务的过程。

而对于反向传播，也就是训练过程中，也如第三章所说，使用无损的 FP4 到 FP8 的反量化来模拟，无缝使用现在成熟的 FP8 混合精度框架。

#### 高效的全词表 OPD 教师调度

这部分可以用一张通俗易懂的图来介绍：

![高效的全词表 OPD 教师调度](./assets/image/opd-teacher-schedule.png)

报告当中强调，真实的训练中，教师极其之多几乎是无限的，而每个教师可能都有万亿级别（相当于 1T）的参数量，每次都全量加载这些权重参数不现实。DS 的做法是把教师模型的参数卸载到分布式存储中，在推理时按需加载，使用第三章中提及过的 ZeRO 来进行切片加载，降低 IO 和 DRAM 压力。

DS 同时提到，即使分片加载，对于词表较大的教师模型来说，逐个固化 logits 也是不可能的。为解决这一问题，工程上的方案是：只缓存最后一层教师模型的 hidden state，当推理时需要用的时候，直接从这些 hidden state 临时做一遍前向计算得到 logits。这部分的思路，与第三章中提到过的在 mHC 优化中的 Recomputation 思路极为相似。报告中也提到，这种临时前向计算的代价几乎可以忽略不计，但省下了大量显存。

DS 还对教师模型的预测头在 GPU 上的占用进行了优化，使用了索引对训练样本进行了排序，这样可以使得在一个 batch 的样本训练时，每个教师模型的预测头只加载一次，而不会重复加载。大体示意如下：

```python
# 一个batch里有100个样本
batch = [x1, x2, x3, ..., x100]

# 每个样本在训练时需要进行指导的教师模型id初始时是乱的
# 假设有2个teacher
teacher = [t1, t2]
# 100个样本训练时需要的专家索引：
batch_teacher_index = [1, 1, 2, ..., 1]

# 把batch按照teacher_index进行排序
sorted_batch = [x1, x2, x100, ...., x3]

# 排序后可以按 teacher 分组顺序处理：
# 先加载 teacher1 的 prediction head，处理所有需要 teacher1 的样本，然后卸载；
# 再加载 teacher2 的 prediction head。
# 这样一个 mini-batch 内每个 teacher head 只加载一次，而且同一时刻显存中最多只保留一个 teacher head。
```

#### 可抢占和容错的部署服务

这一节的内容更加工程化，主要解决分布式 GPU 系统的资源抢占和故障重放问题。

- DeepSeek 的 GPU 集群设计了全集群维度的抢占式任务调度器。用一个我们最熟悉的场景来说：一个 agent 任务期间可能涉及大量的工具调用，在模型发起工具调用和工具调用返回结果之间，GPU 事实上不工作，形成了 GPU 的空闲，这个时候任务调度器就会派发任务予以抢占，这意味着任何 running task 都可能随时被抢占资源。
- 大规模 GPU 集群中，硬件故障极为常见，一个 task 可能上一次还在这个 GPU 上训练，训着训着就可能显卡烧了 / 触点接触不良等等，这个 GPU 就会陷入不可用。

对于这两个问题，DS 的技术方案和数据库的 binlog 恢复思路一致，被称为 Write-Ahead Log：先记日志，再改数据。即使被抢占导致上一中间层找不到 / 硬件故障了导致这一步的数据丢失，也可以通过记录下来的日志进行恢复，简单示意为：

```plaintext
每个 generation request
->
每生成一个新 token
->
立刻 append 到这个 request 的 WAL
```

如果资源被抢占了，没关系，kv cache 还在，可以直接配合 WAL 恢复：

```plaintext
暂停推理
->
保存未完成请求的 KV cache
->
恢复时用 WAL + KV cache 继续 decode
```

如果机器挂了，那么 KV cache 也就丢了，但 WAL 中仍然记录着上一步已经生成跑完的 token，利用已经生成的 token 重新做 prefill 生成 kv cache，然后继续任务：

```plaintext
取得 WAL 里已经生成的 tokens
->
重新跑 prefill
->
重建 KV cache
->
然后继续生成
```

最有意思的是，虽然我们观察这样一个恢复过程会觉得非常合理也非常好，但 DS 的报告不仅说了他们的方案，还把他们为什么一定要做中途恢复，而不是从 0 重新生成的理由说得非常清楚，这里直接引用翻译后的原文：

```plaintext
从零重新生成未完成请求在数学上是不正确的，因为这样会引入长度偏差。
由于较短的响应更有可能在中断中存活，从零重新生成会使模型在每次发生中断时更容易生成较短的序列。
```

对原文这句话进行解释的话就是：

- 一个持续时间越长的序列，越容易因为中途任务抢占或者硬件故障被打断
- 一个短回答序列，不太容易被中断
- 如果每次中断，都从头生成采样，长回答序列就更容易被丢弃或者重来
- 大规模后训练中，任何这种工程上引入的概率都容易带来模型自身的行为偏移
- 模型生成短序列成功率高，生成长序列成功率低，模型会自然地倾向于生成短序列

这部分阐述非常有提示意义，这意味着解决工程上的问题不仅仅要考虑工程本身，还要考虑工程方案是不是本身会引入额外的偏见或者偏好，这对于 LLM 这样一个概率模型更加重要。

#### 面向百万标记上下文的 RL 框架扩展

这一节阐述的是 1M context 下，RL / OPD 数据怎么搬运和加载，能更便宜地做训练的问题。

1M context 的后训练不只有 GPU 计算，还有数据系统问题：

- 超长 rollout，比如一个连续跑了几个小时甚至超过 1 天的大长尾 coding task 的 rollout，而且类似的 rollout 可能有成千上万条。

DeepSeek 的做法是把 rollout data format 拆成两类：

- Lightweight metadata 轻量级元数据，在数据分发时被用于全局 shuffle 和 packing 布局
- Heavy per-token fields 重量级逐 token 的数据，则通过 shared-memory data loader 加载，避免节点内数据重复，并且在 mini-batch 上消费完毕后就立即释放，来降低 CPU / GPU 的内存压力

这个设计可以部分类比到 agent 的 Skills 中，CPU 和 GPU 的内存就好比 agent 的上下文窗口，非常珍贵，能省就应该省。而 metadata 就类似于 Skills 的 name 和 description，这部分可以在 agent 初始化时就全量装载到上下文窗口中。而 `SKILL.md`、`references/samples/scripts` 这些更重的东西则放在硬盘上，agent 自己决定什么时候读取 metadata，什么时候从硬盘读 `SKILL.md`，什么时候执行 `SKILL.md` 里路由的 scripts，动态加载到上下文窗口里。（当然不恰当的地方在于 agent 没有卸载释放已经被加载的 Skill 的机制。）

#### 面向 Agentic AI 的沙箱基础设施

有过自主搭建一个生产级 agent 的同学都能意识到沙箱基础设施的意义，在这里就不再过多赘述。报告提到，DS 搭建了一个生产级沙箱平台——DeepSeek Elastic Compute（DSec）。DSec 由三个 Rust 组件组成——API 网关（Apiserver）、per-host 级别的 agent（Edge）以及集群监控器（Watcher），这些组件通过自定义 RPC 协议互连，并在 3FS 分布式文件系统之上水平扩展。在生产环境中，单个 DSec 集群可管理数十万个并发沙箱实例。

报告还特别提到了 DSec 做了以下四个设计：

1. 统一接口下的四种执行基座。
2. 通过分层存储实现快速镜像加载。
3. 减少重复的页面缓存占用并及时回收内存，缓解容器运行时的自旋锁竞争。
4. 轨迹记录与可抢占安全恢复。

### 5.3 + 5.4 Benchmark 相关

一个模型发布之后，大家都喜欢直接去看参数量，还有各种数据集上的 benchmark 来判断这个模型是不是很强。但在笔者的概念和这一节的导读中，benchmark 上的评分本身不算是一个很有价值的事情，仅能当作参考。理由有如下几点：

1. 模型的预训练和后训练过程中，现有 benchmark 很有可能早已污染进了训练集，我们甚至无法避免这样的恶意揣测：厂商有意在后训练过程中针对公开评测集做了针对性优化。从机器学习的通用原理来讲，只要训练集里混进来了测试集，在测试集上的效果说服力都是要打问号的。
2. 现代 agent harness 工程本身会改良模型自身的输出质量。
3. 大量内部部署版本的开源模型，很可能部署的精度（参见第三章对 FP4-FP32 精度的说明）版本是降低了的，这主要是为了降低成本，但这本身就会带来模型能力的难以量化评估的劣化。
4. DS 自己声称 v4-pro 已经和 anthropic / openai 的最前沿模型差距在 4 个月，而国外有团队声称在自己内部从未披露的数据集上，差距应当在 8 个月。这里不是说国外团队的标准就是准，而是不同的数据集，测出来的水平本身就是不一样的。

所以我们把 5.3 和 5.4 节放在一起，不把重心放在 benchmark 分数上，而是看看这个过程中 DS 做了哪些关键的技术处理和技术判断，在这里做一个总结。（btw，报告原文中小小地阴阳了一波 kimi-k2.6 和 GLM-5.1，说：“We have left some entries blank for K2.6 and GLM-5.1, as their APIs were too busy to return responses to our queries.”）

#### Coding agent 测试方式

报告中说，他们把 Agent 测试分成了几种不同范式：代码仓库修复、终端任务、搜索任务、多工具 / MCP 服务调用、办公任务。所有模型的 code agent 对比评测不是裸模型答题，而是用了内部 evaluation framework，只给了最小工具集：**bash tool + file-edit tool**，最大交互步数 500，最大上下文 512K。由于这套 framework 的形态不可知，工具集又采用了极简化的设计，因此这套评测分数未必能直接推广到 Claude Code、Codex 这类 coding agent 上。

#### 推理强度对 benchmark 的影响

先给出 benchmark 完整评分图：

![Benchmark 图 1](./assets/image/dsv4-benchmark-1.jfif)

![Benchmark 图 2](./assets/image/dsv4-benchmark-2.jfif)

我觉得最有意义的地方在于，虽然开大推理预算对模型的性能几乎都是正面影响，但仍然存在反例，比如 MCPAtlas Public（Scale AI 做的一个真实 MCP 工具调用 benchmark，用真实 MCP servers 测试 LLM agent 的工具使用能力）。

更进一步地，在 agent 任务上的有些评测集上，能观察到开 max 和开 high 之间的差距往往不是那么的大，比如 SWE Verified 和 SWE Pro，前者主要是修复 Github 上的 issue，后者主要是面对专业软件开发任务。这一点在最前沿模型的表现评测中也被证实，从最高档的推理预算下降到略次一档的推理预算，模型表现并没有太多的劣化，而这部分推理预算烧的 token 却又是实打实的。

#### Agentic Search vs RAG

报告原文提到，在 DS 的 Web 端和 App 中，non-think mode 使用的是 RAG；而 thinking mode 使用 agentic search。RAG 和 agentic search 之间的争论早已是 agent 工程中的显学，而 DS 的答案是：agentic search 在复杂任务上明显优于 RAG，并且成本只比标准 RAG 略高。

#### V4 不擅长的场景

DS 还评测了日常办公场景下 V4 的表现，评估了包括金融、教育、法律、技术等，工具包括 Bash 和 web search 在内的 13 个行业。这部分评估是由人来做评分的，评估维度是四个：任务完成度、指令跟随、文本质量、格式美学。

在这里，DS 坦率地承认了 v4-pro max thinking 在以下几个方面的不足：**偶尔忽略具体格式约束、不擅长把长文本压成简短摘要、PPT 视觉设计仍有明显提升空间**。

### DeepSeek 报告的全文小结部分

在全文小结部分，DS 提出了当前的几个限制，以及未来的一些展望：

1. 为了能让 1M context 在工程上可落地，做了大量非常重的工程设计，以及一些目前只在工程上证明有效但没有理论证明的 trick，整体架构非常复杂。未来他们打算做一些更系统、更原则化的研究，把架构“distill down to its most essential designs”，在不损失性能的情况下让架构更优雅。
2. 第四章中提到的 MoE loss spike 问题的解决方案虽然工程上有用，但机制仍然是推测的。未来会研究训练稳定性的基础问题，加强内部指标监控，走向更原则化、可预测的大规模训练稳定方法。
3. 除了 MoE 和稀疏注意力，未来还会探索新的稀疏维度，比如 “more sparse embedding modules”，以进一步提升计算和内存效率，同时不牺牲能力。
4. 会继续研究低延迟的长上下文吞吐，让长上下文交互更友好。
5. 继续押注长周期、多轮、工具化、状态化 agent 任务，做进一步的研究探索。
6. 未来会展开多模态的研究。
