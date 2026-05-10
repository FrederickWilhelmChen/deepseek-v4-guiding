## 第 5 章：Post-Training 把“多个强专家”变成“一个可用统一模型”
### 一句话结论
第五章最关键的变化，是 DeepSeek V4 不再试图用一个混杂的 RL 阶段直接把所有能力揉出来，而是选择了：`先训练多个领域专家，再通过 OPD 把这些能力合并回统一模型`。
### 5.1 的核心路线为什么重要
第五章一开头最值得抓住的问题就是：为什么要从 `mixed RL` 转向 `specialist training + OPD`。
导读里给出的解释很有说服力。因为数学、代码、agent、指令跟随这些方向，并不是天然同向的目标：
- 数学任务希望长推理、严谨、可验证
- 代码任务希望通过测试、能执行、能修复
- agent 任务希望多轮规划、工具调用和环境反馈闭环
- 普通指令跟随和对话又往往要求简洁、稳定、低延迟
如果这些目标在一个统一 RL 锅里同时拉扯，reward 冲突和能力互相污染几乎是必然结果。所以 DeepSeek V4 的判断是：先让专家各自练强，再让统一模型学习这些专家在不同任务上下文里的行为分布。
### Specialist Training：能力形成被拆成独立赛道
第五章 5.1.1 真正展示的是一种后训练分工方法，而不是单个算法名称。它把 base model 之后的能力形成拆成多个领域赛道：
- 领域 SFT
- 领域 RL / GRPO
- reasoning effort 档位设计
- 对 hard-to-verify task 的 GRM 评价
- tool-call schema、interleaved thinking、quick instruction 等产品化行为设计
这说明 V4 的后训练目标已经不只是“让回答更好听”，而是在训练：
- 不同深度的推理模式
- 更稳定的工具调用协议
- 长链路工具使用中的 reasoning continuity
- 面向 agent runtime 的辅助决策能力
这一点很像把“模型行为”当作产品协议和运行时协议的一部分来训练，而不只是语言风格微调。
### OPD：为什么它比简单蒸馏和权重合并更值得重视
第五章真正的主线仍然是 `OPD`。它的重要性在于，学生模型不是只模仿老师已经生成好的答案，而是在 `自己生成的轨迹` 上去学习多个 teacher 的完整分布。
这个设计解决了一个很关键的问题：最终上线运行的是 student，不是 teacher。如果学生只在老师自己走出来的轨迹上学习，它到了自己的状态分布里反而可能学不好。OPD 强调 on-policy，正是在避免这一点。
更进一步，DeepSeek V4 还强调的是 `full-vocabulary logit distillation`。这说明他们想保留的不是某个 top-1 token，而是 teacher 对整个输出空间的偏好结构。这么做更稳，但工程代价极高，也因此直接引出了 5.2 的基础设施章节。
### 5.2：为什么后训练已经变成一整套运行时系统
如果 5.1 讲的是“后训练算法路线”，那 5.2 讲的就是“这套路线怎样在工程上跑得动”。这一节最值得带走的判断是：`后训练已经不是只靠普通 RLHF pipeline 就能支撑的轻量流程，而是一套完整的分布式运行时系统`。
导读里把 5.2 的五个重点压得很清楚：
- `FP4 rollout / teacher / reference forward`：让海量 inference-only forward 跑得更便宜
- `Teacher Scheduling for Full-Vocabulary OPD`：让十多个 teacher 的 full-vocab 蒸馏在显存和 I/O 上可承受
- `Fault-Tolerant Rollout Service`：通过 token-granular WAL 和 KV resume，避免中断把 rollout 数据分布偷偷拉短
- `Million-Token RL Framework`：把超长 rollout 数据拆成 metadata 和 heavy fields，避免 CPU/GPU 内存爆炸
- `Sandbox Infrastructure for Agentic AI`：为 agent RL 和评测提供真实、隔离、可恢复的执行环境
这一节非常能说明 DeepSeek V4 的路线特征：当模型开始追求 agent、长上下文和复杂 reasoning 能力时，训练环境本身就得像一个大规模 runtime，而不是一条简单的监督学习流水线。
### 如何更谨慎地读 5.3 / 5.4 评测
评测章节不值得当作“谁分高谁厉害”的广告表去读。更有价值的读法，是看 DeepSeek 想用哪些任务形态来证明第五章前面提出的路线确实有效。
5.3 更像是在回答“标准 benchmark 上的分数如何”，而 5.4 更像是在回答“真实 coding、tool-use、agent、long-context 任务是否好用”。
真正值得关注的不是具体排行榜名次，而是这些问题：
- 它是不是只在容易刷榜的可验证任务上特别强
- 它是否真的用任务去验证了 interleaved thinking、tool-call schema、sandbox 和 long-context RL 的价值
- 它有没有展示失败边界，而不只是成功样例
从导读里还可以带走一个更重要的提醒：第三方外部评测和发布方自述之间出现差异，本身就是有意义的。它说明 benchmark 叙事常常只能证明“在这套题、这套设置下表现如何”，并不等于对真实世界 agent 任务形成了无可争议的结论。
### 从第五章带走什么
第五章最值得记住的，不是 GRPO、GRM、OPD 这些缩写本身，而是这套路线的组织方式：
1. 先把不同领域能力在各自赛道里练强
1. 再让统一模型在自己的状态分布上学习这些专家
1. 同时建设能支撑 rollout、teacher 调度、容错和 sandbox 的完整基础设施
1. 最后再用 benchmark 和真实任务去验证哪些能力真的落到了产品和运行时层面
从工程视角看，这其实是在把“后训练”从一次单一优化过程，升级成 `能力形成 + 能力整合 + 运行时托底 + 评测验证` 的完整体系。
