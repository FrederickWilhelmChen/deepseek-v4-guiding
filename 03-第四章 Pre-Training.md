## Pre-Training 预训练工程
### 章节导语
在引入了新的算法，并进行了针对性的硬件和软件适配之后。终于到了真正进行模型训练的过程。这一章讨论的重点如下：
1. 模型训练前的数据预处理及数据选择
1. 模型训练的前中后期节奏
1. 模型训练过程中的稳定性问题

### 1. Data Construction：预训练数据选择和组织
DeepSeek 在 V3 预训练数据基础上，构造了一个更多样、更高质量、有效上下文更长的数据集，最终V4 flash的数据集规模超过32T token，V4 pro的数据集规模超过33T。重点包括：
- 过滤批量自动生成和模板化网页内容
报告原文中提到过滤这两类网页内容主要是为了防止model collapse。也就是在DeepSeek看来，现在市面上由LLM生成的模板化的网页内容已经成为垃圾废料，这部分内容如果不做专项清除，会直接污染训练过程中的模型认知。模型会认为这些就是最好最常见的范本，学到失真的内容分布，直接被带偏到跟随这部分样本，这就是所谓model collapse
这其实和我们常讲的 Garbage in，Garbage out是一个道理，已经成为模型厂商之间的普遍共识：模型生成的大量模板化的东西会在训练时污染自己，必须花大力气做数据清洗

- 保留数学和编程语料
DeepSeek对数学语料 / coding 语料 / agentic data / long text（如各类论文和技术报告）更为重视
这里需要额外说明一点，报告里特别强调对长文档的数据。对长上下文窗口而言，语料的组织有两种：（1）把多段较短的上下文（甚至包括AI辅助生成延长后的文本）做打包拼接作为训练语料；（2）直接用长文本做训练语料。对于训练来说，后者显然是更好的，因为他是一个逻辑通顺的自然长序列组织文档而前者往往欠缺这一点。但高质量的长文档语料是相对匮乏的，所以第一种方法本身也在广泛采用

- Tokenizer和特殊组织的packing
DeepSeek沿用了V3的tokenizer，仍然保持128k的词表规模。但是加了一些special token，同时继承 token-splitting 和 FIM 策略，并把不同来源的文档 pack 到合适序列里，减少样本截断并提高token利用率
（1）Token-splitting
Token-splitting 可以理解为一种 tokenizer 鲁棒性训练。由于 DeepSeek tokenizer 中存在标点+换行一类组合 token来提高压缩效率，但也可能让模型对特定 token 边界产生偏置。因此训练时随机拆分一部分组合 token，让模型适应多种等价切分方式，减少由于换行、标点、代码格式等造成的边界敏感性。
（2）FIM（Fill in Middle）策略
这部分其实相当匹配于coding场景。因为在coding场景中大量的情景不是直接在原来的代码尾部做续写，而是在line 和 line之间插入一段新代码。也就是说在训练阶段针对一个固定不动的上文和固定不动的下文，生成中间内容

- Sample-level attention mask
在训练过程中一方面是为了提高效率，另一方面也是为了组装长上下文供训练，往往会把多段不相关的文本packing之后进行训练。这本身是正常做法
但对模型来说，他并不知道这多段文本的确是完全没关系，组装起来只是为了工程优化，他可能从这些本来毫不相干的文本知识之间自己学出来一些奇怪的关联性。为了避免这个问题，DS的方案是：
<text underline="true">*一个 batch sequence 里可以 pack 多个样本，但 attention 上限制不同 sample 之间互相看不到。*</text>
这里涉及到Transformer在decode阶段的基本原理，传统的decode通俗上说是：预测下一个token时，只能看到前面的token，后面的token通过mask的方式屏蔽起来不让模型看到。DS的做法示意图如下：
<image token="NjgMb2qeQo9vcex7VnEcNiEun2d" url="https://internal-api-drive-stream.larkoffice.com/space/api/box/stream/download/authcode/?code=NWZjNjk0NDg4NWY3ZTkwODMyOWIyNmMyMWZmNmE2ZGJfMTQ4NjI2ZjNhYjNiYmE5OWM5OTZiZGQ2ZGQ4MTQ0M2RfSUQ6NzYzODMyMTI0Mjg4NTM2MDgyNV8xNzc4NDM2NTQyOjE3Nzg1MjI5NDJfVjM" width="1448" height="1086" align="center"/>


### 2. Model Setups：Flash 和 Pro 的规模差异
DS发布了flash和pro两个版本，两个版本参数如下：
- Flash
```plaintext
43 层 Transformer
总参数 284B，每 token 激活 13B
CSA top-k = 512
HCA 压缩率： 128
MoE：1 个 shared expert + 256 个 routed experts
每 token 激活 6 个 routed experts
前两层用 dense 的 pure sliding window attention，后续层交替使用 CSA 和 HCA
```

- Pro
```plaintext
61 层 Transformer
总参数 1.6T， 每 token 激活 49B
CSA top-k = 1024
HCA 压缩率 = 128
MoE：1 个 shared expert + 384 个 routed experts
每 token 激活 6 个 routed experts
前两层直接用 HCA，后续层交替使用 CSA 和 HCA
```

其中最两个模型对前两层的注意力构造区别是一个值得注意的点，为什么要这么做，本人总结了以下几个可能的原因，需要强调的是<text underline="true">*纯属个人推测，报告中没有直接说明原因*</text>
1. flash版本更需要低成本省算力，因为滑动窗口不需要做压缩机制，也没有配套压缩机制而来的各种工程处理
1. HCA如第二章所说，属于建立一个粗粒度的全局背景，DS可能认为在flash中没有必要在低层transformer中就建立这样一套机制，而是应该优先看近处的信息，后续高层transformer再引入长程信息
1. 滑动窗口中的attention是稠密的，相较于HCA没有复杂的工程处理，对于flash而言有利于增加稳定性

#### Training Setups 训练节奏
这部分报告原文放在了4.2节 Model Setups下面，主要阐述了三件事：
1. 优化器组合
这部分优化器指的就是第二章提到的AdamW和Muon。报告提到，Flash和Pro采用同样的策略，对大多数参数都使用Muon，而对embedding、prediction head、所有 RMSNorm 权重使用 AdamW
这里体现了DS工程上的取舍，主要矩阵参数使用Muon来提升收敛和稳定性，但一些敏感参数仍然保留了传统AdamW方案
1. 递进的序列长度
Flash和Pro都支持1M context，但在训练上，不是从一开始就直接上1M context，而是一个递进的过程，从 4K -> 16K -> 64K -> 1M
这也比较容易理解，从0开始的模型训练不可能从一开始就上1M context，就像不可能直接让婴儿读四大名著一样，要有一个先学简单的后学难的这样一个过程
1. Sparse attention的引入时机
这部分的描述，可以比较粗略的理解为DS选择在训练的什么阶段真正引入CSA的机制
Flash 的策略是：在训练的前 1T tokens 中用的是dense attention做warm up。训练序列长度扩大至 64K 时，引入 CSA，并在剩余训练中保持。引入CSA时，还要先短暂 warmup CSA 的 indexer，再进入大部分 sparse attention 训练。
Pro 与 Flash 类似，但 dense attention 阶段更长
DS的这部分工程处理意味着两件事：（1）稠密注意力的基座打底过程必不可少，先在较短的文本上把稳定的token表示和分布先学稳定才能后续引入压缩和筛选；（2）indexer的筛选不是一开始就能学会的，他需要从稳定的注意力分布中先学习怎么做选择，如果这一步还没学会，过早上CSA反而会让模型在训练早期就走错路
### 
#### Mitigating Training Instability 训练稳定性问题
DS在报告中提到在训练过程中遇到了 loss spike问题。回滚可以暂时恢复，但不能阻止spike再次发生。
简单解释下loss spike：
有过任何机器学习算法训练经验的人都知道，训练的本质就是不断降低模型预测结果与标签本身之间的差异，这个差异通常用loss来衡量。正常且良好的的训练过程loss应当在总体上是逐步下降的，但loss spike是在本来逐步下降的loss里突然出现了显著上扬的loss尖峰，这个尖峰可能后续就自动在训练中消弭，也有可能直接把训练带崩
<image token="Fes2bVonDorG0NxPwCJcucX0n4e" url="https://internal-api-drive-stream.larkoffice.com/space/api/box/stream/download/authcode/?code=ZWEzMjk0ODQ0NTY3NTQ2ZjlmYjNiOGM1YjRhZDE2MTFfNTg5OTMwYzRjMjU4YmIyMWVhMjI1MmEyZmMwNzkxYTJfSUQ6NzYzODMyMTI0MDA4NzkwNzI5OV8xNzc4NDM2NTQyOjE3Nzg1MjI5NDJfVjM" width="1448" height="1086" align="center"/>

DS从经验角度发现，spike 和 MoE 层里的异常值绑定，而专家路由机制似乎会加剧异常值的出现。于是他们从两个方向处理
1. Anticipatory Routing
为什么DS认为MoE的Expert routing会提高异常值出现几率并引发spike，因为专家路由产生了一个环路：
```plaintext
上一轮中Expert自身由于各种可能的原因出现了短暂的异常，中间层输出了异常值，影响了下一轮训练的模型参数
->
router在受影响的模型参数上计算路由结果，在expert选择上也开始变得异常
->
router行为改变，开始更加偏向把token送到异常expert上
->
闭环发生，整个系统整体开始自强化异常，spike发生
```

DS的解决方案在于打破这个环路，下面是一个简略通俗化的表示：
```plaintext
Expert自身由于各种可能的原因出现了短暂的异常，中间层输出了异常值，这一步仍然可能发生
->
这一轮token应该路由到哪个expert，不由当前轮的模型参数计算得到，而是由若干轮训练之前的的模型参数得到
->
router行为没有直接被这一轮的异常影响
->
闭环在这一轮被打断，后续expert自身的短暂异常可能已自恢复，即使没有自恢复，这个spike的进程也会被大大减速
```

需要强调的是：DS<text underline="true">*不是在所有训练中都做Anticipatory Routing*</text>，而是在loss spike真发生了的时候，先做回滚，然后开启Anticipatory Routing，维持一段时间后，再回到正常训练行为

1. SwiGLU Clamping
Anticipatory Routing 放在后端系统里的类比，更像是熔断、回滚、降级、恢复：当系统已经出现 spike 风险时，先切断异常反馈环路，让训练重新稳定下来
SwiGLU Clamping 的思路则更偏数值稳定性：不是等异常反馈环路形成后再处理，而是在 FFN 的关键中间变量上直接限幅，防止异常激活值被继续放大和扩散。
解释 SwiGLU 需要引入一些额外概念，这里可以简化理解：
- SwiGLU是Transformer中常见的门控激活结构，用来决定哪些信息应该被放大、哪些信息应该被抑制
- SwiGLU 里存在一个乘法门控结构，某个分支如果偶发出现极端大值，和另一个分支相乘后可能进一步放大，形成异常值
- 既然这些异常值可能污染后续计算，那最简单的思路就是直接做限幅（报告原文中设置阈值是10），超过阈值的值直接截断。这样即使中间变量偶发异常，也会被截断在可控范围内，降低异常值扩散并诱发 loss spike 的概率

DS报告中自己也提到，以上提到的两个方案，不是严谨的数学优化，而是一个经验化的工程方案，事实上他们对原因的归纳本身也是经验式而非有数学推导。当然，这两个工程方案本身被证明是有效的

### 3. Evaluations：base model 基模的benchmark
本节主要是对预训练得到的基模的评测结果。如导读最一开始，benchmark本身不是本文关心的重点，但报告中提到的一些现象的确非常有趣值得拿出来说
<image token="GBDDbDeVvowChRxl1RNcvY69nss" url="https://internal-api-drive-stream.larkoffice.com/space/api/box/stream/download/authcode/?code=ZTA0Nzk4NzE0N2MyZjkyM2FkZjc3M2RmN2VlZjc0YjhfNjVmM2U0MGU2Y2FiMzY4OGEwOTczMmYwNDY1ODBhMDFfSUQ6NzYzODMyMTI0MTUzNjUxNTAwMF8xNzc4NDM2NTQyOjE3Nzg1MjI5NDJfVjM" width="1232" height="858" align="center"/>

对于预训练出来的基模，在coding任务上，V4的两个版本和V3.2比性能有参差，并不是一边倒。在BigCodeBench上，V3.2的基模比V4 flash和V4 pro都要强，而在Human Eval上，V4两个版本相较于V3.2都有很大优势。需要解释的是BigCodeBench和Human Eval分别是什么。
- Human Evel更像传统代码生成能力测试：给一个函数签名/说明，让模型补一个通常比较短、相对自包含的 Python 函数。它更考验基础算法、局部代码生成、语法正确性和简单逻辑，通常被视为为短函数级代码生成评测，类似于代码补全任务
- BigCodeBench则更接近现实开发中的小任务：它要求模型理解更复杂的指令，并组合使用大量 Python 库和函数调用，重点挑战模型使用 diverse function calls 和复杂指令的能力。
当然，这里的分数仅仅是基模上的评分，真正开放出来的模型能力都是需要经过大量的后训练
