## ⚙️ General Infrastructure：1M 上下文的系统工程
### 章节导语
算法的创新和进步不能只停留在数学上，如果算法在数学上能证明有效但在GPU上训不动或者训不好带来的大概率是负收益。从DeepSeek V3开始就采用的MoE架构沿用到了V4，但MoE在训练上天生就要比Dense架构要解决更多的工程问题（动态稀疏计算和路由分发）。这一章主要是介绍DeepSeek在MoE架构上把算法真正落地到GPU上的训练上所解决的大量工程问题

### 1. Expert Parallelism：MoE 中的时间组织
#### 简化通俗解释
<image token="VC3gbw7dsozX9bxi2RIcOLGNnLh" url="https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NmI2OGY2NDFhYTRkZjI4MzZiOTZkYzdhODE4YmIwMWRfZjI4ZmFkN2M5ZjEwOWVjYzVjNDQ2NjdmNGUyNmU4ZmJfSUQ6NzYzODIyNzg5NjE5NjI4NzQ1NF8xNzc4NDM2NTM4OjE3Nzg1MjI5MzhfVjM" width="1536" height="1024" align="center"/>


#### 问题背景
如果仔细看MoE模型的参数，除了总参数量标注之外，往往都会带另外一个参数——激活参数量
DeepSeek V3.2 ：全参数671B，激活参数37B
Kimi k2.6 ：全参数 1T，激活参数32B
DeepSeek V4 flash ：284B，激活13B
DeepSeek V4 pro ：1.6T，激活49B
MoE 的优势很清楚：每个 token 只激活少量 expert，所以 activated FLOPs 更低。但真正上线到大规模训练或推理系统后，专家路由和分发带来的GPU之间的通信成本非常大
token 要被分发给不同 expert，expert 结果还要再聚合回来。这个过程中，大量 all-to-all 通信会直接拖住吞吐。
为了解决这个问题，DeepSeek提出了Expert Wave，在一波一波的波次中把通信成本隐式的消解在计算中

#### 轻技术解释
Expert Wave 的核心就是把专家执行拆成小波次：某一波 token 一到就开始算，而不是等全部通信完成再统一处理。这样系统总时间就更接近 <text underline="true">*通信时间和计算时间的最大值*</text>，而不是把几段时间直接相加。
做这个优化的必要性在于：很多场景下，<text underline="true">*计算时间并非显著大于通信时间*</text>。事实上，真实的GPU训练和推理的负载有时可能是细碎的，长度是明显不齐的，计算时间有时甚至能和通信时间打平。通信带来的不仅是延时，还是GPU的空转和资源浪费，通信不只是网络延迟问题，还会造成计算流等待+挤压计算资源
对于GPU，其实<text underline="true">*batch越大越好，*</text>因为可以把矩阵乘法做大，利用率高。最不想看到的其实就是一堆小矩阵计算或者一批里有某些很慢的。小矩阵计算带来GPU空转资源浪费，长尾请求带来一批里大家都在等这个长尾的结束才能卸载。然而在实际后训练中，这种小计算和长尾效应非常普遍

### 2. TileLang：DeepSeek自己的 kernel 平台

#### 问题背景
DeepSeek V4 的热点路径里，有很多并不是标准算子能自然处理好的东西。传统Transformer中，PyTorch和CUDA已经提供了大量调试优化好的算子可以直接拿来用，但对于DeepSeek而言为了成本妥协/工程化需要，已经和传统Transformer训练有了区别。如果完全依赖默认算子，这些路径会被拆成大量细碎 operator；如果全部手写 CUDA，研究迭代速度又会大幅下降。
#### 轻技术解释
在 GPU 编程里，**kernel** 是在GPU上跑的一段计算函数，包括矩阵乘法/激活函数/softmax等等。对于PyTorch + CUDA而言，它的底层有一套算子库，每个算子在底层都使用一个又一个operator来完成计算。每个 operator 调用都有开销，而且中间结果可能要反复读写显存，读写显存本身有其开销浪费
TileLang 在这里承担的是中间层角色：一方面保留研究侧迭代速度，另一方面又能通过 fused kernels、host codegen 和形式化整数分析，把复杂结构逐步压到足够高的执行效率。

- **Fused Kernels** 
把多个小算子尽可能合并整合到一起成一个大算子（并且在DeepSeek架构里可以复用），读一次显存完成多步运算，然后再一口气写回显存，技术思路上和模块化编程是一致的

- **Host Codegen**
kernel的计算本身发生在GPU上，但发起kernel这个操作，实际上是由CPU来操作的，也就是说每次发起一次kernel，CPU自己都要做一些前置工作。当GPU的性能越来越强之后，CPU的这些前置工作开销反而占比越来越大，有的时候甚至CPU喊GPU干活这个过程比GPU把活干完还要慢。Host Codegen负责在kernel编译的时候就提前生成这些前置代码，后续不再在运行时干前置工作而是直接走预编译好的
如果和java代码做一个对比，原来每次请求都要动态反射检查参数、解析 schema、构造调用对象。现在则根据接口定义提前生成 stub，运行时直接走生成好的强类型调用路径
这样优化后DeepSeek发现这部分耗时从几十到几百微秒直接降低到不到一微秒，虽然看着似乎没什么意义，但对于大规模训练和常见细碎kernel运算来说，累计起来仍然非常可观

- **SMT Solver Assisted Formal Integer Analysis**
这部分涉及比较抽象的数学解释。从尽可能通俗的角度来说，<text underline="true">*kernel的编译器需要证明——这个运算是安全的*</text>。如果运算的安全性能够被数学证明，那么编译器就可以放心大胆的做优化
一个 JAVA 代码的类比：
```java
int length = 100
int[] a = new int[length]
int[] b = new int[length]
for (int i = 0; i < 10; i++) {
    if (i*4 + 3 < length && i < length) {
        a[i * 4 + 3] = b[i];
    }
}
```

这段代码做了边界防护，但我们一眼就能看出来这个边界防护并无必要因为根本不会越界。编译器看到这段代码，就可以先委托整数分析器分析这个边界防护是不是真的有必要，如果这个防护代码根本没必要，编译器就会直接无视这段if，直接执行里面的逻辑

为什么这个这么重要，因为kernel运算里包含大量索引计算，带来了越界/转换等价性/地址冲突等问题，必须要一个求解器来做整数分析判断优化是不是安全的。如果不安全，就只能保守处理；反之就可以做激进的优化

### 3. Deterministic Kernels：确保可复现性并约束对模型行为的影响
#### 问题背景
大前提：<text underline="true">*计算机系统的浮点数运算中，实数的加法结合律不成立*</text>
```python
a = 1e20 + (-1e20) + 1
b = 1e20 + 1 + (-1e20)

print(a)
print(b)
```

运行上面一段python代码，a的值是1，而b的值是0。这就是<text underline="true">*浮点数加法不满足结合律*</text>，也就是说在浮点数运算里，即使是同一批数，顺序位置不同，最后的结果是完全不同的
（btw，浮点数运算问题是很多人疑惑的把temperature设置为0了大模型还是每次都可能给不一样的结果的根因）

为什么这件事会这么麻烦又要命：
为了追求 GPU 性能，系统常常会做这些事：
- split-k
- split-KV
- parallel reduction
- atomicAdd
这些技术本身都是合理的，但一旦 partial results 的合并顺序依赖 batch 组织、SM 调度或线程先后顺序，浮点非结合律就会把这些底层差异一点点放大。
<whiteboard token="X5t5wUdAuhSpSVbnI0ucI7I4ncn" width="341" height="549" align="left"/>

在 Dense 模型里，很多时候它只是小扰动；但在 MoE 模型里，这个问题就会变得更严重
从最简单的一个方面说，MoE模型需要根据用户的输入query来路由到正确的专家并激活参数进行推理，但实际上有的时候不同专家之间的路由选择评分差距并不是很大，这样一来即使是浮点数运算带来的误差也可能带来专家路由产生区别。除此之外采样生成、RL rollout、Agent 轨迹里，微小差异可能很快变成离散路径变化。
一个 token 采样错了，后面整条轨迹就可能换世界；一个 route 变了，专家路径就变了。在训练和推理中会出现发现出了问题，回退重置之后就复现不出来错误路径，调试和优化变成非常困难的工作

#### 轻技术解释
DeepSeek V4 的应对思路：
1. Batch-invariant
某个 token 的输出应该和它在 batch 里的位置无关
```plaintext
请求 A: prompt = "1+1="
请求 B: prompt = "巴黎是法国的..."
请求 C：prompt = "pip install -r requirements.txt"

batch = [A]
batch = [B, A]
batch = [A, B]
batch = [A, B, C]

batch-invariant 要求：
A 的每一层输出在以上各种 batch 组织下 bitwise 完全一样（每个bit都一样）
```


普通的kernel运算无法保证这一点，原因为假如一个长序列attention，GPU往往会进行split kv或split reduction
```plaintext
一个长序列的 KV 
↓
SM 0 算前一段 KV 的 attention 
SM 1 算后一段 KV 的 attention 
↓
再把 partial result 合并
SM 0 和 SM 1 哪个先算完是无法预知的，合并过程中顺序一不一样就会出现不一致
```


DeepSeek在工程上的策略是：
设计了两个计算kernel
- First kernel：多段prompt经过Transformer处理后有多个attention sequence，假设一块GPU上有120个SM（计算核心簇）而这一轮wave进来也有120个sequence，处于满wave的状态，那么一个sequence只在gpu的一个SM上做计算，不再做split kv
- Second kernel：如果一个wave里打不满所有SM，假如只来了30个sequences（批训练的最后一批或者就是这一批只有这么多），允许一个sequence分配到多个SM上来降低尾部延迟，但是强制指定不同SM的累加顺序不允许自由组合

1. Determinism
Batch-Invariant关注：<text underline="true">*同一个 token 放在不同 batch 位置，输出是否一样*</text>
Determinism则关注：<text underline="true">*同样输入、同样配置，重复跑多次是否一样*</text>

普通做法：
- 多个 SM 同时写 atomicAdd 到同一个 梯度KV buffer，无法保证先后顺序
- MoE多个专家的权重梯度和token输入梯度的写回combine阶段，多个expert的写回顺序不固定
DeepSeek改进：
- 在注意力的反向传播阶段，不同的 SM 只写自己的那块buffer，不同 SM 之间的buffer区域隔离，最后按照定死的顺序把 buffer 中的内容累加
- 在MoE的反向传播阶段，GPU分为了不同的rank（一个rank上可能只有一个expert也可能有多个expert），每个rank不再是谁先算完谁先写回，而是写回时先把写回的内容强制固定顺序而且每个也有自己固定的buffer，rank A先算完了也只能写回buffer A，不能写进留给 rank B 的 buffer B
- 对于第2章/Architecture中提到的mHC，由于mHC运算的矩阵维度太小（只有24维）不得不做split k，但是每个split计算完之后不直接合并而是单独输出，由独立的kernel做determinism reduction
Deterministic 对 debug 硬件或软件问题意义极大；当训练出现 loss spike 之类异常时，determinism 可以让研究人员更容易定位数值原因并改进模型设计。但这么做也带来了额外的显存和计算开销（这一点报告中没有显式的提到，但也能推理得到这一结果）

### 4. FP4 QAT：把极致成本压缩工程化
前置知识：<text underline="true">*FP4 / FP8 / BF16 / FP32*</text>
#### 问题背景
从FP4到FP32，本质就是用多少位二进制数来表示一个浮点数，很显然位数越多浮点数越精确，FP4精度能表示的浮点数则相当不精确，能表达的数字也比较少。但位数增多带来的是训练/推理时显存消耗，对于现代动辄上百B甚至超过1T的大模型而言，从FP4->FP8->BF16->FP32所消耗的显存可以**近似**认为是翻倍的（实际LLM参数组成很庞杂，这里不多展开），高昂的成本让最初以FP32精度训练和推理的模型逐渐开始降低浮点数精度来压低推理成本
低精度最常见的做法，是模型在高精度浮点数训练完以后，把高精度浮点数压缩成低精度版本进行部署。目前FP8精度在训练和推理上都已经非常常见。但这种方法的问题在于：训练时模型活在高精度世界里，部署时突然被扔进低精度世界，高精度下准确没有表现出问题并不代表低精度下也能同样成立，因此每压缩一步精度都会带来推理质量的劣化。
DeepSeek V4 的思路是反过来：既然低精度是必然的，而且V4后续在全国大范围私有化部署低精度版本是可以预期到的。既然如此，那就让模型提前适应它，在后训练阶段就开始引入低精度，而不是训练完了再压缩。
#### 精度流转图
<whiteboard token="FGzJwouuIhfUE3bP8MqcjtFSnFb" align="left"/>

V4不是全部参数都做FP4量化，只在下面两个参数上使用FP4
- MoE专家权重参数 ：显存访问极为频繁的参数，也是MoE模型参数中的主力
- 第二章 Architecture部分 CSA 的 indexer 和 qk path 
这里需要补充说明与CSA相关的内容，第二章的CSA是将长程历史kv entry做一定比例的压缩，然后挑选最相关的的历史kv entry块。这个挑选就是 indexer 的职责，而相关性评分则是indexer中qk path的职责。从工程上说，CSA是为了节省推理过程中的kv cache消耗，减少全量 attention 扫描长历史带来的计算和内存带宽压力，但CSA这个过程尤其是indexer本身不能太耗资源，否则为了节省推理过程的消耗而大幅增加了途中的消耗并不值得。

具体不同部分的精度分工：

<lark-table rows="10" cols="3" header-row="true" column-widths="244,244,244">

  <lark-tr>
    <lark-td>
      场景 / 对象
    </lark-td>
    <lark-td>
      精度
    </lark-td>
    <lark-td>
      作用
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      Optimizer master weights
    </lark-td>
    <lark-td>
      **FP32**
    </lark-td>
    <lark-td>
      保存高精度主参数，避免小更新被低精度吞掉
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      MoE expert weights 存储/部署目标
    </lark-td>
    <lark-td>
      **FP4**
    </lark-td>
    <lark-td>
      大幅降低 expert 权重显存和访存
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      MoE expert weights 训练计算承载
    </lark-td>
    <lark-td>
      **FP8**
    </lark-td>
    <lark-td>
      FP4 解码后放入 FP8，复用已有 FP8 training framework
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      MoE expert backward 梯度回传
    </lark-td>
    <lark-td>
      梯度经 STE 回到 **FP32 master**
    </lark-td>
    <lark-td>
      量化不可导，用 STE 近似直通
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      部署和采样中的 expert weights
    </lark-td>
    <lark-td>
      **FP4**
    </lark-td>
    <lark-td>
      与线上部署一致，同时减少 memory loading
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      CSA indexer / qk path
    </lark-td>
    <lark-td>
      **FP4**
    </lark-td>
    <lark-td>
      cache、load、multiply 都低精度，降低长上下文 indexer 成本
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      CSA index scores / Top-k selector
    </lark-td>
    <lark-td>
      **BF16**
    </lark-td>
    <lark-td>
      分数本身用较高精度存储提高路由质量
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      某些普通训练/激活/累加路径
    </lark-td>
    <lark-td>
      BF16 / FP8 / FP32 混合
    </lark-td>
    <lark-td>
      取决于 kernel 和数值稳定性需求
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      其他 KV cache 维度，按前文报告描述
    </lark-td>
    <lark-td>
      **FP8**
    </lark-td>
    <lark-td>
      降低 KV cache 存储
    </lark-td>
  </lark-tr>
</lark-table>

- FP32 
主要承担主版本权重（高精度母版权重）和优化器状态的高精度维护。即使前向计算中使用了低精度量化权重，训练时仍需要一个高精度版本作为长期参数基准，用于累积梯度更新、避免低精度量化误差在多轮训练中持续放大
对 MoE expert 权重来说，DeepSeek V4 并不是直接在 FP4 权重上做长期更新，而是保留 FP32 主版本权重；每次模型在做前向计算时，时从 FP32 主版本权重量化到 FP4，再进入后续计算路径。反向传播得到的梯度最终也作用回 FP32 主版本权重。
- FP4 
主要负责MoE expert权重，CSA 的 indexer + qk path
- FP8 
训练中的主要格式。基于FP8的训练框架和硬件能力较为成熟，DeepSeek复用了这套框架
- BF16 
用于表达敏感的中间值，如CSA indexer和qk path算出来的不同块的相关性分数，这部分分数保留较高的精度来提高块选择的质量

### 5. Training Framework：将算法创新纳入训练框架
前置知识：<text underline="true">*第2章 Architecture所有内容*</text>
#### 问题背景
DeepSeek V4 前面已经引入了很多非标准结构：Muon、mHC、CSA/HCA、FP4 QAT、复杂 kernel。算法上的创新，kernel的重写完了还要在GPU上能优化训练，接进现有的分布式训练框架
范式更改带来的问题如下：

<lark-table rows="4" cols="3" header-row="true" column-widths="237,249,249">

  <lark-tr>
    <lark-td>
      范式更改
    </lark-td>
    <lark-td>
      原形态
    </lark-td>
    <lark-td>
      更改后的问题
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      AdamW -> Muon
    </lark-td>
    <lark-td>
      AdamW逐参数更新，可以直接切块并分布式并行（具体方法名为ZeRO，具体做法可自行查阅此处不展开）
    </lark-td>
    <lark-td>
      Muon开始引入权重矩阵更新方向的全矩阵的正交化，必须保持按原矩阵的整体行列结构。AdamW的方式切块会丢失矩阵的方向感知
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      HC -> mHC
    </lark-td>
    <lark-td>
      传统 residual 的工程成本最低；普通 HC 已经引入额外连接状态和映射
    </lark-td>
    <lark-td>
      使用双随机矩阵，引入了额外的归一化 / 投影 等小算子的计算和显存开销，这些小算子开销会显著增加GPU的对齐和优化成本
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      200K context -> 用CSA/HCA支持1M context
    </lark-td>
    <lark-td>
      上下文本身较短，没有相关优化正常训练也ok
    </lark-td>
    <lark-td>
      为了适配1M context的训练和推理，设计出了CSA/HCA，这些新范式需要在GPU并行训练上做适配优化
    </lark-td>
  </lark-tr>
</lark-table>


#### 轻技术解释
- Muon
这一部分在V4技术报告的原文中，写的相当模糊。原文中仅提到，他们仍然会对矩阵做切割和拆分并进行分布式运算，切割方法被称为：Hybrid ZeRO，最后能获得很好的效果，但<text underline="true">*具体是怎么做的，为什么效果好，原文没有做任何解释*</text>。
下面的部分是从kimi于2025.02发表的论文：《Muon is Scalable for LLM Training》（https://arxiv.org/html/2502.16982v1）提出的技术路径，并非原文披露，供参考。
1. 仍然把惯性更新矩阵切片放在不同rank里
1. 但每个rank做Newton-Schulz正交化时，要通过rank间通信把所有rank之间的参数gather起来然后再算，也就是说算的时候仍然是全矩阵算
1. 算完之后每个rank只保留自己rank的update，其他的就全丢掉
<text underline="true">*必须要指出的是，这是kimi在2025.02的成果，时间跨度已经较大。而且从v4报告中有限的陈述中， ds做的不是传统的ZeRO切割，而是他们自定义的切割方式，也就是说ds的思路做法很可能和kimi并不一样，这里只是一个参考。*</text>

- mHC
1. Fused kernel
利用之前提到的聚合kernel计算的方式来把多个小型运算打包在一起，降低显存读写开销，增大单次计算负载
1. Recomputation
传统的前向计算中会保留大量的中间层，并在反向传播时直接利用这些已经算好的中间层（hidden state / layer input / normlized layer input / 各类tensor等等），直接在这些中间层来计算梯度。这样的好处是节省计算成本，但代价是消耗更多的显存，尤其这些中间层往往非常庞大。
Ds的优化方案是，不要全部存所有这些中间层，相反地，把大量的这些计算成本不太高的中间层丢弃，只保留那些一旦丢弃就要重新做高成本计算的中间层，同时保留checkpoint便于恢复中间状态。而在反向传播时，如果需要这些中间层，就局部重放一段前向计算。总的来说，传统方案是用显存换计算，DS的方案则是用计算换显存，扔掉计算成本低的中间层来节省显存，这是显存本身作为紧俏资源的memory-compute tradeoff

- CSA / HCA
CSA / HCA 处理过的attention，经过了 滑动窗口 + 中等比例压缩后top k 筛选后的分块 + 全局高比例压缩出等步骤，这一系列步骤的attention规模不规整，计算也不规整。规模上的不规整会带来切分后不同块的大小不一样，还会带来一个压缩后的kv cache块本身跨了不同的rank。这部分做并行计算需要做特殊的设计，被称为Two-stage，把并行计算分为了两个阶段，
（1）把因为规模不规整带来的跨rank的未经过压缩的kv cache传递到下一个rank上，然后做本地的HCA/CSA压缩，生成和管理压缩后的kv cache
（2）在多个rank之间通信，gather已经压缩好的kv cache并作重组和对齐，后续再在已经重组和对齐好的结果上进一步做 HCA的后续dense attention运算 / CSA qk scoring + top k select + 选中kv cache块的后续运算

- 1M context 带来的大量activation memory
这一部分的问题和mHC中面临的问题非常类似，本质都是前向计算的中间产物过于庞大带来的。mHC中提到中间点checkpoint本身需要保留，但在1M context下，这个checkpoint本身占用也是非常庞大的，带来了大量activation memory。
解决思路也很像，被称为Tensor-Level Checkpoint，实际上是把checkpoint切的更细到了Tensor维度，在Tensor维度决定哪些丢弃等到反向传播时重放，哪些需要保留。技术本质都是以计算换显存做memory-compute tradeoff

#### 本节总结
本节相较于前几节值得单独拉出来一个总结部分，因为这一节真正开始解决第二章提到的技术在GPU中要如何并行训练，并且如何在显存和计算上做取舍，同时利用到了第三章前几节铺垫的GPU维度的基础设施建设。用一个表格来做总结

<lark-table rows="5" cols="2" header-row="true" column-widths="350,350">

  <lark-tr>
    <lark-td>
      问题
    </lark-td>
    <lark-td>
      解决方案
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      Muon 需要矩阵级全局更新，但 ZeRO 要切分状态
    </lark-td>
    <lark-td>
      自定义的Hybrid ZeRO
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      mHC 表达力增强但实现碎片化
    </lark-td>
    <lark-td>
      recomputation + fused kernels
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      CSA / HCA 不规整处理和并行计算
    </lark-td>
    <lark-td>
      Two stage
    </lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>
      activation memory 爆炸
    </lark-td>
    <lark-td>
      tensor-level checkpointing
    </lark-td>
  </lark-tr>
</lark-table>


### 6. Inference Framework：适配长上下文的推理缓存
#### 问题背景
DS引入从算法到工程做了各种优化，虽然带来了推理这个过程的成本节省，但是对于现代大模型而言，成本被划分为input/output/命中缓存/未命中缓存不同情况下的成本分级，KV Cache早已成为成本优化的重中之重，从2025年manus的博客中就已经极度强调KV Cache的命中率，因为命中缓存和没命中缓存的成本差距至少以整数倍计
但对于DeepSeek V4 而言，混合注意力同时引入了：
- CSA 中度压缩KV
- HCA 重度压缩 KV
- Sliding Windows 的滑动局部窗口 KV
- 尚未凑满压缩块的尾部状态
- 稀疏选择器依赖的额外 indexer 状态
这些压缩后的KV Cache已经不再像普通transformer那样规整，这一点在上一节已经做了说明，这使得在原来基于传统Transformer的PageAttention构建的缓存系统中，不能再把所有层的 KV cache 简单统一到同一种分页块管理模型里

#### 解决方案
DS的方案在数学上听起来非常简单直接：
1. CSA（假设压缩系数是m）和HCA（假设压缩系数是n）是两种不同压缩规模的KV Cache块，DS计算m和n的最小公倍数（或是最小公倍数的任意倍数），打包这个数量的原始token成为一个block，这一个block自然而然的生成了整数个CSA KV Cache和HCA KV Cache，让两种压缩粒度在同一个原始 token block 边界上对齐
1. 把滑动窗口的kv / 尾部状态 这些统统打包成state cache，对于这些state cache，每次请求中都分配一块固定大小的缓存block
1. 除此之外的，仍然继续沿用传统的kv cache块
除了以上三点，DS还引入了on-disk KV Cache storage的概念，直接把固定的类似 系统提示词 / MCP tool 连接信息 / git 代码仓库 等等这类基本有着固定的前缀的KV Cache直接落磁盘，prefill的时候直接激活磁盘里这些KV cache，避免agent的过程中这类高度重复的前缀kv cache的重复计算
