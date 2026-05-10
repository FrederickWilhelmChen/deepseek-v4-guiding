## 🧱 Architecture：DeepSeek V4的算法设计
DeepSeek V4为了能较低成本的支持1M context且能力可用，同时做了三件事：怎么读上下文、怎么让信息穿过深层网络、以及怎么更新那些决定模型能力的大矩阵参数。
### 1. CSA / HCA：交替使用的注意力压缩机制
<text underline="true">*前置知识：Transfomer原理*</text>
#### 问题背景
传统 transformer attention 默认把历史 token 当成一块统一的记忆池，理论上都可以被完整访问。这个做法在超长上下文中的问题是：
- 每一步都完整看所有历史，成本太高。
- 但如果粗暴压缩，又容易把真正重要的远程信息一起压没。
- 超长上下文中，很难说得清楚现在问题的答案，会不会在很久之后才又突然用得到
总的来说，长上下文带来的问题，不仅是历史太长，而且还存在历史里只有一小部分现在又用得上，但你事先又不知道是哪一部分。
#### 直观类比
想象你在读一个很长的项目文档库。
- 最近几页你会直接盯着看，因为马上要用。
- 更早的历史，你不会整本重读，但会先看目录和索引，再决定打开哪几章。
- 再远一点，你甚至不会看原文，只保留一个“这个项目大概干过什么”的低分辨率印象。
DeepSeek V4 做的，就是把这种人类阅读方式做进模型内部。
#### 结构图
<whiteboard token="R6eiwKiXbhEcVobuV2DcZcydnOb" align="left"/>

#### 轻技术解释
`Sliding Window` 负责最近上下文。这部分最像工作记忆，必须尽量保真，这部分DeepSeek不做任何压缩。
`CSA` 类似 `Abstract Retrieval`。把历史压成中等粒度的块，再从这些块里选出最相关的部分重新拉回来参与注意力计算。所以它的关键词不是压缩，而是 `selective reading`。从工程直觉上说，这一步和RAG工程有思路上的类似点——压缩+检索+rerank
`HCA` 是全局暴力大比例压缩。它不定位具体某一句，但能让模型始终保留“远处还有什么”这层感觉，类似于人的记忆中长程记忆极为模糊，但它始终存在，尽管你不记得事情的细节，但你会记得有这么个事
三者合起来，才是 DeepSeek V4 对长上下文最关键的回答：不是所有历史都值得同等对待，而应该按信息价值和用途分层处理。

### 2. mHC：混合残差连接的稳定器
<text underline="true">*前置知识：残差连接*</text>
<text underline="true">*本部分理解难度较大，可只看：mHC的简化通俗解释*</text>
#### mHC的简化通俗解释
<image token="M4Efb0Ii1o9JdYxhHJJcijmAnef" url="https://api3-eeft-drive.larkenterprise.com/space/api/box/stream/download/authcode/?code=Yjk4YTFmZDhjYmQzNzkwNDY2MGVhNmRiOWJmZWQ1YTdfODE0NDgxM2NjZjZkZjk5NGExZDZhNThkOTVhYjg2MzRfSUQ6NzYzNzU1NTQ5NDU5NjUxMjk1OV8xNzc4NDM2NTM0OjE3Nzg1MjI5MzRfVjM" width="1672" height="941" align="center"/>

原始残差连接：单行道，所有车只能从一条道上过
HC：没有红绿灯/没有交警/没有车道标志的双向八车道，通行能力大大增强，但所有车能随便超车/换道/转弯，容易车祸/塞车
mHC：给双向八车道加上红绿灯/交警/车道标志，并且加上不同时段的限行，保证车流量受控，不会出车祸

#### 问题背景
残差的意义：**不要直接学习完整目标，而是学习在原输入基础上的偏移量**
传统残差连接的数学直觉是：给了梯度一条更短的回传路径。即使某些层学得不好，模型也可以近似让这个偏移量归0，**最坏情况下，这一层可以什么都不做，让信息原样通过**
```json
y = x + F(x)
这一层输出 = 原输入 + 这一层学到的修正量（即残差）
新的 token 表示 = 原来的 token 表示 + Attention/MLP 计算出来的增量
```

如以上解释，如果这一层layer训练训崩了，训练过程中通过反向传播+梯度下降+归一化等种种手段把这个F(x)压低，原输入x能近似无损的进入下一层
普通 Transformer 的 residual connection 非常稳定，因为它给每一层都保留了一条可以被认为是退路的东西，让深层神经网络训练成为可能。但它的代价是信息路径相对单一，同时可能引入了梯度消失的问题。
Hyper-Connections 在2024-25年被提出，意图是想把这条单一路径变成多条残差路径做混合，让不同类型的信息有更多通行方式；但这直接破坏了残差连接在最坏情况下退化成恒等映射的这种稳定性。2026年的论文指出：不做任何限制的Hyper-Connections会导致训练不稳定

#### mHC对HC的改良
mHC 想保留 HC 的好处：多残差通路、多路径流动、更强表达力。但为了保证训练的稳定性，mHC选择对混合矩阵加约束，尽量保住传统残差连接那种至少可以退回接近恒等传递的稳定性。为此，DeepSeek V4 报告中提到了使用mHC——使用双随机矩阵来限制 HC 中原本自由的连接矩阵。
双随机矩阵具有三个数学性质：所有元素非负，每一行元素累加为 1，每一列元素累加为 1。mHC 的做法可以理解为：训练时仍然学习一个自由矩阵，但真正用于多条残差通路混合之前，会先通过归一化 / 近似投影，把它变成一个近似双随机矩阵。这样做大幅压缩了连接矩阵的自由度，使它更像是在多条残差通路之间做受控重分配，而不是任意放大、抵消或扭曲信息流，从而提升深层训练的稳定性。
至于自由矩阵如何变成双随机矩阵，这又为什么在数学上被证明有用，整个完整推导极为复杂。下面只用一个尽可能通俗的方式解释这一点。


### 3. Muon：大规模训练优化器
<text underline="true">*前置知识：基础Transformer / 神经网络 / 梯度下降 *</text>
<text underline="true">*本部分理解难度较大，可只看：Muon的简化通俗解释*</text>

#### Muon的简化通俗解释
<image token="XWj2ba6Q3oDziSxTBlPcXpAVnNb" url="https://api3-eeft-drive.larkenterprise.com/space/api/box/stream/download/authcode/?code=NDFhNDY3MGY4NWE3ZWIyZjFjZmM2ODlkNGE2MzU5N2NfM2VlMzBlNGJiMGJiYTBkZjc1OWZlMzgwZDZhZGVlMjNfSUQ6NzYzNzU1NTQ5MzU5NDEwNjgyOV8xNzc4NDM2NTM0OjE3Nzg1MjI5MzRfVjM" width="1672" height="941" align="center"/>

主要作用：
1. 避免一次矩阵更新过度集中在少数主方向
1. 让更新步长更可控
1. 相信方向结构，不完全相信幅度比例

#### 问题背景
传统神经网络 / Transformer 在训练过程中对梯度下降使用 AdamW 进行了优化，在这里不展开讲数学过程。
AdamW的优势是：
1. 在 Transformer 训练中通常表现稳定，比 SGD 等优化器对学习率、梯度尺度差异更鲁棒
1. 超参相对不敏感
AdamW的问题：
优化和微调主要是逐元素的，但事实上大规模权重矩阵和梯度矩阵是有整体几何结构的，在整体上有行列空间结构、奇异值分布、条件数和主方向

#### Muon的优化策略
用通俗的说法来解释：
- 对一张excel表里所有单元格的数据，AdamW 是逐个格子微调的
- Muon 先看整张表格的整体倾斜方向，再决定如何整体修正它。
两者都在更新参数，但看问题的尺度不一样。

#### 技术解释
纯技术术语版解释：
- 对二维矩阵参数的 momentum update 做 Newton-Schulz 正交化，让更新方向更 matrix-aware。
一个toy case来解释：
```plaintext
AdamW（这里是极简化版本，忽略了AdamW中的二阶动量和weight decay，也忽略了warm up/decay这些工程处理）：
权重矩阵
Wt = [[1,0,0], [0,1,0], [0,0,1]]

假设反向传播后得到梯度矩阵，经过历史积累获得的惯性更新矩阵为：
Mt = [[3,0,0], [0,1,0], [0,0,0.5]]

假设学习率为0.1，更新后的权重矩阵：
Wt+1 = [[0.7,0,0], [0,0.9,0], [0,0,0.95]]

Muon（同样是极简化演示版本）
权重矩阵
Wt = [[1,0,0], [0,1,0], [0,0,1]]

假设反向传播后得到梯度矩阵，经过历史积累获得的惯性更新矩阵为：
Mt = [[3,0,0], [0,1,0], [0,0,0.5]]

对Mt进行有限轮次的Newton-Schulz近似正交化后的结果是：
Mt' = [[1,0,0],[0,0.852,0],[0,0,0.522]]

假设学习率为0.1，更新后的权重矩阵：
Wt+1 = [[0.9,0,0], [0,0.9148,0], [0,0,0.9478]]
```

对经过适当归一化、且满足收敛条件的非奇异矩阵，Newton-Schulz 迭代可以近似其极分解中的正交因子
<text underline="true">*从过程上看：*</text>
也就是AdamW进行的momentum得到的结果Mt（经过历史积累获得的惯性更新矩阵）是进行Muon的前提，Muon通过Newton-Schulz 迭代把他拉成一个保留矩阵的左右奇异向量所定义的主方向结构，同时把奇异值差异压平的结果，并用这个结果去更新权重向量
<text underline="true">*从效果上看：*</text>
1. 避免一次矩阵更新过度集中在少数主方向
1. 让更新步长更可控
1. 相信方向结构，不完全相信幅度比例
