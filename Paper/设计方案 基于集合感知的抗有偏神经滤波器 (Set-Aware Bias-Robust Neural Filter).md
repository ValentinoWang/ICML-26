# 1. 设计动机与核心理念
在有偏估计（Biased Estimation）场景下，偏差往往表现为数据分布的**整体漂移（Global Shift）**，而非单个样本的异常。传统的 MLP Filter 单独处理每个样本 $x_i$，无法感知“整体均值偏离了多少”。

本方案采用 **Set-based Architecture**，将一轮迭代中的所有候选样本视为一个集合（Set）。网络不仅学习局部特征（Local Features），还通过特征交互学习全局上下文（Global Context），最终输出两部分结果：
1.  **样本权重 (Sample Weights, $w_i$)**：用于降方差和去离群点。
2.  **全局偏差校正 (Global Bias Correction, $\Delta \phi$)**：用于抵消估计器的系统性偏差。

# 2. 整体架构 (Mathematical Framework)

### 2.1 输入层：特征嵌入 (Feature Embedding)
假设第 $t$ 轮生成的候选样本集为 $D_t = \{x_1, x_2, \dots, x_N\}$，其中 $x_i \in \mathbb{R}^D$。

1.  **降维 (PCA)**：沿用 Han 的策略，先将高维数据映射到低维流形，减少计算量。
    $$ z_i = \text{PCA}(x_i) \in \mathbb{R}^d, \quad i=1,\dots,N $$
2.  **初始编码 (Point-wise Encoder)**：将每个样本映射到隐空间。
    $$ h_i^{(0)} = \text{MLP}_{enc}(z_i) \in \mathbb{R}^k $$

### 2.2 核心层：集合交互模块 (Set Interaction Module)
此处引入 **Transformer Encoder Layer** (或 DeepSets 逻辑)，实现样本间的信息交互，感知全局分布。

令 $H^{(l)} = [h_1^{(l)}, \dots, h_N^{(l)}]^\top \in \mathbb{R}^{N \times k}$ 为第 $l$ 层的特征矩阵。

**采用 Self-Attention 机制：**
$$
\begin{aligned}
Q &= H^{(l)}W_Q, \quad K = H^{(l)}W_K, \quad V = H^{(l)}W_V \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{k}}\right)V \\
H^{(l+1)} &= \text{LayerNorm}\left(H^{(l)} + \text{Attention}(Q, K, V)\right)
\end{aligned}
$$
*   *原理说明*：通过 Attention，每个样本 $h_i$ 都能“看到”其他所有样本。如果大部分样本都偏向一侧，网络能通过聚合信息感知到这个偏差，从而调整每个点的特征表示。

### 2.3 输出层：双头机制 (Dual-Head Output)
经过 $L$ 层交互后，得到最终特征 $H^{(L)} = \{h_1^{(L)}, \dots, h_N^{(L)}\}$。输出分为两个分支：

**分支一：局部加权头 (Reweighting Head)**
针对每个样本输出保留概率。
$$ w_i = \sigma(\text{MLP}_{weight}(h_i^{(L)})) \in [0, 1] $$
*   $\sigma$ 为 Sigmoid 函数。

**分支二：全局校正头 (Global Correction Head)**
聚合集合信息，预测一个反向的偏差校正向量。
$$
\begin{aligned}
h_{global} &= \frac{1}{N} \sum_{i=1}^N h_i^{(L)} \quad (\text{Global Average Pooling}) \\
\Delta \phi &= \text{MLP}_{bias}(h_{global}) \in \mathbb{R}^D
\end{aligned}
$$
*   *注意*：这里输出的 $\Delta \phi$ 维度与原始参数 $\theta$ 维度一致。

# 3. 参数更新逻辑 (Parameter Update Dynamics)

在 Filter 推理阶段，我们将通过加权估计和显式校正相结合的方式更新参数。

定义加权统计量（Weighted Statistics）：
$$ \hat{\theta}_{weighted} = \frac{\sum_{i=1}^N w_i \cdot x_i}{\sum_{i=1}^N w_i} $$

最终的参数更新公式为：
$$ \theta_{new} = \hat{\theta}_{weighted} + \Delta \phi $$

*   **物理意义**：
    *   $\hat{\theta}_{weighted}$ 负责利用“好数据”降低方差，并抵抗分布边缘的噪声。
    *   $\Delta \phi$ 负责“硬校正”加权平均后依然无法消除的系统性偏差（Systematic Bias）。

# 4. 损失函数设计 (Optimization Objectives)

为了训练上述 Set-based Filter，我们需要联合优化以下四个 Loss。

$$ \mathcal{L}_{total} = \mathcal{L}_{class} + \lambda_1 \mathcal{L}_{contract} + \lambda_2 \mathcal{L}_{ESS} + \lambda_3 \mathcal{L}_{reg} $$

### 4.1 监督分类损失 ($\mathcal{L}_{class}$)
利用人工构造的标签（与真值距离最近的前 $K\%$ 为 1，其余为 0）监督 $w_i$。
$$ \mathcal{L}_{class} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log w_i + (1-y_i) \log (1-w_i) \right] $$

### 4.2 抗偏收缩损失 ($\mathcal{L}_{contract}$)
这是核心。强制要求“修正后的新参数”必须比“单纯的有偏估计”更接近真值 $\theta_{good}$。
由于是有偏估计，我们不再强求误差小于旧误差，而是直接最小化新参数与真值的距离：
$$ \mathcal{L}_{contract} = \| \theta_{new} - \theta_{good} \|_2^2 = \| (\hat{\theta}_{weighted} + \Delta \phi) - \theta_{good} \|_2^2 $$
*   *梯度回传*：该 Loss 会同时优化 $w_i$（选出重心更准的数据子集）和 $\Delta \phi$（学习反向偏差向量）。

### 4.3 有效样本量正则 ($\mathcal{L}_{ESS}$)
防止 Attention 机制过度关注某几个点（Over-attention），导致样本坍缩。
$$ \mathcal{L}_{ESS} = \max\left(0, \tau - \frac{(\sum w_i)^2}{\sum w_i^2}\right) $$
*   $\tau$ 是目标有效样本数（如 100）。

### 4.4 校正项正则 ($\mathcal{L}_{reg}$)
防止 $\Delta \phi$ 过大导致数值不稳定，鼓励网络优先通过 Reweighting 解决问题。
$$ \mathcal{L}_{reg} = \| \Delta \phi \|_2^2 $$

# 5. 理论与实验的对应关系

该设计直接响应了我们之前的理论推导（有界收敛 UUB）：

1.  **Set-based 交互**：通过 Pooling 操作 $\frac{1}{N}\sum h_i$，网络实质上在估计当前分布的**经验偏差 (Empirical Bias)**。
2.  **$\Delta \phi$ 的作用**：理论推导指出最终误差边界 $R^* \propto \beta$（偏差大小）。通过引入 $\Delta \phi \approx -\mathbf{b}_{sys}$，我们在数学上将有效偏差 $\beta$ 降至最低，从而极大地压缩了最终的误差收敛半径。
3.  **End-to-End 可微**：整个过程（从输入 $Z$ 到 $\theta_{new}$）都是可导的，这意味着我们可以利用少量的 $\theta_{good}$ 数据，反向传播教会 Filter 如何在一个有偏的环境中“生存”。

这个方案将 **Transformer 的上下文感知能力** 与 **控制理论的收缩约束** 完美结合，是解决有偏模型崩溃问题的强力方案。