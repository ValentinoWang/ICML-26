## 我们使用的 Filter 类型与配置

### 核心主推：Set-Aware 校正头（Transformer Backbone，权重仅作辅助）
- 架构：输入集合（候选样本）经 MLP encoder → Transformer encoder（多头注意力）→ 双头输出 `(weights, Δϕ)`。
- 具体结构（以 `filter/set_aware/model.py` 为准）：
  - 编码器：`MLP(dim → hidden → hidden)` + ReLU + Dropout。
  - Transformer：`num_layers` 个 encoder layer，`d_model=hidden, nhead=n_heads, dim_ff=2*hidden, dropout=dropout, activation=gelu`。
  - Weight head：`MLP(hidden → hidden → 1)`，sigmoid 得到 `w_{B,N}`。
  - Correction head：`MLP(hidden → hidden → dim)`，先对 Transformer 输出做均值池化得到全局 `h_global`，再输出 Δϕ（形状 `[B, D]`，推理时扩展为 `[B,1,D]`）。
- 估计策略：**最终估计只用 Δϕ**（权重用于辅助损失/ESS，不参与 θ 估计），即 `theta_new = delta`；在高维/混合场景可用 `theta_new = theta_weighted + delta`（Exp5 d=500 混合更新）。
- 损失：`lambda_class * BCE(weights, labels) + lambda_contract * contraction_loss(theta_new, theta_ref) + lambda_ess * ess_loss(weights) + lambda_reg * ||Δϕ||^2`。典型设置：`lambda_class=0.05`, `lambda_ess=0.01~0.1`, `lambda_contract=1.0`, `lambda_reg=1e-5`（或高维 1e-4~1e-3），`top_ratio=0.2~0.3`。
- 使用范围：Exp1–Exp3、Exp6、Exp7、Exp8（高维/图像场景可加 PCA=50）；Exp5 中低维。高维 500 场景可结合 PCA+混合更新。

### 基线 1：Point-wise MLP 重加权（Standard Filter）
- 架构：独立编码每个候选样本，输出单标量权重 `w_i = σ(MLP(z_i))`，无集合交互、无 Δϕ。
- 具体结构（`filter/standard/model.py`）：`MLP(dim → hidden → hidden)` 编码后接 `MLP(hidden → hidden → 1)`，sigmoid 得到 `w_{B,N}`。
- 估计：`theta_new = weighted_mean(x, w)`，无校正项。
- 损失：`BCE + lambda_contract * contraction(theta_new, theta_ref) + lambda_ess * ess(weights)`。
- 使用范围：所有实验中的“Standard/MLP filter”基线。

### 基线 2：MLP + Correction（无集合交互的全局校正头）
- 架构：Mean-pooling MLP 输出 Δϕ（无 attention）。
- 具体结构：`MLP(dim → hidden → hidden → dim)`，先对集合做均值池化（`pooled = x.mean(dim=1)`），再输出 Δϕ。
- 估计：`theta_new = delta`，不依赖权重；通常无分类/ESS。
- 使用范围：Exp5/Exp6 高维或消融对比。

### 高维/图像场景的附加策略
- PCA 降维：在 Exp5/Exp8 等高维/图像场景，用 PCA=50 前端降维，Set-Aware/MLP 在低维运行，Δϕ 通过逆变换回原空间。
- 混合更新：在极高维（如 Exp5 d=500），可用 `theta_new = theta_weighted + delta`，以加权均值打底、Δϕ 学残差。
- 正则/剪裁：高维时可增大 `lambda_reg`、限制 `correction_clip`，并适当降低收缩系数（ours_contraction）。

### 可视化与快照
- Exp8 提供多类多时间对齐网格：0–9，每类 4 行（GT/No/MLP/Ours），列为 GT, t=1/20/40/60/80/100/150/200；图像路径：`exp8_mnist_recursive/results/exp8_visual_grid_digits.png`。
- 其他曲线/对比见各 `results` 目录（curves/trajectories）。 
