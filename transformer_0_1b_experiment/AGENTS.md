# transformer_0_1b_experiment

## 0. 强制约束（覆盖下文旧描述）

这一节的优先级**高于本文件后面所有旧表述**。如果后文仍出现 `R^2`、`Aff(2)`、`toy`、`玩具化舞台`、二维误差图等说法，**一律视为过时描述，不得再采用**。

本目录后续实验必须满足下面几条：

1. **严禁 toy**  
   不允许再把实验对象简化成 `R^2`、`Aff(2)`、低维代理群、手工二维状态或任何“只为了演示公式”的玩具空间。

2. **必须使用真实 Transformer 模型本体**  
   这里的 `0.1B transformer` 指的是一个**真实的高维 Transformer 模型**，而不是“参数量接近 0.1B 的学习器去拟合 toy 数据”。  
   可以随机初始化，不需要真实数据集，但模型本体必须是真的 Transformer。

3. **流形必须取模型参数空间**  
   本目录中的 `M` 默认指该真实 Transformer 的**参数空间**。  
   `p_t`、`e_t`、候选点、`Δμ_t`、`ΔC_t`、五类通道、Hall 顺序项、共轭修正与对称半校正，默认都应定义在**完整参数空间**里，或在与 `state_dict` 一一对应的高维参数坐标中实现。

4. **默认禁止前置降维**  
   默认取 `\mathcal P_t = id`。  
   不允许一开始就做 PCA、随机投影、低秩代理、二维/三维可视化空间替代、手工块矩阵小空间替代。  
   如需压缩，只能放在理论对象全部定义完成之后，且必须明确记为“最后的数值压缩”，不能改写理论舞台。

5. **如果脚本仍是 toy，则视为不符合本目录要求**  
   任何只在二维或低维合成空间里验证“公式外观”的脚本，都不能再被称为本目录的正式实验实现。

## 1. 这个目录是干什么的

这个目录专门研究一个**接近 `0.1B` 参数量级的 set-transformer 模型**，在统一几何纠偏框架下，对比三条方法曲线：

- `No Filter`
- `Set-Aware`
- `Our method`

这里的重点不是复刻某个旧脚本，而是把**根目录 `AGENTS.md`** 与
[`Engineering-file-reference/model_collapse_diffM_topological_group_unified.pdf`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/Engineering-file-reference/model_collapse_diffM_topological_group_unified.pdf)
中的数学对象，压缩成一个**独立可运行、单脚本、自包含**的实验。

如果你在一个新对话里打开这个目录，这个文件应该足够让你理解：

1. 我们到底在实验什么；
2. `Our method` 的数学定义是什么；
3. 代码中哪些对象对应文稿里的哪些公式；
4. 如何判断实验成功或失败；
5. 如果还不够，要继续去看哪份文稿。

## 2. 权威来源与优先级

这个目录里的方法定义，按下面的顺序解释：

1. 根目录 [`AGENTS.md`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/AGENTS.md)
2. [`Engineering-file-reference/model_collapse_diffM_topological_group_unified.pdf`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/Engineering-file-reference/model_collapse_diffM_topological_group_unified.pdf)
3. [`test1.md`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/test1.md)

这里尤其以第 1、2 项为主，因为它们比 `test1.md` 更强调：

- 先在 `Diff(M)` / `Aut(M, S)` 的**向量场与流**层面定义对象；
- 顺序学习不枚举所有排列，而进入 **Hall 商空间**；
- 校正必须先做 **方向共轭修正**；
- 单步更新必须使用 **对称半校正**；
- 采样、训练、降维都只能是**最后的数值化步骤**。

## 3. 本实验的数学舞台

为了能在一份单文件脚本里数值化，我们用一个最小但严格的局部玩具化舞台：

- 取 `M = R^2`
- 取参考点 `p_t = 0`
- 取局部坐标图 `chi_t = id`
- 把局部允许变换集实现成 `Aff(2)` 的齐次 `3x3` 矩阵

也就是说，一个局部向量场的一阶仿射表达写成

\[
X=(A,b),
\qquad
\widehat X=
\begin{pmatrix}
A & b\\
0 & 0
\end{pmatrix},
\]

其时间一流通过矩阵指数实现：

\[
\operatorname{Exp}_{\mathcal G,t}(X)=\exp(\widehat X).
\]

这一点对应根目录 `AGENTS.md` / PDF 中“**先在向量场与局部流层定义，再最后投到有限维表示**”的原则。

## 4. 健康流、误差与候选集

### 4.1 健康流

健康基准流取为一个稳定收缩：

\[
S=\operatorname{diag}(s_1,s_2), \qquad 0 < s_2 < s_1 < 1.
\]

在代码里它对应 `healthy_diag`。

对应的健康半步生成元写成

\[
\Omega_B =
\begin{pmatrix}
\log S & 0\\
0 & 0
\end{pmatrix}.
\]

### 4.2 局部误差

当前误差写成

\[
e_t \in \mathbb R^2.
\]

实验里所有主图都在监控

\[
\|e_{t+1}\|_2.
\]

### 4.3 候选集与统计量

对每一代，我们先构造一个未筛选候选集

\[
u_{t,1},\dots,u_{t,N_t}\in T_{p_t}M \cong \mathbb R^2,
\]

再定义

\[
\mu_t^{\mathrm{ref}}=\frac1{N_t}\sum_i u_{t,i},
\qquad
C_t^{\mathrm{ref}}=\frac1{N_t}\sum_i (u_{t,i}-\mu_t^{\mathrm{ref}})(u_{t,i}-\mu_t^{\mathrm{ref}})^\top.
\]

给定筛选权重 `w_{t,i}` 后，定义

\[
\mu_t^{\mathrm{sel}}=\sum_i w_{t,i}u_{t,i},
\qquad
C_t^{\mathrm{sel}}=\sum_i w_{t,i}(u_{t,i}-\mu_t^{\mathrm{sel}})(u_{t,i}-\mu_t^{\mathrm{sel}})^\top.
\]

再记

\[
\Delta\mu_t=\mu_t^{\mathrm{sel}}-\mu_t^{\mathrm{ref}},
\qquad
\Delta C_t=C_t^{\mathrm{sel}}-C_t^{\mathrm{ref}}.
\]

这对应 PDF 第 3 节和 `test1.md` 第 3 节。

## 5. 五类有害通道

这是 `Our method` 的核心。我们不是直接让模型猜一个黑箱 correction，而是先显式构造五类通道。

### 5.1 先验偏置通道

若任务或规则本身带显式偏置，记为

\[
X_t^{(\mathrm{prior})}(x)=A_t^{(\mathrm{prior})}x+b_t^{(\mathrm{prior})}.
\]

代码中用 `explicit_a0_diag`、`explicit_b0` 表示。

### 5.2 均值漂移通道

\[
X_t^{(\mu)}(p_t)=\Delta\mu_t.
\]

在 `Aff(2)` 中写成纯平移生成元：

\[
\widehat X_t^{(\mu)}=
\begin{pmatrix}
0 & \Delta\mu_t\\
0 & 0
\end{pmatrix}.
\]

### 5.3 支撑形变通道

定义协方差对齐算子

\[
M_t=(C_t^{\mathrm{sel}}+\lambda I)(C_t^{\mathrm{ref}}+\lambda I)^{-1},
\]

并取局部对数

\[
A_t^{(\Sigma)}=\frac12\log M_t.
\]

对应仿射生成元

\[
\widehat X_t^{(\Sigma)}=
\begin{pmatrix}
A_t^{(\Sigma)} & 0\\
0 & 0
\end{pmatrix}.
\]

### 5.4 曲率—方差漂移通道

健康流在局部坐标中的二阶几何偏置通过 Hessian 与协方差差收缩得到：

\[
\zeta_t=\frac12 D^2F_t(\mu_t^{\mathrm{ref}}):\Delta C_t.
\]

对应纯平移通道

\[
\widehat X_t^{(\zeta)}=
\begin{pmatrix}
0 & \zeta_t\\
0 & 0
\end{pmatrix}.
\]

### 5.5 和乐旋转通道

闭环平行移动或 toy 旋转给出

\[
H_t^{(\mathrm{hol})}\in O(T_{p_t}M),
\qquad
\Omega_t^{(\mathrm{hol})}=\log H_t^{(\mathrm{hol})}.
\]

对应线性旋转向量场

\[
\widehat X_t^{(\mathrm{hol})}=
\begin{pmatrix}
\Omega_t^{(\mathrm{hol})} & 0\\
0 & 0
\end{pmatrix}.
\]

## 6. 顺序学习：只学 Hall 商，不学全排列

这一点是整个方法最容易被写错的地方。

我们不学习所有排列，而只在二阶时学习一个反对称成对次序矩阵的最小摘要。

在当前实验里，我们只保留三组关键二阶括号：

- `[X^(mu), X^(zeta)]`
- `[X^(mu), X^(hol)]`
- `[X^(zeta), X^(hol)]`

于是把顺序头压缩成一个三维向量

\[
w_t = (w_{mz,t}, w_{mh,t}, w_{zh,t}).
\]

对应的二阶有效生成元是

\[
D_t^{(2)}=X_t^{(\mathrm{prior})}+ X_t^{(\mu)}+ X_t^{(\Sigma)}+ X_t^{(\zeta)}+ X_t^{(\mathrm{hol})}+ \frac12 w_{mz,t}[X_t^{(\mu)},X_t^{(\zeta)}]+ \\
\frac12 w_{mh,t}[X_t^{(\mu)},X_t^{(\mathrm{hol})}]+ \frac12 w_{zh,t}[X_t^{(\zeta)},X_t^{(\mathrm{hol})}].
\]

这是根目录 `AGENTS.md` 和 PDF 第 4 节要求的“**默认 step-2**”版本。

## 7. 共轭方向修正与对称半校正

### 7.1 共轭方向修正

若纠正器输出的反向生成元记为 `\hat D_t`，则必须先用方向头 `H_t` 运输到当前框架：

\[
\widetilde D_t=\operatorname{Ad}_{\operatorname{Exp}_{\mathcal G,t}(H_t)}\widehat D_t.
\]

这一步不能省，因为省了会留下一级方向残差。

### 7.2 对称半校正主公式

单步更新不是“直接减 correction”，而是

\[
\Psi_t=B_t^{1/2}
\circ
\operatorname{Exp}_{\mathcal G,t}\!\left(-\frac{\widetilde D_t}{2}\right)
\circ
\operatorname{Exp}_{\mathcal G,t}(D_t^{(K)})
\circ
\operatorname{Exp}_{\mathcal G,t}\!\left(-\frac{\widetilde D_t}{2}\right)
\circ
B_t^{1/2}.
\]

在我们的 `Aff(2)` 实现里，直接写成矩阵乘法：

\[
\Psi_t=\exp(\Omega_B/2)\,
\exp(-\widetilde D_t/2)\,
\exp(D_t^{(2)})\,
\exp(-\widetilde D_t/2)\,
\exp(\Omega_B/2).
\]

然后作用在当前误差上：

\[
e_{t+1}=\rho_t(\Psi_t)(e_t).
\]

这里 `\rho_t` 在 toy 里就是普通仿射作用：

\[
\rho_t(\Psi_t)(e_t)=A_t e_t + u_t,
\qquad
\Psi_t=
\begin{pmatrix}
A_t & u_t\\
0 & 1
\end{pmatrix}.
\]

## 8. 三条方法曲线在本实验中的精确定义

### 8.1 No Filter

- 不训练模型
- 不做加权
- 不做几何校正
- 直接用原始候选均值更新

即：

\[
e_{t+1}^{\mathrm{no}}=\mu_t^{\mathrm{raw}}.
\]

### 8.2 Set-Aware

- 只让 `0.1B` set-transformer 学一个 set-wise correction target
- 不显式输入五个通道摘要
- 不显式走 Hall 顺序压缩
- 不显式走共轭方向修正与对称半校正

它是一个强的 learned baseline，但不是完整几何方法。

### 8.3 Our method

- 先在脚本里显式构造五类通道
- 显式构造 `D_t^(2)`
- 显式构造共轭修正与对称半校正主公式
- Transformer 只在最后一步学习**由完整几何更新导出的 correction target**

也就是说：

1. 理论对象先构造好；
2. Transformer 只负责最后数值逼近；
3. 它不能篡改前面的理论链路。

这正是根目录 `AGENTS.md` 里“**采样、计算、训练与可选降维全部后移**”的原则。

## 9. 当前脚本的输入、目标和输出

### 9.1 输入

每个样本对应一个候选集张量，外加几何摘要。

- `Set-Aware` 输入：
  - 候选点
  - 有效 mask

- `Our method` 输入：
  - 候选点
  - `X^(mu), X^(Sigma), X^(zeta), X^(Hol), X^(0)` 的摘要
  - 参考统计量摘要
  - 前后状态摘要
  - mask

### 9.2 学习目标

这份脚本当前采用的最稳口径是：

- `Set-Aware` 学：
  \[
  \mu_t^{\mathrm{weighted}} - e_{t+1}^{\star}
  \]

- `Our method` 也学：
  \[
  \mu_t^{\mathrm{weighted}} - e_{t+1}^{\star}
  \]

但不同点在于：

- `Set-Aware` 没有显式几何通道输入
- `Our method` 的输入携带完整通道摘要与顺序信息

也就是说，**差别主要放在输入与理论结构，而不是只放在损失名字**。

## 10. 验证口径

主图纵轴统一是：

\[
\|e_{t+1}\|_2
\]

越低越好。

本目录最重要的成功标准是：

1. `device=mps`
2. 参数量接近 `0.1B`
3. 三条曲线都能跑出来：
   - `No Filter`
   - `Set-Aware`
   - `Our method`
4. 尾部 `Our method` 至少优于 `Set-Aware`

## 11. 当前需要看的文件

本目录内最重要的文件：

- 主脚本：
  [`run_transformer_0_1b_experiment.py`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/transformer_0_1b_experiment/run_transformer_0_1b_experiment.py)
- 结果图：
  [`results/transformer_0_1b_comparison.png`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/transformer_0_1b_experiment/results/transformer_0_1b_comparison.png)
- 结果表：
  [`results/transformer_0_1b_metrics.csv`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/transformer_0_1b_experiment/results/transformer_0_1b_metrics.csv)

## 12. 如果这个文件的信息还不够，看哪里

如果你还需要更完整的理论背景，按下面顺序继续看：

1. 根目录：
   [`AGENTS.md`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/AGENTS.md)
   作用：
   - 给出“微分同胚群 / 自同构群舞台 + 向量场通道 + Hall 顺序学习 + 共轭方向修正 + 对称半校正”的总原则

2. 新 PDF：
   [`Engineering-file-reference/model_collapse_diffM_topological_group_unified.pdf`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/Engineering-file-reference/model_collapse_diffM_topological_group_unified.pdf)
   作用：
   - 给出“以向量场为主语”的统一版推导
   - 解释为什么要先在 `Diff(M)` / `Aut(M,S)` 上工作，再最后数值化

3. 补充文稿：
   [`test1.md`](/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/test1.md)
   作用：
   - 提供更直接的 `Aff(r)` / block-matrix 版本公式
   - 对 toy 数值实验落地更方便

如果这三份都还不够，那下一步应该补写一份**本目录专用的 derivation note**，把这个 `0.1B` 实验的所有符号和代码变量逐行对齐。
