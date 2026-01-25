# 实验三 TODO 清单：MT 工业检测场景（MT + YOLO/DAN）

> 本清单针对 `Experiment_Plan_Controlled_Bias_ZH.md` 中的「实验三：工业检测场景（MT + YOLO/DAN）」部分，列出目前尚未完成或需要补充的工作项，方便后续按优先级逐步完成。

---

## 一、方法维度补全（表格行要齐）

目标：在 MT 场景下得到一张完整的对比表，包含：

- Source-only；
- Target-only；
- Pretrain-Finetune-only；
- YOLO baseline（无偏置控制模块）；
- DAN（YOLO + MMD，无偏置控制模块）；
- YOLO + $L_{\text{bias}}$（已完成）；
- YOLO + $L_{\text{bias}} + g_\phi$（完整偏置控制）。

### TODO 1：Source-only 行（MT 上的 YOLO 源域模型表现）

- [x] 从 `Pretrain-Finetune/Results/shared_pretrain/seed_*/...` 中选择或复用 $\theta_{\text{good}}$，在 `MT-tgt_global_test` 上评估 YOLO 源域模型：
  - 指标：mAP50 / mAP50-95 / Precision / Recall；
  - 多种子（1088/2195/4960）下的均值 ± 方差；
  - 输出为一行 “Source-only” baseline，写入实验三表格。

### TODO 2：YOLO baseline 行（无 $L_{\text{bias}}$ / 无 $g_\phi$）

- [ ] 使用与 YOLO + $L_{\text{bias}}$ 完全相同的训练脚本与初始化，但将 $\lambda_{\text{bias}}$ 设为 0（关闭锚点正则），在四个场景下跑 YOLO：
  - few-shot / small / medium / high × 三个种子；
  - 指标同上；
  - 用于公平对比“同一 YOLO 架构下，加不加偏置控制”的差异。

### TODO 3：DAN 行（YOLO + MMD，无偏置控制）

- [ ] 在 `DAN/` 框架下，使用与 MT 实验相同的 few-shot/small/medium/high 场景：
  - 训练 DAN YOLO（YOLO + MMD），不加 $L_{\text{bias}}$ / $g_\phi$；
  - 在 `MT-tgt_global_test` 上评估 mAP 等指标；
  - 按场景和 seed 整理成与 YOLO 结果同一张表中的 “DAN” 行。

### TODO 4：YOLO + $L_{\text{bias}} + g_\phi$（完整偏置控制）

- [x] 在现有 YOLO + $L_{\text{bias}}$ 架构上，设计并实现样本过滤器 $g_\phi$：
  - 输入：per-sample loss / 预测置信度 / feature 等统计；  
  - 输出：样本权重 $w_i \in [0, 1]$；
  - 接入方式：在 detection loss 前乘以 $w_i$（或在 batch 内进行加权汇总）。
  - 实现位置：`Toy/src/yolo_bias_finetune/mlp_filter.py` 中的 `MLPFilter` 与 `PerSampleWeightedDetectionLoss`（即 MLP FiLTER 实现的 g_\phi 模块）：
    - `PerSampleWeightedDetectionLoss` 在 YOLO v8 的 `v8DetectionLoss` 基础上，按样本拆分出检测损失分子 $(N_{\text{box}, i}, N_{\text{cls}, i}, N_{\text{dfl}, i})$；
    - 使用全局分母 $T = \sum_i N_{\text{score}, i}$ 构造 per-sample 特征向量 $f_i = [N_{\text{box}, i}/T, N_{\text{cls}, i}/T, N_{\text{dfl}, i}/T]$，输入到 `MLPFilter`（MLP FiLTER）中，得到样本级权重 $w_i \in [0,1]$；
    - 最终 detection loss 采用加权形式 $L_{\text{det}} = \sum_i w_i \cdot L_{\text{det}}(x_i)$（在实现中通过对分子按样本加权、分母保持与原始 v8DetectionLoss 一致来完成），从而在不改变整体尺度的情况下实现真正的 **per-sample re-weighting**。
- [ ] 在四个 MT 场景上跑 YOLO + $L_{\text{bias}} + g_\phi$：
  - few-shot / small / medium / high × 三个种子；
  - 指标与上述一致；
  - 补齐“完整偏置控制”一行。
  - 运行方式（Baseline 根目录）：
    - 单场景运行示例（few-shot 场景）：
      ```bash
      python Toy/src/yolo_bias_finetune/train_bias_yolo.py \
        --scenario few-shot \
        --lambda-bias 1e-4 \
        --theta-good-seed 1088 \
        --use-filter
      ```
    - 一键跑 four-scenario × three-seed（few-shot/small/medium/high）：
      ```bash
      python Toy/run_yolo_bias_only_experiments.py \
        --lambda-bias 1e-4 \
        --theta-good-seed 1088 \
        --use-filter
      ```
  - 结果目录：`Toy/Results/Bias_only/lambda_0.0001/<scenario>/seed_*/results/`，与 YOLO + $L_{\text{bias}}$ 版本保持一致，可直接用于补齐“完整偏置控制”一行。

---

## 二、曲线与稳定性可视化

目标：不仅给最终 mAP，还要展示收敛行为和稳定性。

### TODO 5：mAP vs epoch 曲线（工业场景版）

- [x] 从 YOLO baseline / YOLO + $L_{\text{bias}}$ / YOLO + $L_{\text{bias}} + g_\phi$ 的 `results.csv` 中抽取若干代表性的种子（例如 each scenario 选一个 seed），绘制：
  - mAP50 vs epoch；
  - few-shot/small 场景尤为重要。
- [x] 对比观察：
  - 当前仓库中尚未完成 YOLO + $L_{\text{bias}} + g_\phi$ 版本的训练，因此本阶段曲线主要对比 YOLO baseline vs YOLO + $L_{\text{bias}}$（基于 seed=2195 的 four-scenario 结果，绘图脚本见 `Baseline/Toy/Analysis/plot_mt_yolo_map50_vs_epoch.py`，运行后会在 `Baseline/Toy/Analysis/figs/` 下生成图像）。
  - few-shot 场景：两种方法在前若干 epoch 都能快速爬升到 mAP50≈0.3 左右，但在持续训练后均出现明显的 late-stage mAP 崩溃（最终 mAP50 回落到 0.1 左右），YOLO + $L_{\text{bias}}$ 的峰值略高但同样存在过拟合/退化现象，提示在极端 few-shot 下还需要更强的过滤器（如 $g_\phi$）或更激进的 early-stopping 策略。
  - small 场景：YOLO baseline 在约 20 个 epoch 内一直徘徊在 mAP50≈0.2–0.5 且较为震荡，而 YOLO + $L_{\text{bias}}$ 在 30–40 个 epoch 内迅速爬升并稳定在 mAP50≈0.85–0.9 区间，且后期几乎无明显震荡，呈现出“收敛更快、更平滑且无 late-stage 崩溃”的典型曲线形态。
  - medium 场景：baseline 需要训练到约 70 个 epoch 后才首次超过 mAP50≈0.8，最终在长时间训练后收敛到≈0.85；相比之下，YOLO + $L_{\text{bias}}$ 在前 10 个 epoch 就能越过 0.8，并在 40 epoch 左右达到≈0.94 的峰值，后期保持在≈0.90 以上，说明在中等规模目标域下，锚点正则可以显著加速收敛并提升最终性能。
  - high 场景：baseline 约在 20 多个 epoch 后才稳定越过 mAP50≈0.8 并缓慢爬升至≈0.90；YOLO + $L_{\text{bias}}$ 则在前 3 个 epoch 内就能达到并超过 0.8，随后快速稳定在≈0.94–0.96 的高位区间，曲线整体较为平滑，没有明显的 late-stage 崩溃，表现为“快速进入高性能区间并在高位保持稳定”。

### TODO 6：bias_loss vs epoch（如有线上记录）

- [ ] 如训练过程中记录了 $L_{\text{bias}}$ 或与 $\theta_{\text{good}}$ 的距离：
  - 绘制 bias_loss 随 epoch 的变化；
  - 对比 few-shot/small/medium/high，不同场景下偏置距离的演化趋势。

---

## 三、工业场景上的敏感性分析（$\theta_{\text{good}}$ / $\lambda_{\text{bias}}$）

目标：在 MT 框架下，补充对 $\theta_{\text{good}}$ 与 $\lambda_{\text{bias}}$ 的敏感性实验，补齐实验五在工业场景中的一部分。

### TODO 7：不同质量的 $\theta_{\text{good}}$ 在 MT small/few-shot 上的影响

- [ ] 构造多个版本的锚点：
  1. 源域 full-data 训练的 $\theta_{\text{good}}$（当前使用版本）；  
  2. 源域 + 噪声（例如 20% label noise）训练出的 $\theta_{\text{good}}$；  
  3. 源域少量样本（例如使用 50% 源数据）训练出的 $\theta_{\text{good}}$；  
  4. 极端坏锚点：来自与目标域分布明显不匹配的源（例如采用不同 domain 的源模型）。
- [ ] 在 MT few-shot / small 场景上，用上述不同 $\theta_{\text{good}}$ 重跑 YOLO + $L_{\text{bias}}$（可选是否加 $g_\phi$），观察：
  - 性能增益是否随锚点质量逐步下降；
  - 在极端坏锚点场景下，方法是“优雅退化”还是灾难性崩溃。

### TODO 8：在 MT 上做简单的 $\lambda_{\text{bias}}$ 扫描

- [ ] 在 few-shot / small 场景中，对 YOLO + $L_{\text{bias}}$ 做几组不同的 $\lambda_{\text{bias}}$：
  - 例如 $\{0, 1\text{e-5}, 1\text{e-4}, 5\text{e-4}\}$；
  - 比较 mAP50 / 稳定性（std）/ bias_loss 等指标随 $\lambda_{\text{bias}}$ 的变化。
- [ ] 在实验报告中给出一个“实用区间建议”，例如：
  - “在 MT 场景下，当 $\lambda_{\text{bias}}$ 位于 [a, b] 时，方法稳定且优于Fine-tune；过大时可能出现过度锚定，性能下降。”

---

## 四、工业场景上的消融实验（实验六的 MT 部分）

目标：在 MT 工业检测任务上，完整呈现消融结果，验证“锚点 + 过滤器”的协同效果（或诚实地界定其边界）。

### TODO 9：MT 场景四行消融表

在 MT few-shot / small / medium / high 四个场景上，给出如下四行并比较：

- [ ] Baseline（YOLO baseline，无 $L_{\text{bias}}$、无 $g_\phi$）；
- [ ] YOLO + $L_{\text{bias}}$ only（当前已有结果）；
- [ ] YOLO + $g_\phi$ only（实现后需跑实验）；
- [ ] YOLO + $L_{\text{bias}} + g_\phi$（完整方法）。

**需要重点观察：**

- 在 few-shot / small（有偏小样本）场景，完整方法是否显著优于单独版本；  
- 是否存在某些场景由 $L_{\text{bias}}$ 主导、某些场景由 $g_\phi$ 主导，从而在论文中给出合理的贡献定位。

---

上述 TODO 完成后，实验三（工业检测场景）就可以从「YOLO + 锚点偏置」的 case study，升级为一个对齐整体可控偏置框架的完整工业验证块，与实验一（理论）、实验二（公开基准）形成真正闭环。 
