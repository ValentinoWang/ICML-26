# ICML 双语写作说明（理论 + 实验 Exp1–Exp8）

目标：使用官方 ICML 模板撰写英文主稿（篇幅受限），中文版本可作为附录或单独文档（不受篇幅约束）。核心依据：
- 理论：`Paper/有偏估计下的一致最终有界性 (Revised Theoretical Derivation).md`
- 实验汇总：`Paper/exp_summary_exp1_to_exp8.md`

## 实验目录（相对于 `autodl-tmp/ICML/STEP2`，均为非门控版本）
- Exp1 偏差来源：`exp1_bias_sources/`（硬偏、Ridge、先验错位）。
- Exp2 偏差敏感性：`exp2_bias_sensitivity/`（幅度/正则/先验偏移/n 曲线）。
- Exp3 数据效率：`exp3_data_efficiency/`（小样本 vs 大样本）。
- Exp4 机制可视化：`exp4_bias_correction_visualization/`（Δϕ 对齐、ESS、权重）。
- Exp5 高维扩展：`exp5_high_dim_scalability/`（d=50/100/500 尾误差）。
- Exp6 架构消融：`exp6_arch_ablation/`（加权/校正/集合交互拆解）。
- Exp7A 方差/Ripple 注意力：`exp7_variance_attention/`。
- Exp7B 递归回归（真实数据）：`exp7_recursive_regression/`。
- Exp8 MNIST 递归漂移：`exp8_mnist_recursive/`（旋转漂移校正）。
- 统一脚本：`run_all_experiments.py`；更多细节与尾部指标见 `Paper/exp_summary_exp1_to_exp8.md`。

## 结构规划（英文主稿，ICML 风格）
1. Abstract（100–150 词，突出有偏递归训练、校正头、集合交互）
2. Introduction（1 页内）：问题背景（biased estimation / recursive drift）、贡献列表（理论 UUB、有偏校正、实证）。
3. Related Work（简短，递归生成、importance weighting、set transformers、bias correction）
4. Theory（0.5–1 页）：从理论文档提炼：
   - 问题设定（有偏估计器、递归更新）
   - 一致最终有界性（关键定理/引理，必要假设）
   - 校正项 Δϕ 与收敛半径关系（保留主要公式，细节放 Appendix）
5. Method（0.5 页）：Set-Aware Bias-Robust Filter 结构（weights + correction），损失项（class/contract/ESS/reg），可提及 Pointwise/MLP 对照。
6. Experiments（2–3 页，精简图）：
   - Exp1/2（偏差来源/敏感性）：各挑 1 图（含 CI），引用路径。
   - Exp3（数据效率）：硬偏差/Ridge 小 vs 大样本关键图（exp3_fig_A/B）。
   - Exp4（机制）：1 图（Δϕ/Attention 对齐）。
   - Exp5/6（高维与消融）：1 图展示尾部误差 vs 维度/变体柱状。
   - Exp7（Ripple 注意力可视化）：3 合 1 图 `exp7_variance_attention/results/exp7_combined.png`。
   - Exp7 递归回归（真实数据）：曲线 `exp7_recursive_regression/results/exp7_curves.png`。
   - Exp8（MNIST 漂移）：曲线/可视化二选一（如 `exp8_curves.png` 或 `exp8_visual_grid.png`）。
   - 表格（可选）：关键尾部指标汇总（从 CSV 提取）。
7. Discussion（0.5 页）：集合交互在高维的局限，Ripple 任务中 Attention 优势，校正头主导。
8. Conclusion & Broader Impact（简短）
9. References
10. Appendix（中文全文、额外公式、全量图表/更多实验）

## 图表与数据（英文主稿精选，需复制到 Figures 目录，保留原名 + 记录原路径）
建议精选（可根据篇幅微调）：
- Exp1/2：`exp1_bias_sources/results/exp1_1.1_const.png` 或 `exp2_bias_sensitivity/results/...` 中 1 张主图。
- Exp3：`exp3_data_efficiency/results/exp3_fig_A_breaking_floor.png`，`exp3_fig_B_data_efficiency.png`
- Exp4：`exp4_bias_correction_visualization/results/exp4_4.1_base.png` 或 4.2 动态图
- Exp5：`exp5_high_dim_scalability/results/exp5_tail_vs_dim.png` 或 `exp5_bias_reduction_rate.png`
- Exp6：`exp6_arch_ablation/results/exp6_trajs.png` 或 `exp6_tail_bar.png`
- Exp7（Ripple）：`exp7_variance_attention/results/exp7_combined.png`
- Exp7 递归回归：`exp7_recursive_regression/results/exp7_curves.png`
- Exp8：`exp8_mnist_recursive/results/exp8_curves.png` 或 `exp8_visual_grid.png`
如需表格，可从各 `.../results/*.csv` 提取尾部指标（简表）。

## 中文部分
- 可在 Appendix 提供全文中文解释，包含更多图表（不受篇幅限制）。
- 结构可平行于英文，但可加入更多细节/补充公式。

## 模板与文件组织
- 模板：使用 ICML 官方模板（icmlXXXX.sty 等）放于 `Paper/Notes/LATEX/ICML/`。
- 主文件：`main.tex`（英文），`main_cn.tex`（中文/附录），共用 `Figures/` 下的图片。
- 图片复制：将上述选定图复制到 `Paper/Notes/LATEX/Figures/`，保留原文件名；在文中引用时注明原始路径（可在注释或表格 footnote 说明）。

## 待确认/注意
- 篇幅控制：英文正文 8–10 页（不含参考/附录）为宜，图表总数约 6–8 张。
- 中文部分可更详尽，附录放置额外证明/图表。
- 是否需要表格汇总关键数字（尾部误差、MSE）？若需要，将从现有 CSV 抽取。
