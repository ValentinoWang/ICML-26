# IJCAI 双语写作说明（理论 + 实验 Exp1–Exp8）

目标：使用 IJCAI 官方模板撰写英文主稿（篇幅受限），中文版本置于附录或单独文档。依据：
- 理论：`Paper/有偏估计下的一致最终有界性 (Revised Theoretical Derivation).md`
- 实验：`Paper/exp_summary_exp1_to_exp8.md`（含 Exp7 递归回归、Exp8 MNIST 漂移）

## 结构规划（英文主稿，IJCAI 风格）
1. Abstract（简洁）
2. Introduction（问题背景、有偏递归学习、贡献）
3. Related Work（简短）
4. Theory（核心定理/假设，细节放附录）
5. Method（Set-Aware + Δϕ，损失）
6. Experiments（精简图，篇幅 7–9 页左右）：
   - Exp1/2：各 1 主图（偏差来源/敏感性）
   - Exp3：数据效率图 A/B
   - Exp4：机制（Δϕ/Attention）
   - Exp5/6：高维/消融主图
   - Exp7（Ripple 注意力）：`exp7_variance_attention/results/exp7_combined.png`
   - Exp7 递归回归：`exp7_recursive_regression/results/exp7_curves.png`
   - Exp8：`exp8_mnist_recursive/results/exp8_curves.png` 或视觉网格
   - 可选简表：尾部指标汇总
7. Discussion（集合交互优势/局限，非单调偏差案例，真实/视觉递归）
8. Conclusion
9. References
10. Appendix（中文、完整公式、更多图表）

## 图表与数据（精选，复制到 Figure 目录，保留原名并注明原路径）
与 ICML 类似，可按篇幅取 6–8 张：
- Exp1/2 主图；Exp3 图 A/B；Exp4 机制图；Exp5/6 主图；Exp7 ripple 组合图；Exp7 回归曲线；Exp8 曲线或可视化。
表格：如需，提取 CSV 尾部数据。

## 表格数据（已拷贝至 `Paper/LaTEX/FormattingGuidelines-IJCAI-ECAI-26/Table/`）
- 主文建议引用：Exp1 尾部指标（硬偏/Ridge/贝叶斯）`table_exp1_tail.csv`；Exp3 小样本 vs 大样本尾部 `table_exp3_data_eff_tail.csv`；Exp5 高维尾误差 `table_exp5_tail_summary.csv`；Exp6 消融尾误差 `table_exp6_bayes_tail_summary.csv`、`table_exp6_ridge_tail_summary.csv`、`table_exp6_complex_tail_summary.csv`；Exp7 递归回归尾部 MSE/范数/耗时 `table_exp7_recursive_tail.csv`；Exp8 漂移尾部 MSE/范数/耗时 `table_exp8_mnist_tail.csv`。
- 附录可选：Exp2 偏差/正则/先验全 sweep `table_exp2_bias_summary.csv`、`table_exp2_ridge_summary.csv`、`table_exp2_prior_summary_n5.csv`、`table_exp2_prior_summary_n50.csv`。

## 中文部分
- 置于附录或单独中文稿，结构平行英文，可更详尽。

## 模板与文件组织
- 模板：使用 IJCAI 官方模板（如 `ijcai*.sty`），置于 `Paper/Notes/LATEX/IJCAI/`。
- 主文件：`main.tex`（英文），`main_cn.tex`（中文/附录）。
- 图片统一放 `Paper/Notes/LATEX/Figure/`，保留原名；在文中可用注释/表格脚注注明原路径。

## STEP2 实验目录（无门控版本）
- Exp1 偏差来源：`autodl-tmp/ICML/STEP2/exp1_bias_sources`
- Exp2 偏差敏感性：`autodl-tmp/ICML/STEP2/exp2_bias_sensitivity`
- Exp3 数据效率：`autodl-tmp/ICML/STEP2/exp3_data_efficiency`
- Exp4 偏差校正可视化：`autodl-tmp/ICML/STEP2/exp4_bias_correction_visualization`
- Exp5 高维可扩展性：`autodl-tmp/ICML/STEP2/exp5_high_dim_scalability`
- Exp6 架构消融：`autodl-tmp/ICML/STEP2/exp6_arch_ablation`
- Exp7 递归回归：`autodl-tmp/ICML/STEP2/exp7_recursive_regression`
- Exp7 Ripple 注意力：`autodl-tmp/ICML/STEP2/exp7_variance_attention`
- Exp8 MNIST 递归：`autodl-tmp/ICML/STEP2/exp8_mnist_recursive`

## 待确认/注意
- 篇幅：英文正文控制在会议要求（7–9 页不含参考/附录）。
- 图数：6–8 张为宜；中文附录可收全量图表。
- 若需表格汇总数字，请指定要包含的指标。***
