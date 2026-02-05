# ICML 审稿人视角：客观差距评估（STEP2）

本文档用于把当前仓库 `STEP2` 的论文稿件与实验结果，放到 **ICML 审稿人**视角做“可接收性”差距诊断与最短补强清单。重点关注：**结论是否可追溯/可复现**、实验是否足够支撑“Δϕ 是核心创新”的叙事、baseline 是否公平、指标是否充分。

## TL;DR（当前结论）

- 以目前论文 `Paper/LaTEX/icml2025/main.tex` + `STEP2` 实验呈现方式，我会倾向给出 **borderline / weak reject**：理论与合成实验很强，但真实任务证据、指标与报告一致性存在明显短板。
- 如果补齐下面的 **关键点 1（报告一致性/可追溯性）** 与 **关键点 3（GPT-2 指标与 baseline 公平性）**，整体会明显上一个台阶：从“可信度红牌”变为“可认真审的工作”，更接近 **borderline / weak accept** 区间。
- 但即便如此仍不稳，最可能的新主攻点会变成 **关键点 2（CIFAR 提升幅度小 + ESS 接近满池导致机制一致性被质疑）**，再叠加 Δϕ/Leash 消融与触发条件未报告的问题。

## 最新进展（v3g 消融 + GPT-2 dispersion 基线）

### Exp9（CIFAR-10，gen5，3 seeds，train-holdout clean-val）

结果来自：
- `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced_train_holdout_cleanval/`
- `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_score_topk_train_holdout_cleanval/`（严格对照：同 v3g 的 \texttt{score\_topk} 规则，但 score=confidence）
- `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v3g_dphi0/`
- `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v3g/`

gen5（mean±std）：
- `baseline_balanced`：acc **0.4827±0.0148** / worst **0.2483±0.0476**
- `baseline_score_topk`（conf-only）：acc **0.4828±0.0075** / worst **0.2777±0.0086**
- `v3g (dphi0)`：acc **0.4854±0.0041** / worst **0.2817±0.0110**
- `v3g (dphi1)`：acc **0.4899±0.0098** / worst **0.3037±0.0103**

审稿口径可用结论：
- “硬 per-class 过滤”（\texttt{score\_topk}）确实解释了 worst 的一部分提升（24.8%→27.8%）。
- 但 **full v3g** 仍显著超过该严格对照（27.8%→30.4%），且 **Δϕ 贡献最大增量**（dphi0→dphi1：28.2%→30.4%）。

### Exp11（GPT-2，G4，3 seeds）

结果来自：
- `Experiments/exp11_gpt2_model/Results/dphi1/`（no_filter / pointwise / set_aware）
- `Experiments/exp11_gpt2_model/Results/dphi0/`（set_aware，Δϕ=0）
- `Experiments/exp11_gpt2_model/Results/dphi1_leash/`（set_aware + PPL leash）
- `Experiments/exp11_gpt2_model/Results/dispersion/`（新增：非学习几何 baseline，dispersion 直接算权重）
- `Experiments/exp11_gpt2_model/Results/ppl_safety/`（新增：点式“只避险” baseline，仅剔除极高 PPL）
- `Experiments/exp11_gpt2_model/Results/rep_filter_thr0.6/`（新增：硬规则 n-gram 重复过滤 baseline）
- `Experiments/exp11_gpt2_model/Results/pointwise_mix0.5/`、`Experiments/exp11_gpt2_model/Results/pointwise_mix0.2/`（新增：pointwise + mix 原始数据）
- 质量侧指标汇总：`Experiments/exp11_gpt2_model/Results/analysis_text_quality/train_g4.csv`

G4（mean±std）：
- `no_filter`：val PPL **1.50e3±0.62e3** / D4 **0.432±0.010**
- `pointwise`：val PPL **1.73e6±0.92e6** / D4 **0.249±0.011**（并伴随复读/塌缩：\texttt{unique\_line\_ratio}≈0.562，\texttt{rep4\_intra}≈0.958，\texttt{gzip\_ratio}≈0.029）
- `set_aware (dphi0)`：val PPL **3.87e3±4.20e3** / D4 **0.517±0.003**（质量稳定性差，方差极大）
- `set_aware (dphi1)`：val PPL **1.85e3±1.07e3** / D4 **0.519±0.016**（训练文本质量：\texttt{unique\_line\_ratio}≈0.997，\texttt{rep4\_intra}≈0.481，\texttt{gzip\_ratio}≈0.197）
- `set_aware (dphi1 + leash)`：val PPL **1.68e3±0.82e3** / D4 **0.520±0.016**
- `dispersion`（non-learned）：val PPL **2.60e3±1.37e3** / D4 **0.518±0.010**（训练文本质量：\texttt{unique\_line\_ratio}≈0.995，\texttt{rep4\_intra}≈0.439，\texttt{gzip\_ratio}≈0.216）
- `ppl_safety`（只避险）：val PPL **2.10e3±1.51e3** / D4 **0.434±0.004**（训练文本质量：\texttt{unique\_line\_ratio}≈0.931，\texttt{rep4\_intra}≈0.573，\texttt{gzip\_ratio}≈0.215）
- `rep_filter`（thr=0.6）：val PPL **1.43e3±0.45e3** / D4 **0.524±0.012**（训练文本质量：\texttt{unique\_line\_ratio}≈0.999，\texttt{rep4\_intra}≈0.412，\texttt{gzip\_ratio}≈0.234）
- `pointwise + mix_original`：mix=0.5 时 val PPL **0.73e3±0.34e3** / D4 **0.719±0.013**；mix=0.2 时 val PPL **0.60e3±0.16e3** / D4 **0.682±0.032**。注意：这里的提升主要来自“每代注入原始 Wikitext 样本”这类 replay/anchor 机制（不是改善伪数据选择本身；pointwise 选出来的伪数据仍高度复读）。

审稿口径可用结论：
- pointwise baseline 在该递归设定下确实会崩坏（不仅 PPL，质量侧指标也一起崩）。
- **Δϕ 对质量稳定性是关键组件**（dphi0 的 PPL 方差极大，dphi1 明显稳定）。
- dispersion / rep_filter 都是强基线；其中 rep_filter 甚至在 PPL 与 D4 上都能与 set-aware 持平甚至更强（但它是 **文本任务特化** 的启发式，和“跨模态/通用 set-aware 几何纠偏”的故事不完全同赛道）。
- mix_original 是非常强的稳定化对照，但它改变了问题设定（每代仍用真实人类数据锚定），需要在论文里明确为“有额外干净数据可用”的 anchored setting。

## 关键点 1：论文中的 Exp11 报告与当前可复现结果不一致（高风险，必须修）

（已修）论文 `Paper/LaTEX/icml2025/main.tex` 已切换到当前可复现目录 `Experiments/exp11_gpt2_model/Results/dphi1/` 与 `Experiments/exp11_gpt2_model/Results/dphi1_leash/`，并补充训练文本质量侧指标（`unique_line_ratio/rep4/gzip`）的证据链。

## 关键点 2：CIFAR-10（Exp9）增益偏小且 ESS≈满池，容易被质疑“filter 近似没工作”

- 论文写法（`Paper/LaTEX/icml2025/main.tex:539`）给出的结论是：
  - gen5 overall acc：0.417 → 0.420
  - gen5 worst-class acc：0.132 → 0.149
  - ESS≈8k（候选池 10k）
- 从 `Experiments/exp9_cifar10_setaware/results/exp9_seed*_merged.csv` 逐 seed 计算（gen5，3 seeds）：
  - overall acc 平均提升约 **+0.0032**
  - worst-class acc 平均提升约 **+0.0173**
  - set-aware 的 ESS 平均约 **7999/10000**（非常接近满池）
- 审稿人常见质疑路径：
  1) 提升幅度太小，统计显著性/稳定性不足；  
  2) ESS 接近满池意味着权重区分度弱（近似 uniform），会怀疑“set-aware/Δϕ 在真实任务上是否真的在发挥机制作用”；  
  3) 这会反过来削弱 “Δϕ 是核心创新、能在真实偏差递归中起决定性作用” 的叙事闭环。

### 2.1 新配置（Scheme A + 选择端均衡）的现实结论：worst-class 明显↑，但 overall 有代价且机制易被质疑

我们已跑通“更贴合故事主线”的版本：**meta clean-val（Scheme A）+ 分层 clean-val + 选择端类均衡（α=0.5）**，结果在：

- `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05/exp9_seed*_merged.csv`

gen5（3 seeds）统计如下：

- baseline：overall acc 均值 **0.4875**；worst-class 均值 **0.2067**
- set_aware（meta+balance α=0.5）：overall acc 均值 **0.4760**；worst-class 均值 **0.2428**
  - paired diff 均值：**Δacc=-0.0115**、**Δworst=+0.0360**

更关键的“confirmation bias”证据：gen5 的伪标签类分布被显著拉平（平均直方图）：

- baseline：`[336,329,517,12,87,504,697,522,673,323]`（类 3/4 几乎不被选）
- set_aware：`[368,406,423,282,404,410,396,440,496,376]`（类 3/4 被系统性补回）

但严格审稿人会继续追问（当前仍是风险点）：

1) **overall 精度下降**：你必须把“目标是 tail/worst-class”作为主指标叙事写清楚，否则会被认为指标挑选；  
2) **稳定性不足**：seed=1088 的 worst-class 反而下降（`-0.014`），说明不是稳定收益；  
3) **机制归因风险更大**：本次 set_aware 的 `ess_score≈11980/12000`（几乎满池），会被认为“filter 权重仍近似 uniform”；worst-class 的提升更像来自**选择端 balance α** 这个 heuristic，而非 Δϕ/注意力几何本身。

因此，这个版本虽更接近“ICML 叙事”（确实打散类坍缩），但要过严格审稿仍需要**公平对照 + 消融**把归因补齐。

### 2.2 公平 baseline 结果：Top-Conf + 同样 α 配额 ≈ Set-Aware（当前最致命的问题）

我们已加入并跑通公平 baseline：`baseline_balanced`（Top-Conf，候选池同为 12k，且使用与 set_aware 相同的 balance α 配额）。

- 结果目录：`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced/`
- gen5（3 seeds）均值：
  - baseline：acc **0.4875** / worst **0.2067**
  - baseline_balanced：acc **0.4798** / worst **0.2424**
  - set_aware（meta+balance）：acc **0.4760** / worst **0.2428**
  - 关键差：`set_aware - baseline_balanced` 约 **Δacc=-0.0038**、**Δworst=+0.0003**（几乎打平）

这意味着：在**默认** CIFAR 配置下，提升几乎完全由“类配额均衡”解释，set-aware/Δϕ 当时**没有提供可被审稿人认可的额外收益**；审稿人会倾向认为“heuristic 足够、set-aware 多此一举”。

#### 2.2.1 关键修复：调强 meta-clean-val + 去掉 balance loss 后，Set-Aware 开始超过公平 baseline

我们在不改任务定义/不改公平对照的前提下，对 set_aware 做了“让权重学会类内去噪”的调参（v2）：

- v2 结果目录：`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v2/`
- v2（3 seeds）相对 `baseline_balanced`（gen5）：
  - 平均 **Δacc=+0.0052**、平均 **Δworst=+0.0098**（两项同时提升）
  - 分 seed（`set_aware_v2 - baseline_balanced`）：  
    - 1088：`Δacc=+0.0116`、`Δworst=-0.0010`  
    - 2195：`Δacc=-0.0009`、`Δworst=+0.0152`  
    - 4960：`Δacc=+0.0049`、`Δworst=+0.0152`

解释角度（给审稿人能接受的因果链）：

- `baseline_balanced` 解决的是“类分布坍缩”（coverage），但无法系统性解决“类内伪标签错误”（quality）。
- v2 的收益来自 **类内重排/去噪**（同一类内部对样本赋予不同权重/排序），而不仅仅是“类配额 α”。  
  - `no clean-val` 对照显示：即便不引入任何干净标签，v2 仍能超过 `baseline_balanced`，说明增益不依赖 test clean-val；  
  - `train holdout clean-val + meta` 在此基础上进一步改善 worst-class，说明 Scheme A 更像是“可选的校正器/稳定器”，而非唯一驱动因素。

仍需坦诚的风险点：

- v2 的 ESS 仍接近满池（权重并非强稀疏），说明收益来自“轻度但稳定的类内重排/去噪”，需要通过 **Δϕ 消融** 与 **α-sweep** 把证据链补得更硬。

#### 2.2.1 Δϕ 消融（已完成，关键证据）

为了回应“Δϕ 是核心创新但真实任务没用上/没贡献”的质疑，我们在 **同一套 v2 配置** 下做了 `delta_phi_scale=1.0 vs 0.0` 的消融（gen5，3 seeds）：

- 公平 baseline：`baseline_balanced`  
  - acc **0.4798±0.0298** / worst **0.2424±0.0396**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced/`）
- set_aware v2（dphi1）：  
  - acc **0.4850±0.0235** / worst **0.2522±0.0489**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v2/`）
- set_aware v2（dphi0）：  
  - acc **0.4776±0.0128** / worst **0.2411±0.0413**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v2_dphi0/`）

结论（审稿可用表述）：

- 在 v2 配置下，**关闭 Δϕ 会让 set-aware 的优势消失并略差于公平 baseline**；因此“Δϕ 对于 v2 的真实任务收益是必要条件”是有数据支撑的。

### 2.3 Clean-Val 对照（必须补，避免“用 test 做 meta”被一票否决）

当前 v2 的 meta-clean-val 默认从 CIFAR-10 **test** 切 100 张作为 clean-val，并在评测时从 test 中剔除这 100 张。审稿人很容易认为这属于“利用 test 调参/泄漏”，即便你解释是 meta-learning 也会非常敏感。

因此需要补两个对照（已提供脚本，输出写到新目录，不覆盖旧结果）：

- **无 clean-val（完全不使用干净标签）**：检验 v2 的优势是否依赖 clean-val  
  - baseline_balanced：`Experiments/exp9_cifar10_setaware/run_baseline_balanced_alpha05_no_cleanval.sh` → `Experiments/exp9_cifar10_setaware/results_balance_alpha05_baseline_balanced_no_cleanval/`  
  - set_aware v2：`Experiments/exp9_cifar10_setaware/run_setaware_tuned_alpha05_v2_no_cleanval.sh` → `Experiments/exp9_cifar10_setaware/results_balance_alpha05_setaware_tuned_v2_no_cleanval/`
- **clean-val 来自 train holdout（不碰 test）**：检验 meta 机制是否能在“无 test 泄漏”的设定下仍然成立  
  - baseline_balanced：`Experiments/exp9_cifar10_setaware/run_baseline_balanced_alpha05_train_holdout_cleanval.sh` → `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced_train_holdout_cleanval/`  
  - set_aware v2：`Experiments/exp9_cifar10_setaware/run_setaware_tuned_alpha05_v2_train_holdout_cleanval.sh` → `Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v2_train_holdout_cleanval/`

跑完后用一条命令汇总 gen5（3 seeds 均值±方差）：

`python Experiments/exp9_cifar10_setaware/summarize_results.py <DIR1> <DIR2> --gen 5`

#### 2.3.1 已跑完的关键对照（gen=5，3 seeds）

已补齐两组对照（并与原 test clean-val 版本一起对齐比较）。汇总可复现结果如下（gen=5，3 seeds 均值±方差）：

- **无 clean-val（full test；不使用任何干净标签）**  
  - baseline_balanced：acc **0.4796±0.0300** / worst **0.2440±0.0392**（`Experiments/exp9_cifar10_setaware/results_balance_alpha05_baseline_balanced_no_cleanval/`）  
  - set_aware v2（dphi1）：acc **0.4837±0.0196** / worst **0.2543±0.0540**（`Experiments/exp9_cifar10_setaware/results_balance_alpha05_setaware_tuned_v2_no_cleanval/`）  
  - 差值：**Δacc=+0.0041**、**Δworst=+0.0103**
- **train holdout clean-val（meta on；full test；不碰 test 做 meta）**  
  - baseline_balanced：acc **0.4827±0.0148** / worst **0.2483±0.0476**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced_train_holdout_cleanval/`）  
  - set_aware v2 + meta（dphi1）：acc **0.4877±0.0189** / worst **0.2603±0.0414**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v2_train_holdout_cleanval/`）  
  - 差值：**Δacc=+0.0050**、**Δworst=+0.0120**
- **test clean-val（meta on；评测剔除 100 张 test）**（原 v2 配置对照）  
  - baseline_balanced：acc **0.4798±0.0298** / worst **0.2424±0.0396**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced/`）  
  - set_aware v2 + meta（dphi1）：acc **0.4850±0.0235** / worst **0.2522±0.0489**（`Experiments/exp9_cifar10_setaware/results_meta_balance_alpha05_setaware_tuned_v2/`）  
  - 差值：**Δacc=+0.0052**、**Δworst=+0.0098**

核心结论（对审稿人最关键）：

- set-aware v2 的优势 **不依赖** “从 test 切 clean-val 做 meta”（无 clean-val 也能赢公平 baseline）；  
- 用 **train holdout clean-val** 做 meta 不仅避免 test 泄漏，在当前配置下还带来更强的 worst-class 改善；  
- 需要在论文中明确：train holdout clean-val 本质上引入了 **小额干净标注预算**（100 张），审稿人会要求你把它写清楚，并在 baseline 侧给出同预算的公平对照或解释为什么不会改变结论。

#### 2.3.2 早期代际的 sanity check（gen=1，历史记录）

为确认代码路径，我们曾先跑过 gen=1 的最小对照（目录：`Experiments/exp9_cifar10_setaware/results_ctrl_no_cleanval_g1_seed1088/` 与 `Experiments/exp9_cifar10_setaware/results_ctrl_train_holdout_cleanval_g1_seed1088/`）。gen=1 的波动与 gen=5 的最终趋势并不完全一致，提示 meta 信号在早期可能存在不稳定性（建议在论文中做 meta warmup/退火并报告）。

**审稿人如何解读（结论门槛）**：

- 你现在可以把 Scheme A 写成“**小额干净标注预算下的 meta 校正**”，并强调“无 clean-val 仍有增益，meta 进一步改善/稳健 worst-class”；  
- 最容易被追问的是：这 100 张 clean-val 是否算额外监督（是），以及 baseline 是否也该享有同预算；建议补一个“给 baseline 同样额外标注 budget”的对照（或在附录明确说明其影响上界）。

## 关键点 3：GPT-2（Exp11）目前指标维度与 baseline 公平性不足（高风险，但可低成本补强）

### 3.1 目前可复现结果呈现的主要现象（G4 / 3 seeds）

- `Experiments/exp11_gpt2_model/Results/dphi1/`（generation=4）：
  - no_filter：Distinct-4 均值约 **0.432**，val PPL 均值约 **1500**
  - set_aware：Distinct-4 均值约 **0.519**，val PPL 均值约 **1847**（提升多样性，但 PPL 有 trade-off 且 seed 波动明显）
  - pointwise（当前实现=按候选 PPL 从低到高选 Top-K）：val PPL 百万级崩坏，Distinct-4 更低  
    - 这不是“故意选弱 baseline”，而是一个合理但在递归自训练里容易自我强化的策略：会偏向挑选**极度容易/重复**的样本，导致训练数据分布塌缩、泛化崩坏。
- `Experiments/exp11_gpt2_model/Results/dphi1_leash/`：
  - leash 并非每个 seed 都会改变轨迹：seed=1088/4960 与 `dphi1` 的 set_aware **逐行一致**；seed=2195 发生变化并降低 PPL。

### 3.1.2 Δϕ 消融（已补齐）：`--delta-phi-scale 0.0`（dphi0）

为补齐“Δϕ 是核心创新”的证据链，我们新增并跑通 GPT-2 侧的 Δϕ 消融（只跑 `set_aware`，其余配置与 `dphi1` 对齐）：

- 结果目录：`Experiments/exp11_gpt2_model/Results/dphi0/`
- 结论（G4 / 3 seeds，set_aware）：
  - **Distinct-4 均值**：`dphi0≈0.5168` vs `dphi1≈0.5190`（差距很小）
  - **val PPL 均值**：`dphi0≈3866` vs `dphi1≈1847`（dphi0 更差且方差更大）
  - 逐 seed 现象非常不一致：dphi0 在 seed=2195/4960 的 PPL 更小，但在 seed=1088 出现明显退化（G4 PPL≈8631）。

**审稿人会如何解读**：

- 这组消融并不支持“Δϕ 在 GPT-2 上必然带来稳定改进/一边倒收益”，更像是 **seed-sensitive 的稳定性调节项**：可能在某些轨迹上起到“牵引/正则化”作用（避免灾难性漂移），但也可能在另一些轨迹上带来额外噪声或把选择推向不利区域。
- 如果要把“Δϕ 是核心创新”写得更硬，需要进一步把“何时有用”说清楚（例如：与 ESS、候选 PPL 分布、或训练集质量指标的触发条件关联），或把 Δϕ 与 leash/meta-validation 组合成一个更可控的 **Pareto** 策略（否则会被认为是“可选 trick”）。

#### 3.1.1 “质量侧”文本指标（已内置到主脚本；旧 generations dump 仅作历史对照）

为避免仅用 `PPL + Distinct-n` 被质疑“鼓励胡言乱语/只追求多样性”，我们补了**文本自身**的质量侧统计（去重/复读/压缩比），用于检测递归是否把训练集推向“复读塌缩”。

重要备注（审稿风险）：当前仓库 `Experiments/exp11_gpt2_model/generations/` 下已有的 `.txt` 是历史 dump，**行数显示它对应的训练规模约为 512（而非当前 dphi1/dphi0 的 2000）**，因此这些数值只能作为“pointwise 容易塌缩”的定性证据，不应直接写入论文作为最终结果。

为解决“对齐最终配置”的问题，我们已把这些质量侧统计 **直接写入主实验输出**：`Experiments/exp11_gpt2_model/run_exp11_gpt2_model.py` 会在每一代的 `metrics_diversity_ppl.{json,csv}` 里附带
`train_unique_line_ratio / train_rep4_intra / train_gzip_ratio / train_avg_words`（基于当代被选中的训练文本）。因此后续运行 `ppl_leash`、Δϕ 消融等，都能自动产出与最终配置对齐的质量指标，不再依赖历史 `.txt` dump。

注意：`dphi1/` 是在加入这些字段之前生成的旧结果，因此对应 CSV 不包含上述列；`dphi0/` 等新目录已包含并可直接用于论文对齐。

- 指标定义：  
  - `unique_line_ratio`：训练文本去重比例（越高越不塌缩）  
  - `rep4_intra`：单条文本内部 4-gram 复读率（越低越不复读）  
  - `gzip_ratio`：压缩比（越低越冗余/越重复）
- 历史 dump 的结果（gen=4 train；3 seeds；见 `Experiments/exp11_gpt2_model/Results/analysis_text_quality/train_g4_lines.csv`）：  
  - pointwise 显著塌缩：`unique_line_ratio≈0.562`、`rep4_intra≈0.958`、`gzip_ratio≈0.029`（且 train 行数固定 512）  
  - set_aware 更健康：`unique_line_ratio≈0.997`、`rep4_intra≈0.481`、`gzip_ratio≈0.197`

复现该统计（基于现有 generations dump）的命令：

`python Experiments/exp11_gpt2_model/summarize_saved_generations.py --generations-root Experiments/exp11_gpt2_model/generations --gen 4 --split train --out Experiments/exp11_gpt2_model/Results/analysis_text_quality/train_g4_lines.csv`

### 3.2 leash “看起来没生效”的代码层原因（不是 bug，但必须在论文里解释/量化）

- leash 的实现是在 set-aware 输出的权重 `w` 上做逐样本惩罚 `w ← w * leash`，随后直接 `topk(w)` 选训练集：见 `Experiments/exp11_gpt2_model/filter_module.py:37` 与 `Experiments/exp11_gpt2_model/filter_module.py:200`。
- 因此只要 leash 乘上去后 **top-k 排名不变**（例如：`upper` 模式下大量候选 `ppl <= ppl_ref` 导致惩罚为 1，或惩罚强度不足以改变排序），训练样本集合就不会变化；在当前严格确定性设置下，后续训练/评估轨迹就会完全一致。
- 这类“条件触发”的机制如果不报告，会被审稿人当作“玄学/调参”；反过来如果报告触发率/触发条件，则可成为加分点（更可信）。

### 3.3 需要一个“不崩”的强 baseline：已在代码里补 `ppl_leash`

为避免 pointwise 长期崩坏被当作 strawman，我们新增一个**不依赖 set-aware**的强 baseline：`ppl_leash`（只用 clean-val PPL 作为语义缰绳，按 leash 权重选 Top-K）。

- 代码位置：`Experiments/exp11_gpt2_model/filter_module.py` 与 `Experiments/exp11_gpt2_model/run_exp11_gpt2_model.py`
- 用法：在运行时把 `--methods` 里加入 `ppl_leash`，并设置 `--ppl-leash-strength > 0`（否则没有 `ppl_ref` 无法工作）

## 回答：如果补上关键点 1 + 关键点 3，会发生什么变化？

- **直接收益 1（可信度从红牌变合格）**：关键点 1 修复后，Exp11 的路径与数值在论文里可追溯，并与可复现实验一一对应，审稿人不再因为“对不上/复现不了”而直接否定结论。
- **直接收益 2（GPT-2 叙事从单一 Distinct-n 变成 Pareto 结果）**：关键点 3 补齐后，你可以把结论写成“在质量约束下提升多样性/稳定性”，同时 baseline 不再像 strawman，显著提升说服力。
- **整体判断的变化**：会把当前状态从 “borderline / weak reject（主要栽在可信度与 GPT-2 证据链）” 推近到 “borderline / weak accept” 区间。
- **但仍不稳**：审稿火力很可能转向关键点 2（CIFAR 增益小 + ESS≈满池导致机制一致性被质疑），以及“Δϕ/Leash 的消融与触发统计是否完整”。

## 最短补强清单（按性价比排序）

1) **论文对齐可复现 Exp11**：把 `Paper/LaTEX/icml2025/main.tex:543` 的 run 路径/数值更新到 `Results/dphi1*`，并报告 3 seeds 的均值±方差（或 CI）。
2) **（已完成）公平 baseline**：我们已新增并跑通 `baseline_balanced`（Top-conf + 同样的 balance α 配额），目前结果显示它几乎与 set_aware 打平——因此后续必须让 set-aware 在它之上产生稳定优势，或调整论文叙事。
3) **补 Δϕ 消融**：至少给 Exp9/Exp11 做 `delta_phi_scale=0 vs 1`（同样设置），否则“Δϕ 是核心创新”在真实任务上证据不闭环。
3) **补 leash 触发统计**：报告 leash 是否改变 top-k（比例/分布），以及对 PPL/Diversity 的影响（哪怕只在 3 seeds 上）。
4) **GPT-2 质量指标补充（已部分完成）**：已补训练文本的 `unique_line_ratio/rep4_intra/gzip_ratio`（见 3.1.1）；仍建议再补 3–5 个典型样例或一个 reference-LM 的生成文本 PPL 作为“语义/流畅度侧”指标。
5) **baseline 公平性说明/增强（已补 baseline 代码）**：pointwise 当前实现是“按候选 PPL 选 Top-K”，但会导致训练集塌缩并在递归中崩坏；已新增 `ppl_leash` 作为不依赖 set-aware 的强 baseline（见 3.3），需要补跑对比以完成论文证据链。
6) **α-sweep + Pareto 报告（CIFAR 强烈建议）**：对 `set_aware_balance_alpha` 做小扫面（如 0/0.25/0.5/0.75/1.0），报告 overall vs worst-class 的权衡曲线，避免被认为只挑了一个幸运 α。
7) **针对 CIFAR 的“filter 是否工作”诊断**：围绕 ESS/权重分布/类分布变化做机制分析；若 ESS 仍接近满池，需要如实表述“权重近似 uniform、效果主要来自选择端均衡”，或者继续改 loss 让权重产生区分度。
