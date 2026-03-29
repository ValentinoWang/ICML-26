# Rebuttal Experiments

这里补的是专门用于 rebuttal 的新增实验，不依赖已有结果。

## 对应审稿人质疑

- `Direct Clean Data Usage`
  - CIFAR-10:
    - `clean_only`: 只用 `N=100` 干净 holdout 从头训练 ResNet-18，不用任何递归合成数据。
    - `clean_ft`: 只用 `N=100` 干净 holdout 反复微调基座 ResNet-18。
    - `clean_mix`: 每代把 `100` 个干净样本直接混进 `4k` 伪标签训练集，不用 SAGE 过滤。
  - GPT-2:
    - `clean_mix`: 每代把 `500` 条干净 reference 直接混进模型生成样本里做 NTP 微调。
    - `clean_only` / `clean_ft`: 每代只用 `N` 条干净 reference 对上一代模型做 NTP 微调，不混入任何生成样本。
- `Simple Batch-Mean Drift Subtraction`
  - CIFAR-10:
    - `batch_mean`: 用候选集算术均值近似“漂移方向”，按 `conf - eta * cos(x_i, mean(x))` 排序选样。
  - GPT-2:
    - `batch_mean`: 用候选 embedding 的均值近似“漂移方向”，按 pointwise 分数减去均值对齐惩罚选样。
- `CIFAR-10 Fraction Sweep`
  - CIFAR-10:
    - `fraction_sweep`: 固定每代总训练样本数为 `4000`，只改变合成伪标签比例 `alpha`。
    - `alpha = 50%`: `2000` pseudo + `2000` clean
    - `alpha = 75%`: `3000` pseudo + `1000` clean
    - `alpha = 90%`: `3600` pseudo + `400` clean
    - `alpha = 97.5%`: `3900` pseudo + `100` clean
    - `alpha = 100%`: `4000` pseudo + `0` clean
  - synthetic 部分统一用 `no-filter / top-confidence` 选样，不用 SAGE，也不用 batch-mean。

## 运行入口

### CIFAR-10

```bash
bash Experiments/Rebuttal/run_cifar_rebuttal.sh
```

或者手动指定：

```bash
python Experiments/Rebuttal/run_cifar_rebuttal.py \
  --device cuda \
  --seeds 1088 2195 4960 \
  --modes set_aware clean_only clean_ft clean_mix batch_mean \
  --clean-set-size 100
```

输出：

- `Experiments/Rebuttal/results/cifar10/rebuttal_cifar_seed{seed}_merged.csv`

核心指标：

- `acc`
- `worst_class_acc`
- `hist_c*` / `sel_pseudo_acc*`

### CIFAR-10 Fraction Sweep

```bash
bash Experiments/Rebuttal/run_cifar_fraction_sweep.sh
```

或者手动指定：

```bash
python Experiments/Rebuttal/run_cifar_rebuttal.py \
  --device cuda \
  --results-dir Experiments/Rebuttal/results/cifar10_fraction_sweep \
  --seeds 1088 2195 4960 \
  --modes fraction_sweep \
  --fraction-sweep-alphas 0.5 0.75 0.9 0.975 1.0 \
  --generations 5 \
  --per-gen-add 4000
```

输出：

- `Experiments/Rebuttal/results/cifar10_fraction_sweep/rebuttal_cifar_seed{seed}_merged.csv`
- `Experiments/Rebuttal/results/summary_fraction_sweep/cifar_fraction_sweep_g5.csv`
- `Experiments/Rebuttal/results/summary_fraction_sweep/cifar_fraction_sweep_g5.md`

核心指标：

- 只看 `Gen=5` 的 `worst_class_acc`
- 总样本数固定 `4000`

### GPT-2

```bash
bash Experiments/Rebuttal/run_gpt2_rebuttal.sh
```

或者手动指定：

```bash
python Experiments/Rebuttal/run_gpt2_rebuttal.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --methods set_aware,clean_mix,batch_mean \
  --clean-ref-size 500
```

如果你要严格贴 reviewer 提问里的 “`500 clean + 2k generated` 直接混训” 单独基线，建议单跑：

```bash
python Experiments/Rebuttal/run_gpt2_rebuttal.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --methods clean_mix \
  --candidate-pool 2000 \
  --train-samples 2000 \
  --clean-ref-size 500 \
  --clean-mix-generated-source first_k
```

如果你要补“每代只用 clean refs 微调，不混任何生成样本”的 strict baseline，建议单跑：

```bash
python Experiments/Rebuttal/run_gpt2_rebuttal.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --methods clean_only \
  --clean-ref-size 500 \
  --clean-support-split validation \
  --eval-split test
```

如果你想和 CIFAR 命名对齐，也可以直接用：

```bash
python Experiments/Rebuttal/run_gpt2_rebuttal.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --methods clean_ft \
  --clean-ref-size 500 \
  --clean-support-split validation \
  --eval-split test
```

如果你想一口气把 `SAGE / clean_mix / clean_only / pointwise / no_filter` 都补齐，建议：

```bash
python Experiments/Rebuttal/run_gpt2_rebuttal.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --methods set_aware,clean_mix,clean_only,pointwise,no_filter \
  --clean-ref-size 500 \
  --clean-support-split validation \
  --eval-split test
```

输出：

- `Experiments/Rebuttal/results/gpt2/{seed}/metrics_diversity_ppl.json`
- `Experiments/Rebuttal/results/gpt2/{seed}/metrics_diversity_ppl.csv`

核心指标：

- `distinct4`
- `val_ppl`
- `train_unique_line_ratio`
- `train_rep4_intra`

快速冒烟验证：

```bash
python Experiments/Rebuttal/run_gpt2_rebuttal.py \
  --device cpu \
  --seeds 1088 \
  --methods set_aware \
  --smoke-mode
```

### GPT-2 Clean-Val Size Ablation

这个实验专门回应“LLM 场景下 clean validation set 大小是否足够小”的质疑。

- 只跑 `set_aware`
- 固定 `candidate_pool=10000`、`train_samples=2000`、`G0->G4`
- 默认 sweep `N_val in {50, 100, 500, 1000}`
- `clean-val` 与 `eval validation` 严格互斥
- `validation` 默认先切成 `block_size=128` 的非重叠 chunks，再抽取 clean-val / eval 子集
- `clean-val` 只用于构造几何漂移方向；不会混入 next-generation 训练文本

运行：

```bash
bash Experiments/Rebuttal/run_gpt2_cleanval_ablation.sh
```

或者手动指定：

```bash
python Experiments/Rebuttal/run_gpt2_cleanval_ablation.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --clean-val-sizes 50,100,500,1000
```

如果要同时补 MAUVE：

```bash
python Experiments/Rebuttal/run_gpt2_cleanval_ablation.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --clean-val-sizes 50,100,500,1000 \
  --run-mauve
```

输出：

- `Experiments/Rebuttal/results/gpt2_cleanval_size/n{N}/{seed}/metrics_diversity_ppl.csv`
- `Experiments/Rebuttal/results/gpt2_cleanval_size/n{N}/mauve/mauve_g0_g4.csv`（若加 `--run-mauve`）
- `Experiments/Rebuttal/results/summary_cleanval_size/gpt2_cleanval_size_final.csv`
- `Experiments/Rebuttal/results/summary_cleanval_size/gpt2_cleanval_size_table.md`

汇总：

```bash
bash Experiments/Rebuttal/summarize_gpt2_cleanval_ablation.sh
```

核心指标：

- `distinct4`
- `val_ppl`
- `mauve`
- `train_rep4_intra`

快速冒烟验证：

```bash
python Experiments/Rebuttal/run_gpt2_cleanval_ablation.py \
  --device cpu \
  --seeds 1088 \
  --clean-val-sizes 100 \
  --smoke-mode
```

### GPT-2 Unsupervised Surrogate Meta-Objective

这个实验专门回应“是否必须依赖 disjoint clean validation set”的质疑。

- 新方法：`unsup_set_aware`
- filter 训练只依赖候选集内部几何散度，不使用 clean-val
- 递归阶段不接触 `validation` split
- 评估改到 `test` split，因此表里的 `val_ppl` 应解释为 held-out PPL
- 默认一起跑 `unsup_set_aware / no_filter / pointwise`

运行：

```bash
bash Experiments/Rebuttal/run_gpt2_unsup_surrogate.sh
```

或者手动指定：

```bash
python Experiments/Rebuttal/run_gpt2_unsup_surrogate.py \
  --device cuda \
  --seeds 1088,2195,4960
```

如果要同时补 MAUVE：

```bash
python Experiments/Rebuttal/run_gpt2_unsup_surrogate.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --run-mauve
```

输出：

- `Experiments/Rebuttal/results/gpt2_unsup_surrogate/{seed}/metrics_diversity_ppl.csv`
- `Experiments/Rebuttal/results/gpt2_unsup_surrogate/mauve/mauve_g0_g4.csv`（若加 `--run-mauve`）
- `Experiments/Rebuttal/results/summary_unsup_surrogate/gpt2_unsup_surrogate_final.csv`
- `Experiments/Rebuttal/results/summary_unsup_surrogate/gpt2_unsup_surrogate_table.md`

汇总：

```bash
bash Experiments/Rebuttal/summarize_gpt2_unsup_surrogate.sh
```

核心指标：

- `distinct4`
- `val_ppl`（这里表示 `test` split 上的 held-out PPL）
- `mauve`
- `train_rep4_intra`

快速冒烟验证：

```bash
python Experiments/Rebuttal/run_gpt2_unsup_surrogate.py \
  --device cpu \
  --seeds 1088 \
  --methods unsup_set_aware \
  --smoke-mode
```

## 汇总脚本

新增汇总入口：

```bash
bash Experiments/Rebuttal/summarize_rebuttal.sh
```

或者手动指定：

```bash
python Experiments/Rebuttal/summarize_rebuttal.py \
  --cifar-dir Experiments/Rebuttal/results/cifar10 \
  --gpt2-dir Experiments/Rebuttal/results/gpt2 \
  --out-dir Experiments/Rebuttal/results/summary
```

默认会自动选择“所有已完成 seed/method 共同拥有的最后一代”做 final table，并同时产出：

- `Experiments/Rebuttal/results/summary/cifar_final.csv`
- `Experiments/Rebuttal/results/summary/gpt2_final.csv`
- `Experiments/Rebuttal/results/summary/cifar_trajectory.csv`
- `Experiments/Rebuttal/results/summary/gpt2_trajectory.csv`
- `Experiments/Rebuttal/results/summary/rebuttal_tables.md`

其中：

- `*_final.csv` 适合直接转 rebuttal 表
- `*_trajectory.csv` 适合补 appendix 里的代际趋势
- `rebuttal_tables.md` 直接给出一版 Markdown 表格

## MAUVE

GPT-2 runner 先产出递归训练的 per-generation 指标；如果 rebuttal 需要补 MAUVE，建议加 `--save-generations`，再复用已有：

```bash
python Experiments/exp11_gpt2_model/mauve_eval.py --help
```

然后把 `Experiments/Rebuttal/results/gpt2/generations/` 下保存的文本喂给现有 MAUVE 评估脚本即可。

## 说明

- 这套 rebuttal runner 尽量复用了 `exp9_cifar10_setaware` 和 `exp11_gpt2_model` 的训练/评估主流程，只在 `Experiments/Rebuttal` 下新增基线分支。
- 为了兼容当前仓库的导入路径，这次同时补了顶层 `filter/` 与 `Tools/` shim，避免 `exp9/exp11` 入口在当前树下直接 import 失败。
