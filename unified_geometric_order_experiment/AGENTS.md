# unified_geometric_order_experiment

## 参考优先级

1. `Engineering-file-reference/unified_collapse_geometric_order.pdf`
2. `test1.md`
3. `Engineering-file-reference/FFinal icml2026.pdf`

当前目录内所有工程实现都以 `1 > 2 > 3` 为准。

## 目录约定

- 主脚本：`run_unified_geometric_order_experiment.py`
- 结果目录：`results/`
- 不把本实验产物写入仓库其他实验目录

## 方法映射

- `No Filter`
  - 原始候选递推，不做重加权，不做显式纠偏。
- `Weight-only`
  - 只做方差控制/重加权，不做显式校正，不做顺序商。
- `Set-Aware`
  - 只看 candidate set 的 set-aware correction baseline。
  - 不引入 `test1.md` 的局部仿射李代数通道分解。
  - 不引入 `D_t^{(K)}` 的顺序商。
  - 不引入共轭方向修正与对称半校正。
- `Our method`
  - 必须按 `test1.md` 实现：
    - `X_t^(mu), X_t^(Sigma), X_t^(zeta), X_t^(Hol), X_t^(0)`
    - 二阶顺序商 `D_t^(2)`
    - 共轭修正 `\tilde D_t = Ad_{exp(H_t)} \hat D_t`
    - 主公式 `Psi_t = exp(Omega_B/2) exp(-\tilde D_t/2) exp(D_t^(K)) exp(-\tilde D_t/2) exp(Omega_B/2)`
    - `theta_{t+1} = rho_t(Psi_t)(theta_t)`

## 人工验收规则

- 不写自动成功判定逻辑。
- 先运行 `--no-plot`，只打印逐步终端输出并写 CSV。
- 由主 agent 人工读取终端输出，确认：
  - `Our method` 的主指标逐步递降/收敛。
  - `Our method` 明显优于 `Set-Aware`。
- 只有人工确认通过后，才运行 `plot` 生成正式图。

## 设备约定

- 支持：`cpu / mps / cuda`
- 本轮基础验收只要求：
  - `cpu`
  - 本机 `mps`

## subagent 记录

- 最多允许 2 个 subagent 陪跑。
- 若启用，必须记录其任务、结论和回收时间。

## 工程操作日志

### 2026-03-26

- 初始化本目录与 `results/`
- 启用 subagent:
  - `geom_mapping`: 陪跑 `test1.md` 到局部仿射 toy experiment 的映射
  - `baseline_reuse`: 陪跑现有基线复用与绘图风格整理
- subagent 回收摘要：
  - `geom_mapping`
    - 建议把 toy 固定到 `Aff(2)` 的 `3x3` 齐次矩阵，并直接落 `X^(mu), X^(Sigma), X^(zeta), X^(Hol), X^(0)`、`D_t^(2)`、`Ad_{exp(H_t)}` 和 `rho_t(Psi_t)`。
  - `baseline_reuse`
    - 建议不要把旧 `standard_filter` 误当成 `Weight-only`；
    - 建议新实验集中到单 runner、统一日志与绘图。
- 主脚本创建：
  - `run_unified_geometric_order_experiment.py`
- 运行记录：
  - `cpu`
    - 命令：`conda run -n base python unified_geometric_order_experiment/run_unified_geometric_order_experiment.py --mode main --device cpu --no-plot --seed 7`
    - 结论：跑通。人工阅读终端逐步输出后确认 `Our method` 在四个场景都整体递降；终点上均优于 `Set-Aware`。
    - 终点：
      - `fixed_additive_bias`: `Set-Aware=0.279333`, `Our=0.171141`
      - `anisotropic_shrinkage`: `Set-Aware=0.204076`, `Our=0.160667`
      - `noncommutative_structural_bias`: `Set-Aware=0.343331`, `Our=0.149487`
      - `small_sample_structural_bias`: `Set-Aware=0.381353`, `Our=0.124357`
  - `mps`
    - 命令：`conda run -n base python unified_geometric_order_experiment/run_unified_geometric_order_experiment.py --mode main --device mps --no-plot --seed 7`
    - 结论：跑通。逐步输出与 `cpu` 一致，满足本轮基础验收。
  - `transformer / mps`
    - 命令：`conda run -n base python unified_geometric_order_experiment/run_unified_geometric_order_experiment.py --mode transformer --device mps --no-plot --seed 7`
    - 改版说明：
      - 删除旧的“单独看一个 set 模型收敛”的 transformer 图。
      - 改成“每个模型尺寸下同时比较 `Set-Aware` 和 `Our method` 的 held-out test MSE”。
      - 图文件仍写到 `results/transformer_convergence.png`，但语义已变。
    - 结论：跑通。人工阅读逐步日志后确认三档模型下两种方法都在同一图里可直接对比，且尾部 `Our method` 明显低于 `Set-Aware`。
    - 尾部：
      - `ST-0.41M`: `Set-Aware=0.002792`, `Our=0.000096`
      - `ST-2.67M`: `Set-Aware=0.015793`, `Our=0.000018`
      - `ST-8.34M`: `Set-Aware=0.018658`, `Our=0.000097`
  - `sweeps / cpu`
    - 命令：`conda run -n base python unified_geometric_order_experiment/run_unified_geometric_order_experiment.py --mode sweeps --device cpu --no-plot --seed 7`
    - 结论：跑通，结果已写入 `results/icml_style_sweeps.csv`。
  - `plot`
    - 命令：`conda run -n base python unified_geometric_order_experiment/run_unified_geometric_order_experiment.py --mode plot --device cpu --seed 7`
    - 结论：正式图已生成。
  - `release_demo / cpu`
    - 命令：`conda run -n base python unified_geometric_order_experiment/run_unified_geometric_order_experiment.py --mode release_demo --device cpu --seed 7`
    - 结论：跑通。按 `Release` demo 的单图风格额外生成 `release_demo_plot.png`。
    - 尾部：
      - `No Filter=0.490139`
      - `Standard Filter=0.329199`
      - `Set-Aware=0.067093`
      - `Our method=0.000000`
  - 人工总验收：
    - 通过。
    - 判定理由：
      - `Our method` 在主实验终端输出中呈清晰递降/收敛；
      - 四个场景终点均优于 `Set-Aware`；
      - 结构偏置两类场景优势明显；
      - `cpu` 与本机 `mps` 均跑通。
