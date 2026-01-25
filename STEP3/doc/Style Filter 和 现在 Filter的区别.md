### Feature-Guided 源域筛选接入 DAN 迁移实验说明

本说明文档把我们刚刚讨论的内容整理成一个实验方案，方便你之后直接照着跑实验和写论文。

---

## 1. 实验目标

- 对比不同 **源域版本**（`full_src` vs Style Filter 后的 `th1/th3/th6`）在 **相同目标域场景** 下的迁移效果。  
- 验证：基于特征的源域数据筛选（Extractor 生成的 MT_src_options）能否 **稳定提升 MT-tgt 缺陷检测的迁移性能**。  
- 所有方法都在同一个 DAN（YOLO+MMD 域适应）框架里训练，在统一测试集 `MT-tgt_global_test` 上评估，整理成结果矩阵。

---

## 2. 数据与版本概念

### 2.1 源域数据版本（Style Filter 输出）

目录：`/root/autodl-tmp/dataset/MT_src_options`

- `0_full_source`：完整源域（未筛选，Baseline）  
  - 结构：`MT-src_train/images, MT-src_val/images`  
  - YOLO 配置：`DAN/config/mt-src-dataset.yaml`（已在用，`path` 指向该目录）
- `th1-MT-src`：基于 Style Filter Top-1 的保守筛选版本  
  - 配置：`th1-MT-src-split-dataset.yaml`（含 `train`, `val`, `nc`, `names`）
- `th3-MT-src`：Style Filter 推荐版本（Top-3）  
  - 配置：`th3-MT-src-split-dataset.yaml`  
- `th2-MT-src`, `th6-MT-src`：对应其他强度的筛选版本（可选用于消融）

这些版本本质上就是：**同一 MT 源域上，不同“Feature-Guided 选样策略”得到的子数据集**。

### 2.2 目标域场景（DAN 已定义）

`DAN/config/dan_training_config.json` 中已有 4 个目标域场景：

- `few-shot` → `mt-tgt-few-shot.yaml`  
- `small` → `mt-tgt-small.yaml`  
- `medium` → `mt-tgt-medium.yaml`  
- `high` → `mt-tgt-high.yaml`

统一测试集（在 Auxiliary 下多处引用）：  

- `test: /root/autodl-tmp/dataset/MT-tgt-split/MT-tgt_global_test/images`

后续所有对比实验都应在这个 `MT-tgt_global_test` 上评估。

---

## 3. DAN 训练脚本如何“接上” MT_src_options

### 3.1 核心入口：`run_dan_trainer.py`

`/root/autodl-tmp/Style_Filter/Baseline/DAN/run_dan_trainer.py` 的角色：

- 它 **不写死数据路径**，而是从 JSON 配置和 dataset YAML 里读；  
- 关键字段在 `DAN/config/dan_training_config.json`：

```json
"data": {
  "active_scenario": "few-shot",
  "base_paths": {
    "source": "/root/autodl-tmp/dataset/MT_src_options/0_full_source",
    "target": "/root/autodl-tmp/dataset/MT-tgt-split"
  },
  "scenarios": {
    "few-shot": {
      "source_subdir": "MT-src_train/images",
      "target_subdir": "MT-tgt_few-shot_train/images",
      "source_config": ".../mt-src-dataset.yaml",
      "target_config": ".../mt-tgt-few-shot.yaml"
    },
    ...
  }
}
```

说明：

- 当前配置已经把源域根目录指向了 `0_full_source`，也就是 **Full Source + DAN** 的迁移设置；  
- 场景切换通过 `--scenarios few-shot / small / medium / high` 完成。

### 3.2 把不同 Style Filter 版本挂载进 DAN

原则：**不改 `run_dan_trainer.py`，只通过配置切换源域路径和 YAML**。

做法有两种：多份配置文件 或 命令行覆盖。

#### 方案 A：多份训练配置 JSON（推荐，清晰）

在 `DAN/config` 下复制多份配置，例如：

- `dan_training_fullsrc.json`（当前 `dan_training_config.json` 的拷贝，保留 `0_full_source`）  
- `dan_training_th1.json`：  
  - `data.base_paths.source` 改为  
    `/root/autodl-tmp/dataset/MT_src_options/th1-MT-src`  
  - `data.scenarios.*.source_config` 改为指向 `th1-MT-src-split-dataset.yaml`（或一个 wrapper yaml）
- `dan_training_th3.json`：类似，把 source 改成 `th3-MT-src`，config 改成 `th3-MT-src-split-dataset.yaml`

然后，你就可以分别在相同场景下运行：

```bash
cd /root/autodl-tmp/Style_Filter/Baseline/DAN

# Full Source + 四个目标场景
python run_dan_trainer.py \
  --config config/dan_training_fullsrc.json \
  --scenarios few-shot small medium high

# Style Filter th3 源域 + 四个目标场景
python run_dan_trainer.py \
  --config config/dan_training_th3.json \
  --scenarios few-shot small medium high

# 如需 th1 / th6，可继续定义对应的配置文件
```

#### 方案 B：在脚本中增加 `--source-root` 覆盖（可选增强）

如果你不想维护多份 JSON，可以在 `run_dan_trainer.py` 的 `apply_argument_overrides` 里增加：

```python
# 在 argument parser 里新增：
override_group.add_argument(
    '--source-root',
    type=str,
    help='覆盖 data.base_paths.source，用于切换不同 MT_src_options'
)

# 在 apply_argument_overrides 中：
if getattr(args, 'source_root', None):
    overrides.setdefault('data', {}).setdefault('base_paths', {})['source'] = args.source_root
```

这样你就可以用统一的 `dan_training_config.json`，通过 CLI 切源域版本：

```bash
python run_dan_trainer.py \
  --config config/dan_training_config.json \
  --scenarios few-shot \
  --source-root /root/autodl-tmp/dataset/MT_src_options/0_full_source   # full_src

python run_dan_trainer.py \
  --config config/dan_training_config.json \
  --scenarios few-shot \
  --source-root /root/autodl-tmp/dataset/MT_src_options/th3-MT-src      # th3
```

（前提是对应的 `source_config` 里 `train/val` 路径与你的 `th*_MT_src_*` 保持一致。）

---

## 4. “结果矩阵”指的是什么

当你把不同源域版本 + 相同 DAN 框架 + 相同目标场景一一跑完后，每个组合都会在 `Results/<scenario>/seed_*/` 下生成：

- `results.csv`  
- `metrics_summary.json`  
- `evaluation/` 下的 mAP/F1 等详细指标与图

你关心的是：**在统一测试集 `MT-tgt_global_test` 上的关键指标（mAP、F1）**。

“结果矩阵”就是把这些结果整理成这样一张表（示例）：

| Source Version      | few-shot mAP | small mAP | medium mAP | high mAP |
|---------------------|-------------:|----------:|-----------:|---------:|
| full_src (0_full)   |      x.xx    |    x.xx   |     x.xx   |   x.xx   |
| th1-MT-src          |      x.xx    |    x.xx   |     x.xx   |   x.xx   |
| th3-MT-src (rec.)   |      x.xx    |    x.xx   |     x.xx   |   x.xx   |
| th6-MT-src          |      x.xx    |    x.xx   |     x.xx   |   x.xx   |

这张表就是你论文里“Feature-Guided Source Data Selection 对 Deep Transfer Learning 的影响”的主实验结果。

---

## 5. 与毕业课题的对应关系

你的题目：**Feature-Guided Data Selection: Improving Deep Transfer Learning for Magnetic Tile Defect Detection**

对应到现在的工程结构：

- **Feature-Guided Data Selection**  
  - Extractor：用 VGG style + PCA + K-means + 高斯 + KL 风险分层，把 MT-src 划成不同版本（th1/th3/...）。  
  - 这些版本就是“用特征引导的源域样本子集”。

- **Deep Transfer Learning**  
  - Baseline/DAN：在 YOLO + MMD 域适应框架下，从 MT-src → MT-tgt (few-shot/small/...) 做迁移；  
  - 通过修改 `data.base_paths.source` + `source_config`，你就能在 **同一套迁移流程** 中对比 full_src vs filtered_src。

- **Improving**  
  - 若结果矩阵中，某些 filtered 源域版本（例如 th3）在 `MT-tgt_global_test` 上显著优于 full_src baseline，那么就可以直接支撑你的题目：  
    “Feature-guided 源域数据选取显著改善了 MT 缺陷检测的迁移性能。”

后续如果你把 `Toy/Theory_and_Experiment_Design.md` 里的偏置控制框架（$\theta_{good}$ + $L_{bias}$ + $g_\phi$）也接进来，这个实验设计可以进一步升级为：  

- “先用 Extractor 做静态的 feature-guided source cleaning”，  
- “再在 DAN/YOLO 训练内部做动态的 feature-guided filtering + 偏置控制”，  

两层一起构成完整的论文方法部分。  

