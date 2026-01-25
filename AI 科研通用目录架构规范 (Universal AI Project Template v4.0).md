# AI 顶会项目工程标准 (V4.0)

> **适用场景**：CVPR / ICCV / NeurIPS / ICLR 等顶会投稿项目
> 
> **核心哲学**：变量即真理（Variables as Truth）。目录结构、文件名、日志必须由统一变量生成，严禁手动命名，确保 100% 可复现与自动化。

---

## 0. 变量体系（Global Unique Truths）

所有目录生成、脚本运行、图表绘制必须依赖以下变量。

### 0.1 必选变量（强制）

- **`${PROJECT}`**：项目名（例 `Defect_Detection`）
    
- **`${DATASET}`**：数据集名（例 `DeepPCB`）
    
- **`${TASK}`**：任务类型（`cls` | `det` | `seg` | `lm` ...）
    
- **`${METHOD}`**：方法名（例 `Ours_ProNet`）
    
- **`${BASELINE}`**：对比方法名（例 `Baseline_YOLOv8`）
    
- **`${BACKBONE}`**：骨干网络（例 `SwinT` | `ResNet50`）
    
- **`${EXP_SET_TAG}`**：实验组唯一标签（定义见 1.1）
    
- **`${SEED}`**：随机种子（例 `42`）
    
- **`${RUN_ID}`**：单次运行ID（`时间戳_哈希`，例 `20260125_a1b2c`，防覆盖）
    

### 0.2 强烈建议变量（复现/消融必备）

- **`${DATA_VER}`**：数据版本（例 `v1.0` | `2026-01-25`）
    
- **`${SPLIT_TAG}`**：切分方案（例 `official` | `kfold5`）
    
- **`${PROTO}`**：评估协议（例 `coco` | `voc07`）
    
- **`${ENV_TAG}`**：环境标签（例 `cu121_torch23`）
    

---

## 1. 命名规范（EXP_SET_TAG）

实验组标签必须能**一眼看穿配置**。

### 1.1 标准格式

Bash

```
${EXP_SET_TAG} = ${METHOD}.${BACKBONE}.${DATASET}.${SPLIT_TAG}.${PROTO}.${HYPER_TAG}
```

### 1.2 示例

- **主实验**：`Ours_ProNet.SwinT.DeepPCB.official.coco.LR1e-4_BS16`
    
- **消融实验**：`Ours_ProNet.SwinT.DeepPCB.official.coco.ABL_no_attention`
    
- **对比实验**：`Baseline_YOLOv8.DarkNet.DeepPCB.official.coco.LR5e-4_BS32`
    

---

## 2. 目录结构（全景图）

Bash

```
autodl-tmp/
├── Pretrained_Zoo/             # [全局区] 预训练权重池 (所有项目共用, 避免重复下载)
│   ├── resnet50-19c8e357.pth
│   └── swin_base_patch4.pth
│
└── ${PROJECT}/
    ├── Common_Utils/           # [工具区] 公共标尺 (严禁放模型逻辑)
    │   ├── io.py               # 统一读写 (json/yaml/csv)
    │   ├── seed.py             # 统一随机种子 (torch/numpy/random)
    │   ├── registry.py         # 维护 _experiments_registry.csv
    │   ├── metrics.py          # 评估指标计算
    │   ├── export_submission.py # ★一键导出匿名代码 (去名字/去log)
    │   └── export_release.py   # ★一键导出开源版本
    │
    ├── Notebooks/              # [沙盒区] 脏活累活 (Git ignore)
    │   ├── debug_dataloader.ipynb
    │   ├── check_arch.py
    │   └── draft_figures/
    │
    ├── Methods/                # [核心区] 方法库 (统一接口)
    │   ├── ${METHOD}/
    │   │   ├── configs/
    │   │   │   ├── default.yaml       # 默认配置
    │   │   │   ├── ablations/         # 消融配置模板
    │   │   │   └── sweeps/            # 超参搜索模板
    │   │   ├── models/                # 模型定义
    │   │   ├── data/                  # 专用 Dataset Wrapper
    │   │   ├── train.py               # 必须支持 --config --seed --output_dir
    │   │   ├── eval.py                # 必须支持 --ckpt --output_dir
    │   │   └── README.md              # 方法独有文档
    │   └── ${BASELINE}/
    │
    ├── Data-pre/               # [数据工程] 预处理与清洗
    │   ├── raw_check/          # 源数据检查
    │   ├── split_scripts/      # 切分脚本
    │   └── outputs/            # ★所有预处理产物
    │       └── ${DATASET}.${DATA_VER}.${SPLIT_TAG}/
    │           ├── splits/     # train.txt, val.txt (最终真理)
    │           └── stats.json  # 类别/尺寸统计
    │
    ├── DataSet/                # [数据湖] 只读原始数据
    │   └── ${DATASET}/
    │       ├── data/           # 物理图片/文本
    │       └── meta/           # ★复现元数据
    │           ├── VERSION.txt         # 数据版本
    │           ├── checksums.json      # 文件指纹 (防数据漂移)
    │           └── protocol.md         # 协议说明 (mAP计算方式等)
    │
    ├── Experiments/            # [实验车间] 产出物
    │   ├── _experiments_registry.csv   # 全局索引表
    │   ├── _notes.md                   # 实验结论笔记
    │   └── ${EXP_SET_TAG}/             # Set 层 (同配置)
    │       ├── _set_config.yaml        # 冻结配置 (除 seed 外全一致)
    │       ├── _set_summary.json       # 聚合结果 (Mean ± Std)
    │       └── S${SEED}.${RUN_ID}/     # Run 层 (原子实验)
    │           ├── weights/            # .pth / .pt
    │           ├── logs/               # tensorboard / wandb
    │           ├── eval/               # 评估结果 (metrics.json, per_class.csv)
    │           ├── env_snapshot/       # 环境快照 (pip freeze)
    │           ├── src_snapshot/       # 代码快照 (备份当前代码)
    │           └── config_backup.yaml  # 最终完整配置
    │
    ├── Assets/                 # [展示区] 项目主页/PPT素材
    │   ├── demo_video.mp4
    │   ├── teaser.gif
    │   └── slides.pptx
    │
    ├── Paper/                  # [交付区] 论文工程
    │   ├── figures/            # PDF/EPS 矢量图
    │   ├── tables/             # LaTeX 表格代码
    │   ├── tex/                # 论文源码 (main.tex)
    │   └── checklist.md        # 投稿前自检清单
    │
    ├── Env/                    # [环境区] 复现环境
    │   ├── environment.yml
    │   ├── requirements.txt
    │   └── Dockerfile
    │
    └── Release/                # [发布区] 导出产物
        ├── submission_anonymous/ # 双盲审稿包
        └── code_release/         # GitHub 开源包
```

---

## 3. 执行标准（Execution Rules）

### 3.1 训练脚本接口（Methods 强制规范）

所有 `Methods/*/train.py` 必须接受以下标准参数，以便被顶层脚本批量调度：

Bash

```
python Methods/Ours/train.py \
    --config Methods/Ours/configs/default.yaml \
    --tag ${EXP_SET_TAG} \
    --seed ${SEED} \
    --output_dir Experiments/${EXP_SET_TAG}/S${SEED}.${RUN_ID}
```

### 3.2 实验防覆盖机制

- **Run ID**: 每次运行生成 `timestamp_hash`，确保 `S42` 重跑时不会覆盖旧的 `S42`（除非你手动想覆盖）。
    
- **Snapshots**: 每次 Run 必须自动备份当前 `Methods/` 代码到 `src_snapshot/`，防止你改了代码后不知道这组权重是哪个版本的代码跑出来的。
    

---

## 4. 常见实验工作流映射

1. **Main Results (刷榜)**
    
    - 创建 `${EXP_SET_TAG}`，固定超参。
        
    - 运行 3-5 个 Seeds。
        
    - 脚本自动读取 `eval/metrics.json`，写入 `Paper/tables/main.tex`。
        
2. **Ablation Study (消融)**
    
    - 复制 default config，修改 `${HYPER_TAG}` 为 `ABL_xxx`。
        
    - 确保除该模块外，其他设置与 Main 完全一致（共享 Data-pre 产物）。
        
3. **Visual Analysis (画图)**
    
    - 在 `Notebooks/` 里调试画图代码。
        
    - 定稿后，读取 `Experiments/.../eval/per_class.csv` 或 `weights`。
        
    - 输出矢量图到 `Paper/figures/`。
        

---

## 5. 顶会投稿自检清单 (Definition of Done)

在打包上传前，请确认：

- [ ] **可复现性**：`Env/` 完整，且任意选一个旧实验能通过 `config_backup.yaml` + `seed` 复现结果。
    
- [ ] **数据一致性**：`DataSet/meta/checksums.json` 校验通过，确定没用错数据版本。
    
- [ ] **匿名化**：运行 `export_submission.py`，确认导出的代码包中无作者姓名、无绝对路径。
    
- [ ] **可视化**：Project Page 需要的 Demo 视频/GIF 已放入 `Assets/`。