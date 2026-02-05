# ICML 代码上传计划（基于 `STEP2/Release` 现状）

目标：在不上传大体量数据/模型/日志的前提下，把**可读、可跑、可复现**的核心实现整理成一个可直接打包上传（例如压缩为 zip）的代码快照。本仓库根目录即为最终上传目录（由原工程的 `STEP2/Release` 整理而来）。

---

## 1) 上传内容分层（“必须 / 可选 / 不上传”）

### 必须上传（reviewer 打开就能看懂/运行）

- 核心方法实现（paper 的“方法部分”对应代码）
  - `filter/`：SAGE-Filter（set-aware 权重 + 显式 drift/Δϕ 校正）
  - `bias/`：偏置源/基线估计器（用于对照/解释）
  - `Common_Utils/`：确定性/画图风格等通用工具（最小必要）
- 最小可运行示例（CPU 分钟级）
  - `scripts/run_regression_demo.sh`
  - `run.sh`
- 环境与复现说明（ICML 复现友好）
  - `requirements.txt`（最小依赖）
  - `README.md`（快速上手）
  - `REPRODUCIBILITY.md`（复现说明/硬件/随机种子）
  - `LICENSE`、`CITATION.cff`
  - `.env.example`（只提供模板，不放真实 key）

### 可选上传（增加透明度，但不强制 reviewer 运行）

- 轻量“实验脚本快照”（只含代码与说明，不带大文件输出）
  - 计划放在 `Experiments/`
  - 保留 `.py/.sh/.md` 等脚本与说明
  - **剔除** `Total_results/`、`results/`、checkpoint、缓存、以及任何数据切片/文本语料
- 兼容层（避免历史脚本 import 失败）
  - `Tools/`：为旧脚本提供 `Tools.deterministic` 的薄封装，转到 `Common_Utils/deterministic.py`
- 扩展依赖（完整实验需要的额外包）
  - `requirements_full.txt`（可选安装，避免污染最小 demo）

### 不上传（明确排除）

- 数据集/缓存（体积大、可能涉及授权）
  - `STEP2/DataSet/`、HF datasets cache、任何 `*.bin/*.pt/*.ckpt/*.safetensors` 等
- 大规模实验输出/日志（体积大且可再生）
  - `STEP2/Experiments/Total_results/`（当前约几十 GB）
  - 各实验目录下的 `results/`、中间 checkpoint、生成文本等
- 论文写作/图片源文件（非代码复现必需，且可能较大）
  - `STEP2/Paper/`、`STEP2/Notebooks/`
- 私密信息
  - 根目录 `.env`（包含真实 API key），以及任何凭证文件

---

## 2) 最终目录结构（上传即打包）

计划整理为：

```text
<repo-root>/
  PLAN.md
  README.md
  REPRODUCIBILITY.md
  LICENSE
  CITATION.cff
  requirements.txt
  requirements_full.txt          # 可选
  .env.example
  run.sh
  scripts/
  bias/
  filter/
  Common_Utils/
  Tools/                         # 可选：兼容旧脚本
  Experiments/                   # 可选：仅脚本快照（无大文件）
```

---

## 3) 执行步骤（先文档、再搬运、再自检）

- [ ] 步骤 A：对仓库根目录做一次“上传体检”
  - [ ] 确认无 `.env`、无大文件、无 `__pycache__/`、无数据/模型权重
  - [ ] 确认 `README.md` + `run.sh` 能跑通最小 demo
- [ ] 步骤 B：搬运（从主工程拣选）可选实验脚本到 `Experiments/`
  - [ ] 仅复制脚本与说明（`.py/.sh/.md` 等）
  - [ ] 排除 `Total_results/`、`results/`、checkpoint、缓存、语料文本等
- [ ] 步骤 C：补齐兼容层与依赖说明
  - [ ] 添加 `Tools/deterministic.py`（兼容旧脚本的 import）
  - [ ] 添加 `requirements_full.txt`（完整实验可选依赖）
- [ ] 步骤 D：最终自检（面向 reviewer）
  - [ ] 在仓库根目录下运行：`pip install -r requirements.txt` + `bash run.sh demo`
  - [ ] 快速检查 `Release/` 目录体积在可接受范围（MB 级）
  - [ ] 生成一个可上传压缩包（由用户自行打包/上传）

---

## 4) 交付标准（ICML reviewer 视角）

- 一条命令可跑：`bash run.sh demo`
- 不需要下载数据/模型也能看到方法在 toy setting 的效果曲线
- 文档能回答 reviewer 常见问题：依赖、随机种子、硬件、输出文件在哪里
- 上传包不含隐私信息与大文件
