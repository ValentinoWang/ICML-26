# Experiments

本目录包含论文/项目对应的各实验脚本。

## 数据目录约定（本地）

本仓库默认不分发原始数据与预处理产物（避免大文件/隐私/许可问题），但为保证路径约定统一，仓库根目录保留两个本地数据目录：

- `DataSet/`：原始数据（只读）。
- `Data-pre/`：预处理与缓存产物（可写）。

它们都被 `.gitignore` 忽略，GitHub 上只会看到占位说明文件。

## 常见数据入口

- 视觉数据（例如 CIFAR-10）脚本一般使用 `--data-root`，默认是实验目录下的 `./data`（torchvision 会自动下载/解压）。
- NLP 数据（例如 Wikitext-103）依赖 HuggingFace `datasets`，缓存位置可通过 `HF_HOME`/`HF_DATASETS_CACHE` 环境变量控制。

建议做法：
- 原始数据尽量放在 `DataSet/` 或将 `DataSet/` 软链接到你的数据盘。
- 预处理产物放 `Data-pre/`，并按实验/数据集分子目录。
