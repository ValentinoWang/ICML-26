# Data-pre (本地预处理产物)

此目录用于放置**数据预处理/清洗/特征缓存**等中间产物。它同样**不随 GitHub 仓库分发**（默认被 `.gitignore` 忽略）。

建议约定：
- 预处理脚本输出到 `Data-pre/<dataset_name>/...`
- 可以按版本/日期分子目录，方便回滚与对齐实验。

示例：
- `Data-pre/cifar10/`
- `Data-pre/wikitext103/`
