"""
ICML 实验包

按数据集/任务类型大致分为三块（+ 一个简单 CIFAR-10 子任务）：

- CityFog 子包（ICML.City→Foggy）：
  - Cityscapes → FoggyCityscapes 目标检测 + L_bias（以及可选 g_phi）实验；
  - 入口脚本参考：
    - ICML.cityfog.run_bias_only_experiments
    - ICML.cityfog.run_bias+filter_experiments

- MT / Toy 子包（ICML.MT）：
  - 多场景 Toy / MT 检测任务上的 YOLO + L_bias / + g_phi 理论实验；
  - 入口脚本参考：
    - ICML.mt.run_bias_only_experiments
    - ICML.mt.run_bias+filter_experiments
    - ICML.mt.run_bias_only_noise_experiments
    - ICML.mt.run_bias+filter_noise_experiments

- CIFAR-10 子包（ICML.CIFAR10）：
  - 在 CIFAR-10 + ResNet-18 + 人工标签噪声上的 g_phi 噪声检测/样本过滤实验；
  - 入口脚本参考：
    - ICML.CIFAR10.train_theta_good_cifar10
    - ICML.CIFAR10.train_gphi_noise_cifar10

- 公共模块：
  - ICML.core.yolo_bias_finetune —— AnchorModel / BiasDetectionTrainer / g_phi 等核心实现；
"""
