# 3-Industry_senario / Long-term TTA

长时测试时自适应（Long-term TTA）模拟脚本，面向 MT 目标域四个稀缺度场景（few-shot / small / medium / high）。在无标签数据流上迭代 50 轮自适应，生成四种方法的结果（模型使用 **源域 θ_good 作为锚点**，避免使用 COCO 预训练权重）：

- **Baseline (Self-Training)**：伪标签自训练
- **TENT (SOTA 1)**：仅最小化预测熵
- **EATA-lite (SOTA 2 简化)**：熵阈值筛选 + L2 正则（模拟 EWC）
- **Ours (Bias_only)**：伪标签自训练 + L_bias 锚定
- **CoTTA (补充)**：EMA 教师伪标签 + KL 一致性（简化版）
- **Stabilized-CoTTA (新)**：CoTTA + L_bias 锚定（EMA 抗噪 + 锚定防漂移）

## 锚点模型 (θ_good)

1. 先在 MT 源域（/root/autodl-tmp/ICML/MT 或 Style_Filter/Baseline/Pretrain-Finetune/MT）按源域训练流程得到 `best.pt`（θ_good），例如 `Results/shared_pretrain/seed_1088/weights/best.pt`。
2. 将该权重复制/重命名到本目录：`cp /root/autodl-tmp/Style_Filter/Baseline/Pretrain-Finetune/Results/shared_pretrain/seed_1088/weights/best.pt anchor_mt.pt`
3. TTA 时用 `--model anchor_mt.pt`（默认即此文件名，未找到会报错）。**不要用 COCO/通用检测权重。**

## 运行 (流式 TTA，主实验入口)

脚本：`run_tta_stream.py`。适应数据 = 各场景 train+val 池（pools/mt_*.txt），监控 = global_test（按场景 YAML 的 test）。方法：Baseline/TENT/EATA-lite/Ours。

| 场景      | 池文件 | 池大小 | Batch | Epochs | 说明 |
|-----------|--------|--------|-------|--------|------|
| few-shot  | pools/mt_few_shot.txt | 全部 few-shot train+val（当前 59 张，含 .jpg） | 2     | 50 | 极端稀缺（小 Batch+长 Epoch） |
| small     | pools/mt_small.txt    | 全部 small train+val（当前 296 张，含 .jpg）  | 4     | 50 | 稀缺 |
| medium    | pools/mt_medium.txt   | 全部 medium train+val（当前 1185 张，含 .jpg） | 8     | 20 | 常规 |
| high      | pools/mt_high.txt     | 全部 high train+val（当前 4742 张，含 .jpg）   | 16    | 10 | 近乎充足 |

池文件生成规则（按 .jpg 计数，seed=2025，可用 `--regen-pools` 重写）：
- few/small/medium/high：各自场景的 train+val 全量（当前 few=59 / small=296 / medium=1185 / high=4742）

示例：
```bash
cd /root/autodl-tmp/ICML/3-Industry_senario
python run_tta_stream.py --methods baseline tent eata bias_only \
  --model anchor_mt.pt --device auto \
  --lr 1e-5 --optimizer adamw --eval-interval 2 \
  --lambda-bias 5e-4 --freeze-backbone \
  --lambda-bias-map few-shot:0.001,small:0.0005,medium:0.0003,high:0.0002
```
结果输出：`Results_stream/<Method>/<scenario>/seed_<seed>/`，包含 `metrics.json`（逐 Epoch）、`config.json` 和 `final_model.pt`。

## 数据拆分与泄露防护

- train/val 池来自 `MT-tgt_<scene>_{train,val}/images`，测试集为 `MT-tgt_global_test/images`，已检查 train/val 与 test 无重叠，train 与 val 也不重叠。
- `run_transductive_tta.py`（直接在 test 上适应）已移除，避免任何数据泄露或误用。主实验只使用本文件的流式 TTA。

## 超参说明

- 默认学习率降为 `1e-5` 以适配小 Batch（few-shot/small）。如需更大 LR，请确保梯度裁剪或自行验证稳定性。
