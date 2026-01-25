# 过滤器规则蒸馏脚本（VOC2007）

本目录包含 g_phi 的规则蒸馏训练脚本，覆盖整图级噪声（per-sample）和框级混合噪声（per-anchor）。

## 噪声数据
- 整图毒（per-sample）：`/root/autodl-tmp/dataset/voc07_noisy_per-sample/voc07_noisy.yaml`
- 框级混合噪声（per-anchor）：`/root/autodl-tmp/dataset/voc07_noisy_per-anchor/voc07_noisy.yaml`
- 生成示例：  
  - `python src/make_noisy_voc.py --noise-mode image --output-root /root/autodl-tmp/dataset/voc07_noisy_per-sample`  
  - `python src/make_noisy_voc.py --noise-mode box --output-root /root/autodl-tmp/dataset/voc07_noisy_per-anchor`

## 脚本（已按子目录拆分）
- `per-sample/train_per-sample_filter_rule_distill.py`：整图级过滤器，默认输出 `Results/distill_rule_per-sample/voc07_noisy/mlp_filter_<anchor>.pt`。（需保证 PYTHONPATH 含 `/root/autodl-tmp`）
- `per-anchor/train_per-anchor_filter_rule_distill.py`：框级过滤器（Top-K 筛选），输出 `Results/distill_rule_per-anchor/voc07_noisy/mlp_filter_per_anchor_<anchor>.pt`。（需保证 PYTHONPATH 含 `/root/autodl-tmp`）

## 默认训练超参（可通过 CLI 覆盖）
- 共同：`max-epochs=10000`，`patience=20`，`lr-phi=1e-3`，`batch-size=64`（蒸馏阶段）。
- per-sample：`loss-thres=0.3`。
- per-anchor：`loss-thres=0.3`，`bad-weight=0.0`，`topk=500`，`pos-scale=1.0`，`neg-scale=5.0`。

## 实现要点
- 共性：冻结 YOLO（anchor_voc_*），只训练 g_phi；BCE 监督，早停；最佳权重存默认目录。
- per-sample：`PerSampleWeightedDetectionLoss` 产生 z = `[loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]`（按样本聚合）；`loss_sum = loss_box+loss_cls+loss_dfl`，大于阈值标 0，否则 1；`MLPFilter`（BN+注意力+MLP），batch<2 跳过以避免 BN 报错。
- per-anchor：`PerAnchorFeatureExtractor` 逐 Anchor 取 z = `[loss_box(CIoU), loss_cls(BCE), 0, conf_diff, iou_diff]`；按 `loss_sum = loss_box+loss_cls` 取 Top-K（默认 500）参与蒸馏，缓解极端负样本/显存；`loss_sum > loss_thres` 标签为 `bad_weight`（默认 0），否则 1；BCE 再乘正负缩放 `pos_scale`/`neg_scale` 平衡比例；`PerAnchorFilter` 为 BN+两层 ReLU + Sigmoid。
- per-anchor-image：与 per-sample 类似但保持早期目录命名。

## 快速冒烟命令
```bash
# per-sample
PYTHONPATH=/root/autodl-tmp \
python src/filter/train_per-sample_filter_rule_distill.py \
  --data /root/autodl-tmp/dataset/voc07_noisy_per-sample/voc07_noisy.yaml \
  --anchor "/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/Anchor/voc07_clean/anchor_voc_yolov8n.pt" \
  --max-epochs 1 --patience 1 --batch-size 4 \
  --save-path "/tmp/mlp_filter_per-sample.pt"

# per-anchor（Top-K + 损失打标 + 正负缩放）
PYTHONPATH=/root/autodl-tmp \
python src/filter/train_per-anchor_filter_rule_distill.py \
  --data /root/autodl-tmp/dataset/voc07_noisy_per-anchor/voc07_noisy.yaml \
  --anchor "/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/Anchor/voc07_clean/anchor_voc_yolov8n.pt" \
  --max-epochs 1 --patience 1 --batch-size 4 --topk 200 \
  --loss-thres 0.3 --bad-weight 0.0 --pos-scale 1.0 --neg-scale 5.0 \
  --save-path "/tmp/mlp_filter_per-anchor.pt"
```

## 默认结果目录
- Per-sample：`Results/distill_rule_per-sample/voc07_noisy/`
- Per-anchor：`Results/distill_rule_per-anchor/voc07_noisy/`
- Anchor-image（遗留）：`Results/distill_rule_per-anchor-image/voc07_noisy/`
