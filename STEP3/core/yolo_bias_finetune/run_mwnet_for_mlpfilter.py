#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 Meta-Weight-Net 思想的 MLP FiLTER (g_phi) 外层训练脚本（阶段二）

目标：
- 在固定 YOLO + L_bias 检测模型参数 θ 的前提下，只优化样本过滤器 g_phi（MLP FiLTER）的参数 φ；
- 使用目标域的验证集（val / test）作为“meta 集”，让 g_phi 学习一套权重 w_i(φ)，
  使得在 1-step 虚拟更新 θ' 之后，meta 集上的检测损失尽可能小（同时通过轻微正则保持 w_i 不偏离 1）。

实现要点（MWNet-style, 1-step）：
- Inner loop：在 train loader 上，用当前 g_phi 产生的 w_i 对检测 loss 加权，计算 L_train(θ, φ)，
  对 θ 求梯度并构造一次虚拟更新 θ' = θ - α ∂L_train/∂θ；
- Outer loop：在 meta loader（val/test）上，用 θ' 计算 L_meta(θ')，并对 φ 反向传播，
  梯度链路为 L_meta(θ'(φ)) → φ，只更新 φ，不更新 θ 本身。

训练完成后，会将 MLP FiLTER 的参数保存到
    Toy/Results/Bias+Filter/<scenario>/seed_<seed>/mlpfilter_meta.pt
后续阶段三可以在 YOLO + L_bias + g_phi 训练时加载该权重作为初始化。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.nn.utils.stateless import functional_call

from ultralytics.models.yolo.detect import DetectionTrainer

# 将 Baseline 根目录加入 sys.path，确保可以导入 ICML 包
THIS_FILE = Path(__file__).resolve()
ICML_ROOT = THIS_FILE.parents[2]  # .../Baseline/ICML
BASELINE_ROOT = ICML_ROOT.parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from ICML.mt.config import build_bias_finetune_config
from ICML.core.yolo_bias_finetune.mlp_filter import MLPFilter, PerSampleWeightedDetectionLoss
from ICML.core.yolo_bias_finetune.train_bias_yolo import build_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段二：在固定 θ 上，用 meta 集训练 MLP FiLTER (g_phi)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="small",
        choices=["few-shot", "small", "medium", "high"],
        help="目标域场景名称（few-shot / small / medium / high）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1088,
        help="选择哪个微调 seed 的 θ 作为固定检测模型（对应 Bias_only/<scenario>/seed_<seed>）",
    )
    parser.add_argument(
        "--theta-checkpoint",
        type=str,
        default=None,
        help=(
            "指定固定检测模型 θ 的 best.pt 路径；"
            "若不指定，则默认使用 "
            "Toy/Results/Bias_only/<scenario>/seed_<seed>/results/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--meta-epochs",
        type=int,
        default=1,
        help="在 meta 集上优化 g_phi 的轮数（每轮遍历一次 meta dataloader）",
    )
    parser.add_argument(
        "--meta-steps",
        type=int,
        default=200,
        help="每个 meta epoch 中最多使用多少个 batch（避免 meta 训练时间过长）",
    )
    parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=32,
        help="meta dataloader 的 batch size（只影响 G_phi 训练，不影响原 YOLO 训练）",
    )
    parser.add_argument(
        "--lr-phi",
        type=float,
        default=1e-3,
        help="MLP FiLTER (g_phi) 的学习率",
    )
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=1e-3,
        help="内层虚拟 GD 步长，用于 θ 的多步更新",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=3,
        help="内层虚拟 GD 步数 K（越大越接近真实 MWNet，计算/显存开销也越大）",
    )
    parser.add_argument(
        "--lambda-keep-rate",
        type=float,
        default=0.01,
        help="约束权重 w_i 不偏离 1 的 L2 正则系数（keep-rate 正则）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 加载阶段一的统一配置，获取数据 yaml（默认 lambda_bias=1e-4）
    cfg = build_bias_finetune_config()
    if args.scenario not in cfg.scenario_data_cfg:
        raise ValueError(f"未知场景: {args.scenario}")
    data_yaml = cfg.scenario_data_cfg[args.scenario]
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")

    # 解析 θ 的 checkpoint 路径
    # 默认与阶段一 (YOLO + L_bias, λ_bias=cfg.lambda_bias) 的结果目录保持一致：
    #   cfg.results_root / lambda_xxx / <scenario>/seed_<seed>/results/weights/best.pt
    lambda_dir = "lambda_0" if cfg.lambda_bias == 0 else f"lambda_{cfg.lambda_bias:g}"

    if args.theta_checkpoint is not None:
        theta_ckpt_path = Path(args.theta_checkpoint)
    else:
        theta_ckpt_path = (
            cfg.results_root
            / lambda_dir
            / args.scenario
            / f"seed_{args.seed}"
            / "results"
            / "weights"
            / "best.pt"
        )
    if not theta_ckpt_path.exists():
        raise FileNotFoundError(f"θ checkpoint 不存在: {theta_ckpt_path}")

    # 构造一个 DetectionTrainer，只为复用其 dataloader 构建逻辑（train_loader/test_loader）
    # 这里使用 θ checkpoint 作为 model 权重，确保数据 pipeline 与阶段一一致。
    # 与阶段三 (Bias+Filter) 训练约定保持一致：
    # Bias+Filter 结果目录根为 cfg.results_root.parent / "Bias+Filter"
    mt_results_root = cfg.results_root.parent  # .../ICML/mt/Results
    bias_filter_root = mt_results_root / "Bias+Filter" / lambda_dir / args.scenario / f"seed_{args.seed}"

    overrides = build_overrides(
        scenario=args.scenario,
        training_cfg=cfg.training,
        data_yaml=data_yaml,
        project_dir=bias_filter_root / "meta_tmp",
        seed=args.seed,
        model_path=theta_ckpt_path,
    )
    # 为 meta 训练设定更小的 batch 和较少的 workers，避免显存压力
    overrides["epochs"] = 1
    overrides["batch"] = args.meta_batch_size
    overrides["workers"] = min(overrides.get("workers", 10), 4)
    overrides["project"] = str(bias_filter_root)
    overrides["name"] = "meta_run"

    trainer = DetectionTrainer(overrides=overrides)
    # 构建模型 + dataloader（world_size=0 视为单 GPU / CPU）
    trainer._setup_train(world_size=0)

    device = trainer.device
    # DetectionTrainer 内部的 model 是 DetectionModel，这里作为 inner/outer 共同使用的基础模型
    det_model = trainer.model.to(device)

    # 使用训练 dataloader 作为 inner batch 来源，验证 dataloader 作为 meta 集
    train_loader = trainer.train_loader
    meta_loader = trainer.test_loader
    if train_loader is None:
        raise RuntimeError("构建 train dataloader 失败：trainer.train_loader 为空")
    if meta_loader is None:
        raise RuntimeError("构建 meta dataloader 失败：trainer.test_loader 为空")

    print(
        f"🧪 开始阶段二 (MWNet-style, multi-step) g_phi 训练：scenario={args.scenario}, seed={args.seed}, "
        f"θ={theta_ckpt_path}, train_batches≈{len(train_loader)}, meta_batches≈{len(meta_loader)}"
    )

    # 构建 MLP FiLTER 和 per-sample loss 封装
    mlp_filter = MLPFilter().to(device)

    # 为 meta 阶段同样提供锚点模型 θ_good，用于构造与 θ_good 预测差异相关的特征
    anchor = None
    try:
        from ICML.core.yolo_bias_finetune.anchor import AnchorModel  # 局部导入避免循环依赖

        anchor = AnchorModel(cfg.theta_good_path, device=device)
    except Exception:
        anchor = None

    per_sample_loss = PerSampleWeightedDetectionLoss(
        det_model,
        mlp_filter,
        anchor_model=anchor,
    )

    optimizer_phi = torch.optim.Adam(mlp_filter.parameters(), lr=args.lr_phi)

    # meta 训练循环：只更新 φ，不更新 θ（det_model）
    # 使用多步近似的 MWNet：在当前 θ 上执行若干步虚拟更新 θ_k，然后在 meta 集上评估 loss(θ_K)，对 φ 反传。
    inner_lr = args.inner_lr  # 内层虚拟更新步长
    inner_steps = max(int(args.inner_steps), 1)  # 虚拟 inner GD 步数（K）
    train_iter = iter(train_loader)
    meta_iter = iter(meta_loader)

    for epoch in range(args.meta_epochs):
        print(f"\n[Meta-Epoch {epoch + 1}/{args.meta_epochs}]")
        step = 0
        running_loss = 0.0

        while step < args.meta_steps:
            if step >= args.meta_steps:
                break

            step += 1
            try:
                batch_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_train = next(train_iter)

            try:
                batch_meta = next(meta_iter)
            except StopIteration:
                meta_iter = iter(meta_loader)
                batch_meta = next(meta_iter)

            batch_train = trainer.preprocess_batch(batch_train)
            batch_meta = trainer.preprocess_batch(batch_meta)

            # 1) inner：在 train loader 上执行 K 步虚拟更新，得到 θ_K
            det_model.train()
            base_named_params = dict(det_model.named_parameters())
            trainable_names = [n for n, p in base_named_params.items() if p.requires_grad]

            # θ_0：以当前 det_model 参数为起点
            theta_k = {n: p for n, p in base_named_params.items()}

            for _ in range(inner_steps):
                # 为了更接近真实 MWNet，每一步 inner 更新使用一个新的 train batch
                try:
                    batch_train = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch_train = next(train_iter)
                batch_train = trainer.preprocess_batch(batch_train)

                # 使用当前 θ_k 进行一次前向与损失计算
                preds_train = functional_call(det_model, theta_k, (batch_train["img"],))
                loss_vec_train, _ = per_sample_loss(preds_train, batch_train)
                inner_loss = loss_vec_train.sum()

                # 对 θ_k 中需要梯度的参数计算一阶梯度
                theta_tensors = [theta_k[n] for n in trainable_names]
                grads_theta = torch.autograd.grad(inner_loss, theta_tensors, create_graph=True)

                # 构造 θ_{k+1}：GD 一步更新
                new_theta_k = {}
                grad_iter = iter(grads_theta)
                for name, p in theta_k.items():
                    if name in trainable_names:
                        g = next(grad_iter)
                        new_theta_k[name] = p - inner_lr * g
                    else:
                        new_theta_k[name] = p
                theta_k = new_theta_k

            # 2) outer：在 meta batch 上使用 θ_K 计算“更接近 mAP 的” meta-loss，并对 φ 反传
            det_model.eval()
            preds_meta = functional_call(det_model, theta_k, (batch_meta["img"],))
            loss_vec_meta, _ = per_sample_loss(preds_meta, batch_meta)

            # loss_vec_meta: [box, cls, dfl]
            # 将分类部分作为主 meta 指标，辅以较小权重的 box/dfl，作为 mAP 的可导 proxy
            loss_box_meta, loss_cls_meta, loss_dfl_meta = loss_vec_meta
            det_loss = loss_cls_meta + 0.5 * (loss_box_meta + loss_dfl_meta)

            # keep-rate 正则：鼓励 w_i 不偏离 1 太远，避免极端 re-weight
            reg = 0.0
            if per_sample_loss.last_weights is not None:
                w = per_sample_loss.last_weights
                reg = args.lambda_keep_rate * ((w - 1.0) ** 2).mean()

            meta_loss = det_loss + reg

            optimizer_phi.zero_grad()
            meta_loss.backward()
            optimizer_phi.step()

            running_loss += float(meta_loss.detach().cpu().item())

            if step % 10 == 0:
                avg = running_loss / step
                print(f"  [meta step {step}/{args.meta_steps}] meta_loss={avg:.4f}")

        avg_epoch_loss = running_loss / max(step, 1)
        print(f"✅ Meta-Epoch {epoch + 1} 完成，平均 meta_loss={avg_epoch_loss:.4f}")

    # 保存 g_phi (MLP FiLTER) 的参数
    out_dir = TOY_ROOT / "Results" / "Bias+Filter" / args.scenario / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlpfilter_meta.pt"
    torch.save(mlp_filter.state_dict(), out_path)
    print(f"\n💾 已保存 MLP FiLTER (g_phi) 参数到: {out_path}")


if __name__ == "__main__":
    main()
