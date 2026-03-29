import os
import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple


def _sanitize_thread_env_var(name: str, default: int = 1) -> None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        os.environ[name] = str(default)
        return
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        os.environ[name] = str(default)
        return
    if value <= 0:
        os.environ[name] = str(default)


for _env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _sanitize_thread_env_var(_env_name)

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp9_cifar10_setaware import run_exp9_cifar10_setaware as cifar9  # noqa: E402


DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "cifar10"
DEFAULT_FRACTION_SWEEP_ALPHAS = [0.5, 0.75, 0.9, 0.975, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuttal CIFAR-10 baselines for direct clean usage and batch-mean aggregation.")
    parser.add_argument("--data-root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["set_aware", "clean_only", "clean_ft", "clean_mix", "batch_mean"],
        choices=["baseline", "set_aware", "clean_only", "clean_ft", "clean_mix", "batch_mean", "fraction_sweep"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1088, 2195, 4960])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--per-gen-add", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gen0-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--direct-ft-epochs", type=int, default=30)
    parser.add_argument("--clean-batch-size", type=int, default=32)
    parser.add_argument("--lr-gen0", type=float, default=0.1)
    parser.add_argument("--lr-finetune", type=float, default=0.02)
    parser.add_argument("--lr-direct-ft", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--baseline-conf-threshold", type=float, default=0.0)
    parser.add_argument("--filter-hidden", type=int, default=256)
    parser.add_argument("--filter-heads", type=int, default=4)
    parser.add_argument("--filter-layers", type=int, default=2)
    parser.add_argument("--filter-dropout", type=float, default=0.1)
    parser.add_argument("--filter-steps", type=int, default=200)
    parser.add_argument("--filter-set-size", type=int, default=2048)
    parser.add_argument("--filter-candidate-pool", type=int, default=12000)
    parser.add_argument("--filter-lr", type=float, default=1e-3)
    parser.add_argument("--filter-wd", type=float, default=1e-4)
    parser.add_argument("--filter-tau", type=float, default=0.2)
    parser.add_argument("--delta-phi-scale", type=float, default=1.0)
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--lambda-proto", type=float, default=0.0)
    parser.add_argument("--lambda-balance", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--proto-topk", type=int, default=200)
    parser.add_argument("--proto-temp", type=float, default=0.2)
    parser.add_argument("--proto-conf-power", type=float, default=1.0)
    parser.add_argument("--set-aware-score-mode", choices=["weight_conf", "weight"], default="weight_conf")
    parser.add_argument(
        "--set-aware-threshold-mode",
        choices=["confidence", "margin_quantile", "score_topk"],
        default="score_topk",
    )
    parser.add_argument("--set-aware-margin-quantile", type=float, default=0.5)
    parser.add_argument("--set-aware-score-floor", type=float, default=0.4)
    parser.add_argument("--set-aware-per-class-k", type=int, default=0)
    parser.add_argument("--set-aware-conf-threshold", type=float, default=0.0)
    parser.add_argument("--set-aware-balance-alpha", type=float, default=0.3)

    parser.add_argument("--meta-clean-val", action="store_true", default=True)
    parser.add_argument("--meta-lambda", type=float, default=1.0)
    parser.add_argument("--meta-inner-lr", type=float, default=0.1)
    parser.add_argument("--meta-every", type=int, default=10)
    parser.add_argument("--meta-set-size", type=int, default=256)
    parser.add_argument("--meta-update-scope", choices=["fc", "all"], default="fc")

    parser.add_argument("--clean-val-source", choices=["train_holdout"], default="train_holdout")
    parser.add_argument("--clean-val-noise-rate", type=float, default=0.0)
    parser.add_argument("--clean-val-noise-seed", type=int, default=0)
    parser.add_argument("--clean-set-size", type=int, default=100)
    parser.add_argument("--clean-val-size", type=int, default=100)
    parser.add_argument("--clean-set-seed", type=int, default=0)
    parser.add_argument("--clean-val-seed", type=int, default=0)
    parser.add_argument("--clean-val-strategy", choices=["stratified", "random"], default="stratified")
    parser.add_argument(
        "--fraction-sweep-alphas",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTION_SWEEP_ALPHAS,
        help="Synthetic fraction alpha for CIFAR fraction sweep, e.g. 0.5 0.75 0.9 0.975 1.0.",
    )

    parser.add_argument("--batch-mean-scale", type=float, default=0.35)
    parser.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def select_clean_indices(
    base_train: torchvision.datasets.CIFAR10,
    unlabeled_idx: List[int],
    args: argparse.Namespace,
    clean_size: int,
) -> List[int]:
    clean_size = min(int(clean_size), len(unlabeled_idx))
    if clean_size <= 0:
        return []
    if args.clean_val_strategy == "stratified":
        return cifar9.stratified_sample_from_indices(
            indices=unlabeled_idx,
            targets=getattr(base_train, "targets", []),
            total=clean_size,
            seed=int(args.clean_set_seed),
            num_classes=10,
        )
    rng = np.random.default_rng(int(args.clean_set_seed))
    return rng.choice(np.asarray(unlabeled_idx, dtype=np.int64), size=clean_size, replace=False).tolist()


def validate_fraction_alphas(alphas: List[float]) -> List[float]:
    out: List[float] = []
    for alpha in alphas:
        a = float(alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError(f"fraction sweep alpha must lie in (0, 1], got {alpha}")
        out.append(a)
    return out


def alpha_to_method(alpha: float) -> str:
    pct = float(alpha) * 100.0
    if abs(pct - round(pct)) < 1e-8:
        return f"frac_{int(round(pct))}"
    text = f"{pct:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"frac_{text}"


def method_to_alpha(method: str) -> float | None:
    if not method.startswith("frac_"):
        return None
    text = method[len("frac_") :].replace("p", ".")
    try:
        pct = float(text)
    except ValueError:
        return None
    return pct / 100.0


def fraction_counts(alpha: float, total_size: int) -> Tuple[int, int]:
    pseudo_count = int(round(float(alpha) * int(total_size)))
    pseudo_count = max(0, min(int(total_size), pseudo_count))
    clean_count = int(total_size) - pseudo_count
    return pseudo_count, clean_count


def expand_modes(modes: List[str], args: argparse.Namespace) -> List[str]:
    expanded: List[str] = []
    alphas = validate_fraction_alphas(list(args.fraction_sweep_alphas))
    for mode in modes:
        if mode == "fraction_sweep":
            expanded.extend(alpha_to_method(alpha) for alpha in alphas)
        else:
            expanded.append(mode)
    return expanded


def build_test_loader(test_set: Dataset, args: argparse.Namespace, seed: int) -> DataLoader:
    dl_seed = cifar9.make_dataloader_seed(seed + 2025)
    return DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=dl_seed["worker_init_fn"],
        generator=dl_seed["generator"],
    )


def fit_classifier(
    model: torch.nn.Module,
    dataset: Dataset,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    desc: str,
    grad_accum_steps: int,
    use_amp: bool,
    seed: int,
) -> None:
    effective_batch = max(1, min(int(batch_size), len(dataset)))
    cifar9.train_classifier(
        model=model,
        dataset=dataset,
        epochs=epochs,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=effective_batch,
        num_workers=num_workers,
        desc=desc,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        seed=seed,
    )


def sigmoid_weights(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float64)
    centered = scores - float(scores.mean())
    scaled = centered / (float(scores.std()) + 1e-8)
    return 1.0 / (1.0 + np.exp(-scaled))


def select_batch_mean(
    preds: Dict[str, np.ndarray],
    top_k: int,
    candidate_pool: int,
    scale: float,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    conf = preds["confidences"]
    order = np.argsort(-conf)
    cand_top = order[: min(len(order), int(candidate_pool))]
    cand_features = preds["features"][cand_top].astype(np.float32)
    cand_labels = preds["pseudo_labels"][cand_top].astype(np.int64)
    cand_conf = conf[cand_top].astype(np.float32)
    cand_true = preds["true_labels"][cand_top].astype(np.int64)

    one_hot = np.eye(10, dtype=np.float32)[cand_labels]
    filter_input = np.concatenate([cand_features, one_hot, cand_conf[:, None]], axis=1)
    mean_vec = filter_input.mean(axis=0, keepdims=True)
    mean_norm = np.linalg.norm(mean_vec, axis=1, keepdims=True) + 1e-8
    mean_vec = mean_vec / mean_norm
    feat_norm = np.linalg.norm(filter_input, axis=1, keepdims=True) + 1e-8
    feat_unit = filter_input / feat_norm
    align = (feat_unit * mean_vec).sum(axis=1)

    scores = cand_conf.astype(np.float64) - float(scale) * align.astype(np.float64)
    weights = sigmoid_weights(scores)
    chosen = np.argsort(-scores, kind="mergesort")[:top_k]
    selected_indices = preds["indices"][cand_top][chosen].tolist()
    selected_labels = cand_labels[chosen].tolist()
    hist = np.bincount(cand_labels[chosen], minlength=10).astype(np.int64)
    ess = float((weights.sum() ** 2) / (np.sum(weights**2) + 1e-8))
    stats = cifar9.compute_selection_stats(
        true_labels=cand_true[chosen],
        pseudo_labels=cand_labels[chosen],
        confidences=cand_conf[chosen],
        num_classes=10,
        weights=weights[chosen],
        scores=scores[chosen],
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), "ess_score": ess, **stats}


def make_train_dataset(
    base_train: torchvision.datasets.CIFAR10,
    train_indices: List[int],
    label_map: Dict[int, int],
    transform,
) -> Dataset:
    return cifar9.PseudoLabeledDataset(base_train, train_indices, label_map, transform=transform)


def run_single_mode(seed: int, mode: str, args: argparse.Namespace) -> List[Dict]:
    cifar9.set_seed(seed)
    device = torch.device(args.device)
    train_tf, eval_tf = cifar9.build_transforms()
    base_train = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=None)
    labeled_idx, unlabeled_idx = cifar9.split_labeled_unlabeled(base_train, per_class=250)
    fraction_alpha = method_to_alpha(mode)
    if fraction_alpha is not None:
        max_clean_count = max(fraction_counts(alpha, args.per_gen_add)[1] for alpha in validate_fraction_alphas(list(args.fraction_sweep_alphas)))
        _, clean_count = fraction_counts(fraction_alpha, args.per_gen_add)
        clean_pool = select_clean_indices(base_train, unlabeled_idx, args, clean_size=max_clean_count)
        clean_indices = clean_pool[:clean_count]
    else:
        clean_indices = select_clean_indices(base_train, unlabeled_idx, args, clean_size=args.clean_set_size)
    clean_index_set = set(clean_indices)
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in clean_index_set]

    clean_eval_set = cifar9.CIFARSubset(base_train, clean_indices, transform=eval_tf, return_index=False)
    test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_tf)
    test_loader = build_test_loader(test_set, args, seed)

    label_map: Dict[int, int] = {}
    for idx in labeled_idx + clean_indices:
        _, target = base_train[idx]
        label_map[int(idx)] = int(target)

    model = torchvision.models.resnet18(num_classes=10).to(device)
    metrics: List[Dict] = []
    if mode == "clean_only":
        clean_train_ds = make_train_dataset(base_train, clean_indices, label_map, train_tf)
        fit_classifier(
            model=model,
            dataset=clean_train_ds,
            epochs=args.direct_ft_epochs,
            device=device,
            lr=args.lr_direct_ft,
            weight_decay=args.weight_decay,
            batch_size=args.clean_batch_size,
            num_workers=args.num_workers,
            desc=f"Clean-only train [seed={seed}]",
            grad_accum_steps=args.grad_accum_steps,
            use_amp=not args.no_amp,
            seed=seed,
        )
        eval_res = cifar9.evaluate(model, test_loader, device)
        final_row = {"train_size": len(clean_train_ds), **eval_res, "pseudo_label_hist": [0] * 10}
        metrics.append({"generation": 0, **final_row})
        for gen in range(1, args.generations + 1):
            metrics.append({"generation": gen, **final_row})
        return metrics

    gen0_ds = make_train_dataset(base_train, labeled_idx, label_map, train_tf)
    fit_classifier(
        model=model,
        dataset=gen0_ds,
        epochs=args.gen0_epochs,
        device=device,
        lr=args.lr_gen0,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc=f"Gen0 train [{mode}][seed={seed}]",
        grad_accum_steps=args.grad_accum_steps,
        use_amp=not args.no_amp,
        seed=seed,
    )

    eval_res = cifar9.evaluate(model, test_loader, device)
    metrics.append({"generation": 0, "train_size": len(gen0_ds), **eval_res, "pseudo_label_hist": [0] * 10})

    for gen in range(1, args.generations + 1):
        if mode == "clean_ft":
            clean_train_ds = make_train_dataset(base_train, clean_indices, label_map, train_tf)
            fit_classifier(
                model=model,
                dataset=clean_train_ds,
                epochs=args.direct_ft_epochs,
                device=device,
                lr=args.lr_direct_ft,
                weight_decay=args.weight_decay,
                batch_size=args.clean_batch_size,
                num_workers=args.num_workers,
                desc=f"Gen{gen} clean-ft [seed={seed}]",
                grad_accum_steps=args.grad_accum_steps,
                use_amp=not args.no_amp,
                seed=seed + gen,
            )
            eval_res = cifar9.evaluate(model, test_loader, device)
            metrics.append({"generation": gen, "train_size": len(clean_train_ds), **eval_res, "pseudo_label_hist": [0] * 10})
            continue

        unlabeled_ds = cifar9.CIFARSubset(base_train, unlabeled_idx, transform=eval_tf, return_index=True)
        preds = cifar9.collect_predictions(
            model=model,
            dataset=unlabeled_ds,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=seed + gen,
        )
        fraction_extra: Dict[str, float] = {}
        if fraction_alpha is not None:
            pseudo_count, clean_count = fraction_counts(fraction_alpha, args.per_gen_add)
            selected_indices, selected_labels, extra = cifar9.select_baseline(
                preds, top_k=pseudo_count, threshold=args.baseline_conf_threshold
            )
            fraction_extra = {
                "fraction_alpha": float(fraction_alpha),
                "fraction_clean_count": float(clean_count),
                "fraction_pseudo_count": float(pseudo_count),
            }
        elif mode == "baseline":
            selected_indices, selected_labels, extra = cifar9.select_baseline(
                preds, top_k=args.per_gen_add, threshold=args.baseline_conf_threshold
            )
        elif mode == "set_aware":
            selected_indices, selected_labels, extra = cifar9.select_set_aware(
                preds=preds,
                top_k=args.per_gen_add,
                threshold=args.set_aware_conf_threshold,
                device=device,
                args=args,
                base_train=base_train,
                eval_transform=eval_tf,
                classifier=model,
                clean_val_set=clean_eval_set,
                seed=seed + 10_000 + gen,
            )
        else:
            selected_indices, selected_labels, extra = select_batch_mean(
                preds=preds,
                top_k=args.per_gen_add,
                candidate_pool=args.filter_candidate_pool,
                scale=args.batch_mean_scale,
            )

        for idx, label in zip(selected_indices, selected_labels):
            label_map[int(idx)] = int(label)
        selected_set = set(selected_indices)
        unlabeled_idx = [idx for idx in unlabeled_idx if idx not in selected_set]

        if fraction_alpha is not None:
            train_indices = list(selected_indices) + clean_indices
        else:
            labeled_idx = labeled_idx + selected_indices
            train_indices = list(labeled_idx)
        if mode == "clean_mix":
            train_indices = train_indices + clean_indices
        train_ds = make_train_dataset(base_train, train_indices, label_map, train_tf)
        fit_classifier(
            model=model,
            dataset=train_ds,
            epochs=args.finetune_epochs,
            device=device,
            lr=args.lr_finetune,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc=f"Gen{gen} train [{mode}][seed={seed}]",
            grad_accum_steps=args.grad_accum_steps,
            use_amp=not args.no_amp,
            seed=seed + gen,
        )
        eval_res = cifar9.evaluate(model, test_loader, device)
        metrics.append({"generation": gen, "train_size": len(train_ds), **eval_res, **extra, **fraction_extra})
    return metrics


def save_seed_config(seed: int, args: argparse.Namespace, merged_path: pathlib.Path) -> None:
    config_path = merged_path.with_name(f"{merged_path.stem}_config.json")
    payload = {"seed": seed, "merged_path": str(merged_path), "args": vars(args)}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    args.fraction_sweep_alphas = validate_fraction_alphas(list(args.fraction_sweep_alphas))
    args.clean_val_size = args.clean_set_size
    args.clean_val_seed = args.clean_set_seed
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    expanded_modes = expand_modes(list(args.modes), args)

    for seed in args.seeds:
        merged_path = results_dir / f"rebuttal_cifar_seed{seed}_merged.csv"
        if cifar9.seed_results_complete(merged_path, modes=expanded_modes, generations=args.generations):
            print(f"Skip seed={seed}: found complete results at {merged_path}")
            continue
        if merged_path.exists() and not args.overwrite_results:
            raise FileExistsError(
                f"Refusing to overwrite existing outputs for seed={seed} under {results_dir}. "
                f"Delete {merged_path}, change --results-dir, or pass --overwrite-results."
            )

        merged_rows: List[Dict[str, float | int | str]] = []
        save_seed_config(seed, args, merged_path)
        for mode in expanded_modes:
            metrics = run_single_mode(seed=seed, mode=mode, args=args)
            merged_rows.extend(cifar9.metrics_to_rows(metrics, method=mode))
            cifar9.write_merged_csv(merged_rows, merged_path)
            print(f"Completed seed={seed}, mode={mode}")
        print(f"Saved merged results to {merged_path}")


if __name__ == "__main__":
    main()
