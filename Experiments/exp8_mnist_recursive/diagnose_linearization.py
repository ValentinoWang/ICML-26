import argparse
import json
import pathlib
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUT_DIR_DEFAULT = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name / "diagnostics"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp8_mnist_recursive.run_exp8_mnist_recursive import (  # noqa: E402
    build_labels,
    load_digits,
    make_candidates,
    rotate_batch,
    set_seed,
)
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss  # noqa: E402
from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402


def _safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec) + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (_safe_norm(a) * _safe_norm(b)))


def _relative_error(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.linalg.norm(ref - est) / _safe_norm(ref))


def _compute_metrics(
    candidates_img: np.ndarray,
    theta_star: np.ndarray,
    weights: np.ndarray,
    embeddings: np.ndarray,
    delta_emb_pred: np.ndarray,
    pca_components: np.ndarray,
) -> Dict[str, float]:
    # Define per-candidate gradient for L(theta)=0.5||theta-theta*||^2 in pixel space.
    grad_per = candidates_img - theta_star[None, :]
    grad_unweighted = grad_per.mean(axis=0)

    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        weights_norm = np.full_like(weights, 1.0 / len(weights))
    else:
        weights_norm = weights / w_sum
    grad_weighted = (weights_norm[:, None] * grad_per).sum(axis=0)

    # Linearized estimate from set statistics: g + J^T (weighted_mean - mean).
    emb_mean = embeddings.mean(axis=0)
    emb_weighted = (weights_norm[:, None] * embeddings).sum(axis=0)
    delta_emb_setstat = emb_weighted - emb_mean
    delta_pixel_setstat = pca_components.T @ delta_emb_setstat
    grad_linear_setstat = grad_unweighted + delta_pixel_setstat

    # Linearized estimate using bias head output: g - J^T Delta_phi_pred.
    delta_pixel_pred = pca_components.T @ delta_emb_pred
    grad_linear_pred = grad_unweighted - delta_pixel_pred

    return {
        "cos_sim_setstat": _cosine(grad_weighted, grad_linear_setstat),
        "rel_error_setstat": _relative_error(grad_weighted, grad_linear_setstat),
        "cos_sim_pred": _cosine(grad_weighted, grad_linear_pred),
        "rel_error_pred": _relative_error(grad_weighted, grad_linear_pred),
        "cos_unweighted": _cosine(grad_weighted, grad_unweighted),
        "rel_error_unweighted": _relative_error(grad_weighted, grad_unweighted),
        "grad_weighted_norm": _safe_norm(grad_weighted),
        "grad_unweighted_norm": _safe_norm(grad_unweighted),
        "delta_pixel_setstat_norm": _safe_norm(delta_pixel_setstat),
        "delta_emb_setstat_norm": _safe_norm(delta_emb_setstat),
        "delta_pixel_pred_norm": _safe_norm(delta_pixel_pred),
        "delta_emb_pred_norm": _safe_norm(delta_emb_pred),
    }


def run_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    target_gens: Sequence[int],
    data: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
    rng = set_seed(seed)
    theta_star = data["theta_star"][args.digit]
    pca = data["pca"]
    theta_no = pca.transform(theta_star[None, ...]).astype(np.float32).squeeze(0)

    model = SetAwareBiasRobustFilter(
        dim=args.pca_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_star_t = torch.from_numpy(theta_no[None, None, :]).float().to(device)

    rows: List[Dict[str, float]] = []
    for gen in range(args.generations):
        theta_no_img = pca.inverse_transform(theta_no)
        imgs = np.tile(theta_no_img, (args.batch_size, 1))
        imgs_rot = rotate_batch(imgs, angle=args.drift_deg)
        theta_hat = imgs_rot.mean(axis=0)
        theta_hat_proj = pca.transform(theta_hat[None, ...]).astype(np.float32).squeeze(0)
        theta_no = theta_hat_proj

        candidates_img = make_candidates(rng, theta_hat, args.candidates_per_gen, args.candidate_noise)
        labels = build_labels(candidates_img, theta_star, top_ratio=args.top_ratio)
        c_proj = pca.transform(candidates_img).astype(np.float32)

        x = torch.from_numpy(c_proj[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        weights, delta = model(x)
        if gen in target_gens:
            metrics = _compute_metrics(
                candidates_img=candidates_img,
                theta_star=theta_star,
                weights=weights.detach().cpu().numpy().squeeze(0),
                embeddings=c_proj,
                delta_emb_pred=delta.detach().cpu().numpy().squeeze(0),
                pca_components=pca.components_,
            )
            metrics.update(
                {
                    "seed": float(seed),
                    "generation": float(gen),
                    "n_candidates": float(args.candidates_per_gen),
                    "pca_dim": float(args.pca_dim),
                }
            )
            rows.append(metrics)

        theta_new = delta
        loss = (
            args.lambda_class * classification_loss(weights, y)
            + args.lambda_contract * contraction_loss(theta_new, theta_star_t)
            + args.lambda_ess * ess_loss(weights, tau=args.tau)
            + args.lambda_reg * correction_reg(delta)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    return rows


def summarize(rows: List[Dict[str, float]], target_gens: Sequence[int]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for gen in target_gens:
        filtered = [r for r in rows if int(r["generation"]) == int(gen)]
        if not filtered:
            continue
        summary[str(gen)] = {}
        for key in [
            "cos_sim_setstat",
            "rel_error_setstat",
            "cos_sim_pred",
            "rel_error_pred",
            "cos_unweighted",
            "rel_error_unweighted",
        ]:
            vals = np.array([r[key] for r in filtered], dtype=np.float64)
            summary[str(gen)][f"{key}_mean"] = float(vals.mean())
            summary[str(gen)][f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return summary


def write_csv(rows: List[Dict[str, float]], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "seed",
        "generation",
        "n_candidates",
        "pca_dim",
        "cos_sim_setstat",
        "rel_error_setstat",
        "cos_sim_pred",
        "rel_error_pred",
        "cos_unweighted",
        "rel_error_unweighted",
        "grad_weighted_norm",
        "grad_unweighted_norm",
        "delta_pixel_setstat_norm",
        "delta_emb_setstat_norm",
        "delta_pixel_pred_norm",
        "delta_emb_pred_norm",
    ]
    with out_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]:.6f}" for k in keys) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose linearized gradient correction on Exp8 (MNIST).")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-noise", type=float, default=0.1)
    parser.add_argument("--candidates-per-gen", type=int, default=128)
    parser.add_argument("--drift-deg", type=float, default=5.0)
    parser.add_argument("--top-ratio", type=float, default=0.3)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--digit", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1088, 2195, 4960])
    parser.add_argument("--pca-seed", type=int, default=1088)
    parser.add_argument("--target-gens", type=str, default="0,4")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_gens = [int(x.strip()) for x in args.target_gens.split(",") if x.strip()]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data = load_digits(seed=args.pca_seed, pca_dim=args.pca_dim)

    all_rows: List[Dict[str, float]] = []
    for seed in args.seeds:
        rows = run_seed(args, seed=seed, device=device, target_gens=target_gens, data=data)
        all_rows.extend(rows)

    summary = summarize(all_rows, target_gens)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exp8_linearization_g0_g4.csv"
    json_path = out_dir / "exp8_linearization_g0_g4.json"
    write_csv(all_rows, csv_path)
    clean_args = {}
    for key, value in vars(args).items():
        clean_args[key] = str(value) if isinstance(value, pathlib.Path) else value
    payload = {"args": clean_args, "summary": summary, "rows": all_rows}
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved Exp8 linearization diagnostics to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
