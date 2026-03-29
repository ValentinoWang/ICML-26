"""
Run Gen0 twice with identical seeds and report whether metrics match.

Usage:
    python gen0_determinism_check.py --seed 1088 --data-root ./data
"""

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import torch
import torchvision

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp9_cifar10_setaware.run_exp9_cifar10_setaware import (  # noqa: E402
    set_seed,
    build_transforms,
    PseudoLabeledDataset,
    train_classifier,
    evaluate,
)


def split_labeled_unlabeled(dataset: torchvision.datasets.CIFAR10, per_class: int) -> Tuple[List[int], List[int]]:
    labeled: List[int] = []
    unlabeled: List[int] = []
    counter = [0] * 10
    for idx, (_, target) in enumerate(dataset):
        if counter[target] < per_class:
            labeled.append(idx)
            counter[target] += 1
        else:
            unlabeled.append(idx)
    return labeled, unlabeled


def run_gen0(seed: int, data_root: pathlib.Path, device: torch.device) -> Dict:
    set_seed(seed)
    train_tf, eval_tf = build_transforms()
    base_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
    labeled_idx, _ = split_labeled_unlabeled(base_train, per_class=250)

    label_map: Dict[int, int] = {}
    for idx in labeled_idx:
        _, target = base_train[idx]
        label_map[idx] = target

    model = torchvision.models.resnet18(num_classes=10).to(device)
    train_ds = PseudoLabeledDataset(base_train, labeled_idx, label_map, transform=train_tf)
    train_classifier(
        model,
        dataset=train_ds,
        epochs=20,
        device=device,
        lr=0.1,
        weight_decay=5e-4,
        batch_size=256,
        num_workers=4,
        desc="Gen0 determinism",
        grad_accum_steps=1,
        use_amp=True,
        seed=seed,
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)
    return evaluate(model, test_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Gen0 determinism by running twice with same seed.")
    parser.add_argument("--seed", type=int, default=1088)
    parser.add_argument("--data-root", type=pathlib.Path, default=ROOT / "data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = run_gen0(args.seed, args.data_root, device)
    m2 = run_gen0(args.seed, args.data_root, device)

    print(json.dumps({"run1": m1, "run2": m2}, indent=2))
    same = m1 == m2
    print(f"Deterministic: {same}")


if __name__ == "__main__":
    main()
