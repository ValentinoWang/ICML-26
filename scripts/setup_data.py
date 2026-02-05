#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _maybe_import_torchvision() -> object:
    try:
        import torchvision  # type: ignore

        return torchvision
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency for vision datasets.\n"
            "Install torchvision first, e.g.:\n"
            "  pip install torch torchvision\n"
            f"Original error: {exc}"
        ) from exc


def _maybe_import_datasets() -> object:
    try:
        import datasets  # type: ignore

        return datasets
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency for HuggingFace datasets.\n"
            "Install datasets first, e.g.:\n"
            "  pip install datasets\n"
            f"Original error: {exc}"
        ) from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_cifar10(torchvision: object, data_root: Path) -> None:
    print(f"[cifar10] data_root={data_root}")
    ensure_dir(data_root)
    torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=None)
    torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=None)
    print("[cifar10] done")


def download_mnist(torchvision: object, data_root: Path) -> None:
    print(f"[mnist] data_root={data_root}")
    ensure_dir(data_root)
    torchvision.datasets.MNIST(root=str(data_root), train=True, download=True, transform=None)
    torchvision.datasets.MNIST(root=str(data_root), train=False, download=True, transform=None)
    print("[mnist] done")


def download_wikitext(datasets: object, hf_home: Path) -> None:
    """Prefetch Wikitext-103 via HuggingFace `datasets`."""

    print(f"[wikitext] hf_home={hf_home}")
    ensure_dir(hf_home)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))

    # Load a small split first (validation) to validate connectivity/cache.
    datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    # Then load train to ensure the full dataset is present.
    datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    print("[wikitext] done")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local dataset directories for this repo.\n\n"
            "Note: DataSet/ and Data-pre/ are LOCAL directories and are not pushed to GitHub.\n"
            "This script downloads common public datasets used by experiments (MNIST/CIFAR-10),\n"
            "and can optionally prefetch Wikitext-103 via HuggingFace datasets.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root (default: inferred from this script path).",
    )
    parser.add_argument(
        "--exp-data-root",
        type=Path,
        default=None,
        help="Where torchvision datasets are stored (default: <repo-root>/Experiments/data).",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="HF_HOME directory used for HuggingFace caches (default: <repo-root>/.hf_cache).",
    )
    parser.add_argument("--cifar10", action="store_true", help="Download CIFAR-10 (used by exp9).")
    parser.add_argument("--mnist", action="store_true", help="Download MNIST (used by exp8).")
    parser.add_argument("--wikitext", action="store_true", help="Prefetch Wikitext-103 (used by exp11).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download everything: --cifar10 --mnist --wikitext",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    exp_data_root: Path = (args.exp_data_root or (repo_root / "Experiments" / "data")).resolve()
    hf_home: Path = (args.hf_home or (repo_root / ".hf_cache")).resolve()

    # Always ensure the local data directories exist (placeholders are committed).
    ensure_dir(repo_root / "DataSet")
    ensure_dir(repo_root / "Data-pre")

    if args.all:
        args.cifar10 = True
        args.mnist = True
        args.wikitext = True

    if not (args.cifar10 or args.mnist or args.wikitext):
        print("Nothing selected. Choose one of: --cifar10/--mnist/--wikitext or use --all.", file=sys.stderr)
        return 2

    if args.cifar10 or args.mnist:
        torchvision = _maybe_import_torchvision()
        if args.cifar10:
            download_cifar10(torchvision, exp_data_root)
        if args.mnist:
            download_mnist(torchvision, exp_data_root)

    if args.wikitext:
        datasets = _maybe_import_datasets()
        download_wikitext(datasets, hf_home)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
