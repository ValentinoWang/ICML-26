"""
Create a stacked figure combining exp8 curves and visual grid for the paper.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
FIGURES_DIR = BASE_FIGURES_DIR / "results"

def load_image(path: pathlib.Path):
    return Image.open(path)


def compose(curves_path: pathlib.Path, grid_path: pathlib.Path, out_path: pathlib.Path) -> None:
    img_curves = load_image(curves_path)
    img_grid = load_image(grid_path)

    # Resize to same width
    target_width = max(img_curves.width, img_grid.width)
    def resize_width(img):
        if img.width == target_width:
            return img
        ratio = target_width / img.width
        new_size = (target_width, int(img.height * ratio))
        return img.resize(new_size, Image.LANCZOS)

    img_curves = resize_width(img_curves)
    img_grid = resize_width(img_grid)

    sep = 30  # pixels
    total_height = img_curves.height + sep + img_grid.height
    canvas = Image.new("RGB", (target_width, total_height), color=(255, 255, 255))
    canvas.paste(img_curves, (0, 0))
    canvas.paste(img_grid, (0, img_curves.height + sep))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=pathlib.Path, default=FIGURES_DIR / "exp8_curves.png")
    parser.add_argument("--grid", type=pathlib.Path, default=FIGURES_DIR / "exp8_visual_grid.png")
    parser.add_argument("--out", type=pathlib.Path, default=FIGURES_DIR / "exp8_enhence.png")
    args = parser.parse_args()
    compose(args.curves, args.grid, args.out)


if __name__ == "__main__":
    main()
