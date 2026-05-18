from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = [
    (30, 30, 30),
    (220, 40, 40),
    (40, 180, 80),
    (40, 90, 220),
    (220, 180, 40),
    (180, 40, 220),
    (40, 200, 200),
]


def draw_sample(size: int, num_classes: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (size, size), COLORS[0])
    mask = Image.new("L", (size, size), 0)
    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    for cid in range(1, num_classes):
        x0 = rng.randint(0, size // 2)
        y0 = rng.randint(0, size // 2)
        x1 = rng.randint(size // 2, size - 1)
        y1 = rng.randint(size // 2, size - 1)
        shape = rng.choice(["rect", "ellipse", "poly"])
        color = tuple(min(255, max(0, c + rng.randint(-20, 20))) for c in COLORS[cid % len(COLORS)])
        if shape == "rect":
            draw_img.rectangle([x0, y0, x1, y1], fill=color)
            draw_mask.rectangle([x0, y0, x1, y1], fill=cid)
        elif shape == "ellipse":
            draw_img.ellipse([x0, y0, x1, y1], fill=color)
            draw_mask.ellipse([x0, y0, x1, y1], fill=cid)
        else:
            pts = [(rng.randint(0, size - 1), rng.randint(0, size - 1)) for _ in range(5)]
            draw_img.polygon(pts, fill=color)
            draw_mask.polygon(pts, fill=cid)
    arr = np.array(image).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), mask


def write_split(root: Path, split: str, n: int, size: int, num_classes: int, seed: int) -> None:
    rng = random.Random(seed)
    img_dir = root / "images" / split
    mask_dir = root / "masks" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        image, mask = draw_sample(size, num_classes, rng)
        img_rel = f"images/{split}/{split}_{i:04d}.png"
        mask_rel = f"masks/{split}/{split}_{i:04d}.png"
        image.save(root / img_rel)
        mask.save(root / mask_rel)
        lines.append(f"{img_rel} {mask_rel}\n")
    with open(root / f"{split}.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic segmentation dataset for ProtoFlow smoke tests")
    parser.add_argument("--root", required=True)
    parser.add_argument("--num-train", type=int, default=24)
    parser.add_argument("--num-val", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    write_split(root, "train", args.num_train, args.size, args.num_classes, args.seed)
    write_split(root, "val", args.num_val, args.size, args.num_classes, args.seed + 1)
    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
