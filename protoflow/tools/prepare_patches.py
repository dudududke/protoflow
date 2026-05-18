from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_files(path: Path) -> list[Path]:
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    stem = image_path.stem
    candidates = []
    for ext in IMAGE_EXTS:
        candidates.append(mask_dir / f"{stem}{ext}")
        candidates.append(mask_dir / f"{stem}_mask{ext}")
        candidates.append(mask_dir / f"{stem}_label{ext}")
    for c in candidates:
        if c.exists():
            return c
    matches = list(mask_dir.rglob(f"{stem}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No mask found for {image_path.name} in {mask_dir}")


def patch_grid(w: int, h: int, patch: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, w - patch + 1), stride))
    ys = list(range(0, max(1, h - patch + 1), stride))
    if not xs or xs[-1] != max(0, w - patch):
        xs.append(max(0, w - patch))
    if not ys or ys[-1] != max(0, h - patch):
        ys.append(max(0, h - patch))
    return [(x, y) for y in ys for x in xs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop large remote-sensing image/mask tiles into patches")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_root = Path(args.out_root)
    out_img_dir = out_root / "images" / args.split
    out_mask_dir = out_root / "masks" / args.split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    images = list_files(image_dir)
    for img_path in tqdm(images, desc=f"patchify-{args.split}"):
        mask_path = find_mask(mask_dir, img_path)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)
        w, h = img.size
        for pi, (x, y) in enumerate(patch_grid(w, h, args.patch_size, args.stride)):
            box = (x, y, x + args.patch_size, y + args.patch_size)
            img_patch = img.crop(box)
            mask_patch = mask.crop(box)
            if args.skip_empty:
                arr = np.array(mask_patch)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                valid = arr != args.ignore_index
                if valid.sum() == 0:
                    continue
            base = f"{img_path.stem}_{pi:04d}"
            img_rel = f"images/{args.split}/{base}.png"
            mask_rel = f"masks/{args.split}/{base}.png"
            img_patch.save(out_root / img_rel)
            mask_patch.save(out_root / mask_rel)
            lines.append(f"{img_rel} {mask_rel}\n")
    split_file = out_root / f"{args.split}.txt"
    with open(split_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {len(lines)} patches and split file {split_file}")


if __name__ == "__main__":
    main()
