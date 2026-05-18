from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegSample:
    image: str
    mask: str
    domain: str = "default"


def read_split_file(root: str, split_file: str, domain: str = "default") -> list[SegSample]:
    path = split_file if os.path.isabs(split_file) else os.path.join(root, split_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    samples: list[SegSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid split line {line_no} in {path}: expected image and mask paths")
            img, mask = parts[0], parts[1]
            dom = parts[2] if len(parts) >= 3 else domain
            samples.append(SegSample(image=img, mask=mask, domain=dom))
    if not samples:
        raise ValueError(f"Split file contains no samples: {path}")
    return samples


class SegmentationFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        samples: Sequence[SegSample],
        transform: Optional[Callable] = None,
        use_nir_as_red: bool = False,
    ):
        self.root = root
        self.samples = list(samples)
        self.transform = transform
        self.use_nir_as_red = use_nir_as_red

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, rel_or_abs: str) -> str:
        return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(self.root, rel_or_abs)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if self.use_nir_as_red and img.mode in {"RGBA", "CMYK"}:
            arr = img.convert("RGBA")
            r, g, b, a = arr.split()
            img = Image.merge("RGB", (a, r, g))
        else:
            img = img.convert("RGB")
        return img

    def _load_mask(self, path: str) -> Image.Image:
        return Image.open(path)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self._resolve(sample.image)
        mask_path = self._resolve(sample.mask)
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {
            "image": image,
            "mask": mask,
            "image_path": image_path,
            "mask_path": mask_path,
            "domain": sample.domain,
        }


def build_samples_from_cfg(dataset_cfg: dict, split_key: str, domain: str = "default") -> list[SegSample]:
    root = dataset_cfg["root"]
    split_file = dataset_cfg.get(split_key)
    if split_file is None:
        raise KeyError(f"dataset.{split_key} is required")
    return read_split_file(root, split_file, domain=domain)
