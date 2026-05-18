from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


@dataclass
class TransformConfig:
    crop_size: int = 512
    min_scale: float = 0.5
    max_scale: float = 2.0
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate90: bool = True
    color_jitter: float = 0.2
    imagenet_norm: bool = True


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resize_pair(image: Image.Image, mask: Image.Image, size: Tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    return image.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST)


def _pad_if_needed(image: Image.Image, mask: Image.Image, target: int, ignore_index: int) -> tuple[Image.Image, Image.Image]:
    w, h = image.size
    pad_w = max(0, target - w)
    pad_h = max(0, target - h)
    if pad_w == 0 and pad_h == 0:
        return image, mask
    new_img = Image.new(image.mode, (w + pad_w, h + pad_h), color=0)
    new_img.paste(image, (0, 0))
    new_mask = Image.new(mask.mode, (w + pad_w, h + pad_h), color=ignore_index)
    new_mask.paste(mask, (0, 0))
    return new_img, new_mask


def _random_crop_pair(image: Image.Image, mask: Image.Image, crop_size: int, ignore_index: int) -> tuple[Image.Image, Image.Image]:
    image, mask = _pad_if_needed(image, mask, crop_size, ignore_index)
    w, h = image.size
    if w == crop_size and h == crop_size:
        return image, mask
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    box = (left, top, left + crop_size, top + crop_size)
    return image.crop(box), mask.crop(box)


def _center_crop_pair(image: Image.Image, mask: Image.Image, crop_size: int, ignore_index: int) -> tuple[Image.Image, Image.Image]:
    image, mask = _pad_if_needed(image, mask, crop_size, ignore_index)
    w, h = image.size
    left = max(0, (w - crop_size) // 2)
    top = max(0, (h - crop_size) // 2)
    box = (left, top, left + crop_size, top + crop_size)
    return image.crop(box), mask.crop(box)


def _color_jitter(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        image = enhancer_cls(image).enhance(factor)
    return image


def image_to_tensor(image: Image.Image, imagenet_norm: bool = True) -> torch.Tensor:
    arr = np.array(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if imagenet_norm:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr).long()


class JointTransform:
    def __init__(self, cfg: TransformConfig, train: bool, ignore_index: int = 255):
        self.cfg = cfg
        self.train = train
        self.ignore_index = ignore_index

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            scale = random.uniform(self.cfg.min_scale, self.cfg.max_scale)
            w, h = image.size
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image, mask = _resize_pair(image, mask, (new_w, new_h))
            image, mask = _random_crop_pair(image, mask, self.cfg.crop_size, self.ignore_index)
            if random.random() < self.cfg.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < self.cfg.vflip_prob:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if self.cfg.rotate90:
                k = random.randint(0, 3)
                if k:
                    image = image.rotate(90 * k, expand=True)
                    mask = mask.rotate(90 * k, expand=True)
            image = _color_jitter(image, self.cfg.color_jitter)
        else:
            if self.cfg.crop_size > 0:
                image, mask = _center_crop_pair(image, mask, self.cfg.crop_size, self.ignore_index)
        return image_to_tensor(image, self.cfg.imagenet_norm), mask_to_tensor(mask)


def resize_logits_to_label(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
