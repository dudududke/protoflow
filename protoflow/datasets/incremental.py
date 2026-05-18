from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset

from protoflow.config import step_info
from protoflow.datasets.generic import SegSample, SegmentationFolderDataset, build_samples_from_cfg, read_split_file
from protoflow.datasets.transforms import JointTransform, TransformConfig


class IncrementalLabelMapper:
    def __init__(
        self,
        known_classes: Sequence[int],
        visible_classes: Sequence[int],
        ignore_index: int = 255,
        background_id: int | None = None,
        map_unseen_to_background: bool = True,
    ):
        self.known_classes = list(known_classes)
        self.visible_classes = set(int(c) for c in visible_classes)
        self.ignore_index = int(ignore_index)
        self.background_id = background_id
        self.map_unseen_to_background = map_unseen_to_background
        self.global_to_local = {int(c): i for i, c in enumerate(self.known_classes)}
        if background_id is not None and background_id in self.global_to_local:
            self.background_local = self.global_to_local[background_id]
        else:
            self.background_local = None

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(mask, fill_value=self.ignore_index)
        valid = mask != self.ignore_index
        for gid in self.visible_classes:
            if gid in self.global_to_local:
                out[(mask == gid) & valid] = self.global_to_local[gid]
        unseen = valid & (out == self.ignore_index)
        if self.map_unseen_to_background and self.background_local is not None:
            out[unseen] = self.background_local
        return out


class IncrementalSegmentationDataset(Dataset):
    def __init__(
        self,
        base_dataset: SegmentationFolderDataset,
        known_classes: Sequence[int],
        visible_classes: Sequence[int],
        ignore_index: int = 255,
        background_id: int | None = None,
        map_unseen_to_background: bool = True,
    ):
        self.base_dataset = base_dataset
        self.known_classes = list(known_classes)
        self.visible_classes = list(visible_classes)
        self.mapper = IncrementalLabelMapper(
            known_classes=known_classes,
            visible_classes=visible_classes,
            ignore_index=ignore_index,
            background_id=background_id,
            map_unseen_to_background=map_unseen_to_background,
        )

    @property
    def samples(self) -> list[SegSample]:
        return self.base_dataset.samples

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        item["mask_global"] = item["mask"].clone()
        item["mask"] = self.mapper(item["mask"])
        item["known_classes"] = torch.tensor(self.known_classes, dtype=torch.long)
        return item


def make_transform(dataset_cfg: dict, train: bool) -> JointTransform:
    aug = dataset_cfg.get("augmentation", {})
    tcfg = TransformConfig(
        crop_size=int(aug.get("crop_size", dataset_cfg.get("crop_size", 512))),
        min_scale=float(aug.get("min_scale", 0.5)),
        max_scale=float(aug.get("max_scale", 2.0)),
        hflip_prob=float(aug.get("hflip_prob", 0.5)),
        vflip_prob=float(aug.get("vflip_prob", 0.5)),
        rotate90=bool(aug.get("rotate90", True)),
        color_jitter=float(aug.get("color_jitter", 0.2 if train else 0.0)),
        imagenet_norm=bool(dataset_cfg.get("imagenet_norm", True)),
    )
    return JointTransform(tcfg, train=train, ignore_index=int(dataset_cfg.get("ignore_index", 255)))


def _visible_classes_for_step(cfg: dict, step: int) -> list[int]:
    info = step_info(cfg, step)
    mode = cfg["protocol"].get("label_visibility", "cumulative")
    if cfg["protocol"].get("type") == "domain_incremental":
        return info.known_classes
    if mode == "cumulative":
        return info.known_classes
    if mode == "current":
        return info.current_classes
    if mode == "all":
        return list(range(int(cfg["dataset"]["num_classes"])))
    raise ValueError("protocol.label_visibility must be one of: cumulative, current, all")


def build_step_dataset(cfg: dict, step: int, train: bool, samples: Sequence[SegSample] | None = None) -> IncrementalSegmentationDataset:
    dcfg = cfg["dataset"]
    info = step_info(cfg, step)
    if samples is None:
        if cfg["protocol"].get("type") == "domain_incremental" and train:
            dom = cfg["protocol"]["domains"][step]
            split_key = dom.get("train_split", "train_split")
            domain_name = dom.get("name", f"step{step}")
            base_samples = build_samples_from_cfg(dcfg, split_key, domain=domain_name)
        else:
            split_key = "train_split" if train else dcfg.get("eval_split_key", "val_split")
            base_samples = build_samples_from_cfg(dcfg, split_key, domain="eval" if not train else "train")
    else:
        base_samples = list(samples)
    transform = make_transform(dcfg, train=train)
    base = SegmentationFolderDataset(
        root=dcfg["root"],
        samples=base_samples,
        transform=transform,
        use_nir_as_red=bool(dcfg.get("use_nir_as_red", False)),
    )
    visible = _visible_classes_for_step(cfg, step)
    return IncrementalSegmentationDataset(
        base_dataset=base,
        known_classes=info.known_classes,
        visible_classes=visible,
        ignore_index=int(dcfg.get("ignore_index", 255)),
        background_id=dcfg.get("background_id", None),
        map_unseen_to_background=bool(dcfg.get("map_unseen_to_background", False)),
    )


def collate_batch(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    masks_global = torch.stack([b["mask_global"] for b in batch], dim=0)
    return {
        "image": images,
        "mask": masks,
        "mask_global": masks_global,
        "image_path": [b["image_path"] for b in batch],
        "mask_path": [b["mask_path"] for b in batch],
        "domain": [b["domain"] for b in batch],
    }


def samples_by_class(dataset: IncrementalSegmentationDataset, class_ids: Sequence[int]) -> dict[int, list[SegSample]]:
    result: dict[int, list[SegSample]] = {int(c): [] for c in class_ids}
    for idx in range(len(dataset.base_dataset.samples)):
        item = dataset.base_dataset[idx]
        mask = item["mask"]
        for local_id, global_id in enumerate(dataset.known_classes):
            if global_id in result and torch.any(mask == global_id):
                result[global_id].append(dataset.base_dataset.samples[idx])
    return result
