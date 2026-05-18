from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from protoflow.datasets.generic import SegSample
from protoflow.datasets.incremental import IncrementalSegmentationDataset, collate_batch
from protoflow.engine.prototype import l2norm


@dataclass
class ReplayMemory:
    budget_per_class: int = 20
    strategy: str = "random"
    samples: dict[int, list[SegSample]] = field(default_factory=dict)

    def all_samples(self) -> list[SegSample]:
        seen: dict[tuple[str, str], SegSample] = {}
        for vals in self.samples.values():
            for s in vals:
                seen[(s.image, s.mask)] = s
        return list(seen.values())

    def state_dict(self) -> dict:
        return {
            "budget_per_class": self.budget_per_class,
            "strategy": self.strategy,
            "samples": {
                int(k): [{"image": s.image, "mask": s.mask, "domain": s.domain} for s in v]
                for k, v in self.samples.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.budget_per_class = int(state.get("budget_per_class", self.budget_per_class))
        self.strategy = state.get("strategy", self.strategy)
        self.samples = {
            int(k): [SegSample(image=x["image"], mask=x["mask"], domain=x.get("domain", "memory")) for x in v]
            for k, v in state.get("samples", {}).items()
        }

    def _mask_contains_class(self, root: str, sample: SegSample, cid: int) -> bool:
        path = sample.mask if os.path.isabs(sample.mask) else os.path.join(root, sample.mask)
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return bool(np.any(mask == int(cid)))

    def _random_select(self, dataset: IncrementalSegmentationDataset, class_ids: Sequence[int]) -> None:
        root = dataset.base_dataset.root
        all_samples = list(dataset.base_dataset.samples)
        for cid in class_ids:
            candidates = [s for s in all_samples if self._mask_contains_class(root, s, int(cid))]
            random.shuffle(candidates)
            selected = candidates[: self.budget_per_class]
            previous = self.samples.get(int(cid), [])
            merged = previous + selected
            unique = []
            seen = set()
            for s in merged:
                key = (s.image, s.mask)
                if key not in seen:
                    unique.append(s)
                    seen.add(key)
            self.samples[int(cid)] = unique[: self.budget_per_class]

    @torch.no_grad()
    def _herding_select(self, dataset: IncrementalSegmentationDataset, class_ids: Sequence[int], model: torch.nn.Module, device: torch.device, ignore_index: int) -> None:
        root = dataset.base_dataset.root
        candidates_by_class = {
            int(cid): [s for s in dataset.base_dataset.samples if self._mask_contains_class(root, s, int(cid))]
            for cid in class_ids
        }
        if not any(candidates_by_class.values()):
            return
        model.eval()
        # Compute image-level class embeddings for candidates. This is intentionally simple and deterministic.
        embeddings: dict[tuple[str, str, int], torch.Tensor] = {}
        temp_dataset = dataset
        loader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)
        sample_keys = [(s.image, s.mask) for s in temp_dataset.base_dataset.samples]
        for idx, batch in enumerate(tqdm(loader, desc="memory-herding", leave=False)):
            image = batch["image"].to(device)
            mask_global = batch["mask_global"].to(device)
            out = model(image)
            feat = out["features"]
            mask_ds = F.interpolate(mask_global.unsqueeze(1).float(), size=feat.shape[-2:], mode="nearest").squeeze(1).long()
            for cid in class_ids:
                pix = mask_ds == int(cid)
                if torch.any(pix):
                    proto = feat.permute(0, 2, 3, 1)[pix].mean(dim=0).detach().cpu()
                    embeddings[(sample_keys[idx][0], sample_keys[idx][1], int(cid))] = l2norm(proto)
        for cid in class_ids:
            embs = []
            sample_list = []
            for s in candidates_by_class[int(cid)]:
                key = (s.image, s.mask, int(cid))
                if key in embeddings:
                    embs.append(embeddings[key])
                    sample_list.append(s)
            if not embs:
                continue
            mat = torch.stack(embs, dim=0)
            center = l2norm(mat.mean(dim=0))
            scores = torch.norm(mat - center.view(1, -1), dim=1)
            order = torch.argsort(scores).tolist()
            selected = [sample_list[i] for i in order[: self.budget_per_class]]
            previous = self.samples.get(int(cid), [])
            merged = previous + selected
            unique = []
            seen = set()
            for s in merged:
                key = (s.image, s.mask)
                if key not in seen:
                    unique.append(s)
                    seen.add(key)
            self.samples[int(cid)] = unique[: self.budget_per_class]

    def update(
        self,
        dataset: IncrementalSegmentationDataset,
        class_ids: Sequence[int],
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        ignore_index: int = 255,
    ) -> None:
        if self.budget_per_class <= 0:
            return
        if self.strategy == "herding" and model is not None and device is not None:
            self._herding_select(dataset, class_ids, model, device, ignore_index)
        elif self.strategy == "random":
            self._random_select(dataset, class_ids)
        else:
            self._random_select(dataset, class_ids)
