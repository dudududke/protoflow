from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        target_np = target.detach().cpu().numpy().astype(np.int64)
        mask = (target_np != self.ignore_index) & (target_np >= 0) & (target_np < self.num_classes)
        pred_np = pred_np[mask]
        target_np = target_np[mask]
        pred_np = np.clip(pred_np, 0, self.num_classes - 1)
        hist = np.bincount(
            self.num_classes * target_np.reshape(-1) + pred_np.reshape(-1),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.matrix += hist

    def compute(self) -> dict:
        hist = self.matrix.astype(np.float64)
        tp = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan, dtype=np.float64), where=union > 0)
        precision = np.divide(tp, pos_pred, out=np.full_like(tp, np.nan, dtype=np.float64), where=pos_pred > 0)
        recall = np.divide(tp, pos_gt, out=np.full_like(tp, np.nan, dtype=np.float64), where=pos_gt > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.full_like(tp, np.nan, dtype=np.float64), where=(precision + recall) > 0)
        oa = tp.sum() / max(1.0, hist.sum())
        return {
            "class_iou": iou.tolist(),
            "mIoU": float(np.nanmean(iou)),
            "OA": float(oa),
            "class_f1": f1.tolist(),
            "mF1": float(np.nanmean(f1)),
            "confusion": self.matrix.tolist(),
        }


def select_mean(values: Sequence[float], indices: Sequence[int]) -> float:
    vals = []
    for i in indices:
        if i < len(values) and not np.isnan(values[i]):
            vals.append(values[i])
    return float(np.mean(vals)) if vals else float("nan")


def add_incremental_metrics(metrics: dict, old_local: Sequence[int], new_local: Sequence[int]) -> dict:
    out = dict(metrics)
    iou = metrics["class_iou"]
    out["mIoU_old"] = select_mean(iou, old_local)
    out["mIoU_new"] = select_mean(iou, new_local)
    out["mIoU_all"] = metrics["mIoU"]
    return out


@dataclass
class ForgettingTracker:
    best_class_iou: dict[int, float] = field(default_factory=dict)

    def update_and_compute(self, global_class_iou: dict[int, float], old_classes: Sequence[int]) -> dict:
        for cid, val in global_class_iou.items():
            if np.isnan(val):
                continue
            self.best_class_iou[cid] = max(float(val), self.best_class_iou.get(cid, float("-inf")))
        forgetting = {}
        for cid in old_classes:
            current = global_class_iou.get(cid, float("nan"))
            best = self.best_class_iou.get(cid, float("nan"))
            if not np.isnan(current) and not np.isnan(best):
                forgetting[cid] = max(0.0, best - current)
        avg = float(np.mean(list(forgetting.values()))) if forgetting else 0.0
        return {"class_forgetting": forgetting, "F": avg}
