from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protoflow.config import step_info
from protoflow.datasets.incremental import build_step_dataset, collate_batch
from protoflow.engine.metrics import ConfusionMatrix, add_incremental_metrics


@torch.no_grad()
def evaluate_model(
    cfg: dict,
    model: torch.nn.Module,
    step: int,
    device: torch.device,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict:
    model.eval()
    dcfg = cfg["dataset"]
    ignore_index = int(dcfg.get("ignore_index", 255))
    info = step_info(cfg, step)
    dataset = build_step_dataset(cfg, step=step, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(cfg["train"].get("eval_batch_size", cfg["train"].get("batch_size", 4))),
        shuffle=False,
        num_workers=num_workers if num_workers is not None else int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    cm = ConfusionMatrix(num_classes=len(info.known_classes), ignore_index=ignore_index)
    for batch in tqdm(loader, desc=f"eval-step{step}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        out = model(images)
        pred = out["logits"].argmax(dim=1)
        cm.update(pred, masks)
    metrics = cm.compute()
    old_local = [info.known_classes.index(c) for c in info.old_classes if c in info.known_classes]
    new_local = [info.known_classes.index(c) for c in info.current_classes if c in info.known_classes]
    metrics = add_incremental_metrics(metrics, old_local=old_local, new_local=new_local)
    metrics["global_class_iou"] = {int(g): metrics["class_iou"][i] for i, g in enumerate(info.known_classes)}
    metrics["known_classes"] = info.known_classes
    metrics["current_classes"] = info.current_classes
    metrics["old_classes"] = info.old_classes
    return metrics
