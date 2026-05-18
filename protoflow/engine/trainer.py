from __future__ import annotations

import copy
import csv
import json
import os
from itertools import cycle
from typing import Any, Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from protoflow.config import class_steps, ensure_dir, first_step_by_class, step_info
from protoflow.datasets.incremental import build_step_dataset, collate_batch
from protoflow.engine.checkpoint import save_checkpoint
from protoflow.engine.evaluator import evaluate_model
from protoflow.engine.losses import distillation_loss, segmentation_loss
from protoflow.engine.memory import ReplayMemory
from protoflow.engine.metrics import ForgettingTracker
from protoflow.engine.prototype import (
    ProtoFlowField,
    PrototypeBank,
    compute_batch_prototypes,
    compute_prototype_losses,
)
from protoflow.engine.scheduler import WarmupPolyLR
from protoflow.engine.seed import set_seed
from protoflow.models import ProtoFlowSegmentor


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        if x != x or x in {float("inf"), float("-inf")}:
            return None
        return x
    return x


def _make_teacher(model: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _max_iters_for_step(cfg: dict, step: int) -> int:
    value = cfg["train"].get("max_iters", 40000)
    if isinstance(value, dict):
        return int(value.get(str(step), value.get(step, value.get("default", 40000))))
    if isinstance(value, list):
        return int(value[min(step, len(value) - 1)])
    return int(value)


def _initial_lr_for_step(cfg: dict, step: int) -> float:
    value = cfg["train"].get("lr", 0.01)
    if isinstance(value, dict):
        return float(value.get(str(step), value.get(step, value.get("default", 0.01))))
    if isinstance(value, list):
        return float(value[min(step, len(value) - 1)])
    return float(value)


def train(cfg: dict) -> str:
    exp = cfg["experiment"]
    output_dir = ensure_dir(exp.get("output_dir", os.path.join("outputs", exp.get("name", "protoflow"))))
    ckpt_dir = ensure_dir(os.path.join(output_dir, "checkpoints"))
    metrics_dir = ensure_dir(os.path.join(output_dir, "metrics"))
    with open(os.path.join(output_dir, "config.resolved.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(cfg), f, indent=2)

    set_seed(int(exp.get("seed", 0)), deterministic=bool(exp.get("deterministic", False)))
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if "num_threads" in cfg["train"]:
        torch.set_num_threads(int(cfg["train"]["num_threads"]))

    first_info = step_info(cfg, 0)
    model = ProtoFlowSegmentor(cfg, num_classes=len(first_info.known_classes)).to(device)
    flow_cfg = cfg.get("protoflow", {})
    flow_field = ProtoFlowField(
        feature_dim=model.feature_dim,
        hidden_dim=int(flow_cfg.get("hidden_dim", 256)),
        time_dim=int(flow_cfg.get("time_dim", 16)),
        use_time=bool(flow_cfg.get("use_time", True)),
    ).to(device)
    bank = PrototypeBank(ema_alpha=float(flow_cfg.get("ema_alpha", 0.1)))
    memory = ReplayMemory(
        budget_per_class=int(cfg["memory"].get("budget_per_class", 20)),
        strategy=cfg["memory"].get("strategy", "random"),
    )
    forgetting_tracker = ForgettingTracker()
    teacher = None
    all_step_metrics: list[dict] = []
    first_steps = first_step_by_class(cfg)
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))

    for step in range(len(class_steps(cfg))):
        info = step_info(cfg, step)
        model.expand_classes(len(info.known_classes))
        model.to(device)
        if step > 0 and teacher is None:
            raise RuntimeError("Teacher model was not created after previous step")

        train_dataset = build_step_dataset(cfg, step=step, train=True)
        mem_samples = memory.all_samples()
        if mem_samples:
            memory_dataset = build_step_dataset(cfg, step=step, train=True, samples=mem_samples)
            combined_dataset = ConcatDataset([train_dataset, memory_dataset])
        else:
            combined_dataset = train_dataset

        loader = DataLoader(
            combined_dataset,
            batch_size=int(cfg["train"].get("batch_size", 8)),
            shuffle=True,
            num_workers=int(cfg["train"].get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_batch,
        )
        if len(loader) == 0:
            raise RuntimeError("Training dataloader has zero batches. Reduce batch_size or add data.")
        loader_iter = cycle(loader)

        params = list(model.parameters()) + list(flow_field.parameters())
        optimizer = torch.optim.SGD(
            params,
            lr=_initial_lr_for_step(cfg, step),
            momentum=float(cfg["train"].get("momentum", 0.9)),
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
            nesterov=bool(cfg["train"].get("nesterov", False)),
        )
        max_iters = _max_iters_for_step(cfg, step)
        scheduler = WarmupPolyLR(
            optimizer,
            base_lr=_initial_lr_for_step(cfg, step),
            max_iters=max_iters,
            power=float(cfg["train"].get("poly_power", 0.9)),
            warmup_iters=int(cfg["train"].get("warmup_iters", 1000)),
            warmup_lr=float(cfg["train"].get("warmup_lr", 1e-4)),
        )

        log_path = os.path.join(metrics_dir, f"train_step_{step}.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(
                csv_f,
                fieldnames=["step", "iter", "lr", "loss", "seg", "dist", "flow", "curve", "sep"],
            )
            writer.writeheader()
            pbar = tqdm(range(max_iters), desc=f"train-step{step}")
            for iteration in pbar:
                model.train()
                flow_field.train()
                batch = next(loader_iter)
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                out = model(images)
                logits = out["logits"]
                features = out["features"]

                loss_seg = segmentation_loss(logits, masks, ignore_index=ignore_index)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)["logits"]
                    loss_dist = distillation_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        old_class_count=len(info.old_classes),
                        temperature=float(cfg["loss"].get("temperature", 2.0)),
                    )
                else:
                    loss_dist = logits.new_tensor(0.0)

                batch_protos = compute_batch_prototypes(
                    features=features,
                    labels=masks,
                    known_classes=info.known_classes,
                    ignore_index=ignore_index,
                )
                proto_losses = compute_prototype_losses(
                    batch_prototypes=batch_protos,
                    bank=bank,
                    flow_field=flow_field,
                    step=step,
                    total_steps=info.total_steps,
                    first_step_by_class=first_steps,
                    use_norm=bool(flow_cfg.get("prototype_norm", True)),
                    use_flow=bool(flow_cfg.get("use_flow", True)),
                    use_curve=bool(flow_cfg.get("use_curve", True)),
                    use_sep=bool(flow_cfg.get("use_sep", True)),
                    margin=float(flow_cfg.get("margin", 0.5)),
                )
                loss = (
                    loss_seg
                    + float(cfg["loss"].get("lambda_dist", 1.0)) * loss_dist
                    + proto_losses.total(
                        lambda_flow=float(cfg["loss"].get("lambda_flow", 1.0)),
                        lambda_curve=float(cfg["loss"].get("lambda_curve", 0.5)),
                        lambda_sep=float(cfg["loss"].get("lambda_sep", 0.1)),
                    )
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(params, max_norm=float(cfg["train"].get("grad_clip", 1.0)))
                optimizer.step()
                lr = scheduler.step(iteration + 1)
                bank.update(step, batch_protos)

                if iteration % int(cfg["train"].get("log_interval", 50)) == 0 or iteration == max_iters - 1:
                    row = {
                        "step": step,
                        "iter": iteration,
                        "lr": lr,
                        "loss": float(loss.detach().cpu()),
                        "seg": float(loss_seg.detach().cpu()),
                        "dist": float(loss_dist.detach().cpu()),
                        "flow": float(proto_losses.flow.detach().cpu()),
                        "curve": float(proto_losses.curve.detach().cpu()),
                        "sep": float(proto_losses.sep.detach().cpu()),
                    }
                    writer.writerow(row)
                    csv_f.flush()
                    pbar.set_postfix({"loss": f"{row['loss']:.3f}", "m": len(mem_samples)})

        step_metrics = evaluate_model(cfg, model, step=step, device=device)
        forget = forgetting_tracker.update_and_compute(step_metrics["global_class_iou"], info.old_classes)
        step_metrics.update(forget)
        step_metrics["step"] = step
        all_step_metrics.append(step_metrics)
        with open(os.path.join(metrics_dir, f"eval_step_{step}.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(step_metrics), f, indent=2)

        save_checkpoint(
            os.path.join(ckpt_dir, f"step_{step}.pt"),
            {
                "step": step,
                "model": model.state_dict(),
                "flow_field": flow_field.state_dict(),
                "prototype_bank": bank.state_dict(),
                "memory": memory.state_dict(),
                "known_classes": info.known_classes,
                "cfg": cfg,
            },
        )

        memory.update(
            train_dataset,
            class_ids=info.known_classes if cfg["protocol"].get("type") == "domain_incremental" else info.current_classes,
            model=model,
            device=device,
            ignore_index=ignore_index,
        )
        teacher = _make_teacher(model).to(device)

    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(
        final_path,
        {
            "step": len(class_steps(cfg)) - 1,
            "model": model.state_dict(),
            "flow_field": flow_field.state_dict(),
            "prototype_bank": bank.state_dict(),
            "memory": memory.state_dict(),
            "known_classes": step_info(cfg, len(class_steps(cfg)) - 1).known_classes,
            "metrics": all_step_metrics,
            "cfg": cfg,
        },
    )
    with open(os.path.join(metrics_dir, "all_steps.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(all_step_metrics), f, indent=2)
    return final_path
