from __future__ import annotations

import argparse
import json
import os

import torch

from protoflow.config import class_steps, load_config, step_info
from protoflow.engine.checkpoint import load_checkpoint
from protoflow.engine.evaluator import evaluate_model
from protoflow.models import ProtoFlowSegmentor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ProtoFlow checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if "num_threads" in cfg["train"]:
        torch.set_num_threads(int(cfg["train"]["num_threads"]))
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    step = args.step if args.step is not None else int(ckpt.get("step", len(class_steps(cfg)) - 1))
    info = step_info(cfg, step)
    model = ProtoFlowSegmentor(cfg, num_classes=len(info.known_classes))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    metrics = evaluate_model(cfg, model, step=step, device=device)
    print(json.dumps(metrics, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
