from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from protoflow.engine.checkpoint import load_checkpoint


def l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prototype curvature and inter-class angle analysis from a run directory")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    ckpt = load_checkpoint(str(run_dir / "checkpoints" / "final.pt"), map_location="cpu")
    bank = ckpt["prototype_bank"]
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    classes = sorted({int(c) for step in bank.values() for c in step.keys()})
    for cid in classes:
        seq = []
        for s in sorted(int(k) for k in bank.keys()):
            proto = bank.get(s, {}).get(cid)
            if proto is not None:
                seq.append((s, l2(proto.float())))
        if len(seq) >= 3:
            curves = []
            for i in range(1, len(seq) - 1):
                k = seq[i + 1][1] - 2 * seq[i][1] + seq[i - 1][1]
                curves.append(float(k.norm(p=2)))
            avg_curve = sum(curves) / len(curves)
        else:
            avg_curve = 0.0
        rows.append({"class_id": cid, "num_steps": len(seq), "avg_curvature": avg_curve})
    with open(out_dir / "prototype_curvature.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "num_steps", "avg_curvature"])
        writer.writeheader()
        writer.writerows(rows)

    final_step = max(int(k) for k in bank.keys())
    final_protos = {int(c): l2(p.float()) for c, p in bank[final_step].items()}
    angle_rows = []
    ids = sorted(final_protos)
    for i, ci in enumerate(ids):
        for cj in ids[i + 1 :]:
            cos = torch.dot(final_protos[ci], final_protos[cj]).clamp(-1, 1)
            angle = float(torch.rad2deg(torch.acos(cos)))
            margin = float(1.0 - cos)
            angle_rows.append({"class_i": ci, "class_j": cj, "angle_deg": angle, "cosine_margin": margin})
    with open(out_dir / "prototype_angles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_i", "class_j", "angle_deg", "cosine_margin"])
        writer.writeheader()
        writer.writerows(angle_rows)
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
