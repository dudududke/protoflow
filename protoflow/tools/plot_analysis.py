from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_curvature(run_dir: Path, out_dir: Path) -> None:
    curv_path = run_dir / "analysis" / "prototype_curvature.csv"
    if not curv_path.exists():
        return
    df = pd.read_csv(curv_path)
    plt.figure(figsize=(6, 4))
    plt.bar(df["class_id"].astype(str), df["avg_curvature"])
    plt.xlabel("Class id")
    plt.ylabel("Average discrete curvature")
    plt.title("Prototype trajectory curvature")
    plt.tight_layout()
    plt.savefig(out_dir / "prototype_curvature.png", dpi=200)
    plt.close()


def plot_angles(run_dir: Path, out_dir: Path) -> None:
    angle_path = run_dir / "analysis" / "prototype_angles.csv"
    if not angle_path.exists():
        return
    df = pd.read_csv(angle_path)
    plt.figure(figsize=(6, 4))
    plt.hist(df["angle_deg"], bins=20, density=True, alpha=0.8)
    plt.xlabel("Inter-class angle (degrees)")
    plt.ylabel("Density")
    plt.title("Final-step prototype angle distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "prototype_angles.png", dpi=200)
    plt.close()


def plot_metrics(run_dir: Path, out_dir: Path) -> None:
    metrics_path = run_dir / "metrics" / "all_steps.json"
    if not metrics_path.exists():
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    steps = [m["step"] for m in metrics]
    miou = [m.get("mIoU_all", m.get("mIoU")) for m in metrics]
    forgetting = [m.get("F", 0.0) for m in metrics]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, miou, marker="o", label="mIoU_all")
    plt.xlabel("Incremental step")
    plt.ylabel("mIoU")
    plt.title("Incremental segmentation performance")
    plt.tight_layout()
    plt.savefig(out_dir / "step_miou.png", dpi=200)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(steps, forgetting, marker="o", label="Forgetting")
    plt.xlabel("Incremental step")
    plt.ylabel("Forgetting")
    plt.title("Forgetting over steps")
    plt.tight_layout()
    plt.savefig(out_dir / "step_forgetting.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot analysis figures from a ProtoFlow run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curvature(run_dir, out_dir)
    plot_angles(run_dir, out_dir)
    plot_metrics(run_dir, out_dir)
    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
