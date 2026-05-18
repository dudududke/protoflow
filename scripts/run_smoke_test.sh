#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m protoflow.tools.make_synthetic --root data/synthetic_protoflow --num-train 4 --num-val 2 --size 32 --num-classes 4 --seed 0
python -m protoflow.tools.train --config configs/smoke.yaml
python -m protoflow.tools.evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoints/final.pt
