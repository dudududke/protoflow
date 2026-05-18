
## Installation

```bash
cd protoflow_reproduce
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Preparing datasets

The project uses a generic image/mask folder format with split files:

```text
data/processed/<dataset>/
  train.txt
  val.txt
  test.txt  # optional
  images/
    xxx.png
  masks/
    xxx.png
```

Each split file contains one sample per line:

```text
images/xxx.png masks/xxx.png
```

Masks must contain integer class ids in `[0, num_classes-1]` and optional ignore id `255`.

For large tiles, use:

```bash
python -m protoflow.tools.prepare_patches \
  --image-dir data/raw/deepglobe/images \
  --mask-dir data/raw/deepglobe/masks \
  --out-root data/processed/deepglobe \
  --split train \
  --patch-size 512 \
  --stride 512
```

Run the command for train/val/test as needed. For Vaihingen/Potsdam, use stride 256 as described in the paper.

## Training commands

### DeepGlobe class-incremental

```bash
python -m protoflow.tools.train --config configs/deepglobe.yaml
```

### Vaihingen class-incremental

```bash
python -m protoflow.tools.train --config configs/vaihingen.yaml
```

### Potsdam class-incremental

```bash
python -m protoflow.tools.train --config configs/potsdam.yaml
```

### LoveDA domain-incremental

```bash
python -m protoflow.tools.train --config configs/loveda.yaml
```

## Evaluation

```bash
python -m protoflow.tools.evaluate --config configs/deepglobe.yaml --checkpoint outputs/deepglobe_protoflow/checkpoints/final.pt
```

## Analysis and plotting

The trainer writes JSON/CSV logs under `outputs/<experiment>/metrics`. Use:

```bash
python -m protoflow.tools.analyze_prototypes --run-dir outputs/deepglobe_protoflow
python -m protoflow.tools.plot_analysis --run-dir outputs/deepglobe_protoflow --out-dir outputs/deepglobe_protoflow/figures
```

## Project layout

```text
protoflow/
  configs/                         YAML configs for all paper protocols and smoke test
  protoflow/
    config.py                      YAML loading and config helpers
    datasets/                      generic folder dataset, incremental wrappers, transforms
    models/                        HRNet-W48, simple CNN, segmentation head
    engine/                        training, evaluation, losses, metrics, memory, prototype bank
    tools/                         CLI tools: train, eval, patchify, synthetic data, analysis plots
  scripts/run_smoke_test.sh         one-command smoke test
  requirements.txt
```