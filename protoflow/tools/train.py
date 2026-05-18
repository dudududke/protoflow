from __future__ import annotations

import argparse

from protoflow.config import load_config
from protoflow.engine.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProtoFlow incremental segmentation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    final_path = train(cfg)
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
