from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    """Raised when a reproduction configuration is invalid."""


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return data


def load_config(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    if "base" in cfg:
        base_path = cfg.pop("base")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base_cfg = load_config(base_path)
        cfg = _deep_update(base_cfg, cfg)
    cfg["config_path"] = os.path.abspath(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    required_top = ["experiment", "dataset", "protocol", "model", "train", "loss", "memory"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ConfigError(f"Missing top-level config keys: {missing}")
    if cfg["protocol"].get("type") not in {"class_incremental", "domain_incremental"}:
        raise ConfigError("protocol.type must be 'class_incremental' or 'domain_incremental'")
    if cfg["dataset"].get("num_classes", 0) <= 0:
        raise ConfigError("dataset.num_classes must be positive")
    if cfg["model"].get("backbone") not in {"hrnet_w48", "simple_cnn"}:
        raise ConfigError("model.backbone must be 'hrnet_w48' or 'simple_cnn'")


@dataclass(frozen=True)
class StepInfo:
    step: int
    known_classes: list[int]
    current_classes: list[int]
    old_classes: list[int]
    total_steps: int


def class_steps(cfg: Dict[str, Any]) -> list[list[int]]:
    ptype = cfg["protocol"].get("type")
    if ptype == "domain_incremental":
        return [list(range(int(cfg["dataset"]["num_classes"]))) for _ in cfg["protocol"]["domains"]]
    steps = cfg["protocol"].get("steps")
    if not isinstance(steps, list) or not steps:
        raise ConfigError("protocol.steps must be a non-empty list for class_incremental")
    parsed = []
    for s in steps:
        if not isinstance(s, list) or not all(isinstance(x, int) for x in s):
            raise ConfigError("Each protocol step must be a list of integer class ids")
        parsed.append(s)
    return parsed


def step_info(cfg: Dict[str, Any], step: int) -> StepInfo:
    steps = class_steps(cfg)
    if step < 0 or step >= len(steps):
        raise ConfigError(f"Invalid step {step}; expected 0..{len(steps)-1}")
    if cfg["protocol"].get("type") == "domain_incremental":
        known = list(range(int(cfg["dataset"]["num_classes"])))
        current = known
        old = known if step > 0 else []
    else:
        known = sorted({c for group in steps[: step + 1] for c in group})
        current = list(steps[step])
        old = sorted({c for group in steps[:step] for c in group})
    return StepInfo(step=step, known_classes=known, current_classes=current, old_classes=old, total_steps=len(steps))


def first_step_by_class(cfg: Dict[str, Any]) -> dict[int, int]:
    if cfg["protocol"].get("type") == "domain_incremental":
        return {c: 0 for c in range(int(cfg["dataset"]["num_classes"]))}
    mapping: dict[int, int] = {}
    for i, group in enumerate(class_steps(cfg)):
        for c in group:
            mapping.setdefault(c, i)
    return mapping


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
