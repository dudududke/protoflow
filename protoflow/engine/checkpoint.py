from __future__ import annotations

import os
from typing import Any, Dict

import torch


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)
