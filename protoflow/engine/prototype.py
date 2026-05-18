from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from protoflow.engine.losses import pairwise_separation_loss


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


class ProtoFlowField(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, time_dim: int = 16, use_time: bool = True):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.use_time = bool(use_time)
        input_dim = self.feature_dim + (self.time_dim if self.use_time else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def time_encoding(self, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau.view(1)
        tau = tau.float().view(-1, 1)
        half = max(1, self.time_dim // 2)
        freq = torch.exp(torch.linspace(0, torch.log(torch.tensor(10000.0, device=tau.device)), half, device=tau.device))
        angles = tau / freq.view(1, -1)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if enc.shape[1] < self.time_dim:
            enc = F.pad(enc, (0, self.time_dim - enc.shape[1]))
        return enc[:, : self.time_dim]

    def forward(self, prototype: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
        if prototype.ndim == 1:
            prototype = prototype.view(1, -1)
        if self.use_time:
            if not torch.is_tensor(tau):
                tau = torch.tensor([tau], device=prototype.device, dtype=prototype.dtype)
            else:
                tau = tau.to(device=prototype.device, dtype=prototype.dtype)
            if tau.numel() == 1 and prototype.shape[0] > 1:
                tau = tau.repeat(prototype.shape[0])
            enc = self.time_encoding(tau).to(dtype=prototype.dtype)
            x = torch.cat([prototype, enc], dim=1)
        else:
            x = prototype
        return self.net(x).view_as(prototype)


@dataclass
class PrototypeBank:
    ema_alpha: float = 0.1
    history: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)

    def update(self, step: int, batch_prototypes: Mapping[int, torch.Tensor]) -> None:
        step = int(step)
        if step not in self.history:
            self.history[step] = {}
        for cid, proto in batch_prototypes.items():
            cid = int(cid)
            detached = proto.detach().float().cpu()
            if cid in self.history[step]:
                self.history[step][cid] = (1.0 - self.ema_alpha) * self.history[step][cid] + self.ema_alpha * detached
            else:
                self.history[step][cid] = detached

    def get(self, step: int, cid: int, device: torch.device | None = None) -> torch.Tensor | None:
        proto = self.history.get(int(step), {}).get(int(cid))
        if proto is None:
            return None
        return proto.to(device=device) if device is not None else proto

    def class_history(self, cid: int) -> list[tuple[int, torch.Tensor]]:
        out = []
        for step in sorted(self.history):
            if int(cid) in self.history[step]:
                out.append((step, self.history[step][int(cid)]))
        return out

    def state_dict(self) -> dict:
        return {int(s): {int(c): p.clone() for c, p in vals.items()} for s, vals in self.history.items()}

    def load_state_dict(self, state: dict) -> None:
        self.history = {int(s): {int(c): p.clone().cpu() for c, p in vals.items()} for s, vals in state.items()}


def compute_batch_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    known_classes: Sequence[int],
    ignore_index: int = 255,
) -> dict[int, torch.Tensor]:
    if labels.shape[-2:] != features.shape[-2:]:
        labels_ds = F.interpolate(labels.unsqueeze(1).float(), size=features.shape[-2:], mode="nearest").squeeze(1).long()
    else:
        labels_ds = labels
    b, c, h, w = features.shape
    feat_flat = features.permute(0, 2, 3, 1).reshape(-1, c)
    label_flat = labels_ds.reshape(-1)
    protos: dict[int, torch.Tensor] = {}
    for local_id, global_id in enumerate(known_classes):
        mask = label_flat == int(local_id)
        if torch.any(mask):
            protos[int(global_id)] = feat_flat[mask].mean(dim=0)
    return protos


def normalized_tau(step: int, first_step: int, total_steps: int) -> float:
    denom = max(1, total_steps - 1 - first_step)
    return float(step - first_step) / float(denom)


@dataclass
class PrototypeLossOutput:
    flow: torch.Tensor
    curve: torch.Tensor
    sep: torch.Tensor

    def total(self, lambda_flow: float, lambda_curve: float, lambda_sep: float) -> torch.Tensor:
        return lambda_flow * self.flow + lambda_curve * self.curve + lambda_sep * self.sep


def compute_prototype_losses(
    batch_prototypes: Mapping[int, torch.Tensor],
    bank: PrototypeBank,
    flow_field: ProtoFlowField | None,
    step: int,
    total_steps: int,
    first_step_by_class: Mapping[int, int],
    use_norm: bool = True,
    use_flow: bool = True,
    use_curve: bool = True,
    use_sep: bool = True,
    margin: float = 0.5,
) -> PrototypeLossOutput:
    if batch_prototypes:
        device = next(iter(batch_prototypes.values())).device
    else:
        device = torch.device("cpu")
    zero = next(iter(batch_prototypes.values())).new_tensor(0.0) if batch_prototypes else torch.tensor(0.0, device=device)
    current: dict[int, torch.Tensor] = {}
    for cid, proto in batch_prototypes.items():
        current[int(cid)] = l2norm(proto) if use_norm else proto

    flow_terms: list[torch.Tensor] = []
    if use_flow and flow_field is not None and step > 0:
        for cid, mu_t in current.items():
            prev = bank.get(step - 1, cid, device=mu_t.device)
            if prev is None:
                continue
            prev = l2norm(prev) if use_norm else prev
            first = int(first_step_by_class.get(cid, 0))
            tau_prev = normalized_tau(step - 1, first, total_steps)
            tau_cur = normalized_tau(step, first, total_steps)
            dt = max(1e-6, tau_cur - tau_prev)
            mu_hat = prev + dt * flow_field(prev, tau_prev).view(-1)
            if use_norm:
                mu_hat = l2norm(mu_hat)
            flow_terms.append(F.mse_loss(mu_hat, mu_t))
    flow_loss = torch.stack(flow_terms).mean() if flow_terms else zero

    curve_terms: list[torch.Tensor] = []
    if use_curve and step >= 2:
        for cid, mu_t in current.items():
            prev = bank.get(step - 1, cid, device=mu_t.device)
            prev2 = bank.get(step - 2, cid, device=mu_t.device)
            if prev is None or prev2 is None:
                continue
            prev = l2norm(prev) if use_norm else prev
            prev2 = l2norm(prev2) if use_norm else prev2
            curve_vec = mu_t - 2.0 * prev + prev2
            curve_terms.append((curve_vec ** 2).sum())
    curve_loss = torch.stack(curve_terms).mean() if curve_terms else zero

    if use_sep and current:
        sep_loss = pairwise_separation_loss(list(current.values()), margin=margin).to(device)
    else:
        sep_loss = zero
    return PrototypeLossOutput(flow=flow_loss, curve=curve_loss, sep=sep_loss)
