from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    old_class_count: int,
    temperature: float = 2.0,
) -> torch.Tensor:
    if old_class_count <= 0:
        return student_logits.new_tensor(0.0)
    s = student_logits[:, :old_class_count]
    t = teacher_logits[:, :old_class_count]
    if s.shape[-2:] != t.shape[-2:]:
        t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
    log_p = F.log_softmax(s / temperature, dim=1)
    q = F.softmax(t / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def pairwise_separation_loss(prototypes: list[torch.Tensor], margin: float) -> torch.Tensor:
    if len(prototypes) < 2:
        if prototypes:
            return prototypes[0].new_tensor(0.0)
        return torch.tensor(0.0)
    mat = torch.stack(prototypes, dim=0)
    dists = torch.cdist(mat, mat, p=2)
    n = dists.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    violations = F.relu(margin - dists[mask]) ** 2
    if violations.numel() == 0:
        return mat.new_tensor(0.0)
    return violations.mean()
