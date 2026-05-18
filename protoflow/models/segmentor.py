from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from protoflow.models.hrnet import HRNetW48
from protoflow.models.simple import SimpleCNNBackbone


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int, num_classes: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.feature_dim = feature_dim
        self.reset_classifier(num_classes)

    @property
    def num_classes(self) -> int:
        return self.classifier.out_channels

    def reset_classifier(self, num_classes: int) -> None:
        self.classifier = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def expand_classes(self, new_num_classes: int) -> None:
        old = self.classifier
        if new_num_classes == old.out_channels:
            return
        if new_num_classes < old.out_channels:
            raise ValueError("Cannot shrink classifier during incremental training")
        new_cls = nn.Conv2d(self.feature_dim, new_num_classes, kernel_size=1)
        nn.init.normal_(new_cls.weight, mean=0.0, std=0.01)
        nn.init.zeros_(new_cls.bias)
        with torch.no_grad():
            new_cls.weight[: old.out_channels].copy_(old.weight)
            new_cls.bias[: old.out_channels].copy_(old.bias)
        self.classifier = new_cls.to(old.weight.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.proj(x)
        logits = self.classifier(feat)
        return logits, feat


class ProtoFlowSegmentor(nn.Module):
    def __init__(self, cfg: dict, num_classes: int):
        super().__init__()
        model_cfg = cfg["model"]
        in_channels = int(model_cfg.get("in_channels", 3))
        if model_cfg["backbone"] == "hrnet_w48":
            self.backbone = HRNetW48(in_channels=in_channels)
            pretrained = model_cfg.get("pretrained", "")
            if pretrained and os.path.exists(pretrained):
                self.backbone.load_pretrained(pretrained, strict=False)
        elif model_cfg["backbone"] == "simple_cnn":
            self.backbone = SimpleCNNBackbone(in_channels=in_channels, width=int(model_cfg.get("simple_width", 32)))
        else:
            raise ValueError(f"Unknown backbone: {model_cfg['backbone']}")
        feature_dim = int(model_cfg.get("feature_dim", 256))
        self.head = SegmentationHead(self.backbone.out_channels, feature_dim, num_classes)
        self.feature_dim = feature_dim

    @property
    def num_classes(self) -> int:
        return self.head.num_classes

    def expand_classes(self, new_num_classes: int) -> None:
        self.head.expand_classes(new_num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        backbone_feat = self.backbone(x)
        logits_low, decoder_feat = self.head(backbone_feat)
        logits = F.interpolate(logits_low, size=input_size, mode="bilinear", align_corners=False)
        return {"logits": logits, "features": decoder_feat, "logits_low": logits_low}
