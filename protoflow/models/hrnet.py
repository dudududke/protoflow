from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.num_inchannels = list(num_inchannels)
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(block)
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
    ) -> nn.Sequential:
        downsample = None
        out_channels = num_channels[branch_index] * block.expansion
        if self.num_inchannels[branch_index] != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [block(self.num_inchannels[branch_index], num_channels[branch_index], downsample=downsample)]
        self.num_inchannels[branch_index] = out_channels
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
    ) -> nn.ModuleList:
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self, block: type[nn.Module]) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None
        num_out_branches = self.num_branches if self.multi_scale_output else 1
        fuse_layers: list[nn.ModuleList] = []
        for i in range(num_out_branches):
            fuse_layer: list[nn.Module] = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(self.num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    convs: list[nn.Module] = []
                    for k in range(i - j):
                        in_ch = self.num_inchannels[j]
                        out_ch = self.num_inchannels[i] if k == i - j - 1 else self.num_inchannels[j]
                        convs.extend(
                            [
                                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                            ]
                        )
                        if k != i - j - 1:
                            convs.append(nn.ReLU(inplace=True))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        return list(self.num_inchannels)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        if self.fuse_layers is None:
            return x
        x_fuse: list[torch.Tensor] = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if j == i:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(fuse_layer[j](x[j]), size=x[i].shape[-2:], mode="bilinear", align_corners=False)
                else:
                    y = y + fuse_layer[j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNetW48(nn.Module):
    """HRNet-W48 feature extractor for semantic segmentation.

    The forward pass returns a fused stride-4 feature map formed by upsampling all stage-4 branches to the highest
    resolution and concatenating them. Channel count is 48+96+192+384 = 720.
    """

    out_channels = 720

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, blocks=4)
        stage1_out_channel = 64 * Bottleneck.expansion

        self.stage2_cfg = dict(num_modules=1, num_branches=2, num_blocks=[4, 4], num_channels=[48, 96], block=BasicBlock)
        num_channels = [c * BasicBlock.expansion for c in self.stage2_cfg["num_channels"]]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = dict(num_modules=4, num_branches=3, num_blocks=[4, 4, 4], num_channels=[48, 96, 192], block=BasicBlock)
        num_channels = [c * BasicBlock.expansion for c in self.stage3_cfg["num_channels"]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = dict(num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[48, 96, 192, 384], block=BasicBlock)
        num_channels = [c * BasicBlock.expansion for c in self.stage4_cfg["num_channels"]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        self.out_channels = sum(pre_stage_channels)
        self.init_weights()

    def _make_layer(self, block: type[nn.Module], inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer: Sequence[int], num_channels_cur_layer: Sequence[int]) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers: list[nn.Module] = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                convs: list[nn.Module] = []
                in_channels = num_channels_pre_layer[-1]
                for j in range(i + 1 - num_branches_pre):
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    convs.extend(
                        [
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        ]
                    )
                    in_channels = out_channels
                transition_layers.append(nn.Sequential(*convs))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config: dict, num_inchannels: Sequence[int], multi_scale_output: bool = True):
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = layer_config["block"]
        modules: list[nn.Module] = []
        channels = list(num_inchannels)
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            module = HighResolutionModule(num_branches, block, num_blocks, channels, num_channels, reset_multi_scale_output)
            modules.append(module)
            channels = module.get_num_inchannels()
        return nn.Sequential(*modules), channels

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]
            cleaned[key] = v
        self.load_state_dict(cleaned, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            x_list.append(self.transition1[i](x))
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if i < self.stage2_cfg["num_branches"]:
                x_list.append(self.transition2[i](y_list[i]))
            else:
                x_list.append(self.transition2[i](y_list[-1]))
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if i < self.stage3_cfg["num_branches"]:
                x_list.append(self.transition3[i](y_list[i]))
            else:
                x_list.append(self.transition3[i](y_list[-1]))
        y_list = self.stage4(x_list)
        h, w = y_list[0].shape[-2:]
        y_upsampled = [y_list[0]] + [F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False) for y in y_list[1:]]
        return torch.cat(y_upsampled, dim=1)
