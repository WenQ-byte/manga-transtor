"""自包含 ResNet34（结构/参数名对齐 torchvision，权重从 detect ckpt 加载，无需 torchvision）"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


def _make_layer(inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * BasicBlock.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * BasicBlock.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * BasicBlock.expansion),
        )
    layers = [BasicBlock(inplanes, planes, stride, downsample)]
    inplanes = planes * BasicBlock.expansion
    layers += [BasicBlock(inplanes, planes) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


class ResNet34(nn.Module):
    """与 torchvision resnet34 结构、参数名一致（无 fc，不需要）"""

    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, 3)
        self.layer2 = _make_layer(64, 128, 4, stride=2)
        self.layer3 = _make_layer(128, 256, 6, stride=2)
        self.layer4 = _make_layer(256, 512, 3, stride=2)
        # torchvision resnet34 含 fc（分类头），ckpt 状态里带该键；forward 不使用
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1000)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet34(pretrained=False) -> ResNet34:
    return ResNet34(pretrained=pretrained)