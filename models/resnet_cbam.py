# 
# Program:
#   ResNet 基础结构 结合 空间（Spatial）、通道（Channel）注意力机制，简写：CBAM.
# 
# Data:
#   2021.4.7
# 
# Author:
#   XingHua.He
# #

import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.utils.model_zoo as model_zoo
from typing import List, Optional, Type, Union


__all__ = ['resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam', 'resnet152_cbam']


def conv3x3(in_planes : int, out_planes : int, stride : int = 1) -> nn.Conv2d:
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes : int, ratio : int = 16) -> None:
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : Tensor) -> Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size : int = 7) -> None:
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes : int, planes : int, stride : int = 1, downsample : Optional[nn.Module] = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x : Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes : int, planes : int, stride : int = 1, downsample :  Optional[nn.Module] = None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x : Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block :Type[Union[BasicBlock, Bottleneck]], layers : List[int], num_classes : int = 9) -> None:
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block : Type[Union[BasicBlock, Bottleneck]], planes : int, blocks : int, stride : int = 1) -> nn.Sequential():
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x : Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_cbam(pretrained : str = '', **kwargs) -> ResNet:

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained != '' and pretrained.endswith('.pt'):
        try:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict)
        except:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'])
    return model


def resnet34_cbam(pretrained : str = '', **kwargs) -> ResNet:

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained != '' and pretrained.endswith('.pt'):
        try:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict)
        except:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'])
    return model


def resnet50_cbam(pretrained : str = '', **kwargs) -> ResNet:

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained != '' and pretrained.endswith('.pt'):
        try:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict)
        except:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'])
    return model


def resnet101_cbam(pretrained : str = '', **kwargs) -> ResNet:

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained != '' and pretrained.endswith('.pt'):
        try:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict)
        except:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'])
    return model


def resnet152_cbam(pretrained : str = '', **kwargs) -> ResNet:

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained != '' and pretrained.endswith('.pt'):
        try:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict)
        except:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'])
    return model
