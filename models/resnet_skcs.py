# 
# Program:
#   
# 
# Data:
#   2021.4.7
# 
# Author:
#   XingHua.He
# #
from typing import List, Type, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
import math

from torch.nn.modules.container import Sequential

__all__ = ['resnet18_skcs']

def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SKCSConv(nn.Module):
    def __init__(self, features: int, WH: int, M: int, G: int, r, stride: int = 1 , L: int = 32) -> None:
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKCSConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        self.cas = nn.ModuleList([])
        self.sas = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
            self.cas.append(nn.Sequential(
                ChannelAttention(features)
            ))
            self.sas.append(nn.Sequential(
                SpatialAttention()
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            out = self.cas[i](fea) * fea
            out = self.sas[i](out) * out
            fea = fea.unsqueeze(dim=1)
            out = out.unsqueeze(dim=1)
            if i == 0:
                outs = out
                feas = fea
            else:
                outs = torch.cat([outs, out], dim=1)
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (outs * attention_vectors).sum(dim=1)
        return fea_v

class SKCSUnit(nn.Module):
    def __init__(self, in_features: int, out_features: int, WH: int, M: int, G: int, r: int, mid_features: int = None, stride: int = 1, L: int = 32) -> None:
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKCSUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKCSConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x: Tensor) -> Tensor:
        fea = self.feas(x)
        return fea + self.shortcut(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
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

class ResNet(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock]], layers: List[int], num_classes: int = 9) -> None:
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 1
        self.stage_1 = nn.Sequential(
            SKCSUnit(128, 512, 32, 2, 8, 2, stride=2),
        )
        self.conv4 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        # 2
        self.stage_2 = nn.Sequential(
            SKCSUnit(256, 512, 16, 2, 8, 2, stride=2),
        )
        self.conv2 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.upsample3 = nn.UpsamplingBilinear2d(size=(32, 32))
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        # 3
        self.stage_3 = nn.Sequential(
            SKCSUnit(512, 512, 8, 2, 8, 2, stride=1),
        )
        self.conv3 = nn.Conv2d(2048, 512, 3, stride=1, padding=1)
        self.upsample4 = nn.UpsamplingBilinear2d(size=(32, 32))
        self.avgpool3 = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: List[int], stride: int = 1) -> Sequential:
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

    def forward(self, x: Tensor) -> Tensor:
        # Input : 1*256*256  / 1*1024*1024
        x = self.conv1(x) #64*128*128 / 64*512*512 / 64*256*256
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x) # 64*64*64 / 64*256*256 / 64*128*128

        x = self.layer1(x) # 64*64*64 / 64*256*256 / 64*128*128
        # features.
        feature1 = self.layer2(x) # 128*32*32 /  128*128*128 / 128*64*64
        feature2 = self.layer3(feature1) # 256*16*16 / 256*64*64 / 256*32*32
        feature3 = self.layer4(feature2) # 512*8*8 / 512*32*32 / 512*16*16

        # output 1
        feature1 = self.stage_1(feature1) # 512*16*16 / 512*64*64 / 512*32*32
        feature1 = self.conv4(feature1) # 新增 512*16*16
        output1 = self.avgpool1(feature1) # 512*1*1
        output1 = output1.view(output1.size(0), -1)
        output1 = self.fc1(output1)
        # output 2
        feature2 = self.stage_2(feature2) # 512*8*8 / 512*32*32 / 512*16*16
        # feature2 = self.upsample3(feature2) # 512*16*16 / 512*64*64 / 512*32*32
        # feature2 = feature2 + feature1 # feature fusion # 512*16*16
        feature2 = torch.cat((feature2, feature1), dim=1) # 1024*16*16
        output2 = self.conv2(feature2) # 512*16*16 / 512*64*64 / 512*16*16
        output2 = self.avgpool2(output2) #512*1*1
        output2 = output2.view(output2.size(0), -1)
        output2 = self.fc2(output2)
        # output 3
        feature3 = self.stage_3(feature3) # 512*8*8 / 512*32*32 / 512*16*16
        # feature3 = self.upsample4(feature3) #512*16*16 / 512*64*64 / 512*32*32
        # feature3 = feature3 + feature2 + feature1 #512*16*16
        feature3 = torch.cat((feature3, feature2, feature1), dim=1) # 2048*16*16 
        output3 = self.conv3(feature3) #512*16*16 / 512*64*64 / 512*32*32
        output3 = self.avgpool3(output3) #512*1*1
        output3 = output3.view(output3.size(0), -1)
        output3 = self.fc3(output3)

        return output1, output2, output3

def resnet18_skcs(pretrained: str = '', **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained != '' and pretrained.endswith('.pt'):
        try:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict)
        except:
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model_state_dict'])
    return model

if __name__=='__main__':
    x = torch.rand(1, 1, 256, 256)
    model = resnet18_skcs()
    output = model(x)
    # criterion = nn.L1Loss()
    # loss = criterion(out, x)
    # loss.backward()
    # print('out shape : {}'.format(out.shape))
    # print('loss value : {}'.format(loss))
    # summary(conv, (64, 32, 32))