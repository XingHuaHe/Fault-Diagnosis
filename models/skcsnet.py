from PIL.Image import NONE
import torch
import torch.nn as nn
from torch.tensor import Tensor
from torchsummary import summary

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SKCSConv(nn.Module):
    def __init__(self, features: int, WH: int, M: int, G: int, r: int, stride: int = 1 , L: int = 32) -> None:
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

class SKCSNet(nn.Module):
    def __init__(self, class_num: int) -> None:
        super(SKCSNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 32x32
        self.stage_1 = nn.Sequential(
            SKCSUnit(64, 256, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKCSUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU(),
            SKCSUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU()
        ) # 32x32
        self.stage_2 = nn.Sequential(
            SKCSUnit(256, 512, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKCSUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU(),
            SKCSUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU()
        ) # 16x16
        self.stage_3 = nn.Sequential(
            SKCSUnit(512, 1024, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKCSUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU(),
            SKCSUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU()
        ) # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

if __name__=='__main__':
    x = torch.rand(1, 3, 128, 128)
    conv = SKCSConv(64, 32, 3, 8, 2)
    out = conv(x)
    model = SKCSNet(class_num=10)
    output = model(x)
    criterion = nn.L1Loss()
    loss = criterion(out, x)
    loss.backward()
    print('out shape : {}'.format(out.shape))
    print('loss value : {}'.format(loss))
    summary(conv, (64, 32, 32))