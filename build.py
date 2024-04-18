from __future__ import print_function
import torch
import sys
import os

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Encoder(nn.Module):
    def __init__(self, depth, widen_factor=1, dropRate=0.):
        super(Encoder, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
    def forward(self, x, grl=False, constant=0.):
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        if grl:
            x = GradReverse.grad_reverse(x, constant)
        return x

class Reconstructor(nn.Module):
    def __init__(self, in_planes=960, out_planes=640):
        super(Reconstructor, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def forward(self,x):
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, enc_out, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(enc_out, num_classes)
        self.nChannels = 640

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self,x):
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.nChannels)
        x = self.fc(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dis_in=2048, dis_mid=256):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dis_in, dis_mid)
        self.fc2 = nn.Linear(dis_mid, 2)
        self.nChannels = 640
    def forward(self, x):
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.nChannels)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        return x

