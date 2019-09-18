import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import conv3x3, conv1x1, norm
from .block import ConvBlock, ResBlock
from .utils import Flatten


class MNISTModule(nn.Module) :
    def __init__(self,
            block,
            layers=1,
            channels=64,
            stride=1,
            coef=1,
            classes=10,
            norm_type="b") :
        super(MNISTModule,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(1,channels,3,1), norm(channels, norm_type), self.relu, nn.Conv2d(channels,channels,4,2,1),
                norm(channels, norm_type), self.relu, nn.Conv2d(channels,channels,4,2,1)
                )
        self.blocks = nn.Sequential(
                *[block(inplanes=channels, planes=channels, stride=stride, coef=coef, norm_type=norm_type) for _ in range(layers)]
                )
        self.linear = nn.Sequential(
                norm(channels, norm_type), self.relu, nn.AdaptiveAvgPool2d((1,1)), Flatten(), nn.Linear(channels,10)
                )

    def forward(self,x) :
        out = self.downsample(x)
        out = self.blocks(out)
        out = self.linear(out)
        return out

    def loss(self) :
        return nn.CrossEntropyLoss()

def mnist_model(block_type="conv", layers=3, norm_type="b") :
    if block_type == "conv" :
        return MNISTModule(block=ConvBlock, layers=layers, norm_type=norm_type)
    elif block_type == "res" :
        return MNISTModule(block=ResBlock, layers=layers, norm_type=norm_type)

