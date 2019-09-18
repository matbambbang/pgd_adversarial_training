############ WORKING ############
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, Bottleneck, ResBlock


def make_block_sequence(block, inplanes=64, planes=64, blocks=2, stride=1) :
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes*block.expansion, stride),
                norm(planes*block.expansion)
                )

    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample))
    nextplanes = planes * block.expansion
    for _ in range(1,blocks) :
        layers.append(block(nextplanes, planes))

    return nn.Sequential(*layers), nextplanes


class CIFAR10Module(nn.Module) :
    def __init__(self, block, layers=1, num_classes=10, init_channel=16, norm_type="b", downsample_type="r") :
        super(CIFAR10Module,self).__init__()
        channel = init_channel
        self.conv = conv3x3(3,channel)
        self.block1 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )
        self.sub1 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block2 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )
        self.sub2 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block3 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channel,num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5)*layers)
                if m.bias is not None : 
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.BatchNorm2d) :
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _subsample(self, inplanes, planes, stride=2, norm_type="b", block_type="r") :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm(planes, norm_type=norm_type)
                )
        return ResBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)

    def forward(self, x) :
        out = self.conv(x)
        out = self.block1(out)
        out = self.sub1(out)

        out = self.block2(out)
        out = self.sub2(out)

        out = self.block3(out)

        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out

    def loss(self) :
        return nn.CrossEntropyLoss()


class CIFARConv(nn.Module) :
    def __init__(self, init_channel=64, expansion=2, norm_type="b") :
        super(CIFARConv,self).__init__()
        self.conv = conv3x3(3, init_channel)
        self.conv1 = self.conv_unit(init_channel, expansion=expansion, norm_type=norm_type)
        init_channel *= 2
        self.conv2 = self.conv_unit(init_channel, expansion=expansion, norm_type=norm_type)
        init_channel *= 2 
        self.conv3 = self.conv_unit(init_channel, expansion=expansion, norm_type=norm_type)
        init_channel *= 2
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512,64)
        self.fc2 = nn.Linear(64,10)

    def conv_unit(self, planes, expansion=2, norm_type="b") :
        unit = nn.Sequential(
                conv3x3(planes, planes),
                nn.MaxPool2d(2, stride=1, padding=1),
                norm(planes, norm_type),
                nn.ReLU(inplace=True),
                conv3x3(planes, planes*expansion),
                nn.MaxPool2d(2, stride=2),
                norm(planes*expansion, norm_type),
                nn.ReLU(inplace=True)
                )
        return unit

    def forward(self, x) :
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def loss(self) :
        return nn.CrossEntropyLoss()


def cifar_model(block_type="res", layers=6, norm_type="b") :
    if block_type == "res" :
        return CIFAR10Module(block=ResBlock, layers=layers, norm_type=norm_type)
    elif block_type == "wres" :
        return CIFAR10Module(block=ResBlock, layers=layers, norm_type=norm_type, init_channel=80)
    elif block_type == "conv" :
        return CIFARConv(norm_type=norm_type)
 
