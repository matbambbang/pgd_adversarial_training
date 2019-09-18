import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, bias=True) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(planes, norm_type="none") :
    if norm_type == "b" :
        return nn.BatchNorm2d(planes)
    if norm_type == "g" :
        return nn.GroupNorm(num_groups=int(planes/2), num_channels=planes)
    return nn.Identity()


class BasicBlock(nn.Module) :
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type="b", **kwargs) :
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.n1 = norm(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.n2 = norm(planes, norm_type)
        self.downsample = downsample
        self.residual = nn.Sequential(
                self.conv1,
                self.n1,
                self.relu,
                self.conv2,
                self.n2
                )

    def forward(self, x) :
        identity = x 
        out = self.residual(x)
        if self.downsample is not None :
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module) :
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs) :
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(inplanes, plnaes)
        self.n1 = norm(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.n2 = norm(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.n3 = norm(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) :
        identity = x
        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.n3(out)

        if self.downsample is not None :
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    def initialize(self) :
        return None

class ConvBlock(nn.Module) :
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_type="b", **kwargs) :
        super(ConvBlock,self).__init__()
        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)

        self.convs = nn.Sequential(
                self.n1,
                self.relu,
                self.conv1,
                self.n2,
                self.relu,
                self.conv2
                )

    def forward(self, x) :
        return self.convs(x)

class ResBlock(nn.Module) :
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type="b", **kwargs) :
        super(ResBlock,self).__init__()
        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)

        self.residual = nn.Sequential(
                self.n1,
                self.relu,
                self.conv1,
                self.n2,
                self.relu,
                self.conv2
                )


    def forward(self, x) :
        shortcut = x

        if self.downsample is not None :
            shortcut = F.relu(self.n1(x))
            shortcut = self.downsample(shortcut)

        out = self.residual(x)

        return out + shortcut

    def initialize(self) :
        return None


