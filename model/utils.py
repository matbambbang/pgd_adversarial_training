import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def binarize(x, threshold=0.5) :
    ones = torch.ones(x.shape)
    zeros = torch.zeros(x.shape)
    k = torch.where(x >= threshold, x, 0)
    k = torch.where(x < threshold, k, 1)
    return k


class BinarizeWrapper(nn.Module) :
    def __init__(self, pre_model) :
        super(BinarizeWrapper,self).__init__()
        self.model = pre_model

    def forward(self ,x, threshold=0.5) :
        k = torch.where(x >= threshold, x, torch.zeros(x.shape, device=x.device))
        k = torch.where(x < threshold, k, torch.ones(x.shape, device=x.device))
        return self.model(k)


class Flatten(nn.Module) :
    def __init__(self) :
        super(Flatten,self).__init__()

    def forward(self, x) :
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
