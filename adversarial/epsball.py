from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from . import AttackBase


class EpsilonAdversary(AttackBase) :
    def __init__(self, model=None, epsilon=0.3, repeat=3, dist="uniform", norm=False, device=None) :
        super(EpsilonAdversary,self).__init__(model, norm)
        self.model = model
        self.epsilon = epsilon
        self.repeat = repeat
        self.dist = dist
        self.device = device or torch.device("cuda:0")

    @overrides
    def loss(self, device=None) :
        device = device or self.device
        self.criterion = nn.CrossEntropyLoss().to(device)
        return self.criterion

    def eps_sampler(self, shape, epsilon=None, repeat=1, dist=None, device=None) :
        eps = epsilon or self.epsilon
        device = device or self.device
        repeat_shape = torch.Size([shape[0]*repeat, *shape[1:]])
        dist = dist or self.dist
        # 1. uniform
        if dist == "uniform" :
            distribution = distributions.uniform.Uniform(-eps,eps)
            noise = distribution.sample(sample_shape=repeat_shape)
        # 2. Gaussian with 0 mean & 1 std
        elif dist == "gaussian" or "normal" :
            distribution = distributions.normal.Normal(0, eps)
            noise = distribution.sample(sample_shape=repeat_shape)

        return noise

    @overrides
    def perturb(self, X, y, model=None, epsilon=None, repeat=None, dist=None, device=None) :
        criterion = self.loss(device)
        model = model or self.model
        epsilon = epsilon or self.epsilon
        device = device or self.device
        repeat = repeat or self.repeat
        model.zero_grad()

        noise = self.eps_sampler(shape=X.size(), epsilon=epsilon, repeat=repeat, dist=dist, device=device).to(device)
        if self.norm :
            X_adv = X.repeat(repeat,1,1,1)
            X_adv = self.inverse_normalize(X_adv)
            X_adv = X_adv + noise
            X_adv = torch.clamp(X_adv, 0., 1.)
            X_adv = self.normalize(X_adv)
        else :
            X_adv = X.repeat(repeat,1,1,1) + noise
            X_adv = torch.clamp(X_adv, 0., 1.)
        return X_adv
