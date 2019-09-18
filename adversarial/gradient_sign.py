from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from . import AttackBase


class FGSM(AttackBase) :
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, **kwargs) :
        super(FGSM,self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.rand = random_start

    @overrides
    def perturb(self, x, y, model=None, bound=None, device=None, **kwargs) :
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand :
            rand_perturb_dist = distributions.uniform.Uniform(-bound,bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, x_nat, bound=bound, inverse_normalized=True)
            if self.discretize :
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else :
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss" :
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
        
        return x_adv.detach()

class LinfPGD(AttackBase) :
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True, device=None, **kwargs) :
        super(LinfPGD,self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.step = step
        self.iter = iters
        self.rand = random_start

    @overrides
    def perturb(self, x, y, model=None, bound=None, step=None, iters=None, x_nat=None, device=None, **kwargs) :
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        model.zero_grad()
        if x_nat is None :
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else :
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand :
            rand_perturb_dist = distributions.uniform.Uniform(-bound,bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat), bound=bound, inverse_normalized=True)
            if self.discretize :
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else :
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        for i in range(iters) :
            pred = model(x_adv)
            if criterion.__class__.__name__ == "NLLLoss" :
                pred = F.softmax(pred, dim=-1)
            loss = criterion(pred, y)
            loss.backward()

            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)
