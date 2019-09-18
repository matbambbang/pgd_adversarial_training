from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from . import AttackBase


class GradientAttack(AttackBase) :
    def __init__(self, model=None, p=2, bound=None, norm=False, discrete=True, device=None, **kwargs) :
        super(GradientAttack,self).__init__(model, norm, discrete, device)
        self.p = p
        self.bound = bound

    @overrides
    def perturb(self, x, y, model=None, p=None, bound=None, device=None, **kwargs) :
        criterion = self.criterion
        model = model or self.model
        p = p or self.p
        bound = bound or self.bound

        model.zero_grad()
        x_nat = x.detach().clone().to(device)
        x_adv = x.detach().clone().requires_grad_(True).to(device)

        pred = model(x_adv)
        loss = criterion(pred, y)
        loss.backward()

        grad = x_adv.grad.data.detach()
        grad_unit = grad / grad.norm(p=p, dim=(2,3), keepdim=True)
        x_adv = self.inverse_normalize(x_adv) + grad_unit * bound
        x_adv = self.clamper(x_adv, x_nat, metric=p, bound=bound, inverse_normalized=True)

        return self.normalize(x_adv).detach()

class ProjectedGradientAttack(AttackBase) :
    def __init__(self, model=None, p=2, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True, device=None, **kwargs) :
        super(ProjectedGradientAttack,self).__init__(model, norm, discrete, device)
        self.p = p
        self.bound = bound
        self.step = step
        self.iters = iters
        self.rand = random_start

    @overrides
    def perturb(self, x, y, model=None, p=None, bound=None, step=None, iters=None, device=None, **kwargs) :
        criterion = self.criterion
        model = model or self.model
        p = p or self.p
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iters
        device = device or self.device

        model.zero_grad()
        x_nat = x.detach().clone().to(device)
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        #TODO: random start criterion
        if self.rand :
            rand_perturb_dist = distributions.uniform.Uniform(-bound,bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            #x_adv = self.clamper(self.inverse_

        for i in range(iters) :
            pred = model(x_adv)
            loss = criterion(pred, y)
            loss.backward()

            grad = x_adv.grad.data.detach()
            grad_unit = grad / grad.norm(p=p, dim=(2,3), keepdim=True)
            x_adv = self.inverse_normalize(x_adv) + grad_unit * step
            x_adv = self.clamper(x_adv, x_nat, metric=p, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)
