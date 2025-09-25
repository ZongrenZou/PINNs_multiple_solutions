import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import utils


class PNN2D(nn.Module):
    
    def __init__(self, units=50, n=100, std=1, activation=torch.tanh):
        super(PNN2D, self).__init__()
        self.n = n
        self.std = std
        weights = [
            nn.Parameter(torch.empty(n, 2, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            # nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, 1), requires_grad=True),
        ]
        bias = [
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            # nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, 1), requires_grad=True),
        ]
        self._weights = nn.ParameterList(weights)
        self._bias = nn.ParameterList(bias)
        self.activation = activation
        self.initialize()

        self.num_params = 0
        for p in self.parameters():
            self.num_params += p.numel()

        self.param_shapes = [p.shape[1:] for p in self.parameters()]
        theta, shapes = utils.flatten(list(self.parameters()))
        self.theta = nn.Parameter(theta.detach(), requires_grad=True)
        self.shapes = shapes
    
    def initialize(self):
        # TODO: customize it
        for w in self._weights:
            nn.init.trunc_normal_(w, mean=0, std=self.std)
        for b in self._bias:
            # nn.init.zeros_(b)
            nn.init.trunc_normal_(b, mean=0, std=self.std)

    def forward(self, x, y):
        var_list = utils.unflatten(self.theta, self.shapes)
        weights = var_list[:len(var_list)//2]
        bias = var_list[len(var_list)//2:]

        if len(x.shape) == 2:
            x = torch.tile(x[None, ...], [self.n, 1, 1])
            y = torch.tile(y[None, ...], [self.n, 1, 1])

        out = torch.cat([x, y], dim=-1)
        for i in range(len(weights) - 1):
            out = torch.einsum("nbi,nij->nbj", out, weights[i]) + bias[i]
            out = self.activation(out)
        out = torch.einsum("nbi,nij->nbj", out, weights[-1]) + bias[-1]
        return out


class PNN2D2(nn.Module):
    
    def __init__(self, units=50, n=100, std=1, activation=torch.tanh):
        super(PNN2D2, self).__init__()
        self.n = n
        self.std = std
        weights = [
            nn.Parameter(torch.empty(n, 2, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, 1), requires_grad=True),
        ]
        bias = [
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, 1), requires_grad=True),
        ]
        self._weights = nn.ParameterList(weights)
        self._bias = nn.ParameterList(bias)
        self.activation = activation
        self.initialize()

        self.num_params = 0
        for p in self.parameters():
            self.num_params += p.numel()

        self.param_shapes = [p.shape[1:] for p in self.parameters()]
        theta, shapes = utils.flatten(list(self.parameters()))
        self.theta = nn.Parameter(theta.detach(), requires_grad=True)
        self.shapes = shapes
    
    def initialize(self):
        # TODO: customize it
        for w in self._weights:
            nn.init.trunc_normal_(w, mean=0, std=self.std)
        for b in self._bias:
            # nn.init.zeros_(b)
            nn.init.trunc_normal_(b, mean=0, std=self.std)

    def forward(self, x, y):
        var_list = utils.unflatten(self.theta, self.shapes)
        weights = var_list[:len(var_list)//2]
        bias = var_list[len(var_list)//2:]

        if len(x.shape) == 2:
            x = torch.tile(x[None, ...], [self.n, 1, 1])
            y = torch.tile(y[None, ...], [self.n, 1, 1])

        out = torch.cat([x, y], dim=-1)
        for i in range(len(weights) - 1):
            out = torch.einsum("nbi,nij->nbj", out, weights[i]) + bias[i]
            out = self.activation(out)
        out = torch.einsum("nbi,nij->nbj", out, weights[-1]) + bias[-1]
        return out



class PNN2D3(nn.Module):
    
    def __init__(self, units=50, n=100, std=1, activation=torch.tanh):
        super(PNN2D3, self).__init__()
        self.n = n
        self.std = std
        weights = [
            nn.Parameter(torch.empty(n, 2, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, 1), requires_grad=True),
        ]
        bias = [
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, 1), requires_grad=True),
        ]
        self._weights = nn.ParameterList(weights)
        self._bias = nn.ParameterList(bias)
        self.activation = activation
        self.initialize()

        self.num_params = 0
        for p in self.parameters():
            self.num_params += p.numel()

        self.param_shapes = [p.shape[1:] for p in self.parameters()]
        theta, shapes = utils.flatten(list(self.parameters()))
        self.theta = nn.Parameter(theta.detach(), requires_grad=True)
        self.shapes = shapes
    
    def initialize(self):
        # TODO: customize it
        for w in self._weights:
            nn.init.trunc_normal_(w, mean=0, std=self.std)
        for b in self._bias:
            # nn.init.zeros_(b)
            nn.init.trunc_normal_(b, mean=0, std=self.std)

    def forward(self, x, y):
        var_list = utils.unflatten(self.theta, self.shapes)
        weights = var_list[:len(var_list)//2]
        bias = var_list[len(var_list)//2:]

        if len(x.shape) == 2:
            x = torch.tile(x[None, ...], [self.n, 1, 1])
            y = torch.tile(y[None, ...], [self.n, 1, 1])

        out = torch.cat([x, y], dim=-1)
        for i in range(len(weights) - 1):
            out = torch.einsum("nbi,nij->nbj", out, weights[i]) + bias[i]
            out = self.activation(out)
        out = torch.einsum("nbi,nij->nbj", out, weights[-1]) + bias[-1]
        return out
