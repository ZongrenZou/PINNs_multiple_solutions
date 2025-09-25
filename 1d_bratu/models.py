import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import utils
    

class PNN(nn.Module):
    
    def __init__(self, units=50, n=100, std=1, factor=1, activation=torch.tanh):
        super(PNN, self).__init__()
        self.n = n
        self.std = std
        self.factor = factor
        weights = [
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
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
            nn.init.normal_(w, mean=0, std=self.std)
        for b in self._bias:
            nn.init.normal_(b, mean=0, std=self.std)

    def forward(self, x):
        var_list = utils.unflatten(self.theta, self.shapes)
        weights = var_list[:len(var_list)//2]
        bias = var_list[len(var_list)//2:]

        if len(x.shape) == 2:
            out = torch.tile(x[None, ...], [self.n, 1, 1])
        else:
            out = x
        for i in range(len(weights) - 1):
            out = torch.einsum("nbi,nij->nbj", out, weights[i]) + bias[i]
            out = self.activation(out)
        out = torch.einsum("nbi,nij->nbj", out, weights[-1]) + bias[-1]
        out = self.factor * out * x * (1 - x)
        return out


class MHNN(nn.Module):
    
    def __init__(self, units=50, n=100, std=1, factor=1, activation=torch.tanh):
        super(MHNN, self).__init__()
        self.n = n
        self.std = std
        self.factor = factor
        weights_shared = [
            nn.Parameter(torch.empty(1, units), requires_grad=True),
            # nn.Parameter(torch.empty(units, units), requires_grad=True),
        ]
        bias_shared = [
            nn.Parameter(torch.empty(1, units), requires_grad=True),
            # nn.Parameter(torch.empty(1, units), requires_grad=True),
        ]
        weights = [
            nn.Parameter(torch.empty(n, units, units), requires_grad=True),
            nn.Parameter(torch.empty(n, units, 1), requires_grad=True),
        ]
        bias = [
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
            nn.Parameter(torch.empty(n, 1, 1), requires_grad=True),
        ]
        self._weights_shared = nn.ParameterList(weights_shared)
        self._bias_shared = nn.ParameterList(bias_shared)
        self._weights = nn.ParameterList(weights)
        self._bias = nn.ParameterList(bias)
        self.activation = activation
        self.initialize()
    
    def initialize(self):
        # TODO: customize it
        for i in range(len(self._weights)):
            nn.init.normal_(self._weights[i], mean=0, std=self.std)
            nn.init.normal_(self._bias[i], mean=0, std=self.std)
        for i in range(len(self._weights_shared)):
            nn.init.normal_(self._weights_shared[i], mean=0, std=self.std)
            nn.init.normal_(self._bias_shared[i], mean=0, std=self.std)

    def forward(self, x):
        weights_shared = self._weights_shared
        bias_shared = self._bias_shared
        weights = self._weights
        bias = self._bias

        if len(x.shape) == 2:
            out = torch.tile(x[None, ...], [self.n, 1, 1])
        else:
            out = x

        for i in range(len(weights_shared)):
            out = torch.einsum("nbi,ij->nbj", out, weights_shared[i]) + bias_shared[i][None, ...]
            out = self.activation(out)
        for i in range(len(weights)-1):
            out = torch.einsum("nbi,nij->nbj", out, weights[i]) + bias[i]
            out = self.activation(out)
        out = torch.einsum("nbi,nij->nbj", out, weights[-1]) + bias[-1]
        out = 1 * out * x * (1 - x)
        return out


class PNN2(nn.Module):
    
    def __init__(self, units=50, n=100, R=1, activation=torch.tanh):
        super(PNN2, self).__init__()
        self.n = n
        self.R = R
        weights = [
            nn.Parameter(torch.empty(n, 1, units), requires_grad=True),
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
            nn.init.uniform_(w, a=-self.R, b=self.R)
        for b in self._bias:
            nn.init.uniform_(b, a=-self.R, b=self.R)

    def forward(self, x):
        var_list = utils.unflatten(self.theta, self.shapes)
        weights = var_list[:len(var_list)//2]
        bias = var_list[len(var_list)//2:]

        if len(x.shape) == 2:
            out = torch.tile(x[None, ...], [self.n, 1, 1])
        else:
            out = x
        for i in range(len(weights) - 1):
            out = torch.einsum("nbi,nij->nbj", out, weights[i]) + bias[i]
            out = self.activation(out)
        out = torch.einsum("nbi,nij->nbj", out, weights[-1]) + bias[-1]
        out = 1 * out * x * (1 - x)
        return out
