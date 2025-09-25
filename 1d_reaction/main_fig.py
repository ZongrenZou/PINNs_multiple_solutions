import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import models
import utils

np.random.seed(99018)
torch.manual_seed(66271)

###################################################################
####################### Initialize data ###########################
###################################################################
ref = sio.loadmat("data/ref_1.mat")
print(ref.keys())


x_test = np.linspace(-0.5, 0.5, 6000).reshape([-1, 1])
print(x_test.shape)


n = 1000
device = torch.device("cuda:0")
model = torch.load("./checkpoints_1/model_case1")
x_test = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
    requires_grad=True,
).to(device)

u_pred = model.forward4(x_test)
u_x_pred = torch.autograd.grad(
    u_pred,
    x_test,
    grad_outputs=torch.ones_like(x_test),
    create_graph=True,
)[0]

