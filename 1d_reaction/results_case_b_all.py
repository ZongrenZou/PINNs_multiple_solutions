import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import models
import utils


###################################################################
####################### Initialize data ###########################
###################################################################

# sparse
x_test = np.linspace(-1, 1, 201).reshape([-1, 1])
print(x_test.shape)


n = 1000
device = torch.device("cuda:0")
model = torch.load("./checkpoints_b/model_1_case3")
x_test = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
    requires_grad=True,
).to(device)

u_pred = model.forward(x_test)
u_x_pred = torch.autograd.grad(
    u_pred,
    x_test,
    grad_outputs=torch.ones_like(x_test),
    create_graph=True,
)[0]
u_pred = u_pred.detach().cpu().numpy()
u_x_pred = u_x_pred.detach().cpu().numpy()


sio.savemat(
    "./results/pinn_case_b_all.mat",
    {
        "x_test": np.linspace(-1, 1, 201).reshape([-1, 1]),
        "u_pred": u_pred,
        "u_x_pred": u_x_pred,
    }
)
