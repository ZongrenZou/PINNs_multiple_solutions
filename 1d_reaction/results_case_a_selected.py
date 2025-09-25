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

ind = [343, 293, 396, 605, 408, 284, 538, 515, 368, 370, 320]

# pick 11
n = 11
device = torch.device("cuda:0")
model = torch.load("./checkpoints_a/model_1_case3")
model.theta = torch.nn.Parameter(model.theta[ind, :].detach(), requires_grad=False)
model.n = n
new_shapes = []
for shape in model.shapes:
    new_shapes += [torch.Size([n, shape[1], shape[2]])]
model.shapes = new_shapes


# denser
x_test_denser = np.linspace(-1, 1, 3201).reshape([-1, 1])
x_test_denser = torch.tensor(
    np.tile(x_test_denser[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
    requires_grad=True,
).to(device)


u_pred_denser = model.forward(x_test_denser)
u_x_pred_denser = torch.autograd.grad(
    u_pred_denser,
    x_test_denser,
    grad_outputs=torch.ones_like(x_test_denser),
    create_graph=True,
)[0]
u_pred_denser = u_pred_denser.detach().cpu().numpy()
u_x_pred_denser = u_x_pred_denser.detach().cpu().numpy()


# densest
x_test_densest = np.linspace(-1, 1, 6401).reshape([-1, 1])
x_test_densest = torch.tensor(
    np.tile(x_test_densest[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
    requires_grad=True,
).to(device)


u_pred_densest = model.forward(x_test_densest)
u_x_pred_densest = torch.autograd.grad(
    u_pred_densest,
    x_test_densest,
    grad_outputs=torch.ones_like(x_test_densest),
    create_graph=True,
)[0]
u_pred_densest = u_pred_densest.detach().cpu().numpy()
u_x_pred_densest = u_x_pred_densest.detach().cpu().numpy()


sio.savemat(
    "./results/pinn_case_a_selected.mat",
    {
        # "x_test": np.linspace(-1, 1, 201).reshape([-1, 1]),
        # "u_pred": u_pred,
        # "u_x_pred": u_x_pred,
        "x_test_denser": np.linspace(-1, 1, 3201).reshape([-1, 1]),
        "u_pred_denser": u_pred_denser,
        "u_x_pred_denser": u_x_pred_denser,
        "x_test_densest": np.linspace(-1, 1, 6401).reshape([-1, 1]),
        "u_pred_densest": u_pred_densest,
        "u_x_pred_densest": u_x_pred_densest,
    }
)
