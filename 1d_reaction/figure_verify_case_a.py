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
x_test = np.linspace(-1, 1, 201).reshape([-1, 1])
print(x_test.shape)
n = 1000
device = torch.device("cuda:0")

xx = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
    requires_grad=True,
).to(device)




data = sio.loadmat("./data/data.mat")
x_test = data["x_test"]
u_test = data["u_test"]

model = torch.load("./checkpoints_a/model_1_case3")
u_pred = model.forward(xx).detach().cpu().numpy()
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u_test, "k--", linewidth=3)
for j in range(n):
    ax.plot(x_test, u_pred[j, ...], "r-", linewidth=0.5)
ax.set_title("Ensemble PINNs")
ax.set_xlim([-1, 1])
ax.set_ylim([-5, 5])
ax.set_aspect(2/10)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
# ax.legend(["The prescribed solution", "PINN solutions"], frameon=False)
fig.savefig("./results/u_case_a_pinn.png", bbox_inches="tight")



data_densest = sio.loadmat("./results/case_a_densest.mat")
x = data_densest["x"].reshape([-1, 1])
y = data_densest["ys"]
print(x.shape, y.shape)

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x, np.sin(6*x)**3, "k--", linewidth=3)
for i in range(11):
    ax.plot(x, y[i, ...], "m-", linewidth=0.5)
ax.set_title("PINN solutions as initial guesses")
ax.set_xlim([-1, 1])
ax.set_ylim([-5, 5])
ax.set_aspect(2/10)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
# ax.legend(["The prescribed solution", "Our solutions"], frameon=False)
fig.savefig("./results/u_case_a_our.png", bbox_inches="tight")

