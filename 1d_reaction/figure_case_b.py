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




data = sio.loadmat("./data/data_case_b.mat")
x_test = data["x_test"]
u_test = data["u_test"]
f_test = data["f_test"]


model = torch.load("./checkpoints_b/model_1_case1")
u_pred = model.forward(xx).detach().cpu().numpy()
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u_test, "k--", linewidth=3)
for j in range(n):
    ax.plot(x_test, u_pred[j, ...], "r-", linewidth=0.5)
ax.set_title("$TruncatedNormal(0, 0.2^2)$")
ax.set_xlim([-1, 1])
ax.set_ylim([-3, 3])
ax.set_aspect(2/6)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
ax.legend(["The prescribed solution", "PINN solutions"], frameon=False)
fig.savefig("./results/u_case_b_case1.png", bbox_inches="tight")


model = torch.load("./checkpoints_b/model_1_case2")
u_pred = model.forward(xx).detach().cpu().numpy()
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u_test, "k--", linewidth=3)
for j in range(n):
    ax.plot(x_test, u_pred[j, ...], "r-", linewidth=0.5)
ax.set_title("$TruncatedNormal(0, 0.5^2)$")
ax.set_xlim([-1, 1])
ax.set_ylim([-10, 10])
ax.set_aspect(2/20)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
# ax.legend(["The prescribed solution", "PINN solutions"], frameon=False)
fig.savefig("./results/u_case_b_case2.png", bbox_inches="tight")


model = torch.load("./checkpoints_b/model_1_case3")
u_pred = model.forward(xx).detach().cpu().numpy()
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u_test, "k--", linewidth=3)
for j in range(n):
    ax.plot(x_test, u_pred[j, ...], "r-", linewidth=0.5)
ax.set_title("$TruncatedNormal(0, 1^2)$")
ax.set_xlim([-1, 1])
ax.set_ylim([-40, 40])
ax.set_aspect(2/80)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
# ax.legend(["The prescribed solution", "PINN solutions"], frameon=False)
fig.savefig("./results/u_case_b_case3.png", bbox_inches="tight")
