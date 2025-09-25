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
data = sio.loadmat("./data/reference.mat")
x_ref = data["x"].flatten() - 0.5
y1 = data["y1"].flatten()
y2 = data["y2"].flatten()
y3 = data["y3"].flatten()
x_test = np.linspace(-0.5, 0.5, 100).reshape([-1, 1])
print(x_test.shape)


n = 1000
device = torch.device("cuda:0")
model = torch.load("./checkpoints_training/model_case2_500")
x_test = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
).to(device)

u_pred = model.forward4(x_test).detach().cpu().numpy()
print(u_pred.shape, x_test.shape)

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for j in range(n):
    ax.plot(x_test.detach().cpu().numpy()[j, ...], u_pred[j, ...], linewidth=0.5)
ax.set_title("PINNs solutions at 500th iteration")
ax.set_ylim([-2, 2])
ax.set_xlim([-0.5, 0.5])
ax.set_aspect(0.25)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
fig.savefig("./results/u_training_1.png", bbox_inches="tight")



n = 1000
device = torch.device("cuda:0")
model = torch.load("./checkpoints_training/model_case2_1500")

u_pred = model.forward4(x_test).detach().cpu().numpy()
print(u_pred.shape, x_test.shape)

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for j in range(n):
    ax.plot(x_test.detach().cpu().numpy()[j, ...], u_pred[j, ...], linewidth=0.5)
ax.set_title("PINNs solutions at 1500th iteration")
ax.set_ylim([-2, 2])
ax.set_xlim([-0.5, 0.5])
ax.set_aspect(0.25)
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
fig.savefig("./results/u_training_2.png", bbox_inches="tight")