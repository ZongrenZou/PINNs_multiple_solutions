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
print(x_test.shape)
u_test = data["u_test"]
f_test = data["f_test"]



model = torch.load("./checkpoints_a/model_0_case3")
u_pred = model.forward(xx).detach().cpu().numpy()
us = [u_pred[:, 20, :]]
for i in range(10000):
    if (i+1) % 50 == 0:
        model = torch.load("./checkpoints_a_training/model_{}".format(str(i+1)))
        u_pred = model.forward(xx).detach().cpu().numpy()
        us += [u_pred[:, 20, :]]
us = np.concatenate(us, axis=-1)
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(201):
    ax.plot(np.linspace(0, 10000, 201), us[i, :], linewidth=0.5)
ax.set_title("Convergence of $u_\\theta(-0.8)$")
ax.set_xlim([0, 10000])
ax.set_ylim([-15, 15])
ax.set_aspect(10000/30)
ax.set_xlabel("# of iterations")
# ax.set_ylabel("$u$")
plt.savefig("./results/case_a_training.png", bbox_inches="tight")

