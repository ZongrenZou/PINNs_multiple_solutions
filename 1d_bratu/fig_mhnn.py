import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from scipy.optimize import fsolve


import models
import utils


data = sio.loadmat("./data/data.mat")
x_test = data["x_test"]


def func(theta, lamb=1):
    return np.sqrt(2*lamb) * np.cosh(theta/4) - theta


theta1 = fsolve(func, 0)
theta2 = fsolve(func, 10)
u1 = -2 * np.log(np.cosh((x_test - 1/2)*theta1/2) / np.cosh(theta1/4))
u2 = -2 * np.log(np.cosh((x_test - 1/2)*theta2/2) / np.cosh(theta2/4))


device = torch.device("cuda:0")

model = torch.load("./checkpoints_mhnn/model_1")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(np.sum(u_pred[:, 50] > 1.5))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "MHPINNs"], frameon=False, loc=2)
ax.set_title("Parameter sharing PINNs")
ax.set_xlabel("$t$")
ax.set_ylim([-0.5, 7.5])
ax.set_xlim([0, 1])
ax.set_aspect(1/8)
fig.savefig("./outputs/mhnn.png", bbox_inches='tight')
