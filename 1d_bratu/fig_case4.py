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


def func(theta, lamb=3.5):
    return np.sqrt(2*lamb) * np.cosh(theta/4) - theta


theta1 = fsolve(func, 0)
theta2 = fsolve(func, 10)
u1 = -2 * np.log(np.cosh((x_test - 1/2)*theta1/2) / np.cosh(theta1/4))
u2 = -2 * np.log(np.cosh((x_test - 1/2)*theta2/2) / np.cosh(theta2/4))


device = torch.device("cuda:0")

model = torch.load("./checkpoints_1/model_case4")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 1.5))


plt.figure(dpi=200)
plt.plot(x_test, u1, "k-", linewidth=2)
plt.plot(x_test, u2, "b-", linewidth=2)
for i in range(10):
    plt.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
plt.ylim([-0.5, 4.5])
plt.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
plt.title("$\lambda = 3.5$")
plt.xlabel("$x$")
plt.savefig("./outputs/case4.png")
plt.close()

