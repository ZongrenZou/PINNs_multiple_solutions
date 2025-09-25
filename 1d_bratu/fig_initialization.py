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
print(theta1, theta2)
u1 = -2 * np.log(np.cosh((x_test - 1/2)*theta1/2) / np.cosh(theta1/4))
u2 = -2 * np.log(np.cosh((x_test - 1/2)*theta2/2) / np.cosh(theta2/4))


device = torch.device("cuda:0")

model = torch.load("./checkpoints_2/model_case8")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))
print(x_test[50])

plt.figure(dpi=200)
plt.plot(x_test, u1, "k-", linewidth=2)
plt.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    plt.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
plt.ylim([-0.5, 4.5])
plt.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
plt.title("$N(0, 0.5^2), c=2$")
# plt.title("U(-2, 2)")
plt.xlabel("$x$")
plt.savefig("./outputs/case8.png")
plt.close()


model = torch.load("./checkpoints_2/model_0_case8")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()


plt.figure(dpi=200)
# plt.plot(x_test, u1, "k-", label="Reference of $u_1$", linewidth=2, alpha=0)
# plt.plot(x_test, u2, "b-", label="Reference of $u_2$", linewidth=2, alpha=0)
for i in range(1000):
    plt.plot(x_test, u_pred[i, ...], "-", linewidth=0.5)
plt.ylim([-0.5, 4.5])
# plt.legend(["PINNs"], frameon=False, loc=2)
plt.title("$N(0, 0.5^2), c=2$")
# plt.title("U(-2, 2)")
plt.xlabel("$x$")
plt.savefig("./outputs/case8_0.png")
plt.close()