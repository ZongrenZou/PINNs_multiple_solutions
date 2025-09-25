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


def u_ref(lamb):
    theta1 = fsolve(lambda theta: func(theta, lamb=lamb), 0)
    theta2 = fsolve(lambda theta: func(theta, lamb=lamb), 10)
    u1 = -2 * np.log(np.cosh((x_test - 1/2)*theta1/2) / np.cosh(theta1/4))
    u2 = -2 * np.log(np.cosh((x_test - 1/2)*theta2/2) / np.cosh(theta2/4))
    return u1, u2


########################################################
################# Case 1: lamb = 3.5 ###################
########################################################
u1, u2 = u_ref(lamb=3.5)
device = torch.device("cuda:0")

model = torch.load("./checkpoints/model_different_lambs_1")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$\lambda = 3.5$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_lambs_1.png", bbox_inches='tight')


########################################################
################## Case 2: lamb = 2 ####################
########################################################
# case 2: lamb = 2
u1, u2 = u_ref(lamb=2)
device = torch.device("cuda:0")

model = torch.load("./checkpoints/model_different_lambs_2")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
# ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$\lambda = 2$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_lambs_2.png", bbox_inches='tight')


########################################################
################## Case 3: lamb = 1 ####################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")

model = torch.load("./checkpoints/model_different_lambs_3")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
# ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$\lambda = 1$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_lambs_3.png", bbox_inches='tight')


########################################################
################# Case 4: lamb = 0.5 ###################
########################################################
u1, u2 = u_ref(lamb=0.5)
device = torch.device("cuda:0")

model = torch.load("./checkpoints/model_different_lambs_4")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
# ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$\lambda = 0.5$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_lambs_4.png", bbox_inches='tight')


########################################################
################# Case 4: lamb = 0.5 ###################
########################################################
device = torch.device("cuda:0")

model = torch.load("./checkpoints/model_different_lambs_0")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(10):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.set_title("Initializations of ten NNs")
# ax.legend(["NNs"], frameon=False, loc=2)
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_lambs_0.png", bbox_inches='tight')
