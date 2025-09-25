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


#####################################################
################### 50th epoch ######################
#####################################################
model = torch.load("./checkpoints_training/model_50")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))
print(x_test[50])

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(1000):
    plt.plot(x_test, u_pred[i, ...], "-", linewidth=0.5)
ax.set_ylim([-0.5, 4.5])
ax.set_xlim([0, 1])
# plt.plot(x_test, u1, "k-", linewidth=2, label="Reference of $u_1$")
# plt.plot(x_test, u2, "b-", linewidth=2, label="Reference of $u_2$")
# plt.legend(frameon=False, loc=2)
ax.set_title("PINN solutions at $50$th iteration")
# plt.title("U(-2, 2)")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
plt.savefig("./outputs/epoch_50.png", bbox_inches="tight")


#####################################################
################### 100th epoch #####################
#####################################################
model = torch.load("./checkpoints_training/model_100")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))
print(x_test[50])

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(1000):
    plt.plot(x_test, u_pred[i, ...], "-", linewidth=0.5)
ax.set_ylim([-0.5, 4.5])
ax.set_xlim([0, 1])
# plt.plot(x_test, u1, "k-", linewidth=2, label="Reference of $u_1$")
# plt.plot(x_test, u2, "b-", linewidth=2, label="Reference of $u_2$")
# plt.legend(frameon=False, loc=2)
ax.set_title("PINN solutions at $100$th iteration")
# plt.title("U(-2, 2)")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
plt.savefig("./outputs/epoch_100.png", bbox_inches="tight")


#####################################################
################### 200th epoch #####################
#####################################################
model = torch.load("./checkpoints_training/model_200")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))
print(x_test[50])

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "-", linewidth=0.5)
ax.set_ylim([-0.5, 4.5])
ax.set_xlim([0, 1])
# plt.plot(x_test, u1, "k-", linewidth=2, label="Reference of $u_1$")
# plt.plot(x_test, u2, "b-", linewidth=2, label="Reference of $u_2$")
# plt.legend(frameon=False, loc=2)
ax.set_title("PINN solutions at $200$th iteration")
# plt.title("U(-2, 2)")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
plt.savefig("./outputs/epoch_200.png", bbox_inches="tight")


#####################################################
############### Convergence of u(0.5) ###############
#####################################################
us = []
for i in range(101):
    model = torch.load("./checkpoints_training/model_{}".format(str(10*i)))
    u_pred = model.forward(
        torch.tensor(np.array([0.5]).reshape([-1, 1]), dtype=torch.float32, device=device),
    ).detach().cpu().numpy()
    us += [u_pred[..., 0, 0]]
us = np.stack(us, axis=0)
print(us.shape)

x = np.linspace(0, 100, 101) * 10
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(1000):
    ax.plot(x, us[:, i], linewidth=0.5)
ax.set_xlabel("# of iterations")
# plt.ylabel("$u_\\theta(0.5)$")
ax.set_ylim([-5, 5])
ax.set_xlim([0, 1000])
ax.set_aspect(100)
ax.set_title("Convergence of $u_\\theta(0.5)$")
plt.savefig("./outputs/training.png", bbox_inches="tight")


#####################################################
########## PINN solutions as initial guess ##########
#####################################################
data = sio.loadmat("./outputs/for_plot.mat")
u1_init = data["u1_init"].T
u2_init = data["u2_init"].T
sol1 = data["sol1"].T
sol2 = data["sol2"].T

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", label="Reference of $u_1$")
ax.plot(x_test, u2, "b-", label="Reference of $u_2$")
ax.plot(x_test, u1_init, "r--", label="A PINN solution of $u_1$")
ax.plot(x_test, u2_init, "y--", label="A PINN solution of $u_2$")
ax.plot(x_test, sol1, "m-.", label="Approximated solution of $u_1$")
ax.plot(x_test, sol2, "c-.", label="Approximated solution of $u_2$")
ax.set_xlabel("$t$")
ax.legend(
    frameon=False, loc=2,
)
# plt.ylabel("$u_\\theta(0.5)$")

ax.set_ylim([-0.5, 7.5])
ax.set_xlim([0, 1])
ax.set_aspect(1/8)
ax.set_title("PINN solutions at 100th iteration as initial guesses")
plt.savefig("./outputs/initial_guesses.png", bbox_inches="tight")


# #####################################################
# ##################### Training ######################
# #####################################################
# us = []
# for i in range(101):
#     model = torch.load("./checkpoints_3/model_{}".format(str(10*i)))
#     theta = model.theta.detach().cpu().numpy()
#     theta2 = np.sum(theta ** 2, axis=-1)
#     us += [theta2]
# us = np.stack(us, axis=0)
# print(us.shape)

# x = np.linspace(0, 100, 101) * 10
# plt.figure(dpi=200)
# for i in range(1000):
#     plt.plot(x, us[:, i], linewidth=0.5)
# plt.xlabel("# of iterations")
# # plt.ylabel("$u_\\theta(0.5)$")
# plt.title("$L_2$ of $\\theta$")
# plt.savefig("./outputs/training2.png")
