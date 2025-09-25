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


def u_ref(lamb):
    theta1 = fsolve(lambda theta: func(theta, lamb=lamb), 0)
    theta2 = fsolve(lambda theta: func(theta, lamb=lamb), 10)
    u1 = -2 * np.log(np.cosh((x_test - 1/2)*theta1/2) / np.cosh(theta1/4))
    u2 = -2 * np.log(np.cosh((x_test - 1/2)*theta2/2) / np.cosh(theta2/4))
    return u1, u2


########################################################
################# Case 2 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case2_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 0.5^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case2_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 0.5^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case2_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 0.5^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 3 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case3_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 1^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case3_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 1^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case3_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 1^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 4 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case4_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 2^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case4_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 2^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case4_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 2^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 5 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case5_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-1, 1)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case5_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-1, 1)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case5_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-1, 1)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 6 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case6_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-2, 2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case6_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-2, 2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case6_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-2, 2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 7 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case7_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-3, 3)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case7_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-3, 3)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case7_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(1000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-3, 3)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 3_5 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case3_5_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 1.5^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case3_5_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 1.5^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case3_5_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$N(0, 1.5^2)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


########################################################
################# Case 6_5 ###################
########################################################
u1, u2 = u_ref(lamb=1)
device = torch.device("cuda:0")


## [1, 50, 50, 1]
name = "case6_5_1_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-2.5, 2.5)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 50, 50, 50, 1]
name = "case6_5_1_50_50_50_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-2.5, 2.5)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')


## [1, 100, 100, 1]
name = "case6_5_1_100_100_1"
model = torch.load("./checkpoints/model_different_initializations_"+name)
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(model.theta.shape)
print(np.sum(u_pred[:, 50] > 3.0))


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.plot(x_test, u1, "k-", linewidth=2)
ax.plot(x_test, u2, "b-", linewidth=2)
for i in range(10000):
    ax.plot(x_test, u_pred[i, ...], "r--", linewidth=2)
ax.set_ylim([-0.5, 4.5])
ax.legend(["Reference of $u_1$", "Reference of $u_2$", "PINNs"], frameon=False, loc=2)
ax.set_title("$U[-2.5, 2.5)$")
ax.set_xlabel("$t$")
ax.set_aspect(0.2)
fig.savefig("./outputs/different_initialization_{}.png".format(name), bbox_inches='tight')