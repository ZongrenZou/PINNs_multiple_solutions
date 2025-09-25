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
################## Compute loss #####################
#####################################################
n = 1000
x_f_train = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)

model = torch.load("./checkpoints_training/model_1000")
u_pred = model.forward(x_f_train).detach().cpu().numpy()
idx = u_pred[:, 50].flatten() < 3


loss = []
loss_1 = []
loss_2 = []
for i in range(101):
    model = torch.load("./checkpoints_training/model_{}".format(str(int(i * 10))))
    u_pred = model.forward(x_f_train)
    u_x_pred = torch.autograd.grad(u_pred.sum(), x_f_train, create_graph=True)[0]
    u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x_f_train, create_graph=True)[0]
    f_pred = u_xx_pred + torch.exp(u_pred)
    _loss = torch.mean(f_pred ** 2, dim=1).detach().cpu().numpy().flatten()
    u_pred = u_pred.detach().cpu().numpy()
    loss += [_loss.reshape([-1, 1])]
    
    # loss_1 += [loss[idx].reshape([-1, 1])]
    # loss_2 += [loss[~idx].reshape([-1, 1])]

# loss_1 = np.concatenate(loss_1, axis=-1)
# loss_2 = np.concatenate(loss_2, axis=-1)
loss = np.concatenate(loss, axis=-1)
fig = plt.figure(dpi=200)
ax = fig.add_subplot()
for i in range(1000):
    ax.semilogy(np.arange(0, 1010, 10), loss[i, :], linewidth=0.5)
ax.set_xlabel("# of iterations")
ax.set_title("PINN losses")
ax.set_xlim([0, 1000])
ax.set_ylim([1e-4, 1e4])
ax.set_aspect(1000/8)
plt.savefig("./outputs/losses.png", bbox_inches="tight")


#####################################################
################### 50th epoch ######################
#####################################################
model = torch.load("./checkpoints_training/model_50")
u_pred = model.forward(x_f_train)
u_x_pred = torch.autograd.grad(u_pred.sum(), x_f_train, create_graph=True)[0]
u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x_f_train, create_graph=True)[0]
f_pred = u_xx_pred + torch.exp(u_pred)
_loss = torch.mean(f_pred ** 2, dim=1).detach().cpu().numpy().flatten()
print(_loss.shape)

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.hist(np.log(_loss), bins=50, density=True)
# ax.set_xlabel("# of iterations")
ax.set_title("Logarithm of PINN losses at $50$th iteration")
ax.set_xlim([-6, 6])
ax.set_ylim([0, 0.4])
ax.set_aspect(12/0.4)
plt.savefig("./outputs/losses_50.png", bbox_inches="tight")



#####################################################
################### 100th epoch #####################
#####################################################
model = torch.load("./checkpoints_training/model_100")
u_pred = model.forward(x_f_train)
u_x_pred = torch.autograd.grad(u_pred.sum(), x_f_train, create_graph=True)[0]
u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x_f_train, create_graph=True)[0]
f_pred = u_xx_pred + torch.exp(u_pred)
_loss = torch.mean(f_pred ** 2, dim=1).detach().cpu().numpy().flatten()
print(_loss.shape)

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.hist(np.log(_loss), bins=50, density=True)
# ax.set_xlabel("# of iterations")
ax.set_title("Logarithm of PINN losses at $100$th iteration")
ax.set_xlim([-6, 6])
ax.set_ylim([0, 0.4])
ax.set_aspect(12/0.4)
plt.savefig("./outputs/losses_100.png", bbox_inches="tight")


#####################################################
################### 200th epoch #####################
#####################################################
model = torch.load("./checkpoints_training/model_200")
u_pred = model.forward(x_f_train)
u_x_pred = torch.autograd.grad(u_pred.sum(), x_f_train, create_graph=True)[0]
u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x_f_train, create_graph=True)[0]
f_pred = u_xx_pred + torch.exp(u_pred)
_loss = torch.mean(f_pred ** 2, dim=1).detach().cpu().numpy().flatten()
print(_loss.shape)

fig = plt.figure(dpi=200)
ax = fig.add_subplot()
ax.hist(np.log(_loss), bins=50, density=True)
# ax.set_xlabel("# of iterations")
ax.set_title("Logarithm of PINN losses at $200$th iteration")
ax.set_xlim([-6, 6])
ax.set_ylim([0, 0.4])
ax.set_aspect(12/0.4)
plt.savefig("./outputs/losses_200.png", bbox_inches="tight")
