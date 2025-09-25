import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time


import models
import utils


np.random.seed(65172)
torch.manual_seed(94611)


###################################################################
####################### Initialize data ###########################
###################################################################
data = sio.loadmat("./data/data.mat")
x_test = data["x_test"]
u_test = data["u_test"]
f_test = data["f_test"]
x_f_train = x_test
f_train = f_test
x_u_train = np.array([-1, 1]).reshape([-1, 1])
u_train = np.array([u_test[0, 0], u_test[-1, 0]]).reshape([-1, 1])
print(u_test.shape)


device = torch.device("cuda:0")
dtype = torch.float32
n = 1000
x_f_train = torch.tensor(
    np.tile(x_f_train[None, ...], [n, 1, 1]), 
    dtype=dtype, 
    requires_grad=True,
).to(device)
f_train = torch.tensor(
    np.tile(f_train[None, ...], [n, 1, 1]), 
    dtype=dtype, 
).to(device)
x_u_train = torch.tensor(
    np.tile(x_u_train[None, ...], [n, 1, 1]), 
    dtype=dtype, 
).to(device)
u_train = torch.tensor(
    np.tile(u_train[None, ...], [n, 1, 1]), 
    dtype=dtype, 
).to(device)


###################################################################
##################### Initialize models ###########################
###################################################################
# Case 1: N(0, 0.5^2)
# Case 2: N(0, 1^2)
# Case 3: N(0, 1.5^2)
std = 1
model = models.MHNN(
    units=50, 
    n=n, 
    std=std, 
).to(device)


torch.save(model, "./checkpoints_a/mhnn_0")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[20000], 
    gamma=0.1,
)
# forward_fn = model.forward4
# forward_fn = model.forward_lb
forward_fn = model.forward


u_pred = forward_fn(
    torch.tensor(x_test, dtype=dtype, device=device),
).detach().cpu().numpy()
# print(np.sum(u_pred[:, 50] > 3))

plt.figure(dpi=200)
for j in range(n):
    plt.plot(x_test, u_pred[j, ...])
# plt.ylim([-0.5, 4.5])
plt.title("Deep ensemble at 0 iteration")
plt.savefig("./outputs/sgd_0.png")
plt.close()


def update():


    def loss_function():
        u = forward_fn(x_f_train)
        u_x = torch.autograd.grad(
            u.sum(), 
            x_f_train,
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), 
            x_f_train,
            create_graph=True,
        )[0]
        kappa = 0.7
        loss_f = torch.mean((0.01 * u_xx + kappa * torch.tanh(u) - f_train) ** 2)
        loss_u = torch.mean((forward_fn(x_u_train) - u_train) ** 2)
        # print(loss_f.dtype)

        return loss_u + loss_f

    
    def closure():
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        return loss

    optimizer.step(closure)
    scheduler.step()
    return loss_function()


def train(): 
    niter = 30000

    for i in range(niter):
        loss = update()
        
        if (i + 1) % 1000 == 0:
            model.eval()
            # current_loss = loss.item()
            # print(i+1, current_loss, flush=True)
            current_loss = loss
            print(
                i+1,
                current_loss.item(),
            )

            # torch.save(model, "./checkpoints/model0")
            model.train()

            u_pred = forward_fn(
                torch.tensor(x_test, dtype=dtype, device=device),
            ).detach().cpu().numpy()
            # print(np.sum(u_pred[:, 50] > 3))

            plt.figure(dpi=200)
            plt.plot(x_test, u_test, "k-")
            for j in range(n):
                plt.plot(x_test, u_pred[j, ...], "--")
            plt.title("Deep ensemble at {} iteration".format(str(i+1)))
            plt.savefig("./outputs/sgd_{}.png".format(str(i+1)))
            plt.close()


t0 = time.time()
train()
t1 = time.time()
print("Elapsed: ", t1 - t0)


torch.save(model, "./checkpoints_a/mhnn_1")
