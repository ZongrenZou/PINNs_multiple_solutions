import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import models
import utils


np.random.seed(442123)
torch.manual_seed(89712)


data = sio.loadmat("./data/data_case1.mat")
x = data["xx"].reshape([-1, 1])
y = data["yy"].reshape([-1, 1])
x_b = data["x_b"].reshape([-1, 1])
y_b = data["y_b"].reshape([-1, 1])
x_test = np.linspace(0, 1, 100)
y_test = 0.5 * np.ones([1])
x_test, y_test = np.meshgrid(x_test, y_test)
x_test = x_test.reshape([-1, 1])
y_test = y_test.reshape([-1, 1])


device = torch.device("cuda:0")
n = 500
x_f_train = torch.tensor(
    np.tile(x[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)
y_f_train = torch.tensor(
    np.tile(y[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)
x_b_train = torch.tensor(
    np.tile(x_b[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)
y_b_train = torch.tensor(
    np.tile(y_b[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)


x_pred = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
).to(device)
y_pred = torch.tensor(
    np.tile(y_test[None, ...], [n, 1, 1]),
    dtype=torch.float32, 
).to(device)


std = 5
model = models.PNN2D(units=50, n=n, std=std).to(device)
torch.save(model, "./checkpoints/model_0_case6_high")
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


def update():
    

    def loss_function():
        u = model.forward(x_f_train, y_f_train)
        u_x = torch.autograd.grad(
            u, 
            x_f_train,
            grad_outputs=torch.ones_like(x_f_train), 
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u_x, 
            x_f_train,
            grad_outputs=torch.ones_like(x_f_train), 
            create_graph=True,
        )[0]
        u_y = torch.autograd.grad(
            u, 
            y_f_train,
            grad_outputs=torch.ones_like(y_f_train), 
            create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            u_y, 
            y_f_train,
            grad_outputs=torch.ones_like(y_f_train), 
            create_graph=True,
        )[0]
        res = 0.05 * (u_xx + u_yy) + u ** 2 - 2 * torch.sin(np.pi*x_f_train) * torch.sin(np.pi*y_f_train)
        u_b = model.forward(x_b_train, y_b_train)

        loss_f = torch.mean(res ** 2) + torch.mean(u_b ** 2)
        return loss_f # + 1e-6 * torch.sum(model.theta ** 2)
    
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
            current_loss = loss.item()
            print(i+1, current_loss, flush=True)

            model.train()

            u_pred = model.forward(
                x_pred, y_pred
            ).detach().cpu().numpy()
            # u_pred = u_pred.reshape([n, 101, 101])
            # x_pred = x_test.reshape([101, 101])
            # y_pred = y_test.reshape([101, 101])

            # u_pred = model.forward(
            #     torch.tensor(x_test, dtype=torch.float32, device=device),
            # ).detach().cpu().numpy()
            # print(np.sum(u_pred[:, 50] > 3))

            plt.figure()
            for j in range(n):
                plt.plot(x_test, u_pred[j], "--")
            # plt.ylim([-0.5, 4.5])
            plt.title("Deep ensemble at {} iteration".format(str(i+1)))
            plt.xlabel("$x$")
            plt.ylabel("$u(x, 0.5)$")
            plt.savefig("./outputs/sgd_x/sgd_{}.png".format(str(i+1)))
            plt.close()

print(model.theta.shape)
train()


torch.save(model, "./checkpoints/model_1_case6_high")
# u_pred = model.forward(
#     torch.tensor(x_test, dtype=torch.float32, device=device),
# ).detach().cpu().numpy()


# plt.figure()
# plt.plot(x_test, u1, "k-")
# plt.plot(x_test, u2, "b-")
# for i in range(n):
#     plt.plot(x_test, u_pred[i, ...], "--")
# plt.ylim([-0.5, 4.5])
# plt.title("Deep ensemble")
# plt.savefig("./outputs/sgd.png")
# plt.show()

