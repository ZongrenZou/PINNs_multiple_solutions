import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import models
import utils


np.random.seed(2220191)
torch.manual_seed(98291)

###################################################################
####################### Initialize data ###########################
###################################################################

x_test = np.linspace(-0.5, 0.5, 100).reshape([-1, 1])
print(x_test.shape)


device = torch.device("cuda:0")
n = 1000
x_f_train = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)

x_d = np.array([-0.3]).reshape([-1, 1])
x_d_train = torch.tensor(
    np.tile(x_d[None, ...], [n, 1, 1]),
    dtype=torch.float32,
    requires_grad=True,
).to(device)


###################################################################
##################### Initialize models ###########################
###################################################################
eps = 0.06
# Case 1: N(0, 0.1^2)
# Case 2: N(0, 1^2)
std = 1
model = models.PNN(units=50, n=n, std=std).to(device)

torch.save(model, "./checkpoints_3/model_0")
optimizer = torch.optim.Adam(
    [model.theta],
    lr=1e-3,
    weight_decay=0,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[5000], 
    gamma=0.1,
)
forward_fn = model.forward4


u_pred = forward_fn(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
# print(np.sum(u_pred[:, 50] > 3))

plt.figure()
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
        loss_f = torch.mean((eps * u_xx - u * u_x + u) ** 2)
        loss_d = loss_f.detach() * torch.mean(torch.exp(-forward_fn(x_d_train)))

        loss = loss_f + loss_d
        return loss

    
    def closure():
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        return loss

    optimizer.step(closure)
    scheduler.step()
    return loss_function()
        


def train(): 
    niter = 20000

    for i in range(niter):
        loss = update()
        
        if (i + 1) % 500 == 0:
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
                torch.tensor(x_test, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
            # print(np.sum(u_pred[:, 50] > 3))

            plt.figure()
            for j in range(n):
                plt.plot(x_test, u_pred[j, ...], "--")
            plt.title("Deep ensemble at {} iteration".format(str(i+1)))
            plt.savefig("./outputs/sgd_{}.png".format(str(i+1)))
            plt.close()

print(model.theta.shape)
train()


torch.save(model, "./checkpoints_3/model_1")
