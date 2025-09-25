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

# np.random.seed(33321)
# torch.manual_seed(3333)
np.random.seed(19381)
torch.manual_seed(9872)

data = sio.loadmat("./data/data.mat")
x_test = data["x_test"]
u1 = data["u1"]
u2 = data["u2"]


device = torch.device("cuda:0")
n = 1000
x_f_train = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)


std = 1
model = models.MHNN(
    units=50, 
    n=n, 
    std=std,
    factor=1,
).to(device)

# torch.save(model, "./checkpoints_mhnn/model_0")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0,
)

u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()
print(np.sum(u_pred[:, 50] > 3))

plt.figure()
for j in range(n):
    plt.plot(x_test, u_pred[j, ...])
plt.ylim([-0.5, 4.5])
plt.title("Deep ensemble at 0 iteration")
plt.savefig("./outputs/mhnn/sgd_0.png")
plt.close()


def update():
    

    def loss_function():
        u = model.forward(x_f_train)
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
        loss_f = torch.mean((u_xx + 1 * torch.exp(u)) ** 2)
        # loss_f = torch.mean((u_xx + torch.exp(u)) ** 2  / torch.sum((u_xx + torch.exp(u)) ** 2, dim=1).detach()[:, None, :])
        return loss_f # + 1e-6 * torch.sum(model.theta ** 2)
    
    def closure():
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        return loss

    optimizer.step(closure)
    return loss_function()
        


def train(): 
    niter = 20000

    for i in range(niter):
        loss = update()
        
        if (i + 1) % 1000 == 0:
            model.eval()
            current_loss = loss.item()
            print(i+1, current_loss, flush=True)

            # torch.save(model, "./checkpoints_3/model_{}".format(str(i+1)))
            model.train()

            u_pred = model.forward(
                torch.tensor(x_test, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
            print(np.sum(u_pred[:, 50] > 3))

            plt.figure()
            for j in range(n):
                plt.plot(x_test, u_pred[j, ...])
            plt.ylim([-0.5, 4.5])
            plt.title("Deep ensemble at {} iteration".format(str(i+1)))
            plt.savefig("./outputs/mhnn/sgd_{}.png".format(str(i+1)))
            plt.close()


t0 = time.time()
train()
t1 = time.time()
print("Elapsed:", t1 - t0)


# torch.save(model, "./checkpoints_mhnn/model_1")
u_pred = model.forward(
    torch.tensor(x_test, dtype=torch.float32, device=device),
).detach().cpu().numpy()


plt.figure()
plt.plot(x_test, u1, "k-")
plt.plot(x_test, u2, "b-")
for i in range(n):
    plt.plot(x_test, u_pred[i, ...], "--")
plt.ylim([-0.5, 4.5])
plt.title("Deep ensemble")
plt.savefig("./outputs/sgd.png")
plt.show()

