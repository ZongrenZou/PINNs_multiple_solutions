import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import models
import utils

# np.random.seed(33321)
# torch.manual_seed(3333)
# np.random.seed(1133)
# torch.manual_seed(8888)

data = sio.loadmat("./data/data.mat")
x_test = data["x_test"]
u1 = data["u1"]
u2 = data["u2"]


device = torch.device("cuda:0")
n = 10000
x_f_train = torch.tensor(
    np.tile(x_test[None, ...], [n, 1, 1]), 
    dtype=torch.float32, 
    requires_grad=True,
).to(device)


def run(case_name, scale):
    print("*************************************")
    print(case_name, scale)
    print("*************************************")
    if case_name in ["case1", "case2", "case3", "case4", "case3_5"]:
        model = models.PNN(
            units=100, 
            n=n, 
            std=scale,
        ).to(device)
    else:
        model = models.PNN2(units=100, n=n, R=scale).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0,
    )
    

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

                # torch.save(model, "./checkpoints/model0")
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
                plt.savefig("./outputs/sgd/sgd_{}.png".format(str(i+1)))
                plt.close()

    # start training
    train()
    torch.save(model, "./checkpoints/model_different_initializations_{}_1_100_100_1".format(case_name))
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


"""
Study of different initializations for PINNs

Primary cases:
Case 1: N(0, 0.1^2)
Case 2: N(0, 0.5^2)
Case 3: N(0, 1^2)
Case 4: N(0, 2^2)
Case 5: Uniform(-1, 1)
Case 6: Uniform(-2, 2)
Case 7: Uniform(-3, 3)
Case 3_5: N(0, 1.5^2)
Case 6_5: Uniform(-2.5, 2.5)

Additional cases:
Case 8: N(0, 0.1^2) multiplied by 10
"""

# run(case_name="case1", scale=0.1)
# run(case_name="case2", scale=0.5)
# run(case_name="case3", scale=1)
# run(case_name="case4", scale=2)
# run(case_name="case5", scale=1)
# run(case_name="case6", scale=2)
# run(case_name="case7", scale=3)
run(case_name="case3_5", scale=1.5)
run(case_name="case6_5", scale=2.5)
