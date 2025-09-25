import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


def loss_function(model, x, y, noise, std):
    y_pred = model.forward(x)
    log_likeli = -torch.sum((y - y_pred) ** 2 / 2 / noise ** 2)
    log_prior = 0
    for p in model.parameters():
        log_prior += -torch.sum(p ** 2 / 2 / std ** 2)
    return -log_likeli - log_prior


def flatten(var_list):
    shapes = []
    vs = []
    n = var_list[0].shape[0]
    for v in var_list:
        shapes += [v.shape]
        vs += [v.reshape([n, -1])]
    vs = torch.concat(vs, dim=1)
    return vs, shapes


def unflatten(vs, shapes):
    var_list = []
    idx = 0
    for shape in shapes:
        length = shape[1] * shape[2]
        var_list += [vs[:, idx:idx+length].reshape(shape)]
        idx += length
    return var_list





# def plot_uq(x, y, mu, sd):
#     plt.
#     plt.plot(x, y, "k-")
#     plt.fill_between(inputs, preds_mean + preds_std,
#                      preds_mean - preds_std, color=colors[2], alpha=0.3)
#     plt.fill_between(inputs, preds_mean + 2. * preds_std,
#                      preds_mean - 2. * preds_std, color=colors[2], alpha=0.2)
#     plt.fill_between(inputs, preds_mean + 3. * preds_std,
#                      preds_mean - 3. * preds_std, color=colors[2], alpha=0.1)


        



# def loss_function(model, x, y, noise, std):
#     likeli_dist = Normal(loc=y, scale=noise)
#     prior_dist = Normal(loc=0, scale=std)
#     y_pred = model.forward(x)

#     log_likeli = torch.sum(likeli_dist.log_prob(y_pred)) / 1000
#     log_prior = 0
#     for p in model.parameters():
#         log_prior += torch.sum(prior_dist.log_prob(p)) / 1000
#     return -log_likeli - 0 * log_prior
